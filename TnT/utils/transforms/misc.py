import numpy as np, json, tqdm, torch, random, copy
from pprint import pprint
import statistics
from torchvision.transforms import Normalize, Compose, InterpolationMode
from monai.transforms import Resize, Spacing
from TnT.utils.transforms import LabelEncoder
from monai.transforms import (
    RandAffined,
    RandAffine,
    RandFlip,
    RandRotate90,
    RandSpatialCrop,
    RandGaussianNoise,
    RandAdjustContrast,
    RandGaussianSmooth,
    RandScaleIntensity,
    RandShiftIntensity,
    RandBiasField,
    RandHistogramShift,
    NormalizeIntensity,
)
from monai.data import MetaTensor
from scipy.ndimage import binary_dilation, binary_erosion, grey_dilation, grey_erosion, generate_binary_structure

def get_inference_transforms(patch_size_vx):
    return Compose([
            MaybeToTensor(),
            ClipCtaIntensities(),
            MaybeResize(size=patch_size_vx),
            BinarizeAneuChannel(),
            BinarizeVesselChannel(),
            AdaNorm.make(ct_clipped=True),
        ])

def get_train_test_transforms(patch_size_vx):
    test_transforms = Compose([
        LabelEncoder(),
        MaybeToTensor(),
        ClipCtaIntensities(),
        MaybeResize(size=patch_size_vx),
        BinarizeAneuChannel(),
        BinarizeVesselChannel(),
        AdaNorm.make(ct_clipped=True),
    ])
    
    train_transforms = Compose([
        LabelEncoder(),
        MaybeToTensor(),
        ClipCtaIntensities(),
        MaybeResize(patch_size_vx),
        BinarizeAneuChannel(),
        BinarizeVesselChannel(),
        
        # ---- Spatial stuff ----
        RandomFlipLaterality(0.5),
        
        NoisyCoordinates(0, 0.2, 0.9),
        
        # ---- Custom stuff ----
        RandomMask(0.2),
        RandomNonCorrespondingMask(0.2),
        RandomNonCorrespondingMorph(0.2),

        # ---- Intensity-only transforms: image channel exclusively ----
        AdaNorm.make(ct_clipped=True),
        ImageTransformWrapper(
            RandGaussianNoise(prob=0.2, mean=0.0, std=0.05),
            apply_to=['image']
        ),
        ImageTransformWrapper(
            RandGaussianSmooth(prob=0.15, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
            apply_to=['image']
        ),
        ImageTransformWrapper(
            RandAdjustContrast(prob=0.2, gamma=(0.7, 1.5)),
            apply_to=['image']
        ),
        ImageTransformWrapper(
            RandScaleIntensity(prob=0.2, factors=0.1),
            apply_to=['image']
        ),
        ImageTransformWrapper(
            RandShiftIntensity(prob=0.2, offsets=0.1),
            apply_to=['image']
        ),
        ImageTransformWrapper(
            RandBiasField(prob=0.1, coeff_range=(0.0, 0.3)),
            apply_to=['image']
        ),
        ImageTransformWrapper(
            RandHistogramShift(prob=0.1, num_control_points=(3, 5)),
            apply_to=['image']
        ),
        
    ])
    
    return train_transforms, test_transforms

class BinarizeVessels():
    def __call__(self, dct):
        binarized = dct['vessel_mask']!=0
        dct['vessel_mask']=binarized.astype(np.uint8)
        return dct
    
class ClipCtaIntensities():
    def __init__(self, lower=-200, upper=800):
        self.lower=lower
        self.upper=upper
        
    def __call__(self, dct):
        if dct['modality']=='CTA':
            dct['image'][0] = torch.clip(dct['image'][0], self.lower, self.upper)
        return dct
    
class BinarizeAneus():
    def __call__(self, dct):
        binarized = dct['location_mask']!=0
        dct['location_mask']=binarized.astype(np.uint8)
        return dct
    
class BinarizeVesselChannel():
    def __call__(self, dct):
        dct['image'][2, :] = (dct['image'][2, :]!=0).to(torch.float32)
        return dct
    
class BinarizeAneuChannel():
    def __call__(self, dct):
        dct['image'][1, :] = (dct['image'][1, :]!=0).to(torch.float32)
        return dct
    
class MaybeToTensor():
    def __init__(self, dtype=torch.float32):
        self.convertable = ['image', 'location_mask', 'coords', 'vessel_mask', 'type_mask', 'location_a', 'location_v', 'laterality']
        self.dtype = dtype
        
    def __call__(self, dct):
        for k, v in dct.items():
            if k == "location": dct[k] = {kk:torch.tensor(vv, dtype=self.dtype) for kk, vv in v.items()}
            elif k in self.convertable:
                if isinstance(v, np.ndarray): dct[k] = torch.from_numpy(v).to(self.dtype)
                else: dct[k] = torch.tensor(v).to(self.dtype)
        return dct
    
class MaybeResize():
    def __init__(self, size=64):
        spatial_size = (size, size, size)
        self.msk_resize = Resize(spatial_size=spatial_size, mode='nearest')
        self.img_resize = Resize(spatial_size=spatial_size, mode='trilinear', anti_aliasing=True)
        self.convertable = ['image', 'location_mask', 'vessel_mask', 'type_mask']
        self.mask_keys = ['location_mask', 'vessel_mask', 'type_mask']

    def __call__(self, dct):
        for k, v in dct.items():
            if k not in self.convertable:
                continue

            if k == 'image' and v.dim() == 4:
                # channel-first [3, D, H, W]: raw image + aneu mask + vessel mask
                c0 = self.img_resize(v[0].unsqueeze(0)).squeeze(0)
                c1 = self.msk_resize(v[1].unsqueeze(0)).squeeze(0)
                c2 = self.msk_resize(v[2].unsqueeze(0)).squeeze(0)
                dct[k] = torch.stack([c0, c1, c2], dim=0)
            elif k in self.mask_keys:
                dct[k] = self.msk_resize(v.unsqueeze(0)).squeeze(0)
            elif v.dim() == 3:
                dct[k] = self.img_resize(v.unsqueeze(0)).squeeze(0)
            else:
                raise RuntimeError(f"MaybeResize: unexpected shape {tuple(v.shape)} for key '{k}'")

        return dct
    
class AdaNorm():
    def __init__(self, mr_mean, mr_std, ct_mean, ct_std):
        self.mr_mean, self.mr_std = mr_mean, mr_std
        self.ct_mean, self.ct_std = ct_mean, ct_std

    def __call__(self, dct):
        if dct['modality'] == 'CTA':
            mean, std = self.ct_mean, self.ct_std
        elif dct['modality'] == 'MRA':
            mean, std = self.mr_mean, self.mr_std
        else:
            raise ValueError('Unknown modality')

        if dct['image'].dim() == 3:
            dct['image'] = (dct['image'] - mean) / std
        else:
            dct['image'][0, :] = (dct['image'][0, :] - mean) / std

        return dct
    
    @staticmethod
    def make(ct_clipped=True):
        if ct_clipped: return AdaNorm(105.64747174811804, 128.1358337638573, 63.98932940740848, 279.77159725605)
        else: return AdaNorm(105.64747174811804, 128.1358337638573, -111.44454449849296, 648.8878738938635)
    
    @staticmethod
    def compute_mean_std(dataset):
        mr_total_sum = 0.0
        mr_total_sq_sum = 0.0
        mr_total_voxels = 0
        
        ct_total_sum = 0.0
        ct_total_sq_sum = 0.0
        ct_total_voxels = 0
        
        ct_clipped_sum = 0.0
        ct_clipped_sq_sum = 0.0
        ct_clipped_voxels = 0

        for idx in tqdm.tqdm(range(len(dataset)), 'Comp mean-std'):
            # if dataset returns (volume, mask) or dicts, adjust this line accordingly
            vol = dataset[idx]['image']
            mod = dataset[idx]['modality']

            if mod == 'MRA':
                mr_total_sum += vol.sum().item()
                mr_total_sq_sum += (vol ** 2).sum().item()
                mr_total_voxels += vol.size
            elif mod == 'CTA':
                ct_total_sum += vol.sum().item()
                ct_total_sq_sum += (vol ** 2).sum().item()
                ct_total_voxels += vol.size
                
                vol = np.clip(vol, -200, 800)
                ct_clipped_sum += vol.sum().item()
                ct_clipped_sq_sum += (vol ** 2).sum().item()
                ct_clipped_voxels += vol.size
            else: raise ValueError('Unkwown modality')

        mr_mean = mr_total_sum / mr_total_voxels
        mr_std = (mr_total_sq_sum / mr_total_voxels - mr_mean ** 2) ** 0.5
        
        ct_tot_mean = ct_total_sum / ct_total_voxels
        ct_tot_std = (ct_total_sq_sum / ct_total_voxels - ct_tot_mean ** 2) ** 0.5
        
        ct_clp_mean = ct_clipped_sum / ct_clipped_voxels
        ct_clp_std = (ct_clipped_sq_sum / ct_clipped_voxels - ct_clp_mean ** 2) ** 0.5

        with open('mean-std_mr-ct.json', 'w') as f:
            json.dump({'mr': {'mean':mr_mean, 'std':mr_std}, 'ct_total':{'mean':ct_tot_mean, 'std': ct_tot_std}, 'ct_clipped':{'mean':ct_clp_mean, 'std': ct_clp_std}}, f, indent=4)
    
class ImageTransformWrapper():
    """Wrapper to allow usage of monai transforms on our data structure
    """
    def __init__(self, trans, apply_to=['image', 'aneu', 'vessel']):
        self.trans = trans
        self.map = {'image':0, 'aneu':1, 'vessel':2}
        self.apply_to = apply_to
    
    def __call__(self, dct):
        if self.apply_to == 'all':
            dct['image'] = self.trans(dct['image'])
        else:
            for channel in self.apply_to:
                idx = self.map[channel]
                dct['image'][idx] = self.trans(dct['image'][idx])
        return dct
    
    def __repr__(self):
        return f'TnT.utils.transforms.ImageTransformWrapper object wrapping {self.trans.__repr__()} onto channels {self.apply_to}'
    
class RandomMask():
    """Mask out random regions with correspondence across channels, if object is a dict with a 4D image, else mask out random region in 3D tensor
    """
    def __init__(self, prob):
        self.prob=prob
        
    @property
    def execute(self):
        return random.choices([True, False], weights=[self.prob, 1-self.prob], k=1)[0]
    
    def _gen_random_msk(self, shape):
        centers = [random.choice(range(s)) for s in shape]
        shapes = [round(random.choice(range(s-c))/2) for s, c in zip(shape, centers)]
        msk = torch.zeros(shape, dtype=torch.bool)
        msk[centers[0]-shapes[0]:centers[0]+shapes[0], centers[1]-shapes[1]:centers[1]+shapes[1], centers[2]-shapes[2]:centers[2]+shapes[2]]=True
        return msk
        
    def __call__(self, dct):
        if not self.execute: return dct
        if isinstance(dct, torch.Tensor):
            assert len(dct.shape)==3, 'TnT.utils.transforms.RandomMask only supports execution on 3D tensors or dicts'
            dct[self._gen_random_msk(dct.shape)]=0
            return dct
        elif isinstance(dct, dict):
            assert len(dct['image'].shape)==4, 'TnT.utils.transforms.RandomMask only supports execution on 3D tensors or dicts'
            msk = self._gen_random_msk(dct['image'].shape[1:])
            for i in range(3):
                dct['image'][i, msk]=0
            return dct
        else: raise AssertionError('TnT.utils.transforms.RandomMask only supports execution on 3D tensors or dicts')
        
        
class RandomNonCorrespondingMask():
    """Mask out random regions, without correspondence across channels
    """
    def __init__(self, prob):
        self.prob=prob
        self.trans = RandomMask(prob)
        
    @property
    def execute(self):
        return random.choices([True, False], weights=[self.prob, 1-self.prob], k=1)[0]
        
    def __call__(self, dct):
        if not self.execute: return dct
        assert isinstance(dct, dict), 'TnT.utils.transforms.RandomNonCorrespondingMask only supports execution on dicts with 4D images'
        assert len(dct['image'].shape)==4, 'TnT.utils.transforms.RandomNonCorrespondingMask only supports execution on dicts with 4D images'
        for i in range(3):
            dct['image'][i, :] = self.trans(dct['image'][i, :])
        return dct
    
class RandomNonCorrespondingMorph():
    def __init__(self, prob, operator_diameter=[3, 5, 7]):
        self.prob = prob
        self.operator_diameter = operator_diameter
        
    @property
    def execute(self):
        return random.choices([True, False], weights=[self.prob, 1-self.prob], k=1)[0]
    
    @property
    def method(self):
        return random.choice(['erosion', 'dilation'])
    
    @property
    def struct(self):
        return generate_binary_structure(rank=3, connectivity=random.choice(self.operator_diameter) if isinstance(self.operator_diameter, list) else self.operator_diameter)
    
    def _apply_grey_morph(self, tensor):
        if self.execute:
            arr = tensor.numpy()
            if self.method == 'erosion': arr = grey_erosion(arr, structure=self.struct)
            elif self.method == 'dilation': arr = grey_dilation(arr, structure=self.struct)
            return torch.from_numpy(arr)
        else: return tensor
        
    def _apply_bin_morph(self, tensor):
        if self.execute:
            arr = tensor.numpy()
            if self.method == 'erosion': arr = binary_erosion(arr, structure=self.struct)
            elif self.method == 'dilation': arr = binary_dilation(arr, structure=self.struct)
            return torch.from_numpy(arr)
        else: return tensor
    
    def __call__(self, dct):
        if not self.execute: return dct
        assert isinstance(dct, dict), 'TnT.utils.transforms.RandomNonCorrespondingMask only supports execution on dicts with 4D images'
        assert len(dct['image'].shape)==4, 'TnT.utils.transforms.RandomNonCorrespondingMask only supports execution on dicts with 4D images'
        dct['image'][0, :] = self._apply_grey_morph(dct['image'][0, :])
        dct['image'][1, :] = self._apply_bin_morph(dct['image'][1, :])
        dct['image'][2, :] = self._apply_bin_morph(dct['image'][2, :])
        return dct
    
class Resample(): # to median
    def __init__(self, spacing):
        self.spacing = spacing
        self.msk_resampler = Spacing(spacing, mode="nearest")
        self.img_resampler = Spacing(spacing, mode="bilinear")
        
    def __call__(self, dct):
        img, spacing = dct["image"], dct['spacing']
        affine = torch.diag(torch.tensor([*spacing, 1.0], dtype=torch.float64))
        if len(img.shape)==4:
            c0 = self.img_resampler(MetaTensor(img[0, :].unsqueeze(0), affine=affine)).squeeze(0)
            c1 = self.msk_resampler(MetaTensor(img[1, :].unsqueeze(0), affine=affine)).squeeze(0)
            c2 = self.msk_resampler(MetaTensor(img[2, :].unsqueeze(0), affine=affine)).squeeze(0)
            img = torch.stack([c0, c1, c2], dim=0)
            dct["image"]=img
            dct["spacing"]=self.spacing
        else:
            img = self.img_resampler(MetaTensor(img.unsqueeze(0), affine=affine)).squeeze(0)
            dct["image"]=img
            dct["spacing"]=self.spacing

        return dct
    
    @staticmethod
    def make():
        return Resample((0.38999998569488525, 0.39000001549720764, 0.25004658102989197))
    
    @staticmethod
    def compute_median(dataset):
        spaces_d = []
        spaces_h = []
        spaces_w = []
        for i in range(len(dataset)):
            d, h, w = dataset[i]["spacing"]
            spaces_d.append(d)
            spaces_h.append(h)
            spaces_w.append(w)
        return statistics.median(spaces_d), statistics.median(spaces_h), statistics.median(spaces_w)
class RandomResample():
    def __init__(self, prob=0.8):
        self.resampler = Resample.make()
        self.prob = prob
    
    @property
    def execute(self):
        return random.choices([True, False], weights=[self.prob, 1-self.prob], k=1)[0]
    
    def __call__(self, dct):
        if self.execute: return self.resampler(dct)
        else: return dct

class RandomFlipLaterality():
    def __init__(self, prob):
        self.prob = prob
        # 4D array: [C, H, D, W]
        self.channels_laterality_dimension = 3
        self.laterality_dimension = 2
        
    @property
    def execute(self):
        return random.choices([True, False], weights=[self.prob, 1-self.prob], k=1)[0]
    
    def _flip_laterality(self, obj):
        tmp = obj[-2].clone() if torch.is_tensor(obj) else obj[-2]
        obj[-2] = obj[-1]
        obj[-1] = tmp
        return obj
    
    def __call__(self, dct):
        if self.execute: 
            dct['coords'][2] = 1-dct['coords'][2]
            if dct['image'].dim() == 3:
                dct['image'] = torch.flip(dct['image'], dims=[self.laterality_dimension])
                dct['laterality'] = self._flip_laterality(dct['laterality'])
            
            else:
                dct['image'] = torch.flip(dct['image'], dims=[self.channels_laterality_dimension])
                dct['laterality'] = self._flip_laterality(dct['laterality'])
            return dct
        else: return dct

class NoisyCoordinates():
    def __init__(self, mean=0, std=0.2, prob=0.9):
        self.mean = mean
        self.std = std
        self.prob = prob
        
    @property
    def execute(self):
        return random.choices([True, False], weights=[self.prob, 1-self.prob], k=1)[0]
    
    def __call__(self, dct):
        if self.execute:
            dct["coords"]= torch.clip(dct["coords"]+torch.randn_like(dct["coords"]) * self.std + self.mean, 0, 1)
        return dct