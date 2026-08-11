import numpy as np, json, tqdm, torch, random, copy
from pprint import pprint
import statistics
from torchvision.transforms import Normalize, Compose, InterpolationMode
from monai.transforms import Resize, Spacing
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

def get_train_test_transforms(patch_size_vx):
    test_transforms = Compose([
        MaybeToTensor(),
        MaybeResize(size=patch_size_vx),
        BinarizeAneuChannel(),
        BinarizeVesselChannel(),
        AdaNorm.make(),
    ])
    
    train_transforms = Compose([
        MaybeToTensor(),
        MaybeResize(patch_size_vx),
        BinarizeAneuChannel(),
        BinarizeVesselChannel(),
        
        # ---- Spatial stuff ----
        RandomFlipLaterality(0.5),
        
        # ---- Custom stuff ----
        RandomMask(0.2),
        RandomNonCorrespondingMask(0.2),
        RandomNonCorrespondingMorph(0.2),

        # ---- Intensity-only transforms: image channel exclusively ----
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
        AdaNorm.make(),
    ])
    
    return train_transforms, test_transforms

class BinarizeVessels():
    def __call__(self, dct):
        binarized = dct['vessel_mask']!=0
        dct['vessel_mask']=binarized.astype(np.uint8)
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
        self.convertable = ['image', 'location_mask', 'coords', 'vessel_mask', 'type_mask', 'location']
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
    def make():
        return AdaNorm(311.7783241351976, 226.50190118254997, -125.36350339401929, 647.2861290527815)
    
    @staticmethod
    def compute_mean_std(dataset):
        mr_total_sum = 0.0
        mr_total_sq_sum = 0.0
        mr_total_voxels = 0
        
        ct_total_sum = 0.0
        ct_total_sq_sum = 0.0
        ct_total_voxels = 0

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
            else: raise ValueError('Unkwown modality')

        mr_mean = mr_total_sum / mr_total_voxels
        mr_std = (mr_total_sq_sum / mr_total_voxels - mr_mean ** 2) ** 0.5
        
        ct_mean = ct_total_sum / ct_total_voxels
        ct_std = (ct_total_sq_sum / ct_total_voxels - ct_mean ** 2) ** 0.5

        with open('mean-std_mr-ct.json', 'w') as f:
            json.dump({'mr': {'mean':mr_mean, 'std':mr_std}, 'ct':{'mean':ct_mean, 'std': ct_std}}, f, indent=4)
            
        return mr_mean, mr_std, ct_mean, ct_std
    
class LateralityInvariance():
    def __init__(self):
        self.locmap = {
            0: 0,
            1: 1,
            2: 1,
            3: 2,
            4: 2,
            5: 3,
            6: 3,
            7: 4,
            8: 5,
            9: 6,
            10: 6,
            11: 7,
            12: 7,
            13: 8,
            14: 8,
            15: 9,
            16: 9,
            17: 10,
            18: 11,
            19: 11,
            20: 12,
            21: 12,
            22: 13,
            23: 13,
            24: 14,
            25: 14,
            26: 15,
            27: 15,
            28: 16,
            29: 16,
            30: 17,
            31: 17,
            32: 18,
            33: 18,
            34: 19,
            35: 19,
            36: 20,
            37: 21,
            38: 21,
            39: 22,
            40: 22,
            41: 23,
            42: 23,
            43: 24,
            44: 24,
            45: 25,
            46: 25,
            47: 26,
            48: 26,
            49: 27,
            50: 27,
            51: 28,
            52: 28
        }
        self.latmap = { # stored as [R, L]
            0: [0, 0],
            1: [1, 0],
            2: [0, 1],
            3: [1, 0],
            4: [0, 1],
            5: [1, 0],
            6: [0, 1],
            7: [0, 0],
            8: [0, 0],
            9: [1, 0],
            10: [0, 1],
            11: [1, 0],
            12: [0, 1],
            13: [1, 0],
            14: [0, 1],
            15: [1, 0],
            16: [0, 1],
            17: [0, 0],
            18: [1, 0],
            19: [0, 1],
            20: [1, 0],
            21: [0, 1],
            22: [1, 0],
            23: [0, 1],
            24: [1, 0],
            25: [0, 1],
            26: [1, 0],
            27: [0, 1],
            28: [1, 0],
            29: [0, 1],
            30: [1, 0],
            31: [0, 1],
            32: [1, 0],
            33: [0, 1],
            34: [1, 0],
            35: [0, 1],
            36: [0, 0],
            37: [1, 0],
            38: [0, 1],
            39: [1, 0],
            40: [0, 1],
            41: [1, 0],
            42: [0, 1],
            43: [1, 0],
            44: [0, 1],
            45: [1, 0],
            46: [0, 1],
            47: [1, 0],
            48: [0, 1],
            49: [1, 0],
            50: [0, 1],
            51: [1, 0],
            52: [0, 1],
        }
        self.n_locs, self.n_lats = self.get_n_locs_lats()
        
    def __call__(self, dct):
        if isinstance(dct, dict):
            loc = self.locmap[dct['location']]
            lat = self.latmap[dct['location']]
            hot = [0]*self.n_locs
            hot[loc]=1 # offset due to indexing
            
            return {
                        "aneurysm": hot,
                        "vessel": [],
                        "laterality": lat
                    } 
        else:
            loc = self.locmap[dct]
            lat = self.latmap[dct]
        
            hot = [0]*self.n_locs
            hot[loc]=1
            
            return {
                        "aneurysm": hot,
                        "vessel": [],
                        "laterality": lat
                    } 
        
    @staticmethod
    def get_n_locs_lats():
        return 29, 2
        return 29, 2
class DecodeTarget():
    def __init__(self):
        self.map = {
            0:0,
            1:1,
            2:3,
            3:5, 
            4:7,
            5:8,
            6:9,
            7:11,
            8:13,
            9:15,
            10:17,
            11:18,
            12:20,
            13:22,
            14:24,
            15:26,
            16:28,
            17:30,
            18:32,
            19:34,
            20:36,
            21:37,
            22:39,
            23:41,
            24:43,
            25:45,
            26:47,
            27:49,
            28:51,
            28:51
        }
        self.excl_from_lat = [0, 7, 8, 17, 36]
        self.lit_loc_lookup = {
            0: 'background',
            1: "1.1 VA trunk",
            2: "1.2 PICA trunk",
            3: "1.3 VA-PICA junction",
            4: "1.4 BA trunk",
            5: "1.5 VA-BA junction",
            6: "1.6 AICA trunk",
            7: "1.7 BA-AICA junction",
            8: "1.8 SCA trunk",
            9: "1.9 BA-SCA junction",
            10: "1.10 BA tip",
            11: "2.1 P1P2",
            12: "2.2 P3P4",
            13: "3.1 ICA infraclinoid C1-C5",
            14: "3.2 ICA C6-OA-junction",
            15: "3.3 ICA C6-nonOA",
            16: "3.4 ICA C7-Pcom-junction",
            17: "3.5 ICA C7-AChA-junction",
            18: "3.6 ICA C7-nonBranch",
            19: "3.7 ICA C7-terminus",
            20: "4.1 Acom complex",
            21: "4.2 A1",
            22: "4.3 A2",
            23: "4.4 A3",
            24: "4.5 Distal ACA branches",
            25: "5.1 M1 trunk",
            26: "5.2 M1 early bifurcation",
            27: "5.3 M1-M2 junction",
            28: "5.4 Distal-M2M3",
        }
    
    def __call__(self, obj):
        if len(obj.shape)==1:
            return self._conv_row(obj)
        elif len(obj.shape)==2:
            classes = []
            for i in range(obj.shape[0]):
                classes.append(self._conv_row(obj[i]))
            return classes
        else: raise RuntimeError(f'Expected object to have 1 dimension if unbatched or 2 dimensions if batched, but received {len(obj.shape)} dimensions instead')
            
    def _conv_row(self, row):
        loc_28, prob_loc = max(enumerate(row[:29]), key=lambda x: x[1])
        loc_50 = self.map[loc_28]
        lat, prob_lat = max(enumerate(row[29:]), key=lambda x: x[1])
        
        if loc_50 not in self.excl_from_lat: # should be redundant as the model should learn not to assign laterality in these cases.
            loc_50 += lat
        
        loc_lit, lat_lit = self._to_literal(loc_28, lat)
        
        return loc_lit, lat_lit, loc_50
    
    def _to_literal(self, loc, lat):
        if loc not in self.excl_from_lat:
            lit_lat = 'R' if lat == 0 else 'L'
        else: lit_lat = 'N/A'
        lit_loc = self.lit_loc_lookup[loc]
        return lit_loc, lit_lat
    
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
        
class LateralityInvarianceForVessels():
    def __init__(self):
        self.locmap = {
            0:0,
            1: 1, # na
            2: 2, # r
            3: 2, # l
            4: 3,  # r
            6: 3, # l
            5: 4, # r
            7: 4, # l
            8: 5, # r
            9: 5, # l
            10: 6, # na
            11: 7, # r
            12: 7, # l
            13: 8, # r
            14: 8, # l
            15: 9, # na
            16: 10, # na
            17: 11, # r
            19: 11, # l
            18: 12, # r
            20: 12, # l
            21: 13,
            22: 13,
            23: 14,
            24: 14,
            25: 15,
            26: 15,
            27: 16,
            28: 16,
            29: 17,
            30: 17,
            31: 18,
            32: 18,
            33: 19,
            34: 19,
            35: 20,
            36: 20
        }
        self.latmap = { # encoded [R, L]
            0: [0,0],
            1: [0,0], # na
            2: [1,0], # r
            3: [0,1], # l
            4: [1,0],  # r
            6: [0,1], # l
            5: [1,0], # r
            7: [0,1], # l
            8: [1,0], # r
            9: [0,1], # l
            10: [0,0], # na
            11: [1,0], # r
            12: [0,1], # l
            13: [1,0], # r
            14: [0,1], # l
            15: [0,0], # na
            16: [0,0], # na
            17: [1,0], # r
            19: [0,1], # l
            18: [1,0], # r
            20: [0,1], # l
            21: [1,0],
            22: [0,1],
            23: [1,0],
            24: [0,1],
            25: [1,0],
            26: [0,1],
            27: [1,0],
            28: [0,1],
            29: [1,0],
            30: [0,1],
            31: [1,0],
            32: [0,1],
            33: [1,0],
            34: [0,1],
            35: [1,0],
            36: [0,1]
        }
        self.n_locs_a, self.n_locs_v, self.n_lats = self.get_n_locs_lats()
        self.associated_aneus = {
                    0:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    1:[0,0,0,0,1,1,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    2:[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    3:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
                    4:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
                    5:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                    6:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                    7:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0],
                    8:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                    9:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                    10:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                    11:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                    12:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                    13:[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    14:[0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    15:[0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    16:[0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    17:[0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    18:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0],
                    19:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    20:[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                }
        
    def __call__(self, dct):
        
        if isinstance(dct, dict):
            loc = self.locmap[dct['location']]
            lat = self.latmap[dct['location']]
            hot = [0]*self.n_locs_v
            hot[loc]=1 
            
            loc_dct = {
                        "aneurysm": self.associated_aneus[loc],
                        "vessel": hot,
                        "laterality": lat
                    }
            return loc_dct
        else:
            loc = self.locmap[dct]
            lat = self.latmap[dct]
        
            hot = [0]*self.n_locs_v
            hot[loc]=1
            
            loc_dct = {
                        "aneurysm": self.associated_aneus[loc],
                        "vessel": hot,
                        "laterality": lat
                    } 
            return loc_dct
        
    @staticmethod
    def get_n_locs_lats():
        return 28, 21, 2
    
class DecodeTargetForVessels():
    def __init__(self):
        self.map = {
            0:'background',
            1:'BA',
            2:'P1P2',
            3:'ICA-C6-C7', 
            4:'M1',
            5:'Pcom',
            6:'Acom',
            7:'A1A2',
            8:'A3',
            9:'3rd-A2',
            10:'3rd-A3',
            11:'M2',
            12:'M3',
            13:'P3P4',
            14:'VA',
            15:'SCA',
            16:'AICA',
            17:'PICA',
            18:'AChA',
            19:'OA',
            20:'ICA-C1-C5'
        }
        self.lit_loc_lookup = {
            "background": 0,
            "BA": 1,
            "R-P1P2": 2,
            "L-P1P2": 3,
            "R-ICA-C6-C7": 4,
            "R-M1": 5,
            "L-ICA-C6-C7": 6,
            "L-M1": 7,
            "R-Pcom": 8,
            "L-Pcom": 9,
            "Acom": 10,
            "R-A1A2": 11,
            "L-A1A2": 12,
            "R-A3": 13,
            "L-A3": 14,
            "3rd-A2": 15,
            "3rd-A3": 16,
            "R-M2": 17,
            "R-M3": 18,
            "L-M2": 19,
            "L-M3": 20,
            "R-P3P4": 21,
            "L-P3P4": 22,
            "R-VA": 23,
            "L-VA": 24,
            "R-SCA": 25,
            "L-SCA": 26,
            "R-AICA": 27,
            "L-AICA": 28,
            "R-PICA": 29,
            "L-PICA": 30,
            "R-AChA": 31,
            "L-AChA": 32,
            "R-OA": 33,
            "L-OA": 34,
            "R-ICA-C1-C5": 35,
            "L-ICA-C1-C5": 36
        }
        self.rev_map = {v:k for k, v in self.map.items()}
        self.ignore_lat = ['background', 'BA', 'Acom', '3rd-A2', '3rd-A3']
        
    def __call__(self, obj):
        if len(obj.shape)==1:
            return self._conv_row(obj)
        elif len(obj.shape)==2:
            classes = []
            for i in range(obj.shape[0]):
                classes.append(self._conv_row(obj[i]))
            return classes
        else: raise RuntimeError(f'Expected object to have 1 dimension if unbatched or 2 dimensions if batched, but received {len(obj.shape)} dimensions instead')
            
    def _conv_row(self, row):
        loc_20, prob_loc = max(enumerate(row[:21]), key=lambda x: x[1])
        loc_20_lit = self.map[loc_20]
        lat, prob_lat = max(enumerate(row[21:]), key=lambda x: x[1])
        lat_lit = ['R', 'L'][lat]

        if loc_20_lit not in self.ignore_lat:
            loc_lit = lat_lit+'-'+loc_20_lit
        else: loc_lit = loc_20_lit
        
        return loc_lit, lat_lit, self.lit_loc_lookup[loc_lit]

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
            if dct['image'].dim() == 3:
                dct['image'] = torch.flip(dct['image'], dims=[self.aterality_dimension])
                dct['location']['laterality'] = self._flip_laterality(dct['location']['laterality'])
            else:
                dct['image'] = torch.flip(dct['image'], dims=[self.channels_laterality_dimension])
                dct['location']['laterality'] = self._flip_laterality(dct['location']['laterality'])
            return dct
        else: return dct