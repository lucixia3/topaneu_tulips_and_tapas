from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import os, tqdm, SimpleITK as sitk, json, numpy as np, random, torch
from scipy.ndimage import label
from TnT.utils.transforms import LateralityInvariance
from pprint import pprint

class TopAneuDS(Dataset):
    def __init__(self, source, transforms=None, load_type_mask=False, cases=None):
        self.src = Path(source)
        self._check_complete(cases)
        self.transforms = transforms
        self.load_tm = load_type_mask
        
    def _check_complete(self, cases=None):
        assert os.path.exists(self.src/'images') and os.path.exists(self.src/'location_jsons') and os.path.exists(self.src/'location_masks') and os.path.exists(self.src/'type_masks') and os.path.exists(self.src/'vessel_masks'), 'Could not find all required diretories'
        if cases is None: self.cases = sorted(os.listdir(self.src/'images')) # sorted for consistency
        else: self.cases = cases
        for img in self.cases:
            fn = img.replace('_0000.', '.')
            fn_json = fn.replace('.nii.gz', '.json')
            assert os.path.exists(self.src/'images'/img) and os.path.exists(self.src/'location_jsons'/fn_json) and os.path.exists(self.src/'location_masks'/fn) and os.path.exists(self.src/'type_masks'/fn) and os.path.exists(self.src/'vessel_masks'/fn), f'Could not find all required images for case {fn} -> Image: {os.path.exists(self.src/'images'/img)}, Json: {os.path.exists(self.src/'location_jsons'/fn_json)}, Location Mask: {os.path.exists(self.src/'location_masks'/fn)}, Type Mask: {os.path.exists(self.src/'type_masks'/fn)}, Vessel Mask: {os.path.exists(self.src/'vessel_masks'/fn)}'
        
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        img = self.cases[idx]
        fn = img.replace('_0000.', '.')
        fn_json = fn.replace('.nii.gz', '.json')
        
        vol = sitk.GetArrayFromImage(sitk.ReadImage(self.src/'images'/img))
        l_msk = sitk.GetArrayFromImage(sitk.ReadImage(self.src/'location_masks'/fn))
        v_msk = sitk.GetArrayFromImage(sitk.ReadImage(self.src/'vessel_masks'/fn))
        if self.load_tm: t_msk = sitk.GetArrayFromImage(sitk.ReadImage(self.src/'type_masks'/fn))
        with open(self.src/'location_jsons'/fn_json, 'r') as f:
            l_json = json.load(f)
            
        dct = {
            'image': vol,
            'location_mask': l_msk,
            'type_mask': t_msk if self.load_tm else None,
            'vessel_mask': v_msk,
            'locations': l_json['locations'],
            'modality': 'MRA' if '_mr_' in fn else 'CTA',
            'id': fn.replace('.nii.gz', '')
        }
        
        if self.transforms:
            dct = self.transforms(dct)
            
        return dct
    
class TopAneu_TnTs2_DS(Dataset):
    ########################### builtins
    def __init__(self, source, transforms=None, cases=None):
        self.image_ds = TopAneuDS(source, transforms=None, load_type_mask=False, cases=cases)
        self.aneus = None
        self.transforms = transforms
        self.encode_location = LateralityInvariance()
        
    def __len__(self):
        return len(self.aneus)
    
    def __getitem__(self, idx):
        if not self.is_patched: self.preprocess()
        smp = self.aneus[idx]
        img_smp = self.image_ds[smp['idx']]
        
        multichannel_img = np.stack( # 4D array: [C, D, H, W]
            [
                self._center_crop(img_smp['image'], smp['coords'], smp['size']),
                self._center_crop(img_smp['location_mask'], smp['coords'], smp['size']),
                self._center_crop(img_smp['vessel_mask'], smp['coords'], smp['size'])
            ], axis=0
        )
        
        coords_in_vbb = [
            smp['coords'][0]-smp['vbb'][0][0],
            smp['coords'][1]-smp['vbb'][1][0],
            smp['coords'][2]-smp['vbb'][2][0]
        ]
        
        dct = {
            'image': multichannel_img,
            'location': smp['location'], # multihot
            'coords': np.array(coords_in_vbb, dtype=int)/np.array(smp['vbb.shape'], dtype=int), # relative
            'modality': smp['modality'], # string
            'id': img_smp['id']
        }
        
        if self.transforms:
            dct = self.transforms(dct)
        
        return dct
    ###########################
    ########################### properties
    @property
    def is_patched(self):
        return self.aneus is not None
    ########################### 
    ########################### privates
    def _center_crop(self, img, coords, size):
        sz = round(size/2)
        shape = img.shape
        
        coords_lower = []
        coords_upper = []
        for i, d in enumerate(coords):
            d_size = shape[i]
            upper = d+sz
            lower = d-sz
            
            if upper<=d_size and lower >=0:
                coords_lower.append(lower)
                coords_upper.append(upper)
            elif upper > d_size:
                diff = upper-d_size
                coords_upper.append(d_size)
                coords_lower.append(lower-diff)
            elif lower < 0:
                diff = lower
                coords_lower.append(0)
                coords_upper.append(upper+abs(diff))
        
        crop = img[
            int(coords_lower[0]):int(coords_upper[0]),
            int(coords_lower[1]):int(coords_upper[1]),
            int(coords_lower[2]):int(coords_upper[2])
        ]            
        return crop
    ###########################
    ########################### publics
    def preprocess(self):
        if self.is_patched: return
        self.aneus = []
        for i in tqdm.tqdm(range(len(self.image_ds)), desc='Patching'):
            sample = self.image_ds[i]
            if not any(sample['locations']): continue
            
            cc, n = label(sample['location_mask'])
            
            vbb_coords = np.argwhere(sample['vessel_mask']) # VBB = Vessel Bounding Box
            vbb_d = [int(np.min(vbb_coords[0])), int(np.max(vbb_coords[0]))]
            vbb_h = [int(np.min(vbb_coords[1])), int(np.max(vbb_coords[1]))]
            vbb_w = [int(np.min(vbb_coords[2])), int(np.max(vbb_coords[2]))]
            
            for obj in range(1, n+1):
                smp = {
                    'idx': i, # the base image idx in the base dataset
                    'coords': np.mean(np.argwhere(cc==obj), axis=0).tolist(), # the centroid
                    'size': 64, # the size of the crop window maybe make this dynamic later on.
                    'location': self.encode_location(np.median(sample['location_mask'][cc==obj])),
                    'modality': sample['modality'],
                    'vbb': [vbb_d, vbb_h, vbb_w],
                    'vbb.shape': [int(vbb_d[1]-vbb_d[0]), int(vbb_h[1]-vbb_h[0]), int(vbb_w[1]-vbb_w[0])],
                    'id': sample['id']
                }
                self.aneus.append(smp)
        
    def split(self, folds='0.8-0.2', random_seed=42):
        folds = [float(val) for val in folds.split('-')]
        assert sum(folds)==1, 'Sum of folds must be equal to 1'
        random.seed(random_seed)
        rndm_cases = self.image_ds.cases.copy()
        random.shuffle(rndm_cases)
        n_cases = len(self.image_ds)
        fold_ds = []
        lower = 0
        for fold in folds:
            upper = round(n_cases*fold)
            upper = upper if upper <= n_cases else n_cases
            subset = rndm_cases[lower:upper]
            fold_ds.append(TopAneu_TnTs2_DS(self.image_ds.src, transforms=None, cases=subset))
        return fold_ds
    
    def save(self, path):
        saveable = {
            'cases': self.image_ds.cases,
            'source': str(self.image_ds.src),
            'aneus': self.aneus
        }
        path = str(path)+'.json' if not str(path).endswith('.json') else str(path)
        with open(path, 'w') as file:
            json.dump(saveable, file, indent=4)
        
    @staticmethod
    def load(path, transforms=None):
        path = str(path)+'.json' if not str(path).endswith('.json') else str(path)
        with open(path, 'r') as file:
            loaded = json.load(file)
        ds = TopAneu_TnTs2_DS(source=loaded['source'], transforms=transforms, cases=loaded['cases'])
        ds.aneus = loaded['aneus']
        return ds
    ###########################
    
def TnTs2_collate(batch):
    # Assumes all data is already a tensor, if not will attempt to cast to tensor
    images = []
    coords = []
    locations = []
    modalities = []
    ids = []
    for sample in batch:
        images.append(sample['image'])
        coords.append(sample['coords'])
        locations.append(sample['location'])
        modalities.append(sample['modality'])
        ids.append(sample['id'])
        
    return {
        'image': torch.stack(images, dim=0),
        'coords': torch.stack(coords, dim=0),
        'location': torch.stack(locations, dim=0),
        'modality': modalities, # just a basic list
        'id': ids # just a basic list
    }
