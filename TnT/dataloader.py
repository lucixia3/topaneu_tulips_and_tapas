from torch.utils.data import DataLoader
from pathlib import Path
import os, tqdm, SimpleITK as sitk, json, numpy as np
from scipy.ndimage import label
from transforms import LateralityInvariance

class TopAneuDS(DataLoader):
    def __init__(self, source, transforms=None, load_type_mask=False):
        self.src = Path(source)
        self._check_complete()
        self.transforms = transforms
        self.load_tm = load_type_mask
        
    def _check_complete(self):
        assert os.path.exists(self.src/'images') and os.path.exists(self.src/'location_jsons') and os.path.exists(self.src/'location_masks') and os.path.exists(self.src/'type_masks') and os.path.exists(self.src/'vessel_masks'), 'Could not find all required diretories'
        self.cases = sorted(os.listdir(self.src)) # sorted for consistency
        for img in tqdm.tqdm(self.cases):
            fn = img.replace('_0000.', '.')
            fn_json = fn.replace('.nii.gz', 'json')
            assert os.path.exists(self.src/'images'/img) and os.path.exists(self.src/'location_jsons'/fn_json) and os.path.exists(self.src/'location_masks'/fn) and os.path.exists(self.src/'type_masks'/fn) and os.path.exists(self.src/'vessel_masks'/fn), f'Could not find all required images for case {fn}'
        
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        img = self.cases[idx]
        fn = img.replace('_0000.', '.')
        fn_json = fn.replace('.nii.gz', 'json')
        
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
            'modality': 'MRA' if '_mr_' in fn else 'CTA'
        }
        
        if self.transforms:
            dct = self.transforms(dct)
            
        return dct
    
class TopAneu_TnTs2_DS(DataLoader):
    def __init__(self, source, transforms=None):
        self.image_ds = TopAneuDS(source, transforms=None, load_type_mask=False)
        self._extract_aneus()
        self.transforms = transforms
        self.encode_location = LateralityInvariance()
        
    def __len__(self):
        return len(self.aneus)
    
    def _extract_aneus(self):
        self.aneus = []
        for i in range(len(self.image_ds)):
            sample = self.image_ds[i]
            if not any(sample['locations']): continue
            
            cc, n = label(sample['location_mask'])
            
            for obj in range(1, n+1):
                smp = {
                    'idx': i, # the base image idx in the base dataset
                    'coords': np.mean(np.argwhere(cc==obj), axis=0), # the centroid
                    'size': 64, # the size of the crop window maybe make this dynamic later on.
                    'locations': self.encode_location(np.median(sample['location_mask'][cc==obj])),
                    'modality': sample['modality']
                }
                self.aneus.append(smp)
                
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
            coords_lower[0]:coords_upper[0],
            coords_lower[1]:coords_upper[1],
            coords_lower[2]:coords_upper[2]
        ]            
        return crop
    
    def __getitem__(self, idx):
        smp = self.aneus[idx]
        img_smp = self.image_ds[smp['idx']]
        
        dct = {
            'image': self._center_crop(img_smp['image'], smp['coords'], smp['size']),
            'location_mask': self._center_crop(img_smp['location_mask'], smp['coords'], smp['size'])!=0, # binarized
            'vessel_mask': self._center_crop(img_smp['vessel_mask'], smp['coords'], smp['size'])!=0, # binarized
            'location': smp['location'], # multihot
            'coords': smp['coords']/img_smp['image'].shape, # relative
            'modality': smp['modality'] # string
        }
        
        if self.transforms:
            dct = self.transforms(dct)
        
        return dct