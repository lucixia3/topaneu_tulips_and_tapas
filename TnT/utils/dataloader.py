from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import os, tqdm, SimpleITK as sitk, json, numpy as np, random, torch, copy
from scipy.ndimage import label, binary_erosion
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
        
        sitk_im = sitk.ReadImage(self.src/'images'/img)
        vol = sitk.GetArrayFromImage(sitk_im)
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
            'location': l_json['locations'],
            'modality': 'MRA' if '_mr_' in fn else 'CTA',
            'id': fn.replace('.nii.gz', ''),
            'spacing': sitk_im.GetSpacing()
        }
        
        if self.transforms:
            dct = self.transforms(dct)
            
        return dct
    
    @staticmethod
    def load(path, transforms=None):
        path = str(path)+'.json' if not str(path).endswith('.json') else str(path)
        with open(path, 'r') as file:
            loaded = json.load(file)
        ds = TopAneuDS(source=loaded['source'], transforms=transforms, cases=loaded['cases'])
        return ds
    
class TopAneu_TnTs2_DS(Dataset):
    ########################### builtins
    def __init__(self, source, transforms=None, cases=None, patch_size_mm=50):
        self.image_ds = TopAneuDS(source, transforms=None, load_type_mask=False, cases=cases)
        self.aneus = []
        self.transforms = transforms
        self.patch_size_mm = patch_size_mm
        
    def __len__(self):
        return len(self.aneus)
    
    def __getitem__(self, idx):
        if not self.is_patched: self.preprocess()
        smp = self.aneus[idx]
        img_smp = self.image_ds[smp['idx']]
        
        multichannel_img = np.stack( # 4D array: [C, H, D, W]
            [
                self._center_crop(img_smp['image'], smp['coords'], img_smp['spacing'], self.patch_size_mm),
                self._center_crop(img_smp['location_mask'], smp['coords'], img_smp['spacing'], self.patch_size_mm),
                self._center_crop(img_smp['vessel_mask'], smp['coords'], img_smp['spacing'], self.patch_size_mm)
            ], axis=0
        )
        
        if smp['location']['aneurysm'][0]==1: # if it is one the bg patches need to gen a random sphere
            multichannel_img = self._put_random_sphere_as_aneu(multichannel_img, img_smp["spacing"])
        
        coords_in_vbb = [
            smp['coords'][0]-smp['vbb'][0][0],
            smp['coords'][1]-smp['vbb'][1][0],
            smp['coords'][2]-smp['vbb'][2][0]
        ]
        
        dct = {
            'image': multichannel_img,
            'location_v': [lbl for lbl in np.unique(multichannel_img[2]).tolist() if lbl != 0],
            'location_a': smp['location'], # multihot
            'coords': np.array(coords_in_vbb, dtype=int)/np.array(smp['vbb.shape'], dtype=int), # relative
            'modality': smp['modality'], # string
            'id': img_smp['id'],
            'spacing': img_smp['spacing']
        }
        
        if self.transforms:
            dct = self.transforms(dct)
        
        return dct
    ###########################
    ########################### properties
    @property
    def is_patched(self):
        return len(self.aneus)>0
    ########################### 
    ########################### privates
    def _center_crop(self, img, coords, spacing, size_mm):
        sz_mm = round(size_mm/2)
        sz = [round(sz_mm/sp) for sp in spacing]
        shape = img.shape
        
        coords_lower = []
        coords_upper = []
        for c_ax, sh_ax, sz_ax in zip(coords, shape, sz):

            upper = c_ax+sz_ax
            lower = c_ax-sz_ax
            
            if upper<=sh_ax and lower >=0:
                coords_lower.append(lower)
                coords_upper.append(upper)
            elif upper > sh_ax:
                diff = upper-sh_ax
                coords_upper.append(sh_ax)
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
    
    def _center_mask_out(self, img, coords, spacing, size_mm):
        sz_mm = round(size_mm/2)
        sz = [round(sz_mm/sp) for sp in spacing]
        shape = img.shape
        
        coords_lower = []
        coords_upper = []
        for c_ax, sh_ax, sz_ax in zip(coords, shape, sz):

            upper = c_ax+sz_ax
            lower = c_ax-sz_ax
            
            if upper<=sh_ax and lower >=0:
                coords_lower.append(lower)
                coords_upper.append(upper)
            elif upper > sh_ax:
                diff = upper-sh_ax
                coords_upper.append(sh_ax)
                coords_lower.append(lower-diff)
            elif lower < 0:
                diff = lower
                coords_lower.append(0)
                coords_upper.append(upper+abs(diff))
        
        img[
            int(coords_lower[0]):int(coords_upper[0]),
            int(coords_lower[1]):int(coords_upper[1]),
            int(coords_lower[2]):int(coords_upper[2])
        ] = 0      
        return img
    
    def _make_bg_patches(self, n):
        bg_patches = []
        for i in tqdm.tqdm(range(n), desc='Generating random bg patches'):
            rndm_img_idx = random.choice(range(len(self)))
            cur_smp = self.aneus[rndm_img_idx]
            img_smp = self.image_ds[cur_smp['idx']]
            masked = self._center_mask_out(copy.deepcopy(img_smp['vessel_mask']), cur_smp['coords'], img_smp['spacing'], self.patch_size_mm)
            
            possible_seeds = np.argwhere(masked)
            
            seed = possible_seeds[random.choice(range(possible_seeds.shape[0])), :]
            
            vbb_coords = np.argwhere(img_smp['vessel_mask']) # VBB = Vessel Bounding Box
            vbb_d = [int(np.min(vbb_coords[0])), int(np.max(vbb_coords[0]))]
            vbb_h = [int(np.min(vbb_coords[1])), int(np.max(vbb_coords[1]))]
            vbb_w = [int(np.min(vbb_coords[2])), int(np.max(vbb_coords[2]))]
            
            smp = {
                    'idx': i, # the base image idx in the base dataset
                    'coords': seed.tolist(), # the centroid
                    'location': 0,
                    'modality': img_smp['modality'],
                    'vbb': [vbb_d, vbb_h, vbb_w],
                    'vbb.shape': [int(vbb_d[1]-vbb_d[0]), int(vbb_h[1]-vbb_h[0]), int(vbb_w[1]-vbb_w[0])],
                    'id': img_smp['id']
                }
            
            bg_patches.append(smp)
        return bg_patches
            
    def _put_random_sphere_as_aneu(self, img, spacing):
        diameter_in_mm = random.choice(np.arange(2, 10.5, 0.25)) # generates a random aneurysm diameter in mms
        diameter = max(spacing)*diameter_in_mm # converts into voxels bit lossy though since the spacing is non isometric, yielding basically an odd shape
        diameter = min(min(img.shape), round(diameter))
        obj = np.zeros((diameter, diameter, diameter), dtype=bool)
        radius = diameter / 2
        center = (diameter - 1) / 2  # e.g. for diameter=3: center=1.0

        z, y, x = np.ogrid[:diameter, :diameter, :diameter]
        dist_sq = (x - center)**2 + (y - center)**2 + (z - center)**2

        obj[dist_sq <= radius**2] = True
        obj = obj.astype(np.uint8)
        size_sph = obj.shape
        center = [round(s/2) for s in img.shape[1:]]
        lowers = [c-round(s) for c, s in zip(center, size_sph)]
    
        img[1, lowers[0]:lowers[0]+size_sph[0], lowers[1]:lowers[1]+size_sph[1], lowers[2]:lowers[2]+size_sph[2]] = obj
        return img
    
    ###########################
    ########################### publics
    def preprocess(self, include_bg=False, max_items=-1):
        if self.is_patched: return
        self.aneus = []
        for i in tqdm.tqdm(range(len(self.image_ds)), desc='Patching'):
            sample = self.image_ds[i]
            if not any(sample['location']): continue
            
            cc, n = label(sample['location_mask'])
            
            vbb_coords = np.argwhere(sample['vessel_mask']) # VBB = Vessel Bounding Box
            vbb_d = [int(np.min(vbb_coords[0])), int(np.max(vbb_coords[0]))]
            vbb_h = [int(np.min(vbb_coords[1])), int(np.max(vbb_coords[1]))]
            vbb_w = [int(np.min(vbb_coords[2])), int(np.max(vbb_coords[2]))]
            
            for obj in range(1, n+1):
                smp = {
                    'idx': i, # the base image idx in the base dataset
                    'coords': np.mean(np.argwhere(cc==obj), axis=0).tolist(), # the centroid
                    'location': np.median(sample['location_mask'][cc==obj]),
                    'modality': sample['modality'],
                    'vbb': [vbb_d, vbb_h, vbb_w],
                    'vbb.shape': [int(vbb_d[1]-vbb_d[0]), int(vbb_h[1]-vbb_h[0]), int(vbb_w[1]-vbb_w[0])],
                    'id': sample['id']
                }
                self.aneus.append(smp)
            if i == max_items:break
        
        n_real_patches = len(self.aneus) # store number of actual patches.
        
        extra_patches = []
        if include_bg:
            n_bg = round(n_real_patches*include_bg)
            extra_patches += self._make_bg_patches(n_bg)
            
        self.aneus+=extra_patches
        
    def split(self, folds='0.8-0.2', random_seed=42):
        folds = [float(val) for val in folds.split('-')]
        assert sum(folds)==1, 'Sum of folds must be equal to 1'
        random.seed(random_seed)
        rndm_cases = self.image_ds.cases.copy()
        random.shuffle(rndm_cases)
        n_cases = len(self.image_ds)
        fold_ds = []
        subsets = []
        lower = 0
        for fold in folds:
            upper = round(n_cases*fold)+lower
            upper = upper if upper <= n_cases else n_cases
            subset = rndm_cases[lower:upper]
            fold_ds.append(TopAneu_TnTs2_DS(self.image_ds.src, transforms=None, cases=subset, patch_size_mm=self.patch_size_mm))
            subsets.append(subset)
            lower=upper
        
        for i, sub in enumerate(subsets):
            other = []
            for j, ss in enumerate(subsets):
                if j == i: continue
                other += ss
            assert not any([s in other for s in sub]), 'Found leakage between sets.'
            
        if self.is_patched:
            for fold in fold_ds:
                for aneu in self.aneus:
                    if aneu['id']+'_0000.nii.gz' in fold.image_ds.cases:
                        aneu['idx'] = [i for i, c in enumerate(fold.image_ds.cases) if c == aneu['id']+'_0000.nii.gz'][0]
                        fold.aneus.append(aneu)
        
        return fold_ds
    
    def save(self, path):
        saveable = {
            'cases': self.image_ds.cases,
            'source': str(self.image_ds.src),
            'aneus': self.aneus,
            'patch_size_mm': self.patch_size_mm
        }
        path = str(path)+'.json' if not str(path).endswith('.json') else str(path)
        with open(path, 'w') as file:
            json.dump(saveable, file, indent=4)
    
    def append(self, to_add):
        if to_add.image_ds.src != self.image_ds.src: raise RuntimeError('Sources dont match between datasets, cant append')
        self.aneus += to_add.aneus
        self.image_ds.cases += to_add.image_ds.cases
        
    @staticmethod
    def load(path, transforms=None):
        path = str(path)+'.json' if not str(path).endswith('.json') else str(path)
        with open(path, 'r') as file:
            loaded = json.load(file)
        ds = TopAneu_TnTs2_DS(source=loaded['source'], transforms=transforms, cases=loaded['cases'])
        ds.aneus = loaded['aneus'] if loaded['aneus'] is not None else []
        ds.patch_size_mm = loaded['patch_size_mm']
        return ds
    
    @staticmethod
    def join(folds):
        if len(folds) == 1: return folds[0]
        new_ds = folds[0]
        for i in range(1, len(folds)):
            new_ds.append(folds[i])
        return new_ds
    ###########################
    
class TopAneu_TnTs2_DS_for_vessel_pt(Dataset):
    ########################### builtins
    def __init__(self, source, transforms=None, cases=None, patch_size_mm=50):
        self.image_ds = TopAneuDS(source, transforms=None, load_type_mask=False, cases=cases)
        self.aneus = []
        self.transforms = transforms
        self.patch_size_mm = patch_size_mm
        
    def __len__(self):
        return len(self.aneus)
    
    def __getitem__(self, idx):
        if not self.is_patched: self.preprocess()
        smp = self.aneus[idx]
        img_smp = self.image_ds[smp['idx']]
        
        multichannel_img = np.stack( # 4D array: [C, H, D, W]
            [
                self._center_crop(img_smp['image'], smp['coords'], img_smp['spacing'], self.patch_size_mm),
                self._center_crop(img_smp['location_mask'], smp['coords'], img_smp['spacing'], self.patch_size_mm),
                self._center_crop(img_smp['vessel_mask'], smp['coords'], img_smp['spacing'], self.patch_size_mm)
            ], axis=0
        )
        
        multichannel_img = self._put_random_sphere_as_aneu(multichannel_img, img_smp["spacing"])
        
        coords_in_vbb = [
            smp['coords'][0]-smp['vbb'][0][0],
            smp['coords'][1]-smp['vbb'][1][0],
            smp['coords'][2]-smp['vbb'][2][0]
        ]
        
        dct = {
            'image': multichannel_img,
            'location_v': [smp['location']], # multihot
            'location_a': None,
            'coords': np.array(coords_in_vbb, dtype=int)/np.array(smp['vbb.shape'], dtype=int), # relative
            'modality': smp['modality'], # string
            'id': img_smp['id'],
            'spacing': img_smp['spacing']
        }
        
        if self.transforms:
            dct = self.transforms(dct)
        
        return dct
    ###########################
    ########################### properties
    @property
    def is_patched(self):
        return len(self.aneus)>0
    ########################### 
    ########################### privates
    def _center_crop(self, img, coords, spacing, size_mm):
        sz_mm = round(size_mm/2)
        sz = [round(sz_mm/sp) for sp in spacing]
        shape = img.shape
        
        coords_lower = []
        coords_upper = []
        for c_ax, sh_ax, sz_ax in zip(coords, shape, sz):

            upper = c_ax+sz_ax
            lower = c_ax-sz_ax
            
            lower = max(lower, 0)
            upper = min(upper, sh_ax)
            
            if upper - lower < min(2*sz_ax, sh_ax):
                lower, upper = 0, sh_ax
            
            coords_lower.append(lower)
            coords_upper.append(upper)
        
        crop = img[
            int(coords_lower[0]):int(coords_upper[0]),
            int(coords_lower[1]):int(coords_upper[1]),
            int(coords_lower[2]):int(coords_upper[2])
        ]            
        return crop
    
    def _center_mask_out(self, img, coords, spacing, size_mm):
        sz_mm = round(size_mm/2)
        sz = [round(sz_mm/sp) for sp in spacing]
        shape = img.shape
        
        coords_lower = []
        coords_upper = []
        for c_ax, sh_ax, sz_ax in zip(coords, shape, sz):

            upper = c_ax+sz_ax
            lower = c_ax-sz_ax
            
            if upper<=sh_ax and lower >=0:
                coords_lower.append(lower)
                coords_upper.append(upper)
            elif upper > sh_ax:
                diff = upper-sh_ax
                coords_upper.append(sh_ax)
                coords_lower.append(lower-diff)
            elif lower < 0:
                diff = lower
                coords_lower.append(0)
                coords_upper.append(upper+abs(diff))
        
        img[
            int(coords_lower[0]):int(coords_upper[0]),
            int(coords_lower[1]):int(coords_upper[1]),
            int(coords_lower[2]):int(coords_upper[2])
        ] = 0      
        return img
            
    def _put_random_sphere_as_aneu(self, img, spacing):
        diameter_in_mm = random.choice(np.arange(2, 10.5, 0.25)) # generates a random aneurysm diameter in mms
        diameter = max(spacing)*diameter_in_mm # converts into voxels bit lossy though since the spacing is non isometric, yielding basically an odd shape
        diameter = min(min(img.shape if len(img.shape)<4 else img.shape[1:]), round(diameter))
        diameter = max(diameter, 1)
        
        obj = np.zeros((diameter, diameter, diameter), dtype=bool)
        radius = diameter / 2
        center = (diameter - 1) / 2  # e.g. for diameter=3: center=1.0

        z, y, x = np.ogrid[:diameter, :diameter, :diameter]
        dist_sq = (x - center)**2 + (y - center)**2 + (z - center)**2

        obj[dist_sq <= radius**2] = True
        obj = obj.astype(np.uint8)
        size_sph = obj.shape
        center = [s//2 for s in img.shape[1:]]
        lowers = [max(0, min(c - s // 2, img.shape[1:][i] - s))
              for i, (c, s) in enumerate(zip(center, size_sph))]

        img[1,
            lowers[0]:lowers[0]+size_sph[0],
            lowers[1]:lowers[1]+size_sph[1],
            lowers[2]:lowers[2]+size_sph[2]] = obj
        return img
    
    ###########################
    ########################### publics
    def preprocess(self, patches_per_vloc=42, max_items=-1):
        if self.is_patched: return
        for i in tqdm.tqdm(range(len(self.image_ds)), desc='Making Patches from Vessels'):
            sample = self.image_ds[i]
            
            vessel_mask = sample['vessel_mask']
            
            vbb_coords = np.argwhere(vessel_mask) # VBB = Vessel Bounding Box
            vbb_d = [int(np.min(vbb_coords[0])), int(np.max(vbb_coords[0]))]
            vbb_h = [int(np.min(vbb_coords[1])), int(np.max(vbb_coords[1]))]
            vbb_w = [int(np.min(vbb_coords[2])), int(np.max(vbb_coords[2]))]
            
            vessel_cls = np.unique(vessel_mask)
            
            for vc in vessel_cls:
                if vc==0:continue # skip bg
                binarized = vessel_mask==vc
                surface = binarized & ~binary_erosion(binarized)
                for obj in range(patches_per_vloc):
                    possible_seeds = np.argwhere(surface)
                    seed = possible_seeds[random.choice(range(possible_seeds.shape[0])), :]
                    smp = {
                        'idx': i, # the base image idx in the base dataset
                        'coords': seed.tolist(), # the centroid
                        'location': vc,
                        'modality': sample['modality'],
                        'vbb': [vbb_d, vbb_h, vbb_w],
                        'vbb.shape': [int(vbb_d[1]-vbb_d[0]), int(vbb_h[1]-vbb_h[0]), int(vbb_w[1]-vbb_w[0])],
                        'id': sample['id']
                    }
                    self.aneus.append(smp)
                
            if i == max_items:break
        
    def split(self, folds='0.8-0.2', random_seed=42):
        folds = [float(val) for val in folds.split('-')]
        assert sum(folds)==1, 'Sum of folds must be equal to 1'
        random.seed(random_seed)
        rndm_cases = self.image_ds.cases.copy()
        random.shuffle(rndm_cases)
        n_cases = len(self.image_ds)
        fold_ds = []
        subsets = []
        lower = 0
        for fold in folds:
            upper = round(n_cases*fold)+lower
            upper = upper if upper <= n_cases else n_cases
            subset = rndm_cases[lower:upper]
            fold_ds.append(TopAneu_TnTs2_DS_for_vessel_pt(self.image_ds.src, transforms=None, cases=subset, patch_size_mm=self.patch_size_mm))
            subsets.append(subset)
            lower=upper
        
        for i, sub in enumerate(subsets):
            other = []
            for j, ss in enumerate(subsets):
                if j == i: continue
                other += ss
            assert not any([s in other for s in sub]), 'Found leakage between sets.'
            
        if self.is_patched:
            for fold in fold_ds:
                for aneu in self.aneus:
                    if aneu['id']+'_0000.nii.gz' in fold.image_ds.cases:
                        fold.aneus.append(aneu)
        
        return fold_ds
    
    def save(self, path):
        saveable = {
            'cases': self.image_ds.cases,
            'source': str(self.image_ds.src),
            'aneus': self.aneus,
            'patch_size_mm': self.patch_size_mm
        }
        path = str(path)+'.json' if not str(path).endswith('.json') else str(path)
        with open(path, 'w') as file:
            json.dump(saveable, file, indent=4)
        
    @staticmethod
    def load(path, transforms=None):
        path = str(path)+'.json' if not str(path).endswith('.json') else str(path)
        with open(path, 'r') as file:
            loaded = json.load(file)
        ds = TopAneu_TnTs2_DS_for_vessel_pt(source=loaded['source'], transforms=transforms, cases=loaded['cases'])
        ds.aneus = loaded['aneus'] if loaded['aneus'] is not None else []
        ds.patch_size_mm = loaded['patch_size_mm']
        return ds
    ###########################
    
def TnTs2_collate(batch):
    # Assumes all data is already a tensor, if not will attempt to cast to tensor
    images = []
    coords = []
    locations = []
    modalities = []
    ids = []
    vlocs = []
    for sample in batch:
        images.append(sample['image'])
        coords.append(sample['coords'])
        locations.append(sample['location'])
        modalities.append(sample['modality'])
        ids.append(sample['id'])
        if "vloc" in sample.keys(): vlocs.append(sample["vloc"])
        
    return {
        'image': torch.stack(images, dim=0),
        'coords': torch.stack(coords, dim=0),
        'location': {'aneurysm': torch.stack(locations, dim=0)},
        'modality': modalities, # just a basic list
        'id': ids, # just a basic list
        "vloc": torch.stack(vlocs, dim=0) if any(vlocs) else None
    }

def TnTs2_collate_dev(batch):
    # Assumes all data is already a tensor, if not will attempt to cast to tensor
    images = []
    coords = []
    locations = []
    modalities = []
    ids = []
    vlocs = []
    for sample in batch:
        images.append(sample['image'])
        coords.append(sample['coords'])
        locations.append(sample['location'])
        modalities.append(sample['modality'])
        ids.append(sample['id'])
        if "vloc" in sample.keys(): vlocs.append(sample["vloc"])
    parsed = {} 
    for k in locations[0].keys():
        parsed[k] = torch.stack([loc[k] for loc in locations], dim=0)
        
    return {
        'image': torch.stack(images, dim=0),
        'coords': torch.stack(coords, dim=0),
        'location': parsed,
        'modality': modalities, # just a basic list
        'id': ids, # just a basic list
        "vloc": torch.stack(vlocs, dim=0) if any(vlocs) else None
    }