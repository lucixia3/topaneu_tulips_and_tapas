import numpy as np, json, tqdm, torch
from torchvision.transforms import Normalize, Compose, Resize

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
            if k in self.convertable:
                if isinstance(v, np.ndarray): dct[k] = torch.from_numpy(v).to(self.dtype)
                else: dct[k] = torch.tensor(v).to(self.dtype)
        return dct
    
class MaybeResize(): ## gen by claude, cause my implementation was buggy
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.convertable = ['image', 'location_mask', 'vessel_mask', 'type_mask']
        self.mask_keys = ['location_mask', 'vessel_mask', 'type_mask']

    def __call__(self, dct):
        for k, v in dct.items():
            if k not in self.convertable:
                continue

            added_channel = False
            if v.dim() == 3:          # (D, H, W) -> add channel dim
                v = v.unsqueeze(0)
                added_channel = True

            # pick interpolation mode per key type
            kwargs = dict(self.kwargs)
            if k in self.mask_keys:
                kwargs['mode'] = 'nearest'  # override whatever was passed
            resize = Resize(*self.args, **kwargs)

            v = resize(v)

            if added_channel:
                v = v.squeeze(0)

            dct[k] = v
        return dct
    
class AdaNorm():
    def __init__(self, mr_mean, mr_std, ct_mean, ct_std):
        self.norm_mr = Normalize([mr_mean], [mr_std])
        self.norm_ct = Normalize([ct_mean], [ct_std])
        
    def __call__(self, dct):
        if dct['modality']=='CTA':
            if len(dct['image'])==3: dct['image'] = self.norm_ct(dct['image'])
            else: dct['image'][0, :] = self.norm_ct(dct['image'][0, :])

        elif dct['modality'] == 'MRA':
            if len(dct['image'])==3: dct['image'] = self.norm_mr(dct['image'])
            else: dct['image'][0, :] = self.norm_mr(dct['image'][0, :])
            
        else: raise ValueError('Unknown modality')
        
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
            50: 27
        }
        self.latmap = { # stored as [R, L]
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
            50: [0, 1]
        }
    def __call__(self, dct):
        if isinstance(dct, dict):
            loc = self.locmap[dct['locations']]
            lat = self.latmap[dct['locations']]
            hot = [0]*27
            hot[loc]=1
            hot += lat
            
            dct['locations'] = hot
            return dct
        else:
            loc = self.locmap[dct]
            lat = self.latmap[dct]
        
            hot = [0]*27
            hot[loc]=1
            hot += lat
            
            return hot
        
class DecodeTarget():
    def __init__(self):
        self.map = {
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
            27:49
        }
        self.excl_from_lat = [7, 8, 17, 36]
    
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
        loc, logit_loc = max(enumerate(row[:27]), key=lambda x: x[1])
        lat, logit_lat = max(enumerate(row[27:]), key=lambda x: x[1])
        if loc not in self.excl_from_lat:
            loc += lat
        return loc