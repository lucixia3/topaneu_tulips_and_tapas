import numpy as np
from torchvision.transforms import ToTensor

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
        loc = self.locmap(dct['locations'])
        lat = self.latmap(dct['locations'])
        
        hot = [0]*27
        hot[loc]=1
        hot += lat
        
        dct['locations'] = hot
        return dct