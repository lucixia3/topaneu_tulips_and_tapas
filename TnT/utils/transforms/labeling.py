import numpy as np
class LabelEncoder():
    def __init__(self):
        self.n_locs_v, self.n_locs_a, self.n_lats = self.get_locs_lats()
    
    @staticmethod
    def get_locs_lats():
        return 21, 29, 2
    
    def __call__(self, dct):
        ## first transform all vessels, the value for location_v should be a list with n vessels in the frame, store all vessels
        vessels = {}
        for v in dct['location_v']:
            cur = [0]*self.n_locs_v
            cur[LAT_INV_VES[v]]=1
            vessels[v]=cur
        
        ## second transform aneurysm locations, if none present use the vessels to make synthetic hierarchical targets.
        
        if dct['location_a'] is None:
            aneu = LAT_INV_VES_ASSOCIATED_ANEUS[LAT_INV_VES[list(vessels.keys())[0]]]
        else:
            aneu = [0]*self.n_locs_a
            #print(len(aneu), LAT_INV_ANEU[dct['location_a']], dct['location_a'])
            aneu[LAT_INV_ANEU[dct['location_a']]] = 1
            
        ## third transform laterality
        if dct['location_a'] is None:
            lat = LAT_FOR_VES[list(vessels.keys())[0]]
        else:
            lat = LAT_FOR_ANEU[dct['location_a']]
            
        ## fourth unify vessels
        if len(vessels)==1: vessel = vessels[list(vessels.keys())[0]]
        else:
            vessel = [0]*self.n_locs_v
            for k, v in vessels.items():
                for i, pres in enumerate(v):
                    if pres==1: vessel[i]=1
            
        dct['location_a'] = np.asarray(aneu, dtype=np.float32)
        dct['location_v'] = np.asarray(vessel, dtype=np.float32)
        dct['laterality'] = np.asarray(lat, dtype=np.float32)
        
        return dct

class DecodeVessel():
    def __init__(self):
        self.map = LIT_VLOC_21
        self.lit_loc_lookup = LIT_VLOC_36
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

class DecodeAneu():
    def __init__(self):
        self.map = REV_LAT_INV_ANEU
        self.excl_from_lat = [0, 7, 8, 17, 36]
        self.lit_loc_lookup = LIT_ALOC
    
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
    
LIT_VLOC_21 = {
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

LIT_VLOC_36 = {
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

LAT_INV_ANEU = {
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

LAT_FOR_ANEU = { # stored as [R, L]
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

LIT_ALOC = {
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

REV_LAT_INV_ANEU = {
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

LAT_FOR_VES = { # encoded [R, L]
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

LAT_INV_VES = {
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

LAT_INV_VES_ASSOCIATED_ANEUS = {
                    0: [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    1: [0,0,0,0,1,1,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    2: [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    3: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                    4: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
                    5: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                    6: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                    7: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                    8: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                    9: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                    10:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                    11:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
                    12:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                    13:[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    14:[0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    15:[0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    16:[0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    17:[0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    18:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0],
                    19:[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    20:[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                }








