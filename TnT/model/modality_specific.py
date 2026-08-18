import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib as pl
import datetime
import os

from monai.networks.nets import resnet50#, resnet18
from TnT.model.stage2 import TnTS2

class TnTS2_Specific(nn.Module):
    def __init__(self, mr_model_path, ct_model_path):
        super().__init__()
        self.ct = TnTS2()
        self.ct.load(ct_model_path)
        self.mr = TnTS2()
        self.mr.load(mr_model_path)
    
    def classify(self, patch, coords, modalities, target='aneu'):
        unbatched = isinstance(modalities, str)
        if unbatched: raise RuntimeError("Unbatched data is not supported")
        
        lats = []
        locs = []
        for i, mod in enumerate(modalities):
            if mod == 'MRA':
                if target=='vessel': lat, loc, _ = self.mr(patch[i].unsqueeze(0), coords[i].unsqueeze(0), [mod])
                else: lat, _, loc = self.mr(patch[i].unsqueeze(0), coords[i].unsqueeze(0), [mod])
                lats.append(lat)
                locs.append(loc)
            elif mod == 'CTA':
                if target=='vessel': lat, loc, _ = self.ct(patch[i].unsqueeze(0), coords[i].unsqueeze(0), [mod])
                else: lat, _, loc = self.ct(patch[i].unsqueeze(0), coords[i].unsqueeze(0), [mod])
                lats.append(lat)
                locs.append(loc)
            else: raise ValueError(f'Received unknown modality {mod}')
        
        lat = torch.concat(lats, dim=0)
        loc = torch.concat(locs, dim=0)  
            
        lat_sigmoid, loc_sigmoid = F.sigmoid(lat), F.sigmoid(loc)
        
        assigned_lat = torch.zeros_like(lat_sigmoid, dtype=torch.uint8)
        assigned_loc = torch.zeros_like(loc_sigmoid, dtype=torch.uint8)
        for b_item in range(lat_sigmoid.shape[0]):
            loc = torch.argmax(loc_sigmoid[b_item, :]).item()
            lat = torch.argmax(lat_sigmoid[b_item, :]).item()
            assigned_lat[b_item, lat]=1
            assigned_loc[b_item, loc]=1
        
        return assigned_lat, assigned_loc
    
    def eval(self):
        self.mr.eval()
        self.ct.eval()
        #return self

    def to(self, item):
        self.mr.to(item)
        self.ct.to(item)
        #return self