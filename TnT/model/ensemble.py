import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib as pl
import datetime
import os
from TnT.model.stage2 import TnTS2

from monai.networks.nets import resnet50#, resnet18

class Ensemble(nn.Module):
    """Builds an ensemple by averaging probabilities across models

    Args:
        nn (_type_): _description_
    """
    def __init__(self, n_locs_v=21, n_locs_a=29, n_lats=2):
        super().__init__()
        self.n_locs_v, self.n_locs_a, self.n_lats = n_locs_v, n_locs_a, n_lats
        self.models = []
        
    def add(self, model):
        assert model.n_locs_a == self.n_locs_a
        assert model.n_locs_v == self.n_locs_v
        assert model.n_lats == self.n_lats
        self.models.append(model)
        
    def forward(self, patch, coords, modalities):
        lats, locs_a, locs_v = [], [], []
        for m in self.models:
            lat, loc_v, loc_a = m(patch, coords, modalities)
            lats.append(lat)
            locs_v.append(loc_v)
            locs_a.append(loc_a)
        return lats, locs_v, locs_a
    
    def classify(self, patch, coords, modalities, target='aneu'):
        unbatched = isinstance(modalities, str)
        if unbatched: raise RuntimeError("Unbatched data is not supported")
        lats, locs = [], []
        for m in self.models:
            if target == 'vessel': lat, loc, _ = m(patch, coords, modalities)
            else: lat, _, loc = m(patch, coords, modalities)
            lat_sigmoid, loc_sigmoid = F.sigmoid(lat), F.sigmoid(loc)
            lats.append(lat_sigmoid)
            locs.append(loc_sigmoid)
            
        lat_sigmoid, loc_sigmoid = torch.mean(torch.stack(lats, dim=0), dim=0), torch.mean(torch.stack(locs, dim=0), dim=0)
        
        assigned_lat = torch.zeros_like(lat_sigmoid, dtype=torch.uint8)
        assigned_loc = torch.zeros_like(loc_sigmoid, dtype=torch.uint8)
        for b_item in range(lat_sigmoid.shape[0]):
            loc = torch.argmax(loc_sigmoid[b_item, :]).item()
            lat = torch.argmax(lat_sigmoid[b_item, :]).item()
            assigned_lat[b_item, lat]=1
            assigned_loc[b_item, loc]=1
        
        return assigned_lat, assigned_loc
    
    def save(self, pth, overwrite=False):
        pth = pl.Path(pth)
        for i, m in enumerate(self.models):
            dir = f'Model_{i}'
            m.save(pth/dir, overwrite)
    
    def load(self, pth):
        pth=pl.Path(pth)
        for m in [m for m in sorted(os.listdir(pth), key=lambda x: int(x.split('_')[-1])) if 'Model_' in m]:
            cur_model = TnTS2(self.n_locs, self.n_lats)
            cur_model.load(pth/m)
            self.models.append(cur_model)
            
    def to(self, dev):
        [m.to(dev) for m in self.models]