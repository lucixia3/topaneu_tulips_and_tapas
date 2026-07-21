import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib as pl
import datetime
import os
from TnT.utils.transforms import LateralityInvariance

from monai.networks.nets import resnet50#, resnet18

class TnTS2(nn.Module):
    def __init__(self, n_locs, n_lats):
        super().__init__()
        self.n_locs, self.n_lats = n_locs, n_lats
        self.bb = resnet50(spatial_dims=3, n_input_channels=3) # outputs [B, 400]
        self.laterality = nn.Linear(404, self.n_lats)
        self.location = nn.Linear(404, self.n_locs) # in pretraining is the vessel classes
        
    def forward(self, patch, coords, modalities):
        unbatched = isinstance(modalities, str)
        if unbatched:
            modalities = [modalities]
            patch = patch.unsqueeze(0)
            coords = coords.unsqueeze(0)

        x = self.bb(patch)
        modality_flag = torch.tensor([m == 'MRA' for m in modalities], dtype=torch.uint8, device=coords.device).unsqueeze(1)
        x = torch.concat([x, coords, modality_flag], dim=-1)
        lat = self.laterality(x)
        loc = self.location(x)

        if unbatched:
            lat = lat.squeeze(0)
            loc = loc.squeeze(0)

        return lat, loc
    
    def classify(self, patch, coords, modalities):
        unbatched = isinstance(modalities, str)
        if unbatched: raise RuntimeError("Unbatched data is not supported")
        lat, loc = self.forward(patch, coords, modalities)
        lat_sigmoid, loc_sigmoid = F.sigmoid(lat), F.sigmoid(loc)
        
        assigned_lat = torch.zeros_like(lat_sigmoid, dtype=torch.uint8)
        assigned_loc = torch.zeros_like(loc_sigmoid, dtype=torch.uint8)
        for b_item in range(lat_sigmoid.shape[0]):
            loc = torch.argmax(loc_sigmoid[b_item, :]).item()
            lat = torch.argmax(lat_sigmoid[b_item, :]).item()
            assigned_lat[b_item, loc]=1
            assigned_loc[b_item, lat]=1
        
        return assigned_lat, assigned_loc
    
    def save(self, pth, overwrite=False):
        pth = pl.Path(pth)
        if os.path.exists(pth) and not overwrite:
            override = datetime.datetime.now().strftime(r'TnTS2_from-%H:%M:%S-%d.%m.%y')
            print(f"INFO: {pth} already exists, using {pth.parent/override} instead")
            pth=pth.parent/override
        if not os.path.exists(pth): os.mkdir(pth)
        torch.save(self.bb.state_dict(), pth/'bb.pth')
        torch.save(self.location.state_dict(), pth/'head.pth')
        torch.save(self.laterality.state_dict(), pth/'laterality.pth')
        
    def load(self, pth):
        pth=pl.Path(pth)
        self.bb.load_state_dict(torch.load(pth/'bb.pth'))
        self.location.load_state_dict(torch.load(pth/'location.pth'))
        self.laterality.load_state_dict(torch.load(pth/'laterality.pth'))
    
    @staticmethod
    def from_pretrained(pth, n_locs, n_lats):
        pth=pl.Path(pth)
        model = TnTS2(n_locs, n_lats)
        try: model = model.load(pth) ## will error if loc/lat missmatch
        except: ## instead only load backbone and laterality head, build location head from scratch
            model.bb.load_state_dict(torch.load(pth/'bb.pth'))
            model.laterality.load_state_dict(torch.load(pth/'laterality.pth'))
        return model
    
    def loss(self, patch, coords, modalities, targets):
        unbatched = isinstance(modalities, str)
        if unbatched:
            modalities = [modalities]
            patch = patch.unsqueeze(0)
            coords = coords.unsqueeze(0)

        lat, loc = self.forward(patch, coords, modalities)
        
        loc_loss = F.cross_entropy(loc, targets[:, :self.n_locs])
        lat_loss = F.cross_entropy(lat, targets[:, self.n_locs:])
        return loc_loss+lat_loss

class TnTS2Loss(nn.Module):
    def __init__(self, n_locs):
        super().__init__()
        self.n_locs = n_locs
        
    def loss(self, lat, loc, targets):        
        loc_loss = F.cross_entropy(loc, targets[:, :self.n_locs])
        lat_loss = F.cross_entropy(lat, targets[:, self.n_locs:])
        return loc_loss+lat_loss