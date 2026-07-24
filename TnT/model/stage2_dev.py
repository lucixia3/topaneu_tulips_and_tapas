import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib as pl
import datetime
import os
from TnT.utils.transforms import LateralityInvariance

from monai.networks.nets import resnet50#, resnet18

class TnTS2(nn.Module):
    def __init__(self, n_locs_a, n_locs_v, n_lats):
        super().__init__()
        self.n_locs_v, self.n_locs_a, self.n_lats = n_locs_v, n_locs_a, n_lats
        self.bb = resnet50(spatial_dims=3, n_input_channels=3) # outputs [B, 400]
        self.laterality = nn.Linear(404, self.n_lats)
        self.location_a = nn.Linear(404, self.n_locs_a)
        self.location_v = nn.Linear(404+self.n_locs_a, self.n_locs_v) # in pretraining is the vessel classes
        
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
        loc_a = self.location_a(x)
        loc_v = self.location_v(torch.concat([x, loc_a], dim=-1))

        if unbatched:
            lat = lat.squeeze(0)
            loc_a = loc_a.squeeze(0)
            loc_v = loc_v.squeeze(0)

        return lat, loc_a, loc_v
    
    def classify(self, patch, coords, modalities):
        unbatched = isinstance(modalities, str)
        if unbatched: raise RuntimeError("Unbatched data is not supported")
        lat, loc_a, loc_v = self.forward(patch, coords, modalities)
        lat_sigmoid, loc_a_sigmoid, loc_v_sigmoid = F.sigmoid(lat), F.sigmoid(loc_a), F.sigmoid(loc_v)
        
        assigned_lat = torch.zeros_like(lat_sigmoid, dtype=torch.uint8)
        assigned_loc_a = torch.zeros_like(loc_a_sigmoid, dtype=torch.uint8)
        assigned_loc_v = torch.zeros_like(loc_v_sigmoid, dtype=torch.uint8)
        for b_item in range(lat_sigmoid.shape[0]):
            loc_a = torch.argmax(loc_a_sigmoid[b_item, :]).item()
            loc_v = torch.argmax(loc_v_sigmoid[b_item, :]).item()
            lat = torch.argmax(lat_sigmoid[b_item, :]).item()
            assigned_lat[b_item, lat]=1
            assigned_loc_a[b_item, loc_a]=1
            assigned_loc_v[b_item, loc_v]=1
        
        return assigned_lat, assigned_loc_a, assigned_loc_v
    
    def save(self, pth, overwrite=False):
        pth = pl.Path(pth)
        if os.path.exists(pth) and not overwrite:
            override = datetime.datetime.now().strftime(r'TnTS2_from-%H:%M:%S-%d.%m.%y')
            print(f"INFO: {pth} already exists, using {pth.parent/override} instead")
            pth=pth.parent/override
        if not os.path.exists(pth): os.makedirs(pth)
        torch.save(self.bb.state_dict(), pth/'bb.pth')
        torch.save(self.location_a.state_dict(), pth/'location_a.pth')
        torch.save(self.location_v.state_dict(), pth/'location_v.pth')
        torch.save(self.laterality.state_dict(), pth/'laterality.pth')
        
    def load(self, pth):
        pth=pl.Path(pth)
        self.bb.load_state_dict(torch.load(pth/'bb.pth'))
        self.location_a.load_state_dict(torch.load(pth/'location_a.pth'))
        self.location_v.load_state_dict(torch.load(pth/'location_v.pth'))
        self.laterality.load_state_dict(torch.load(pth/'laterality.pth'))
    
    def loss(self, patch, coords, modalities, targets):
        unbatched = isinstance(modalities, str)
        if unbatched:
            modalities = [modalities]
            patch = patch.unsqueeze(0)
            coords = coords.unsqueeze(0)

        lat, loc_a, loc_v = self.forward(patch, coords, modalities)
        
        loc_a_loss = F.cross_entropy(loc_a, targets["aneurysm"].to(loc_a.device))
        lat_loss = F.cross_entropy(lat, targets["laterality"].to(lat.device))
        if "vessel" in targets.keys():
            loc_v_loss = F.binary_cross_entropy_with_logits(loc_v, targets["vessel"].to(loc_v.device))
            return loc_a_loss+lat_loss+loc_v_loss
        else: return loc_a_loss+lat_loss

class TnTS2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, lat, loc_a, loc_v, targets):      
        loc_a_loss = F.cross_entropy(loc_a, targets["aneurysm"].to(loc_a.device))
        lat_loss = F.cross_entropy(lat, targets["laterality"].to(lat.device))
        if "vessel" in targets.keys():
            loc_v_loss = F.binary_cross_entropy_with_logits(loc_v, targets["vessel"].to(loc_v.device))
            return loc_a_loss+lat_loss+loc_v_loss
        else: return loc_a_loss+lat_loss