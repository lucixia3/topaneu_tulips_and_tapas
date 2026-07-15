import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib as pl
import datetime
import os

from monai.networks.nets import resnet50#, resnet18

class TnTS2(nn.Module):
    def __init__(self, n_classes=29):
        super().__init__()
        self.bb = resnet50(spatial_dims=3, n_input_channels=3) # outputs [B, 400]
        self.head = nn.Sequential(
            nn.Linear(404, n_classes), # +4 input for [D, H, W, is_mr]
        )
        
    # def forward(self, patch, coords, modalities):
    #     if isinstance(modalities, str): modalities=[modalities]
    #     x = self.bb(patch)
    #     x = torch.concat([x, coords, torch.tensor([m=='MRA' for m in modalities], dtype=torch.uint8, device=coords.device).unsqueeze(1)], dim=-1)
    #     x = self.head(x)
    #     return x
    def forward(self, patch, coords, modalities):
        unbatched = isinstance(modalities, str)
        if unbatched:
            modalities = [modalities]
            patch = patch.unsqueeze(0)
            coords = coords.unsqueeze(0)

        x = self.bb(patch)
        modality_flag = torch.tensor([m == 'MRA' for m in modalities], dtype=torch.uint8, device=coords.device).unsqueeze(1)
        x = torch.concat([x, coords, modality_flag], dim=-1)
        x = self.head(x)

        if unbatched:
            x = x.squeeze(0)

        return x
    
    def classify(self, patch, coords, modalities):
        unbatched = isinstance(modalities, str)
        sigmoid = F.sigmoid(self.forward(patch, coords, modalities))
        if unbatched: sigmoid = sigmoid.unsqueeze(0)
        assigned = torch.zeros_like(sigmoid, dtype=torch.uint8)
        for b_item in range(sigmoid.shape[0]):
            loc = torch.argmax(sigmoid[b_item, :27]).item()
            lat = torch.argmax(sigmoid[b_item, 27:]).item()
            assigned[b_item, loc]=1
            assigned[b_item, 27+lat]=1
        if unbatched: assigned=assigned.squeeze(0)
        return assigned
    
    def save(self, pth, overwrite=False):
        pth = pl.Path(pth)
        if os.path.exists(pth) and not overwrite:
            override = datetime.datetime.now().strftime(r'TnTS2_from-%H:%M:%S-%d.%m.%y')
            print(f"INFO: {pth} already exists, using {pth.parent/override} instead")
            pth=pth.parent/override
        if not os.path.exists(pth): os.mkdir(pth)
        torch.save(self.bb.state_dict(), pth/'bb.pth')
        torch.save(self.head.state_dict(), pth/'head.pth')
        
    def load(self, pth):
        pth=pl.Path(pth)
        self.bb.load_state_dict(torch.load(pth/'bb.pth'))
        self.head.load_state_dict(torch.load(pth/'head.pth'))
        
    def loss(self, patch, coords, modalities, targets):
        x = self.forward(patch, coords, modalities)
        
        loc_loss = F.cross_entropy(x[:, :27], targets[:, :27])
        lat_loss = F.cross_entropy(x[:, 27:], targets[:, 27:])
        
        return loc_loss+lat_loss