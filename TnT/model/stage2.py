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
            nn.Sigmoid() # sigmoid for BCE loss
        )
        
    def forward(self, patch, coords, modalities):
        x = self.bb(patch)
        x = torch.concat([x, coords, torch.tensor([m=='MRA' for m in modalities], dtype=torch.uint8, device=coords.device).unsqueeze(1)], dim=1)
        x = self.head(x)
        return x
    
    def classify(self, patch, coords, modalities):
        sigmoid = self.forward(patch, coords, modalities)
        if len(sigmoid.shape)==1: sigmoid.unsqueeze(0)
        cls = torch.zeros_like(sigmoid, dtype=torch.uint8)
        # assign loc
        max_prob = torch.argmax(sigmoid[:, :27], dim=1)
        cls[:, max_prob] = 1
        # assign lat
        max_prob = torch.argmax(sigmoid[:, 27:], dim=1)
        cls[:, max_prob+27] = 1
        return cls
    
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