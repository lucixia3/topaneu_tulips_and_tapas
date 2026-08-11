from TnT.utils.transforms import get_train_test_transforms, DecodeTarget
from TnT.utils.dataloader import TopAneu_TnTs2_DS
import SimpleITK as sitk, numpy as np, torch
from scipy.ndimage import label

class InferencePipeline():
    def __init__(self, model, patch_size_vx=64, patch_size_mm=35, device='cuda'):
        self.s2_model = model
        self.patch_size_mm = patch_size_mm
        _, self.s2_transforms = get_train_test_transforms(patch_size_vx)
        self.decoder = DecodeTarget()
        self.device = device
        self.s2_model.to(self.device)
        self.s2_model.eval()
    
    @torch.no_grad()  
    def __call__(self, sample: sitk.Image, modality: str):
        if isinstance(sample, sitk.Image):
            image, vmask, lmask = self._stage1(sample)
        elif isinstance(sample, dict):
            image=sample['image']
            vmask=sample['vessel_mask']!=0
            lmask=sample['location_mask']!=0
        
        if np.any(lmask):
            s2ds = CaseDL(image, vmask, lmask, modality, self.s2_transforms, sample.GetSpacing() if isinstance(sample, sitk.Image) else sample['spacing'], self.patch_size_mm)
            for i in range(len(s2ds)):
                smp = s2ds[i]
                pred = self._stage2(smp)
                s2ds.add_label(i, pred)
            output = s2ds.make_mask()
        else: output = np.zeros_like(vmask)
        
        if isinstance(sample, sitk.Image): 
            outimg = sitk.GetImageFromArray(output.astype(np.uint8))
            outimg.CopyInformation(sample)
            return outimg
        else: return output.astype(np.uint8)
        
    def _stage1(self, sample: sitk.Image):
        raise NotImplementedError('Needs to return a tuple of (\n   image: (as it was passed to the funciton, no mutations or transformations applied), \n    location_mask: (binary), \n    vessel_mask: (binary)\n)')
    
    def _stage2(self, smp: dict) -> int:
        pred_lat, pred_loc = self.s2_model.classify(smp['image'].unsqueeze(0).to(self.device), smp['coords'].unsqueeze(0).to(self.device), [smp['modality']])
        pred_lat=pred_lat.detach().to('cpu')
        pred_loc=pred_loc.detach().to('cpu')
        preds = torch.concat([pred_loc, pred_lat], dim=-1)
        assert len(preds.shape)==2, 'only implemented for batched data'
        decoded_label = self.decoder(preds)[0][2]
        return decoded_label
        
class CaseDL(TopAneu_TnTs2_DS):
    def __init__(self, image, vmask, lmask, modality, transforms, spacing, patch_size_mm):
        self.image = image
        self.vmask = vmask
        self.lmask = lmask
        self.modality = 'CTA' if 'ct' in modality.lower() else 'MRA'
        self.transforms = transforms
        self.cc, self.n = label(self.lmask)
        self.spacing = spacing
        self.patch_size_mm = patch_size_mm
        vbb_coords = np.argwhere(self.vmask) # VBB = Vessel Bounding Box
        vbb_d = [int(np.min(vbb_coords[0])), int(np.max(vbb_coords[0]))]
        vbb_h = [int(np.min(vbb_coords[1])), int(np.max(vbb_coords[1]))]
        vbb_w = [int(np.min(vbb_coords[2])), int(np.max(vbb_coords[2]))]
        self.vbb = [vbb_d, vbb_h, vbb_w]
        self.vbb_shape = [int(vbb_d[1]-vbb_d[0]), int(vbb_h[1]-vbb_h[0]), int(vbb_w[1]-vbb_w[0])]
        
        self.assigned_labels = {}
        
    def __getitem__(self, idx):     
        coords = np.mean(np.argwhere(self.cc==idx+1), axis=0).tolist()   
        multichannel_img = np.stack( # 4D array: [C, H, D, W]
            [
                self._center_crop(self.image, coords, self.spacing, self.patch_size_mm),
                self._center_crop(self.lmask, coords, self.spacing, self.patch_size_mm),
                self._center_crop(self.vmask, coords, self.spacing, self.patch_size_mm)
            ], axis=0
        )
        coords_in_vbb = [
            coords[0]-self.vbb[0][0],
            coords[1]-self.vbb[1][0],
            coords[2]-self.vbb[2][0]
        ]
        
        dct = {
            'image': multichannel_img,
            'coords': np.array(coords_in_vbb, dtype=int)/np.array(self.vbb_shape, dtype=int), # relative
            'modality': self.modality, # string
            'spacing': self.spacing
        }
        
        if self.transforms:
            dct = self.transforms(dct)
        
        return dct

    def __len__(self):
        return self.n

    def add_label(self, idx, lbl):
        assert isinstance(lbl, int)
        self.assigned_labels[idx]=lbl
        assert len(self.assigned_labels)<=len(self)
        
    def make_mask(self):
        output = np.zeros_like(self.cc, dtype=np.uint8)
        for idx, lbl in self.assigned_labels.items():
            output[self.cc==idx+1]=lbl
        return output