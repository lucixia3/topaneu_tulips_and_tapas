from TnT.utils.transforms import get_train_test_transforms, DecodeTarget
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
            spacing = sample.GetSpacing()
            stack = torch.stack(self._stage1(sample), dim=0).numpy()
        elif isinstance(sample, dict):
            spacing = sample['spacing']
            stack = [sample['image'], sample['location_mask']!=0, sample['vessel_mask']!=0] ## fallback for s2 development purposes
            stack = np.stack(stack, axis=0)
            
        cc, n = label(stack[1]) # labels the vessel mask channel
        if n == 0: # early exit if no segmentation results
            #print('No aneurysms segmented')
            output =  np.zeros_like(stack[1])
        
        else:
            output = np.zeros_like(stack[1]) # the image to write the things to
            for comp in range(1, n+1):
                cur_stack = stack.copy()
                cur_stack[1] = cc==comp
                predicted_label = self._stage2(cur_stack, spacing, modality)
                #print(predicted_label)
                output[cc==comp]=predicted_label[2]
        
        if isinstance(sample, sitk.Image): 
            outimg = sitk.GetImageFromArray(output)
            outimg.CopyInformation(sample)
            return outimg
        else: return output
    
        
    def _stage1(self, sample: sitk.Image):
        raise NotImplementedError('Needs to return a tuple of (\n   image: (as it was passed to the funciton, no mutations or transformations applied), \n    location_mask: (binary), \n    vessel_mask: (binary)\n)')
    
    def _stage2(self, sample, spacing, modality):
        # centroid in image space
        centroid = np.mean(np.argwhere(sample[2]), axis=0).tolist()
        
        # vessel bbox
        vbb_coords = np.argwhere(sample[2]) # VBB = Vessel Bounding Box
        vbb_d = [int(np.min(vbb_coords[0])), int(np.max(vbb_coords[0]))]
        vbb_h = [int(np.min(vbb_coords[1])), int(np.max(vbb_coords[1]))]
        vbb_w = [int(np.min(vbb_coords[2])), int(np.max(vbb_coords[2]))]
        vbb = [vbb_d, vbb_h, vbb_w]
        
        # crop in image space
        cropped_stack = []
        for i in range(3):
            cropped_stack.append(self._center_crop(sample[i], centroid, spacing, self.patch_size_mm))
        cropped_stack = np.stack(cropped_stack, axis=0)
        
        # translate centorid to vessel space for inference
        coords_in_vbb = [
            centroid[0]-vbb[0][0],
            centroid[1]-vbb[1][0],
            centroid[2]-vbb[2][0]
        ]
        
        dct = {
                    'image': torch.from_numpy(cropped_stack),
                    'coords': torch.tensor(np.array(coords_in_vbb, dtype=int)/np.array([int(vbb_d[1]-vbb_d[0]), int(vbb_h[1]-vbb_h[0]), int(vbb_w[1]-vbb_w[0])], dtype=int)), # relative
                    'modality': modality, # string
                    'spacing': spacing
                }
        
        # transform
        dct = self.s2_transforms(dct)
        
        # compute the thang
        pred_lat, pred_loc = self.s2_model.classify(dct['image'].unsqueeze(0).to(self.device), dct['coords'].unsqueeze(0).to(self.device), [dct['modality']])
        pred_lat=pred_lat.detach().to('cpu')
        pred_loc=pred_loc.detach().to('cpu')
        
        decoded_label = self.decoder(torch.concat([pred_loc, pred_lat], dim=-1))[0]
        return decoded_label

    def _center_crop(self, img, coords, spacing, size_mm):
            sz_mm = round(size_mm/2)
            sz = [round(sz_mm/sp) for sp in spacing]
            shape = img.shape
            
            coords_lower = []
            coords_upper = []
            for c_ax, sh_ax, sz_ax in zip(coords, shape, sz):
    
                upper = c_ax+sz_ax
                lower = c_ax-sz_ax
                
                if upper<=sh_ax and lower >=0:
                    coords_lower.append(lower)
                    coords_upper.append(upper)
                elif upper > sh_ax:
                    diff = upper-sh_ax
                    coords_upper.append(sh_ax)
                    coords_lower.append(lower-diff)
                elif lower < 0:
                    diff = lower
                    coords_lower.append(0)
                    coords_upper.append(upper+abs(diff))
            
            crop = img[
                int(coords_lower[0]):int(coords_upper[0]),
                int(coords_lower[1]):int(coords_upper[1]),
                int(coords_lower[2]):int(coords_upper[2])
            ]            
            return crop