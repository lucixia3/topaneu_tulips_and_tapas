import os
import numpy as np, SimpleITK as sitk
import torch
from monai.transforms import Flip

from TnT.utils.transforms import *

OUTPUT_DIR = 'test_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_test_sample():
    img = np.load('TnT/utils/test/img.npy')
    with open('TnT/utils/test/sample.json', 'r') as f:
        smp = json.load(f)
    smp['image'] = img
    smp['coords'] = np.array(smp['coords'])
    return smp

def _save(name, tensor):
    arr = tensor.numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    arr = np.transpose(arr, (1, 2, 3, 0))
    path = os.path.join(OUTPUT_DIR, f'{name}.nii.gz')
    img = sitk.GetImageFromArray(arr)
    sitk.WriteImage(img, path)
    print(f"[{name}] saved resulting image to {path}")


 
def test_to_tensor_transform():
    smp = load_test_sample()
    print(f"[to_tensor] image type before: {type(smp['image'])}, dtype: {smp['image'].dtype}, shape: {smp['image'].shape}")
    out = MaybeToTensor()(smp)
    print(f"[to_tensor] image type after: {type(out['image'])}, dtype: {out['image'].dtype}, shape: {tuple(out['image'].shape)}")
    _save('to_tensor', out['image'])
 
 
def test_resize_transform(size=64):
    smp = MaybeToTensor()(load_test_sample())
    shape_before = tuple(smp['image'].shape)
    out = MaybeResize(size=size)(smp)
    shape_after = tuple(out['image'].shape)
    print(f"[resize] image shape before: {shape_before} -> after: {shape_after} (target size={size})")
    _save('resize', out['image'])
 
 
def test_adanorm_transform():
    smp = MaybeToTensor()(load_test_sample())
    ch0_before = smp['image'][0].clone()
    mean_before, std_before = ch0_before.mean().item(), ch0_before.std().item()
    out = AdaNorm.make()(smp)
    ch0_after = out['image'][0]
    mean_after, std_after = ch0_after.mean().item(), ch0_after.std().item()
    print(f"[adanorm] modality={out['modality']} | raw channel mean/std before: {mean_before:.4f}/{std_before:.4f} -> after: {mean_after:.4f}/{std_after:.4f}")
    _save('adanorm', out['image'])
 
 
def test_binarize_vessel_channel_transform():
    smp = MaybeToTensor()(load_test_sample())
    smp['image'][2, :] = torch.rand_like(smp['image'][2, :]) * 5  # force non-binary values
    out = BinarizeVesselChannel()(smp)
    unique_vals = torch.unique(out['image'][2, :])
    print(f"[binarize_vessel_channel] unique values in vessel channel after binarization: {unique_vals.tolist()}")
    _save('binarize_vessel_channel', out['image'])
 
 
def test_binarize_aneu_channel_transform():
    smp = MaybeToTensor()(load_test_sample())
    smp['image'][1, :] = torch.rand_like(smp['image'][1, :]) * 5  # force non-binary values
    out = BinarizeAneuChannel()(smp)
    unique_vals = torch.unique(out['image'][1, :])
    print(f"[binarize_aneu_channel] unique values in aneu channel after binarization: {unique_vals.tolist()}")
    _save('binarize_aneu_channel', out['image'])
 
 
def test_image_transform_wrapper():
    smp = MaybeToTensor()(load_test_sample())
    before = smp['image'].clone()
    wrapper = ImageTransformWrapper(Flip(spatial_axis=0), apply_to=['image', 'aneu', 'vessel'])
    out = wrapper(smp)
    changed = not torch.equal(before, out['image'])
    print(f"[wrapper] applied Flip(spatial_axis=0) to all channels | tensor changed: {changed} | shape: {tuple(out['image'].shape)}")
    _save('wrapper', out['image'])
 
 
def test_random_mask_transform():
    smp = MaybeToTensor()(load_test_sample())
    smp['image'][1, :] = 1.0
    smp['image'][2, :] = 1.0
    trans = RandomMask(prob=1.0)  # force execution
    out = trans(smp)
    n_zeroed_raw = (out['image'][0] == 0).sum().item()
    n_zeroed_aneu = (out['image'][1] == 0).sum().item()
    n_zeroed_vessel = (out['image'][2] == 0).sum().item()
    print(f"[random_mask] zeroed voxels -> raw:{n_zeroed_raw} aneu:{n_zeroed_aneu} vessel:{n_zeroed_vessel} (aneu/vessel counts should match, mask is corresponding across channels)")
    _save('random_mask', out['image'])
 
 
def test_random_noncorresponding_mask_transform():
    smp = MaybeToTensor()(load_test_sample())
    smp['image'][1, :] = 1.0
    smp['image'][2, :] = 1.0
    trans = RandomNonCorrespondingMask(prob=1.0)  # force execution
    out = trans(smp)
    zeroed_aneu = (out['image'][1] == 0)
    zeroed_vessel = (out['image'][2] == 0)
    identical_mask = torch.equal(zeroed_aneu, zeroed_vessel)
    print(f"[random_noncorresponding_mask] aneu zeroed voxels: {zeroed_aneu.sum().item()} | vessel zeroed voxels: {zeroed_vessel.sum().item()} | masks identical across channels: {identical_mask} (should be False)")
    _save('random_noncorresponding_mask', out['image'])
 
 
def test_random_noncorresponding_morph_transform():
    smp = MaybeToTensor()(load_test_sample())
    smp['image'][1, :] = 1.0
    smp['image'][2, :] = 1.0
    before = smp['image'].clone()
    trans = RandomNonCorrespondingMorph(prob=1.0)  # force execution
    out = trans(smp)
    changed_raw = not torch.equal(before[0], out['image'][0])
    changed_aneu = not torch.equal(before[1], out['image'][1])
    changed_vessel = not torch.equal(before[2], out['image'][2])
    print(f"[random_noncorresponding_morph] channel changed -> raw:{changed_raw} aneu:{changed_aneu} vessel:{changed_vessel}")
    _save('random_noncorresponding_morph', out['image'])
 
 
def test_resample_transform():
    smp = MaybeToTensor()(load_test_sample())
    shape_before = tuple(smp['image'].shape)
    spacing_before = smp['spacing']
    trans = Resample.make()
    out = trans(smp)
    shape_after = tuple(out['image'].shape)
    spacing_after = out['spacing']
    print(f"[resample] spacing before: {spacing_before} -> after: {spacing_after} (target: {trans.spacing})")
    print(f"[resample] shape before: {shape_before} -> after: {shape_after}")
    _save('resample', out['image'])
 
 
def test_random_resample_transform():
    smp = MaybeToTensor()(load_test_sample())
    shape_before = tuple(smp['image'].shape)
    trans = RandomResample(prob=1.0)  # force execution
    out = trans(smp)
    shape_after = tuple(out['image'].shape)
    print(f"[random_resample] executed (prob=1.0) | shape before: {shape_before} -> after: {shape_after} | spacing after: {out['spacing']}")
    _save('random_resample', out['image'])


def test_all_transforms():
    testsample = load_test_sample()
    test_aneu_label_transform()
    test_vessel_label_transform()
 
    test_to_tensor_transform()
    test_resize_transform()
    test_adanorm_transform()
    test_binarize_vessel_channel_transform()
    test_binarize_aneu_channel_transform()
    test_image_transform_wrapper()
    test_random_mask_transform()
    test_random_noncorresponding_mask_transform()
    test_random_noncorresponding_morph_transform()
    test_resample_transform()
    test_random_resample_transform()

def test_aneu_label_transform():
    encoder = LateralityInvariance()
    decoder = DecodeTarget()
    
    for i in range(51):
        
        enc = np.array(encoder(i))
        
        dec = decoder(enc)
        
        print(f'Testing Aneu encoding/decoding: input={i}; output={dec}; Success={i==dec[-1]}')
        
def test_vessel_label_transform():
    encoder = LateralityInvarianceForVessels()
    decoder = DecodeTargetForVessels()
    
    for i in range(37):
        
        enc = np.array(encoder(i))
        
        dec = decoder(enc)
        
        print(f'Testing Vessel encoding/decoding: input={i}; output={dec}; Success={i==dec[-1]}')