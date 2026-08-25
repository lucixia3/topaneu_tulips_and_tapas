from pprint import pprint
from pathlib import Path
import os, datetime, torch, tqdm
import SimpleITK as sitk
from TnT.utils.dataloader import TopAneu_TnTs2_DS, TnTs2_collate, DataLoader
from TnT.utils.transforms import get_train_test_transforms, DecodeAneu, DecodeVessel, Resample, RandomResample, RandomNonCorrespondingMask, RandomNonCorrespondingMorph, RandomMask, AdaNorm, Compose, MaybeToTensor, MaybeResize, BinarizeAneuChannel, BinarizeVesselChannel, ImageTransformWrapper
from TnT.model.stage2 import TnTS2
from TnT.trainer.base import BasicTrainer
from monai.transforms import (
    RandAffined,
    RandAffine,
    RandFlip,
    RandRotate90,
    RandSpatialCrop,
    RandGaussianNoise,
    RandAdjustContrast,
    RandGaussianSmooth,
    RandScaleIntensity,
    RandShiftIntensity,
    RandBiasField,
    RandHistogramShift,
    NormalizeIntensity,
)
import numpy as np
from TnT.utils.transforms.labeling import LAT_INV_ANEU

if __name__ == '__main__':
    PATCH_SIZE_VX = 64 # to avoid oom error on local
    BATCH_SIZE = 4
    EARLY_STOP_PATCHING = False
    
    ## Prep trans
    train_transforms, transforms = get_train_test_transforms(PATCH_SIZE_VX)
    
    
    train = TopAneu_TnTs2_DS.load('val.json', None)
    train.preprocess(include_bg=False, max_items=-1)
    
    
    smp=train[42]
    img = smp['image']
    for i in range(3):
        sitk.WriteImage(sitk.GetImageFromArray(img[i]), f'channel_{i}.nii.gz')
    print('Aneu Class:', smp['location_a'])
    print('Vessel Class:', smp['location_v'])
    print('Coords:', smp['coords'])
    print('iid:', smp['id'])

    train.transforms=transforms
    smp=train[42]
    img = smp['image'].numpy()
    for i in range(3):
        sitk.WriteImage(sitk.GetImageFromArray(img[i]), f'T_channel_{i}.nii.gz')
    print('T Aneu Class:', smp['location_a'])
    print('T Vessel Class:', smp['location_v'])
    print('T Coords:', smp['coords'])
    print('T iid:', smp['id'])
    
    img = sitk.ReadImage("/home/tue20260926/Repos/TopAneu-26/topaneu_release/vessel_masks/topaneu_center2_mr_085.nii.gz")
    arr = sitk.GetArrayFromImage(img)
    vbb_coords = np.argwhere(arr) # VBB = Vessel Bounding Box
    vbb_d = [int(np.min(vbb_coords[:, 0])), int(np.max(vbb_coords[:, 0]))]
    vbb_h = [int(np.min(vbb_coords[:, 1])), int(np.max(vbb_coords[:, 1]))]
    vbb_w = [int(np.min(vbb_coords[:, 2])), int(np.max(vbb_coords[:, 2]))]
    vbb_msk = np.zeros_like(arr)
    vbb_msk[vbb_d[0]:vbb_d[1], vbb_h[0]:vbb_h[1], vbb_w[0]:vbb_w[1]]=1
    vbb_img=sitk.GetImageFromArray(vbb_msk)
    vbb_img.CopyInformation(img)
    sitk.WriteImage(vbb_img, 'topaneu_center2_mr_085_vbb.nii.gz')
    
    
    # cnt = np.zeros(29)
    # for i in tqdm.tqdm(range(len(train))):
    #     cnt[LAT_INV_ANEU[train[i]['location_a']]] += 1
    
    # print(cnt)
    # cnt /= len(train)
    # print(cnt, np.sum(cnt))
    # cnt = np.abs(cnt-1).tolist()
    # print(cnt)
    # np.save('/home/tue20260926/Repos/topaneu_tulips_and_tapas/TnT/trainer/aneu_class_weights.npy', cnt)
