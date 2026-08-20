from pprint import pprint
from pathlib import Path
import os, datetime, torch, tqdm
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
    train_transforms = None
    
    ## load splits
    if not os.path.exists('tuning-val.json'):
        train = TopAneu_TnTs2_DS.load('val.json', None)
        train.preprocess(include_bg=0.2, max_items=1 if EARLY_STOP_PATCHING else -1)
        train.save('tuning-val.json')
    else: train = TopAneu_TnTs2_DS.load('tuning-val.json', None)
    
    
    
    for i in range(len(train)):
        print(train[i]['coords'])
    
    
    # cnt = np.zeros(29)
    # for i in tqdm.tqdm(range(len(train))):
    #     cnt[LAT_INV_ANEU[train[i]['location_a']]] += 1
    
    # print(cnt)
    # cnt /= len(train)
    # print(cnt, np.sum(cnt))
    # cnt = np.abs(cnt-1).tolist()
    # print(cnt)
    # np.save('/home/tue20260926/Repos/topaneu_tulips_and_tapas/TnT/trainer/aneu_class_weights.npy', cnt)
