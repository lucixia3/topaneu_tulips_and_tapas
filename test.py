from pprint import pprint
from pathlib import Path
import torch
from TnT.utils.dataloader import TopAneu_TnTs2_DS, TnTs2_collate, DataLoader
from TnT.utils.transforms import DecodeTarget, LateralityInvariance, AdaNorm, Compose, MaybeToTensor, MaybeResize, BinarizeAneuChannel, BinarizeVesselChannel, ImageTransformWrapper
from TnT.model.stage2 import TnTS2
from TnT.trainer.base import Trainer
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

if __name__ == '__main__':
    transform = LateralityInvariance()
    decode = DecodeTarget()
    for i in range(1, 51):
        
        batch = {'location': i}
        print('cls 50', batch)
        enc = transform(batch)
        print('cls enc', enc['location'])
        dec = decode(torch.tensor(batch['location']))
        print('dec', dec)