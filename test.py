from pprint import pprint
from pathlib import Path
import torch, os, json, numpy as np
import SimpleITK as sitk
from TnT.utils.dataloader import TopAneu_TnTs2_DS, TnTs2_collate, DataLoader
from TnT.utils.transforms import DecodeTarget, LateralityInvariance, AdaNorm, Compose, MaybeToTensor, MaybeResize, BinarizeAneuChannel, BinarizeVesselChannel, ImageTransformWrapper
from TnT.model.stage2 import TnTS2
from TnT.trainer.base import Trainer
from TnT.utils.test.transforms import test_all_transforms
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
def get_image_info(img: sitk.Image) -> dict:
    return {
        "shape": img.GetSize(),          # (x, y, z)
        "spacing": img.GetSpacing(),      # (sx, sy, sz)
        "origin": img.GetOrigin(),        # (ox, oy, oz)
        "orientation": sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img.GetDirection()) # 9-element direction cosine matrix (flattened)
    }

if __name__ == '__main__':
    test_all_transforms()