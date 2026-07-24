from pprint import pprint
from pathlib import Path
import os
from TnT.utils.dataloader import TopAneu_TnTs2_DS, TnTs2_collate, DataLoader, TopAneu_TnTs2_DS_for_vessel_pt
from TnT.utils.transforms import get_train_test_transforms, DecodeTargetForVessels, LateralityInvarianceForVessels, Resample, RandomResample, RandomNonCorrespondingMask, RandomNonCorrespondingMorph, RandomMask, AdaNorm, Compose, MaybeToTensor, MaybeResize, BinarizeAneuChannel, BinarizeVesselChannel, ImageTransformWrapper
from TnT.model.stage2_dev import TnTS2
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

if __name__ == '__main__':
    PATCH_SIZE_VX = 64 # to avoid oom error on local
    PATCH_PER_VLOC = 1
    EARLY_STOP_PATCHING = True
    
    ## Do splits
    # ds = TopAneu_TnTs2_DS("/home/tue20260926/Data/topaneu_deployment")
    # folds = ds.split('0.8-0.1-0.1', 42)
    
    # for id, fold in zip(['train', 'test', 'val'], folds):
    #     fold.save(id)
    
    ## Prep trans
    train_transforms, transforms = get_train_test_transforms(PATCH_SIZE_VX)
    
    ## load splits
    if not os.path.exists('pre-ves-train.json'):
        train = TopAneu_TnTs2_DS_for_vessel_pt.load('train.json', train_transforms)
        train.preprocess(patches_per_vloc=PATCH_PER_VLOC, max_items=1 if EARLY_STOP_PATCHING else -1)
        train.save('pre-ves-train.json')
    else: train = TopAneu_TnTs2_DS_for_vessel_pt.load('pre-ves-train.json', train_transforms)
    
    if not os.path.exists('pre-ves-val.json'):
        val = TopAneu_TnTs2_DS_for_vessel_pt.load('val.json', transforms)
        val.preprocess(patches_per_vloc=PATCH_PER_VLOC, max_items=1 if EARLY_STOP_PATCHING else -1)
        val.save('pre-ves-val.json')
    else: val = TopAneu_TnTs2_DS_for_vessel_pt.load('pre-ves-val.json', transforms)
    
    if not os.path.exists('pre-ves-test.json'):
        test = TopAneu_TnTs2_DS_for_vessel_pt.load('test.json', transforms)
        test.preprocess(patches_per_vloc=PATCH_PER_VLOC, max_items=1 if EARLY_STOP_PATCHING else -1)
        test.save('pre-ves-test.json')
    else: test = TopAneu_TnTs2_DS_for_vessel_pt.load('pre-ves-test.json', transforms)
    
    ## PrEP DL
    train_dl = DataLoader(train, batch_size=4, shuffle=True)
    val_dl = DataLoader(val, batch_size=4, shuffle=True)
    test_dl = DataLoader(test, batch_size=4, shuffle=False)
    
    ## setup objs
    trainer = BasicTrainer()
    model = TnTS2(*LateralityInvarianceForVessels.get_n_locs_lats())
    
    ## train or load
    model = trainer.train(model, train_dl, val_dl, 1, None)

    ## QnD test
    print('#'*20, 'Training ACC', '#'*20)
    train = TopAneu_TnTs2_DS_for_vessel_pt.load('pre-ves-train.json', transforms)
    train_dl = DataLoader(train, batch_size=4, shuffle=True)
    acc = trainer.test(model, train_dl, decoder=DecodeTargetForVessels())
    print('#'*20, 'Validation ACC', '#'*20)
    acc = trainer.test(model, val_dl, decoder=DecodeTargetForVessels())
    print('#'*20, 'Testing ACC', '#'*20)
    acc = trainer.test(model, test_dl, decoder=DecodeTargetForVessels())