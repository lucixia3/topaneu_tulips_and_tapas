from pprint import pprint
from pathlib import Path
import os, datetime, torch
from TnT.utils.dataloader import TopAneu_TnTs2_DS, TnTs2_collate, DataLoader
from TnT.utils.transforms import DecodeTarget, LateralityInvariance,  Resample, RandomResample, RandomNonCorrespondingMask, RandomNonCorrespondingMorph, RandomMask, AdaNorm, Compose, MaybeToTensor, MaybeResize, BinarizeAneuChannel, BinarizeVesselChannel, ImageTransformWrapper
from TnT.model.stage2 import TnTS2
from TnT.trainer.n_fold import NFoldTrainer
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
    BATCH_SIZE = 4
    EARLY_STOP_PATCHING = False
    
    ## Do splits
    # ds = TopAneu_TnTs2_DS("/home/tue20260926/Data/topaneu_deployment")
    # folds = ds.split('0.8-0.1-0.1', 42)
    
    # for id, fold in zip(['train', 'test', 'val'], folds):
    #     fold.save(id)
    
    ## Prep trans
    transforms = Compose([
        MaybeToTensor(),
        Resample.make(),
        MaybeResize(size=PATCH_SIZE_VX),
        BinarizeAneuChannel(),
        BinarizeVesselChannel(),
        AdaNorm.make(),
    ])
    
    train_transforms = Compose([
        MaybeToTensor(),
        RandomResample(0.8),
        MaybeResize(PATCH_SIZE_VX),
        BinarizeAneuChannel(),
        BinarizeVesselChannel(),
        
        # ---- Custom stuff ----
        RandomMask(0.2),
        RandomNonCorrespondingMask(0.2),
        RandomNonCorrespondingMorph(0.2),
        
        
        # ---- Spatial transforms: must apply identically to image + all masks ----
        ImageTransformWrapper(
            RandFlip(prob=0.5, spatial_axis=0),
            apply_to='all'
        ),
        ImageTransformWrapper(
            RandFlip(prob=0.5, spatial_axis=1),
            apply_to='all'
        ),
        ImageTransformWrapper(
            RandRotate90(prob=0.5, spatial_axes=(0, 1)),
            apply_to='all'
        ),
        ImageTransformWrapper(
            RandAffine(
                prob=0.3,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                padding_mode='border',
            ),
            apply_to='all'
        ),

        # ---- Intensity-only transforms: image channel exclusively ----
        ImageTransformWrapper(
            RandGaussianNoise(prob=0.2, mean=0.0, std=0.05),
            apply_to=['image']
        ),
        ImageTransformWrapper(
            RandGaussianSmooth(prob=0.15, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
            apply_to=['image']
        ),
        ImageTransformWrapper(
            RandAdjustContrast(prob=0.2, gamma=(0.7, 1.5)),
            apply_to=['image']
        ),
        ImageTransformWrapper(
            RandScaleIntensity(prob=0.2, factors=0.1),
            apply_to=['image']
        ),
        ImageTransformWrapper(
            RandShiftIntensity(prob=0.2, offsets=0.1),
            apply_to=['image']
        ),
        ImageTransformWrapper(
            RandBiasField(prob=0.1, coeff_range=(0.0, 0.3)),
            apply_to=['image']
        ),
        ImageTransformWrapper(
            RandHistogramShift(prob=0.1, num_control_points=(3, 5)),
            apply_to=['image']
        ),
        AdaNorm.make(),
    ])
    
    ## load splits
    if not os.path.exists('tuning-train.json'):
        train = TopAneu_TnTs2_DS.load('train.json', train_transforms)
        train.preprocess(include_bg=0.2, max_items=1 if EARLY_STOP_PATCHING else -1)
        train.save('tuning-train.json')
    else: train = TopAneu_TnTs2_DS.load('tuning-train.json', train_transforms)
    
    if not os.path.exists('tuning-val.json'):
        val = TopAneu_TnTs2_DS.load('val.json', transforms)
        val.preprocess(max_items=1 if EARLY_STOP_PATCHING else -1)
        val.save('tuning-val.json')
    else: val = TopAneu_TnTs2_DS.load('tuning-val.json', transforms)
    
    if not os.path.exists('tuning-test.json'):
        test = TopAneu_TnTs2_DS.load('test.json', transforms)
        test.preprocess(max_items=1 if EARLY_STOP_PATCHING else -1)
        test.save('tuning-test.json')
    else: test = TopAneu_TnTs2_DS.load('tuning-test.json', transforms)
    
    ## PrEP DL
    train.append(val)
    #train_dl = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    #val_dl = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test, batch_size=1, shuffle=False)
    
    ## setup objs
    trainer = NFoldTrainer()
    model = TnTS2.from_pretrained('/home/tue20260926/Repos/topaneu_tulips_and_tapas/_pretrain/15_epochs', *LateralityInvariance.get_n_locs_lats())
    
    ## train or load
    model = trainer.train(model=model, ds=train, train_trans=train_transforms, val_trans=transforms, epochs=20, early_stop=5)

    ## QnD test
    print('#'*20, 'Testing ACC', '#'*20)
    acc = trainer.test(model, test_dl)