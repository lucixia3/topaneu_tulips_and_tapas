from pprint import pprint
from pathlib import Path
from TnT.utils.dataloader import TopAneu_TnTs2_DS, TnTs2_collate, DataLoader
from TnT.utils.transforms import AdaNorm, Compose, MaybeToTensor, MaybeResize, BinarizeAneuChannel, BinarizeVesselChannel, ImageTransformWrapper
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
    ## Do splits
    # ds = TopAneu_TnTs2_DS("/home/tue20260926/Data/topaneu_deployment")
    # folds = ds.split('0.8-0.1-0.1', 42)
    # for id, fold in zip(['train', 'test', 'val'], folds):
    #     fold.preprocess()
    #     fold.save(id)
    
    ## Prep trans
    transforms = Compose([
        MaybeToTensor(),
        AdaNorm.make(),
        MaybeResize(size=64),
        BinarizeAneuChannel(),
        BinarizeVesselChannel(),
    ])
    
    train_transforms = Compose([
        MaybeToTensor(),
        MaybeResize(size=64),
        BinarizeAneuChannel(),
        BinarizeVesselChannel(),
        # ---- Spatial transforms: must apply identically to image + all masks ----
        ImageTransformWrapper(
            RandSpatialCrop(roi_size=(64, 64, 64), random_size=False),
            apply_to='all'
        ),
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
    train = TopAneu_TnTs2_DS.load('train.json', train_transforms)
    val = TopAneu_TnTs2_DS.load('val.json', transforms)
    test = TopAneu_TnTs2_DS.load('test.json', transforms)
    
    ## PrEP DL
    train_dl = DataLoader(train, batch_size=4, shuffle=True)
    val_dl = DataLoader(val, batch_size=4, shuffle=True)
    test_dl = DataLoader(test, batch_size=4, shuffle=False)
    
    ## setup objs
    trainer = Trainer()
    model = TnTS2()
    
    ## train or load
    #model = trainer.train(model, train_dl, val_dl, 10, 5)
    model.load('best_val_loss')

    ## QnD test
    print('#'*20, 'Training ACC', '#'*20)
    train = TopAneu_TnTs2_DS.load('train.json', transforms)
    train_dl = DataLoader(train, batch_size=4, shuffle=True)
    acc = trainer.test(model, train_dl)
    print('#'*20, 'Validation ACC', '#'*20)
    acc = trainer.test(model, val_dl)
    print('#'*20, 'Testing ACC', '#'*20)
    acc = trainer.test(model, test_dl)