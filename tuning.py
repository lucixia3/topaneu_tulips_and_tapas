from pprint import pprint
from pathlib import Path
import os, datetime, torch
from TnT.utils.dataloader import TopAneu_TnTs2_DS, TnTs2_collate_dev, DataLoader
from TnT.utils.transforms import get_train_test_transforms,  Resample, RandomResample, RandomNonCorrespondingMask, RandomNonCorrespondingMorph, RandomMask, AdaNorm, Compose, MaybeToTensor, MaybeResize, BinarizeAneuChannel, BinarizeVesselChannel, ImageTransformWrapper
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
    train_transforms, transforms = get_train_test_transforms(PATCH_SIZE_VX)
    
    ## load splits
    if not os.path.exists('tuning-train.json'):
        train = TopAneu_TnTs2_DS.load('train.json', train_transforms)
        train.preprocess(include_bg=0.2, max_items=1 if EARLY_STOP_PATCHING else -1)
        train.save('tuning-train.json')
    else: train = TopAneu_TnTs2_DS.load('tuning-train.json', train_transforms)
    train.wdir = 'tuning-train'
    
    if not os.path.exists('tuning-val.json'):
        val = TopAneu_TnTs2_DS.load('val.json', transforms)
        val.preprocess(max_items=1 if EARLY_STOP_PATCHING else -1)
        val.save('tuning-val.json')
    else: val = TopAneu_TnTs2_DS.load('tuning-val.json', transforms)
    val.wdir = 'tuning-val'
    
    if not os.path.exists('tuning-test.json'):
        test = TopAneu_TnTs2_DS.load('test.json', transforms)
        test.preprocess(max_items=1 if EARLY_STOP_PATCHING else -1)
        test.save('tuning-test.json')
    else: test = TopAneu_TnTs2_DS.load('tuning-test.json', transforms)
    test.wdir = 'tuning-test'
    
    ## PrEP DL
    #train.append(val)
    train_dl = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test, batch_size=1, shuffle=False)
    
    ## setup objs
    trainer = BasicTrainer()
    model = TnTS2.from_pretrained('/home/tue20260926/Repos/topaneu_tulips_and_tapas/_pretrain/TnTS2_pretraining_from-13:40:57-04.08.26/best_val_loss')
    
    # ## train or load
    model = trainer.train(model, train_dl, val_dl, 10, 5)#(model=model, ds=train, train_trans=train_transforms, val_trans=transforms, epochs=20, early_stop=5)

    ## QnD test
    print('#'*20, 'Testing ACC', '#'*20)
    acc = trainer.test(model, test_dl)