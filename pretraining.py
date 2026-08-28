from pprint import pprint
from pathlib import Path
import os, datetime
from tqdm import tqdm
from TnT.utils.dataloader import TopAneu_TnTs2_DS, TnTs2_collate, DataLoader, TopAneu_TnTs2_DS_for_vessel_pt
from TnT.utils.transforms import DecodeVessel, get_train_test_transforms, Resample, RandomResample, RandomNonCorrespondingMask, RandomNonCorrespondingMorph, RandomMask, AdaNorm, Compose, MaybeToTensor, MaybeResize, BinarizeAneuChannel, BinarizeVesselChannel, ImageTransformWrapper
from TnT.model.stage2 import TnTS2, TnTS2Loss
from TnT.trainer.accelerated import AccelTrainer
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
    PATCH_SIZE_VX = 64 # resizing in transforms
    PATCH_SIZE_MM = 35 # extracting from image
    PATCH_PER_VLOC = 8 # 4*36 = 144 samples per available image !! yields hella training samples
    EARLY_STOP_PATCHING = 1
    BATCH_SIZE = 8
    
    # ## Do splits
    # ds = TopAneu_TnTs2_DS("/home/20260926/datasets/TopAneu-26", patch_size_mm=PATCH_SIZE_MM)
    # folds = ds.split('0.8-0.1-0.1', 42)
    
    # for id, fold in zip(['train', 'test', 'val'], folds):
    #     fold.save(id)

    ## Prep trans
    train_transforms, transforms = get_train_test_transforms(PATCH_SIZE_VX)
    
    ## load splits
    if not os.path.exists('pre-ves-val.json'):
        val = TopAneu_TnTs2_DS_for_vessel_pt.load('val.json', transforms)
        val.preprocess(patches_per_vloc=1, max_items=1 if EARLY_STOP_PATCHING else -1)
        val.save('pre-ves-val.json')
    else: val = TopAneu_TnTs2_DS_for_vessel_pt.load('pre-ves-val.json', transforms)
    val.wdir = "pre-ves-val"
        
    if not os.path.exists('pre-ves-train.json'):
        train = TopAneu_TnTs2_DS_for_vessel_pt.load('train.json', train_transforms)
        train.preprocess(patches_per_vloc=PATCH_PER_VLOC, max_items=1 if EARLY_STOP_PATCHING else -1)
        train.save('pre-ves-train.json')
    else: train = TopAneu_TnTs2_DS_for_vessel_pt.load('pre-ves-train.json', train_transforms)
    train.wdir = "pre-ves-train"
    
    if not os.path.exists('pre-ves-test.json'):
        test = TopAneu_TnTs2_DS_for_vessel_pt.load('test.json', transforms)
        test.preprocess(patches_per_vloc=1, max_items=1 if EARLY_STOP_PATCHING else -1)
        test.save('pre-ves-test.json')
    else: test = TopAneu_TnTs2_DS_for_vessel_pt.load('pre-ves-test.json', transforms)
    test.wdir = "pre-ves-test"
    
    train.patch_size_mm=PATCH_SIZE_MM
    val.patch_size_mm=PATCH_SIZE_MM
    test.patch_size_mm=PATCH_SIZE_MM
    
    ## PrEP DL
    train_dl = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=TnTs2_collate)
    val_dl = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True, collate_fn=TnTs2_collate)
    test_dl = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=TnTs2_collate)
    
    ## setup objs
    trainer = BasicTrainer(lr=1e-3)
    model = TnTS2()
    loss = TnTS2Loss(model.n_locs_v, model.n_locs_a)
    
    ## train or load
    model = trainer.train(model, train_dl, val_dl, 200, 10, wdir=Path(datetime.datetime.now().strftime(r'TnTS2_pretraining_from-%H:%M:%S-%d.%m.%y')), loss=loss)

    ## QnD test
    print('#'*20, 'Testing ACC', '#'*20)
    acc = trainer.test(model, test_dl, decoder=DecodeVessel(), target='vessel')