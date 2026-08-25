from pprint import pprint
from pathlib import Path
import os, datetime, torch
from TnT.utils.dataloader import TopAneu_TnTs2_DS, TnTs2_collate, DataLoader, TopAneuDS
from TnT.utils.transforms import get_train_test_transforms, DecodeAneu, DecodeVessel, Resample, RandomResample, RandomNonCorrespondingMask, RandomNonCorrespondingMorph, RandomMask, AdaNorm, Compose, MaybeToTensor, MaybeResize, BinarizeAneuChannel, BinarizeVesselChannel, ImageTransformWrapper
from TnT.model.stage2 import TnTS2
from TnT.trainer.base import BasicTrainer
from TnT.model.modality_specific import TnTS2_Specific
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
    pt_source = '/home/tue20260926/Repos/topaneu_tulips_and_tapas/_pretrain/new_architecture/new'
    
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
    
    ## separate modalities
    ct_tr, mr_tr = train.separate_by_modality()
    ct_val, mr_val = val.separate_by_modality()
    
    ## setup working variables
    wdir = Path(datetime.datetime.now().strftime(r'TnTS2_modespec_training_from-%H:%M:%S-%d.%m.%y'))
    
    ## CT
    train_dl = DataLoader(ct_tr, batch_size=BATCH_SIZE, shuffle=True, collate_fn=TnTs2_collate)
    val_dl = DataLoader(ct_val, batch_size=BATCH_SIZE, shuffle=True, collate_fn=TnTs2_collate)
    trainer_ct = BasicTrainer()
    model_ct = TnTS2()
    model_ct.load(pt_source)
    model_ct = trainer_ct.train(model=model_ct, train_dl=train_dl, val_dl=val_dl, epochs=20, early_stop=5, use_aneu_class_balancing=False, wdir=wdir/'CT')#(model=model, ds=train, train_trans=train_transforms, val_trans=transforms, epochs=20, early_stop=5)

    ## MR
    train_dl = DataLoader(mr_tr, batch_size=BATCH_SIZE, shuffle=True, collate_fn=TnTs2_collate)
    val_dl = DataLoader(mr_val, batch_size=BATCH_SIZE, shuffle=True, collate_fn=TnTs2_collate)
    trainer_mr = BasicTrainer()
    model_mr = TnTS2()
    model_mr.load(pt_source)
    model_mr = trainer_mr.train(model=model_mr, train_dl=train_dl, val_dl=val_dl, epochs=20, early_stop=5, use_aneu_class_balancing=False, wdir=wdir/'MR')#(model=model, ds=train, train_trans=train_transforms, val_trans=transforms, epochs=20, early_stop=5)
    
    ## QnD test
    trainer_ct.wdir=trainer_ct.wdir.parent
    test = TopAneuDS('test.json')
    print('#'*20, 'Testing TopAneu performance', '#'*20)
    model = TnTS2_Specific(model_mr, model_ct)
    acc = trainer_ct.test_TopAneu(model, test)