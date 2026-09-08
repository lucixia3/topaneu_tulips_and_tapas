from TnT.model.stage2 import TnTS2
from TnT.utils.transforms import *
from TnT.utils.dataloader import TopAneuDS, TnTs2_collate, TopAneu_TnTs2_DS
from TnT.evaluation.topaneu26 import TopAneu26LikeEvaluator
from TnT.pipeline.inference import InferencePipeline
import json, os
from pathlib import Path
if __name__ == '__main__':
    PATCH_SIZE_VX = 64 # to avoid oom error on local
    PATCH_SIZE_MM = 35
    
    S1_path = '/home/tue20260926/Models/TopAneu-26/Stage1/nnUNetTrainer_single_encoder_mixed__nnUNetPlans__3d_fullres'
    S2_path = '/home/tue20260926/Repos/topaneu_tulips_and_tapas/TnTS2_training_from-08:25:58-05.09.26/best_val_loss'
    
    # S2_path = {
    #     'ct':'/home/tue20260926/Repos/topaneu_tulips_and_tapas/TnTS2_training_from-10:16:59-25.08.26/CT/best_val_loss',
    #     'mr':'/home/tue20260926/Repos/topaneu_tulips_and_tapas/TnTS2_training_from-10:16:59-25.08.26/MR/best_val_loss'
    # }
    
    trnsfrm = Compose([
            MaybeToTensor(),
            ClipCtaIntensities(),
            MaybeResize(size=PATCH_SIZE_VX),
            BinarizeVesselChannel(),
            BinarizeAneuChannel(),
            AdaNorm.make(True),
    ])
    
    ## Prep pipeline
    pl = InferencePipeline(S1_path, S2_path, PATCH_SIZE_VX, PATCH_SIZE_MM, use_tta=True, transforms=trnsfrm)
    
    ## Prep data
    test = TopAneuDS.load('test.json')
    ## Prep evaluator
    outdir = Path(S2_path).parent/'s1'
    ev = TopAneu26LikeEvaluator(pl, outdir, precomp_predictions='/home/tue20260926/Data/TNT/s1/infersTs_altTrainer_best')
    res, agg, avg, disc = ev.eval_ds(test)#ev.eval_list(test.src, test.cases)
    with open(f'{outdir}/classification_failure.json', 'w') as f:
            json.dump(disc, f, indent=4)
    
    os.makedirs(outdir, exist_ok=True)
    ev.plot(agg, f'{outdir}/per_clas')
    ev.re_eval_by_modality(res, outdir)
    with open(f'{outdir}/per_cls.json', 'w') as f:
        json.dump(avg, f, indent=4)

    ## Prep pipeline
    pl = InferencePipeline(S1_path, S2_path, PATCH_SIZE_VX, PATCH_SIZE_MM, use_tta=True, transforms=trnsfrm)
       
    ## Prep data
    test = TopAneuDS.load('test.json')
    ## Prep evaluator
    outdir = Path(S2_path).parent/'perfect'
    ev = TopAneu26LikeEvaluator(pl, outdir, precomp_predictions=None)
    res, agg, avg, disc = ev.eval_ds(test)#ev.eval_list(test.src, test.cases)
    with open(f'{outdir}/classification_failure.json', 'w') as f:
            json.dump(disc, f, indent=4)
    
    os.makedirs(outdir, exist_ok=True)
    ev.plot(agg, f'{outdir}/per_clas')
    ev.re_eval_by_modality(res, outdir)
    with open(f'{outdir}/per_cls.json', 'w') as f:
        json.dump(avg, f, indent=4)