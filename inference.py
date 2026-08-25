from TnT.model.stage2 import TnTS2
from TnT.utils.transforms import get_train_test_transforms
from TnT.utils.dataloader import TopAneuDS, TnTs2_collate
from TnT.evaluation.topaneu26 import TopAneu26LikeEvaluator
from TnT.pipeline.inference import InferencePipeline
import json, os

if __name__ == '__main__':
    PATCH_SIZE_VX = 64 # to avoid oom error on local
    PATCH_SIZE_MM = 35
    
    S1_path = '/home/tue20260926/Models/TopAneu-26/Stage1/nnUNetTrainer_single_encoder_mixed__nnUNetPlans__3d_fullres'
    # S2_path = '/home/tue20260926/Repos/topaneu_tulips_and_tapas/_tune/TnTS2_training_from-12:41:52-18.08.26_clipping_and_new_stds/best_val_loss'
    
    S2_path = {
        'ct':'/home/tue20260926/Repos/topaneu_tulips_and_tapas/TnTS2_training_from-10:16:59-25.08.26/CT/best_val_loss',
        'mr':'/home/tue20260926/Repos/topaneu_tulips_and_tapas/TnTS2_training_from-10:16:59-25.08.26/MR/best_val_loss'
    }
    
    ## Prep pipeline
    pl = InferencePipeline(S1_path, S2_path, PATCH_SIZE_VX, PATCH_SIZE_MM, use_tta=True)
    
    ## Prep data
    test = TopAneuDS.load('test.json')
    ## Prep evaluator
    outdir = 'results_tta_exp'
    ev = TopAneu26LikeEvaluator(pl, outdir, use_perfect_segmentations=True)
    res, agg, avg, disc = ev.eval_ds(test)#ev.eval_list(test.src, test.cases)
    with open(f'{outdir}/classification_failure.json', 'w') as f:
            json.dump(disc, f, indent=4)
    
    # with open('res_agg.json', 'w') as f:
    #     json.dump(agg, f, indent=4)
    # with open('res_agg.json', 'r') as f:
    #     agg = json.load(f)
    
    os.makedirs(outdir, exist_ok=True)
    ev.plot(agg, f'{outdir}/per_clas')
    ev.re_eval_by_modality(res, outdir)
    with open(f'{outdir}/per_cls.json', 'w') as f:
        json.dump(avg, f, indent=4)
    
    # ## Prep data
    # test = TopAneuDS.load('train.json')
    # ## Prep evaluator
    # ev = TopAneu26LikeEvaluator(pl, 'train_acc')
    # _ = ev.eval(test)
        
    # ## Prep data
    # test = TopAneuDS.load('val.json')
    # ## Prep evaluator
    # ev = TopAneu26LikeEvaluator(pl, 'val_acc')
    # _ = ev.eval(test)