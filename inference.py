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
    #S2_path = '/home/tue20260926/Repos/topaneu_tulips_and_tapas/_tune/TnTS2_training_from-12:41:52-18.08.26_clipping_and_new_stds/best_val_loss'
    
    S2_path = {
        'ct':'/home/tue20260926/Repos/topaneu_tulips_and_tapas/_tune/TnTS2_training_from-14:14:06-18._modalityspecific/CT/best_val_loss',
        'mr':'/home/tue20260926/Repos/topaneu_tulips_and_tapas/_tune/TnTS2_training_from-14:14:06-18._modalityspecific/MR/best_val_loss'
    }
    
    ## Prep pipeline
    pl = InferencePipeline(S1_path, S2_path, PATCH_SIZE_VX, PATCH_SIZE_MM)
    
    ## Prep data
    test = TopAneuDS.load('test.json')
    ## Prep evaluator
    ev = TopAneu26LikeEvaluator(pl, None)
    res, agg, avg = ev.eval_list(test.src, test.cases)
    
    # with open('res_agg.json', 'w') as f:
    #     json.dump(agg, f, indent=4)
    # with open('res_agg.json', 'r') as f:
    #     agg = json.load(f)
    
    os.makedirs('results_with_s1_modspec', exist_ok=True)
    ev.plot(agg, 'results_with_s1_modspec/per_class_heatmap.png')
    ev.re_eval_by_modality(res, 'results_with_s1_modspec')
    with open('results_with_s1_modspec/per_cls.json', 'w') as f:
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