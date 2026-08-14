from TnT.model.stage2 import TnTS2
from TnT.utils.transforms import get_train_test_transforms
from TnT.utils.dataloader import TopAneuDS, TnTs2_collate
from TnT.evaluation.topaneu26 import TopAneu26LikeEvaluator
from TnT.pipeline.inference import InferencePipeline

if __name__ == '__main__':
    PATCH_SIZE_VX = 64 # to avoid oom error on local
    PATCH_SIZE_MM = 35
    BATCH_SIZE = 4
    EARLY_STOP_PATCHING = False
    
    
    
    ## Prep model
    model = TnTS2()
    model.load('/home/tue20260926/Repos/topaneu_tulips_and_tapas/_pretrain/new_architecture/new')
    
    ## Prep pipeline
    pl = InferencePipeline(model, PATCH_SIZE_VX, PATCH_SIZE_MM)
    
    ## Prep data
    test = TopAneuDS.load('test.json')
    ## Prep evaluator
    ev = TopAneu26LikeEvaluator(pl, 'test_acc')
    _ = ev.eval(test)
    
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