import os
from TnT.trainer.nnUNetTrainer_single_encoder_baseline import find_dir as baseline
from TnT.trainer.nnUNetTrainer_single_encoder_mixed import find_dir as mixed
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import torch


def get_s1(path, device, folds=(0,1,2,3,4,)):
    os.environ['nnUNet_extTrainer'] = mixed()
    predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=torch.device(device),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=False,
            )
        
    predictor.initialize_from_trained_model_folder(
        path,
        use_folds=folds,
        checkpoint_name="checkpoint_best.pth",
        
    )
    return predictor