# nnunetv2/training/nnUNetTrainer/nnUNetTrainer_single_encoder.py

# ============================================================
# Imports
# ============================================================
# Standard libs + PyTorch + nnU-Net internals.
# This trainer is a customized single-encoder baseline that:
#   1) splits training data into CTA vs MRA streams
#   2) alternates CTA/MRA iterations during training/validation
#   3) logs modality-specific Dice during validation
import os, re, warnings, multiprocessing, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from time import time,sleep
from typing import List, Optional, Dict, Any

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.configuration import default_num_processes
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

# Transforms
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
# Transforms, END

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

print("Loaded nnUNetTrainer from:", __file__)
# ============================================================
# Modality identification (CTA vs MRA)
# ============================================================
# We infer modality from the case identifier string (filename/key).
# Convention:
#   - return 0 for CT(A)
#   - return 1 for MR(A)
# NOTE: If no pattern matches, default to 0 (CT) to be safe.
_PAT_MR = re.compile(r"(?:^mr[a]?_|_mr[a]?_|_mr[a]?$)", re.IGNORECASE)
_PAT_CT = re.compile(r"(?:^ct[a]?_|_ct[a]?_|_ct[a]?$)", re.IGNORECASE)

def find_dir():
    return os.path.dirname(os.path.realpath(__file__))

def _infer_mid_from_identifier(identifier: str) -> int:
    s = str(identifier)
    if _PAT_MR.search(s): return 1
    if _PAT_CT.search(s): return 0
    return 0


class ApplyTransformToChannels(BasicTransform):
    """
    Apply a transform only to selected channels of data_dict["data"].

    Assumes:
        data_dict["data"] shape is (C, X, Y, Z) or (C, X, Y) for a single sample.
    This is the format commonly used inside batchgenerators/nnUNet sample transforms.

    Example:
        ApplyTransformToChannels(GaussianNoiseTransform(...), channels=(0,))
    """

    def __init__(self, transform: BasicTransform, channels):
        super().__init__()
        self.transform = transform
        self.channels = list(channels)

    def __call__(self, **data_dict):
        if "data" not in data_dict:
            return data_dict

        data = data_dict["data"]
        if data is None:
            return data_dict

        if not isinstance(data, np.ndarray):
            # if somehow tensor appears, convert handling accordingly
            # but normally batchgenerators transforms receive numpy arrays here
            return data_dict

        if data.ndim < 2:
            return data_dict

        # Copy only selected channels
        selected = data[self.channels].copy()

        # Run wrapped transform on selected channels only
        tmp_dict = dict(data_dict)
        tmp_dict["data"] = selected
        tmp_dict = self.transform(**tmp_dict)

        # Write transformed channels back
        out = data.copy()
        out[self.channels] = tmp_dict["data"]
        data_dict["data"] = out
        return data_dict


class nnUNetTrainer_single_encoder_baseline(nnUNetTrainer):
    """
    Single-encoder/decoder baseline trainer.

    Core idea:
    - Use ONE shared network for both CTA and MRA (same encoder/decoder).
    - During training, build two separate dataloader streams (CTA/MRA) and
      alternate batches iteration-wise.
    - During validation, compute modality-specific mean foreground Dice and
      log them separately (CTA vs MRA), while still calling the parent hooks
      to keep standard nnU-Net logging/early stopping behavior intact.
    """

    # ============================================================
    # Small utilities
    # ============================================================
    def _is_global_rank0(self) -> bool:
        """
        Returns True if this process is the global rank 0 (DDP main process),
        so we only print/log once.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                return torch.distributed.get_rank() == 0
            except Exception:
                return True
        return True

    def _patch_progress_plot(self):
        """
        Monkey-patch nnU-Net's logger plotting function to also plot
        modality-specific curves (mean_fg_dice_cta / mean_fg_dice_mra)
        if present in my_fantastic_logging.
        """
        import matplotlib.pyplot as plt
        from os.path import join as _join

        def _plot_progress_png_extended(_self, output_folder):
            # Pick the primary keys we expect to exist
            keys = [k for k in ('train_losses', 'val_losses', 'mean_fg_dice') if k in _self.my_fantastic_logging]
            if not keys: return

            # Determine the latest epoch index available for all required curves
            epoch = min(len(_self.my_fantastic_logging[k]) for k in keys) - 1
            if epoch < 0: return

            fig, ax_all = plt.subplots(3, 1, figsize=(22, 36))
            ax = ax_all[0]
            ax2 = ax.twinx()
            x = list(range(epoch + 1))

            # Loss curves
            ax.plot(x, _self.my_fantastic_logging['train_losses'][:epoch + 1], label='loss_tr')
            ax.plot(x, _self.my_fantastic_logging['val_losses'][:epoch + 1], label='loss_val')

            # Dice curves (overall + modality-specific if present)
            if 'mean_fg_dice' in _self.my_fantastic_logging:
                ax2.plot(x, _self.my_fantastic_logging['mean_fg_dice'][:epoch + 1], label='dice', color='g')
            if 'mean_fg_dice_cta' in _self.my_fantastic_logging:
                ax2.plot(x, _self.my_fantastic_logging['mean_fg_dice_cta'][:epoch + 1], '--', label='dice_cta')
            if 'mean_fg_dice_mra' in _self.my_fantastic_logging:
                ax2.plot(x, _self.my_fantastic_logging['mean_fg_dice_mra'][:epoch + 1], '--', label='dice_mra')

            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')

            # Epoch time curve
            ax_all[1].plot(
                x,
                [i - j for i, j in zip(
                    _self.my_fantastic_logging['epoch_end_timestamps'][:epoch + 1],
                    _self.my_fantastic_logging['epoch_start_timestamps']
                )][:epoch + 1]
            )
            ax_all[1].set_ylabel('time [s]')

            # LR curve
            ax_all[2].plot(x, _self.my_fantastic_logging['lrs'][:epoch + 1])
            ax_all[2].set_ylabel('lr')

            for a in ax_all:
                a.set_xlabel('epoch')

            plt.tight_layout()
            fig.savefig(_join(output_folder, 'progress.png'))
            plt.close()

        # Bind to the logger instance
        self.logger.plot_progress_png = _plot_progress_png_extended.__get__(self.logger, type(self.logger))

    def _log_with_nan(self, key: str, value, epoch: int):
        """
        Ensure a logging list exists and is long enough up to `epoch`,
        then write value (or NaN if value is None).
        This prevents index errors when some epochs don't have a modality metric.
        """
        import numpy as np
        log = self.logger.my_fantastic_logging
        lst = log.setdefault(key, [])
        while len(lst) <= epoch:
            lst.append(np.nan)
        lst[epoch] = np.nan if value is None else float(value)

    # ============================================================
    # Initialization
    # ============================================================
    def initialize(self):
        """
        Initialize trainer state.
        - Call parent initialize() to build plans/network/loss/optim, etc.
        - Override some hyperparams via environment variables (optional).
        - Infer dataset class + number of input channels.
        - Patch logger plotting for modality-specific curves.
        """
        if self.was_initialized:
            raise RuntimeError("initialize called twice")
        super().initialize()

        # Optional overrides via environment variables:
        #   NUM_EPOCHS: override total epochs
        #   RECALL_OVERSAMPLE: oversample foreground percent (nnU-Net style)
        #   FINETUNE_LR: override initial lr
        try:
            self.num_epochs = int(os.getenv("NUM_EPOCHS", self.num_epochs))
            self.configuration_manager.num_epochs = self.num_epochs
        except Exception:
            pass

        self.oversample_foreground_percent = float(
            os.getenv("RECALL_OVERSAMPLE", str(self.oversample_foreground_percent))
        )

        try:
            self.initial_lr = float(os.getenv("FINETUNE_LR", str(getattr(self, "initial_lr", 0.01))))
        except Exception:
            pass

        # Track input channels + dataset reader class
        self.num_input_channels = determine_num_input_channels(
            self.plans_manager, self.configuration_manager, self.dataset_json
        )
        self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        self.was_initialized = True

        # Rank-0 only debug print
        if self._is_global_rank0():
            cur = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            print(
                f"[SingleEncoder-SupOnly] world={int(os.getenv('WORLD_SIZE','1'))} "
                f"device={self.device} (current={cur}) "
                f"epochs={self.num_epochs} oversample={self.oversample_foreground_percent} lr={self.initial_lr}"
            )

        # Extend progress plotting
        self._patch_progress_plot()

    # ============================================================
    # Dataloaders: split CTA/MRA and build two training streams
    # ============================================================
    def get_dataloaders(self):
        """
        Build dataloaders.

        Training:
        - Split training identifiers into CTA and MRA based on key naming.
        - Create two nnUNetDataLoader instances, each with identical transforms
          but different datasets (CTA-only vs MRA-only).
        - Wrap each dataloader with SingleThreadedAugmenter or
          NonDetMultiThreadedAugmenter.

        Validation:
        - Keep the parent-style "full" validation loader (mt_val_full).
        - Additionally build CTA-only and MRA-only validation loaders so that
          `run_training()` can alternate modality during validation iterations.
        """
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        ds_scales = self._get_deep_supervision_scales()

        # nnU-Net helper that decides rotation ranges, dummy 2D aug etc.
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # Build transforms for train/val (nnU-Net standard)
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, ds_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label
        )
        val_transforms = self.get_validation_transforms(
            ds_scales, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label
        )

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        # Split training keys by modality id
        keys_tr_all = list(dataset_tr.identifiers)
        keys_tr_cta = [k for k in keys_tr_all if _infer_mid_from_identifier(k) == 0]
        keys_tr_mra = [k for k in keys_tr_all if _infer_mid_from_identifier(k) == 1]

        # If one modality is missing, fall back to parent behavior.
        if len(keys_tr_cta) == 0 or len(keys_tr_mra) == 0:
            return super().get_dataloaders()

        ds_cls = self.dataset_class

        # Build CTA-only and MRA-only training datasets
        dataset_cta = ds_cls(
            self.preprocessed_dataset_folder, keys_tr_cta,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
        )
        dataset_mra = ds_cls(
            self.preprocessed_dataset_folder, keys_tr_mra,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
        )

        # Build base nnUNetDataLoader objects (sampling/cropping etc.)
        dl_cta = nnUNetDataLoader(
            dataset_cta, self.batch_size, initial_patch_size, self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling
        )
        dl_mra = nnUNetDataLoader(
            dataset_mra, self.batch_size, initial_patch_size, self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling
        )

        # -------- Split VAL keys into CTA/MRA (optional) --------
        keys_val_all = list(dataset_val.identifiers)
        keys_val_cta = [k for k in keys_val_all if _infer_mid_from_identifier(k) == 0]
        keys_val_mra = [k for k in keys_val_all if _infer_mid_from_identifier(k) == 1]

        dataset_val_cta = ds_cls(
            self.preprocessed_dataset_folder, keys_val_cta,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
        ) if len(keys_val_cta) else None

        dataset_val_mra = ds_cls(
            self.preprocessed_dataset_folder, keys_val_mra,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
        ) if len(keys_val_mra) else None

        # Full validation loader (kept for compatibility / fallback)
        dl_val_full = nnUNetDataLoader(
            dataset_val, self.batch_size,
            self.configuration_manager.patch_size, self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling
        )

        # Modality-specific validation loaders (used only if available)
        dl_val_cta = nnUNetDataLoader(
            dataset_val_cta, self.batch_size,
            self.configuration_manager.patch_size, self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling
        ) if dataset_val_cta is not None else None

        dl_val_mra = nnUNetDataLoader(
            dataset_val_mra, self.batch_size,
            self.configuration_manager.patch_size, self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling
        ) if dataset_val_mra is not None else None

        # -------- Wrap loaders with augmenter workers --------
        allowed = get_allowed_n_proc_DA()
        pin = (self.device.type == 'cuda')

        def _wrap(dl, np_):
            """
            Wrap a DataLoader with batchgenerators augmenter.
            - If np_ == 0 -> SingleThreadedAugmenter
            - Else -> NonDetMultiThreadedAugmenter
            """
            if dl is None:
                return None
            if np_ == 0:
                return SingleThreadedAugmenter(dl, None)
            return NonDetMultiThreadedAugmenter(
                dl, None, num_processes=np_, num_cached=max(6, np_ // 2),
                seeds=None, pin_memory=pin, wait_time=0.002
            )

        # Training augmenters
        if allowed == 0:
            aug_cta = _wrap(dl_cta, 0)
            aug_mra = _wrap(dl_mra, 0)
            mt_val_full = _wrap(dl_val_full, 0)
            val_aug_cta = _wrap(dl_val_cta, 0)
            val_aug_mra = _wrap(dl_val_mra, 0)
        else:
            aug_cta = _wrap(dl_cta, allowed)
            aug_mra = _wrap(dl_mra, allowed)

            # Validation often needs fewer workers to avoid overhead
            val_np = max(1, allowed // 2)
            mt_val_full = _wrap(dl_val_full, val_np)
            val_aug_cta = _wrap(dl_val_cta, val_np)
            val_aug_mra = _wrap(dl_val_mra, val_np)

        # Warm up iterators to catch worker crashes early
        _ = next(iter(aug_cta)); _ = next(iter(aug_mra)); _ = next(iter(mt_val_full))
        if val_aug_cta is not None:
            _ = next(iter(val_aug_cta))
        if val_aug_mra is not None:
            _ = next(iter(val_aug_mra))

        # Save references for alternating training/validation logic
        self._train_aug_cta = aug_cta
        self._train_aug_mra = aug_mra
        self._val_aug_cta = val_aug_cta
        self._val_aug_mra = val_aug_mra

        # Which modality to start with (for train loader returned here)
        self._alt_start = os.getenv("ALT_START", "cta").lower()

        # Try to set iterations-per-epoch based on chosen start loader length
        try:
            start_loader = self._train_aug_cta if self._alt_start == "cta" else self._train_aug_mra
            self.num_iterations_per_epoch = len(start_loader)
        except Exception:
            pass

        # Return a training loader (for base compatibility) + full val loader
        return (self._train_aug_cta if self._alt_start == "cta" else self._train_aug_mra), mt_val_full
    
    #Added by JZhang, 20260306, only apply intensity augmentation on channel0
    @staticmethod
    def get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=None,
            is_cascaded=False,
            foreground_labels=None,
            regions=None,
            ignore_label=None,
    ):
        return nnUNetTrainer_single_encoder_baseline.get_training_transforms_image_only_aug(
            patch_size=patch_size,
            rotation_for_DA=rotation_for_DA,
            deep_supervision_scales=deep_supervision_scales,
            mirror_axes=mirror_axes,
            do_dummy_2d_data_aug=do_dummy_2d_data_aug,
            use_mask_for_norm=use_mask_for_norm,
            is_cascaded=is_cascaded,
            foreground_labels=foreground_labels,
            regions=regions,
            ignore_label=ignore_label,
            image_channels=(0,)  # only raw image gets intensity augmentation
        )

    @staticmethod
    def get_training_transforms_image_only_aug(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm,
            is_cascaded,
            foreground_labels,
            regions,
            ignore_label,
            image_channels
        ):
        transforms = []

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        # Spatial transforms should affect all channels together
        transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA,
                p_scaling=0.2,
                scaling=(0.7, 1.4),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        # Intensity transforms: image channels only
        transforms.append(
            RandomTransform(
                ApplyTransformToChannels(
                    GaussianNoiseTransform(
                        noise_variance=(0, 0.1),
                        p_per_channel=1,
                        synchronize_channels=True
                    ),
                    channels=image_channels
                ),
                apply_probability=0.1
            )
        )

        transforms.append(
            RandomTransform(
                ApplyTransformToChannels(
                    GaussianBlurTransform(
                        blur_sigma=(0.5, 1.0),
                        synchronize_channels=False,
                        synchronize_axes=False,
                        p_per_channel=0.5,
                        benchmark=True
                    ),
                    channels=image_channels
                ),
                apply_probability=0.2
            )
        )

        transforms.append(
            RandomTransform(
                ApplyTransformToChannels(
                    MultiplicativeBrightnessTransform(
                        multiplier_range=BGContrast((0.75, 1.25)),
                        synchronize_channels=False,
                        p_per_channel=1
                    ),
                    channels=image_channels
                ),
                apply_probability=0.15
            )
        )

        transforms.append(
            RandomTransform(
                ApplyTransformToChannels(
                    ContrastTransform(
                        contrast_range=BGContrast((0.75, 1.25)),
                        preserve_range=True,
                        synchronize_channels=False,
                        p_per_channel=1
                    ),
                    channels=image_channels
                ),
                apply_probability=0.15
            )
        )

        transforms.append(
            RandomTransform(
                ApplyTransformToChannels(
                    SimulateLowResolutionTransform(
                        scale=(0.5, 1),
                        synchronize_channels=False,
                        synchronize_axes=True,
                        ignore_axes=ignore_axes,
                        allowed_channels=None,
                        p_per_channel=0.5
                    ),
                    channels=image_channels
                ),
                apply_probability=0.25
            )
        )

        transforms.append(
            RandomTransform(
                ApplyTransformToChannels(
                    GammaTransform(
                        gamma=BGContrast((0.7, 1.5)),
                        p_invert_image=1,
                        synchronize_channels=False,
                        p_per_channel=1,
                        p_retain_stats=1
                    ),
                    channels=image_channels
                ),
                apply_probability=0.1
            )
        )

        transforms.append(
            RandomTransform(
                ApplyTransformToChannels(
                    GammaTransform(
                        gamma=BGContrast((0.7, 1.5)),
                        p_invert_image=0,
                        synchronize_channels=False,
                        p_per_channel=1,
                        p_retain_stats=1
                    ),
                    channels=image_channels
                ),
                apply_probability=0.3
            )
        )

        # Mirror should also affect all channels together
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(
                MaskImageTransform(
                    apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                    channel_idx_in_seg=0,
                    set_outside_to=0,
                )
            )

        transforms.append(RemoveLabelTansform(-1, 0))

        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ),
                    apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ),
                    apply_probability=0.2
                )
            )

        if regions is not None:
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(
                DownsampleSegForDSTransform(ds_scales=deep_supervision_scales)
            )
        print("Using custom training transforms")
        return ComposeTransforms(transforms)

    # Added by JZhang, 20260306, only apply intensity augmentation on channel0, END
    
    
    # ============================================================
    # Helpers for target alignment and modality tensor creation
    # ============================================================
    def _safe_one_hot(self, lbl: torch.Tensor, C: int, ref: torch.Tensor) -> torch.Tensor:
        """
        Convert label map to one-hot (B,C,...) safely.
        Ensures integer type and clamps labels into [0, C-1].
        """
        if lbl.ndim == 5 and lbl.size(1) == 1:
            lbl = lbl[:, 0]
        lbl = lbl.to(dtype=torch.long).clamp_(min=0, max=C - 1)
        B, *sp = lbl.shape
        oh = torch.zeros((B, C, *sp), device=ref.device, dtype=ref.dtype)
        oh.scatter_(1, lbl[:, None], 1)
        return oh

    def _resize_seg_like(self, seg: torch.Tensor, size_hwz):
        """
        Nearest-neighbor resize of segmentation tensor to match output spatial size.
        Used when deep supervision outputs have different resolutions.
        """
        if tuple(seg.shape[2:]) == tuple(size_hwz): return seg
        seg_f = seg.float()
        seg_f = F.interpolate(seg_f, size=size_hwz, mode='nearest')
        return seg_f.to(seg.dtype)

    def _align_targets_to_outputs(self, outputs, targets):
        """
        Align target list to output list (deep supervision).
        For each output tensor, select the target tensor with the closest spatial size,
        then resize to match exactly.
        Returns:
          - list of aligned targets if outputs is list/tuple
          - single aligned target otherwise
        """
        outs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        tlist = targets if isinstance(targets, (list, tuple)) else [targets]
        aligned, used = [], set()

        for o in outs:
            oshp = tuple(o.shape[2:])
            idx = None

            # Prefer exact spatial match
            for j, t in enumerate(tlist):
                if j in used: continue
                if tuple(t.shape[2:]) == oshp:
                    idx = j
                    break

            # Otherwise pick closest by L1 distance over spatial dims
            if idx is None:
                diffs = [(j, sum(abs(a-b) for a, b in zip(t.shape[2:], oshp)))
                         for j, t in enumerate(tlist) if j not in used]
                idx = min(diffs, key=lambda x: x[1])[0] if diffs else 0

            used.add(idx)
            aligned.append(self._resize_seg_like(tlist[idx], oshp))

        return aligned if isinstance(outputs, (list, tuple)) else aligned[0]

    def _make_mid_from_keys(self, batch: dict, B: int) -> torch.Tensor:
        """
        Create a modality-id tensor for the batch from batch['keys'].
        Output: (B,) int64 tensor in {0,1} (CTA=0, MRA=1).
        If keys missing or mismatch length, defaults to zeros.
        """
        keys = batch.get('keys', None)
        if keys is None:
            return torch.zeros((B,), dtype=torch.long, device=self.device)

        if isinstance(keys, (list, tuple)):
            ks = list(keys)
        elif isinstance(keys, np.ndarray):
            ks = keys.tolist()
        elif torch.is_tensor(keys):
            ks = keys.detach().cpu().tolist()
        else:
            ks = [keys] * B

        mids = [_infer_mid_from_identifier(str(k)) for k in ks]
        if len(mids) != B:
            return torch.zeros((B,), dtype=torch.long, device=self.device)
        return torch.tensor(mids, dtype=torch.long, device=self.device)

    # ============================================================
    # Train step (supervised segmentation only)
    # ============================================================
    def train_step(self, batch: dict) -> dict:
        """
        One training step:
        - Move data/target to device
        - Forward through network
        - Compute nnU-Net loss (handles deep supervision)
        - AMP if enabled
        - Gradient clipping + optimizer step
        Returns dict with scalar loss for logging.
        """
        data, target = batch['data'], batch['target']
        data = data.to(self.device, non_blocking=True)
        target = [i.to(self.device, non_blocking=True) for i in target] if isinstance(target, list) \
            else target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        use_amp = (self.device.type == 'cuda' and getattr(self, "grad_scaler", None) is not None)

        def _forward():
            output = self.network(data)
            L_sup = self.loss(
                output if isinstance(output, (list, tuple)) else [output],
                self._align_targets_to_outputs(output, target)
            )
            return L_sup

        if use_amp:
            with torch.autocast(self.device.type, enabled=True):
                loss = _forward()
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss = _forward()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': float(getattr(loss, "detach", lambda: loss)().cpu().numpy())}

    # ============================================================
    # Validation step (compute tp/fp/fn + modality id)
    # ============================================================
    def validation_step(self, batch: dict) -> dict:
        """
        One validation step:
        - Forward + loss
        - Convert output to probabilities (softmax or sigmoid depending on regions)
        - Build one-hot target (handling ignore label mask)
        - Compute tp/fp/fn per class with nnU-Net utility
        - Return arrays for epoch-end aggregation, plus `mid` for CTA/MRA split
        """
        data, target = batch['data'], batch['target']
        data = data.to(self.device, non_blocking=True)
        target = [i.to(self.device, non_blocking=True) for i in target] if isinstance(target, list) \
            else target.to(self.device, non_blocking=True)

        # modality id per sample in the batch
        mid = self._make_mid_from_keys(batch, data.shape[0])

        # Forward + loss (AMP on GPU)
        if self.device.type == 'cuda':
            with torch.autocast(self.device.type, enabled=True):
                output = self.network(data)
                l = self.loss(
                    output if isinstance(output, (list, tuple)) else [output],
                    self._align_targets_to_outputs(output, target)
                )
        else:
            output = self.network(data)
            l = self.loss(
                output if isinstance(output, (list, tuple)) else [output],
                self._align_targets_to_outputs(output, target)
            )

        # Use highest-resolution output for metric computation
        if isinstance(output, (list, tuple)):
            output_eval = output[0]
            target_eval = self._align_targets_to_outputs(output, target)[0]
        else:
            output_eval = output
            target_eval = self._align_targets_to_outputs(output, target)

        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
        axes = list(range(2, output_eval.ndim))
        C = self.label_manager.num_segmentation_heads

        # Convert logits -> probabilities
        if self.label_manager.has_regions:
            net_output = torch.sigmoid(output_eval).float()
        else:
            net_output = torch.softmax(output_eval, dim=1).float()

        # Build ignore mask and make sure target_eval is in correct format
        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                # class-label map with ignore label value
                if target_eval.ndim == 5 and target_eval.size(1) == 1:
                    target_lbl = target_eval[:, 0].long()
                elif target_eval.ndim == 4:
                    target_lbl = target_eval.long()
                else:
                    target_lbl = target_eval.long()
                mask = (target_lbl != self.label_manager.ignore_label).unsqueeze(1).bool()
                target_eval = target_lbl
            else:
                # region target format: last channel is ignore mask
                ignore_ch = target_eval[:, -1:]
                mask = (~ignore_ch.bool())
                target_eval = target_eval[:, :-1]
        else:
            mask = None

        # Convert target to one-hot if needed
        if target_eval.ndim != 5 or (target_eval.ndim == 5 and target_eval.size(1) == 1):
            y_onehot = self._safe_one_hot(target_eval, C, output_eval).bool()
        else:
            y_onehot = target_eval.bool()

        tp, fp, fn, _ = get_tp_fp_fn_tn(net_output, y_onehot, axes=axes, mask=mask)

        return {
            'loss': float(l.detach().cpu().numpy()),
            'tp_hard': tp.detach().cpu().numpy(),
            'fp_hard': fp.detach().cpu().numpy(),
            'fn_hard': fn.detach().cpu().numpy(),
            'mid': mid.detach().cpu().numpy()
        }

    # ============================================================
    # Validation epoch end: aggregate CTA/MRA dice + keep parent hooks
    # ============================================================
    def on_validation_epoch_end(self, val_outputs: List[dict]):
        """
        Aggregate validation outputs across steps (and across DDP ranks if enabled).

        Goals:
        1) Compute per-class Dice (foreground only) overall and separately for CTA/MRA.
        2) Log modality-specific mean foreground Dice into logger.my_fantastic_logging.
        3) Print per-class Dice summary (overall/CTA/MRA) on rank0.
        4) Call super().on_validation_epoch_end(...) to preserve nnU-Net behavior
           (standard curves, EMA logic, scheduler, etc.).
        """
        import numpy as np, torch
        is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        dev = self.device if isinstance(self.device, torch.device) else torch.device(self.device)

        # Gather local arrays for this rank
        if len(val_outputs):
            mids_local = np.concatenate([o['mid'] for o in val_outputs if 'mid' in o], axis=0)
            tps_local  = np.concatenate([o['tp_hard'] for o in val_outputs], axis=0)
            fps_local  = np.concatenate([o['fp_hard'] for o in val_outputs], axis=0)
            fns_local  = np.concatenate([o['fn_hard'] for o in val_outputs], axis=0)
        else:
            # Empty validation (rare) -> create empty placeholders
            C_fg = len(self.label_manager.foreground_labels)
            C = self.label_manager.num_segmentation_heads if self.label_manager.has_regions else C_fg
            mids_local = np.zeros((0,), np.int64)
            tps_local  = np.zeros((0, C), np.float64)
            fps_local  = np.zeros_like(tps_local)
            fns_local  = np.zeros_like(tps_local)

        # DDP: all_gather variable-length arrays (pad to max per rank)
        if is_dist:
            W = torch.distributed.get_world_size()

            def _gather_1d(a_np: np.ndarray, dtype=torch.long):
                t = torch.as_tensor(a_np, device=dev, dtype=dtype)
                n = torch.tensor([t.shape[0]], device=dev, dtype=torch.long)
                sizes = [torch.zeros_like(n) for _ in range(W)]
                torch.distributed.all_gather(sizes, n)
                sizes = [int(s.item()) for s in sizes]
                maxn = max(sizes) if sizes else 0
                if t.shape[0] < maxn:
                    pad = torch.zeros((maxn - t.shape[0],), device=dev, dtype=t.dtype)
                    t = torch.cat([t, pad], 0)
                bufs = [torch.zeros_like(t) for _ in range(W)]
                torch.distributed.all_gather(bufs, t)
                outs = [b[:sz].detach().cpu().numpy() for b, sz in zip(bufs, sizes)]
                return np.concatenate(outs, 0) if len(outs) else np.zeros((0,), np.int64)

            def _gather_2d(a_np: np.ndarray, dtype=torch.float64):
                C_ = a_np.shape[1] if a_np.size else 0
                t = torch.as_tensor(a_np, device=dev, dtype=dtype)
                n = torch.tensor([t.shape[0]], device=dev, dtype=torch.long)
                sizes = [torch.zeros_like(n) for _ in range(W)]
                torch.distributed.all_gather(sizes, n)
                sizes = [int(s.item()) for s in sizes]
                maxn = max(sizes) if sizes else 0
                if t.shape[0] < maxn:
                    pad = torch.zeros((maxn - t.shape[0], t.shape[1]), device=dev, dtype=t.dtype)
                    t = torch.cat([t, pad], 0)
                bufs = [torch.zeros_like(t) for _ in range(W)]
                torch.distributed.all_gather(bufs, t)
                outs = [b[:sz].detach().cpu().numpy() for b, sz in zip(bufs, sizes)]
                if len(outs):
                    return np.concatenate(outs, 0)
                return np.zeros((0, C_), np.float64) if C_ > 0 else np.zeros((0, 0), np.float64)

            mids = _gather_1d(mids_local, dtype=torch.long)
            tps  = _gather_2d(tps_local, dtype=torch.float64)
            fps  = _gather_2d(fps_local, dtype=torch.float64)
            fns  = _gather_2d(fns_local, dtype=torch.float64)
        else:
            mids, tps, fps, fns = mids_local, tps_local, fps_local, fns_local

        # Remove background channel for standard class-based segmentation
        if (not self.label_manager.has_regions) and tps.shape[1] >= 2:
            tps2, fps2, fns2 = tps[:, 1:], fps[:, 1:], fns[:, 1:]
            class_offset = 1  # foreground class indices start at 1 in original logits
        else:
            tps2, fps2, fns2 = tps, fps, fns
            class_offset = 0

        num_fg_classes = tps2.shape[1] if tps2.ndim == 2 else 0

        def _dice_per_class(mask: np.ndarray):
            """
            Compute per-class Dice for the subset defined by mask.
            Returns (dice_per_class, mean_dice).
            """
            if not mask.any() or num_fg_classes == 0:
                return np.full((num_fg_classes,), np.nan, dtype=np.float64), None
            tp_sum = tps2[mask].sum(0)
            fp_sum = fps2[mask].sum(0)
            fn_sum = fns2[mask].sum(0)
            den = (2.0 * tp_sum + fp_sum + fn_sum)
            dice_cls = np.where(den > 0, (2.0 * tp_sum) / den, np.nan)
            mean_dice = float(np.nanmean(dice_cls)) if np.isfinite(dice_cls).any() else None
            return dice_cls, mean_dice

        dice_all_cls, md_all = _dice_per_class(np.ones_like(mids, dtype=bool))
        dice_cta_cls, md_cta = _dice_per_class(mids == 0)
        dice_mra_cls, md_mra = _dice_per_class(mids == 1)

        # Rank0: log + print modality-specific metrics
        if self._is_global_rank0():
            self._log_with_nan('mean_fg_dice_cta', md_cta, self.current_epoch)
            self._log_with_nan('mean_fg_dice_mra', md_mra, self.current_epoch)

            if md_cta is not None:
                self.print_to_log_file(f"[Val/CTA] mean_fg_dice={md_cta:.4f}", also_print_to_console=True)
            if md_mra is not None:
                self.print_to_log_file(f"[Val/MRA] mean_fg_dice={md_mra:.4f}", also_print_to_console=True)

            # Also log per-class Dice (overall/CTA/MRA)
            for local_cls_idx in range(num_fg_classes):
                global_cls_idx = local_cls_idx + class_offset

                d_all = dice_all_cls[local_cls_idx]
                d_cta = dice_cta_cls[local_cls_idx] if local_cls_idx < len(dice_cta_cls) else np.nan
                d_mra = dice_mra_cls[local_cls_idx] if local_cls_idx < len(dice_mra_cls) else np.nan

                self._log_with_nan(
                    f'dice_cls_{global_cls_idx}_overall',
                    (None if np.isnan(d_all) else float(d_all)),
                    self.current_epoch
                )
                self._log_with_nan(
                    f'dice_cls_{global_cls_idx}_cta',
                    (None if np.isnan(d_cta) else float(d_cta)),
                    self.current_epoch
                )
                self._log_with_nan(
                    f'dice_cls_{global_cls_idx}_mra',
                    (None if np.isnan(d_mra) else float(d_mra)),
                    self.current_epoch
                )

                msg = (
                    f"[Val/Class {global_cls_idx}] "
                    f"Dice_all={('nan' if np.isnan(d_all) else f'{float(d_all):.4f}')} "
                    f"CTA={('nan' if np.isnan(d_cta) else f'{float(d_cta):.4f}')} "
                    f"MRA={('nan' if np.isnan(d_mra) else f'{float(d_mra):.4f}')}"
                )
                self.print_to_log_file(msg, also_print_to_console=True)

        # Prepare outputs for parent aggregation:
        # Parent expects per-step arrays shaped like (classes,)
        # Here we collapse step-level tp/fp/fn into sums and remove background.
        vo_for_super = []
        for o in val_outputs:
            o2 = o.copy()
            for k in ('tp_hard', 'fp_hard', 'fn_hard'):
                if k not in o2: 
                    continue
                arr = np.asarray(o2[k])
                if arr.ndim == 2:
                    arr = arr.sum(axis=0)
                if (not self.label_manager.has_regions) and arr.shape[0] >= 2:
                    arr = arr[1:]
                o2[k] = arr
            vo_for_super.append(o2)

        # Keep standard nnU-Net behavior (lr scheduling hooks, plots, etc.)
        super().on_validation_epoch_end(vo_for_super)
        
    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], 4))
        self.print_to_log_file('val_loss',   np.round(self.logger.my_fantastic_logging['val_losses'][-1], 4))
        self.print_to_log_file('Val mean_fg_dice', np.round(self.logger.my_fantastic_logging['mean_fg_dice'][-1], 4))
        self.print_to_log_file('Pseudo dice', [np.round(i, 4) for i in self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])

        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], 2)} s"
        )
        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self._is_global_rank0():           # was: local_rank == 0
            self.logger.plot_progress_png(self.output_folder)
        self.current_epoch += 1
        
    # ============================================================
    # Actual (full-volume) validation: export predictions per modality
    # ============================================================
    def perform_actual_validation(self, save_probabilities: bool = False):
        """
        Full-volume validation inference (sliding window) and metric computation.

        Differences from base:
        - Split validation keys into CTA/MRA groups and save predictions into:
            <output_folder>/validation_cta/
            <output_folder>/validation_mra/
        - Compute metrics separately per group and print final mean dice.
        - Still supports cascaded setups (adds previous-stage seg channels).
        - Uses multiprocessing pool to export predictions and optionally resample
          for next stage configurations.
        """
        self.network.eval()
        self.set_deep_supervision_enabled(False)

        predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
            perform_everything_on_device=True, device=self.device,
            verbose=False, verbose_preprocessing=False, allow_tqdm=False
        )
        base_net = self.network.module if hasattr(self.network, "module") else self.network
        predictor.manual_initialization(
            base_net, self.plans_manager, self.configuration_manager, None,
            self.dataset_json, self.__class__.__name__, self.inference_allowed_mirroring_axes
        )

        try:
            with multiprocessing.get_context("spawn").Pool(default_num_processes) as pool:
                workers = [w for w in pool._pool]
                _, val_keys_all = self.do_split()

                # Group validation cases by modality
                groups = {0: [], 1: []}
                for k in val_keys_all:
                    groups[_infer_mid_from_identifier(k)].append(k)

                results = []
                world, rank = (
                    (torch.distributed.get_world_size(), torch.distributed.get_rank())
                    if (torch.distributed.is_available() and torch.distributed.is_initialized())
                    else (1, 0)
                )

                # Loop over each modality group and run inference
                for mid, keys_this_mod in {k: v for k, v in groups.items() if len(v) > 0}.items():
                    subname = "validation_cta" if mid == 0 else "validation_mra"
                    out_dir = join(self.output_folder, subname)
                    maybe_mkdir_p(out_dir)

                    # DDP: shard keys across ranks
                    keys_rank = keys_this_mod[rank::world]

                    # Next stage handling (cascades)
                    next_stages = self.configuration_manager.next_stage_names
                    if next_stages is not None:
                        _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

                    dataset_val = self.dataset_class(
                        self.preprocessed_dataset_folder, keys_rank,
                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
                    )

                    for i, k in enumerate(dataset_val.identifiers):
                        # Backpressure: keep pool queue bounded
                        proceed = not check_workers_alive_and_busy(pool, workers, results, allowed_num_queued=2)
                        while not proceed:
                            sleep(0.1)
                            proceed = not check_workers_alive_and_busy(pool, workers, results, allowed_num_queued=2)

                        self.print_to_log_file(f"[Val/{'CTA' if mid==0 else 'MRA'}] predicting {k}")

                        data, _, seg_prev, properties = dataset_val.load_case(k)
                        data = data[:]

                        # Cascaded: concatenate previous-stage one-hot channels
                        if self.is_cascaded:
                            seg_prev = seg_prev[:]
                            data = np.vstack((
                                data,
                                convert_labelmap_to_one_hot(
                                    seg_prev, self.label_manager.foreground_labels,
                                    output_dtype=data.dtype
                                )
                            ))

                        # Convert to torch tensor for predictor
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            data_t = torch.from_numpy(data)

                        out_name = join(out_dir, k)
                        pred = predictor.predict_sliding_window_return_logits(data_t).float().cpu()

                        # Export prediction asynchronously
                        results.append(pool.starmap_async(
                            export_prediction_from_logits,
                            ((pred, properties, self.configuration_manager, self.plans_manager,
                              self.dataset_json, out_name, save_probabilities),)
                        ))

                        # Optional: export prediction resampled to "next stage" target shapes
                        if next_stages is not None:
                            for n in next_stages:
                                next_cfg = self.plans_manager.get_configuration(n)
                                exp_pre  = join(self.preprocessed_dataset_folder_base, next_cfg.data_identifier)
                                dataset_cls = infer_dataset_class(exp_pre)
                                try:
                                    tmp = dataset_cls(exp_pre, [k])
                                    d, _, _, _ = tmp.load_case(k)
                                except FileNotFoundError:
                                    self.print_to_log_file(f"Next stage {n} missing preprocessed {k}.")
                                    continue
                                tgt_shape = d.shape[1:]
                                out_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                                out_trunc  = join(out_folder, k)
                                results.append(pool.starmap_async(
                                    resample_and_save,
                                    ((pred, tgt_shape, out_trunc, self.plans_manager, self.configuration_manager,
                                      properties, self.dataset_json, default_num_processes, dataset_cls),)
                                ))

                    # Ensure all exports are finished for this modality
                    _ = [r.get() for r in results]
                    results.clear()

                # DDP barrier before computing final metrics on rank0
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.barrier()

                # Rank0: compute metrics separately for CTA and MRA folders
                if self._is_global_rank0():
                    gt_dir = join(self.preprocessed_dataset_folder_base, 'gt_segmentations')
                    for mid in sorted([m for m in groups if len(groups[m]) > 0]):
                        subname = "validation_cta" if mid == 0 else "validation_mra"
                        out_dir = join(self.output_folder, subname)
                        if not os.path.isdir(out_dir):
                            continue

                        metrics = compute_metrics_on_folder(
                            gt_dir, out_dir, join(out_dir, 'summary.json'),
                            self.plans_manager.image_reader_writer_class(),
                            self.dataset_json["file_ending"],
                            self.label_manager.foreground_regions if self.label_manager.has_regions
                            else self.label_manager.foreground_labels,
                            self.label_manager.ignore_label, chill=True,
                            num_processes=default_num_processes * (
                                torch.distributed.get_world_size()
                                if (torch.distributed.is_available() and torch.distributed.is_initialized())
                                else 1
                            )
                        )
                        mname = "CTA" if mid == 0 else "MRA"
                        self.print_to_log_file(
                            f"[Final Val/{mname}] Mean Dice: {metrics['foreground_mean']['Dice']:.4f}",
                            also_print_to_console=True
                        )
        finally:
            # Clean caches and restore deep supervision for training
            compute_gaussian.cache_clear()
            self.set_deep_supervision_enabled(True)

    # ============================================================
    # Training loop: alternate CTA/MRA every iteration
    # ============================================================
    def run_training(self):
        """
        Custom training loop (instead of parent) to enforce iteration-wise alternation.

        Main behavior:
        - steps_per_epoch can be overridden by env STEPS_PER_EPOCH.
        - For each epoch:
            * Iterate `steps_per_epoch` times.
            * Alternate between CTA and MRA batches (it % 2).
            * Run train_step() and log loss via tqdm.
        - Validation:
            * If modality-specific val loaders exist, alternate CTA/MRA similarly.
            * Otherwise fallback to parent-style self.dataloader_val.
        - Calls parent hooks (on_train_start/on_epoch_start/on_train_epoch_end/etc.)
          so that lr schedulers, logging, checkpointing remain consistent.
        """
        self.on_train_start()

        # Override per-epoch iteration count if provided
        steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH", str(getattr(self, "num_iterations_per_epoch", 250))))
        self.num_iterations_per_epoch = steps_per_epoch

        aug_cta = getattr(self, "_train_aug_cta", None)
        aug_mra = getattr(self, "_train_aug_mra", None)

        # If CTA/MRA split not available, revert to default training loop
        if (aug_cta is None) or (aug_mra is None):
            return super().run_training()

        # tqdm visualization options (rank0 only)
        show_pbar      = (tqdm is not None) and self._is_global_rank0() and bool(int(os.getenv("PROGRESS", "1")))
        POSTFIX_EVERY  = int(os.getenv("PBAR_POSTFIX_EVERY", "20"))
        MININTERVAL    = float(os.getenv("PBAR_MININTERVAL", "0.5"))
        MINITERS       = int(os.getenv("PBAR_MINITERS", "1"))
        SMOOTHING      = float(os.getenv("PBAR_SMOOTHING", "0.3"))

        # Which modality to start with for training alternation
        start_mod      = os.getenv("ALT_FIRST", "cta").lower()

        # Validation modality loaders (optional)
        val_cta = getattr(self, "_val_aug_cta", None)
        val_mra = getattr(self, "_val_aug_mra", None)
        val_alt_first = os.getenv("VAL_ALT_FIRST", "cta").lower()

        # Determine availability of both modality streams
        try:
            len_cta = len(aug_cta)
            len_mra = len(aug_mra)
        except Exception:
            len_cta = len_mra = 1

        both     = (len_cta > 0 and len_mra > 0)
        only_cta = (len_cta > 0 and len_mra == 0)
        only_mra = (len_mra > 0 and len_cta == 0)

        # Epoch loop
        for epoch in range(self.current_epoch, self.num_epochs):
            if self._is_global_rank0():
                mode_desc = "ALT-ITER CTA/MRA" if both else ("CTA-only" if only_cta else "MRA-only")
                self.print_to_log_file(
                    f"[Epoch {epoch+1}/{self.num_epochs}] {mode_desc} | steps={steps_per_epoch}",
                    also_print_to_console=True
                )

            it_cta = iter(aug_cta) if len_cta > 0 else None
            it_mra = iter(aug_mra) if len_mra > 0 else None

            self.on_epoch_start()
            self.on_train_epoch_start()

            # Progress bar (optional)
            pbar = tqdm(
                total=steps_per_epoch, desc=f"Epoch {epoch+1}/{self.num_epochs}",
                disable=not show_pbar, leave=True, dynamic_ncols=True,
                mininterval=MININTERVAL, miniters=MINITERS, smoothing=SMOOTHING, unit="it"
            ) if show_pbar else None

            train_outputs = []
            last_step_loss = float("nan")
            want_cta_first = (start_mod == "cta")

            def _next_from(which: str):
                """
                Get next batch from CTA or MRA stream.
                If iterator ends, re-create it (infinite cycling).
                """
                nonlocal it_cta, it_mra
                if which == "cta":
                    if it_cta is None:
                        return None
                    try:
                        return next(it_cta)
                    except StopIteration:
                        it_cta = iter(aug_cta)
                        return next(it_cta)
                else:
                    if it_mra is None:
                        return None
                    try:
                        return next(it_mra)
                    except StopIteration:
                        it_mra = iter(aug_mra)
                        return next(it_mra)

            # -------- Training iterations (alternate modality) --------
            for it in range(steps_per_epoch):
                if both:
                    want_cta = (it % 2 == 0) if want_cta_first else (it % 2 == 1)
                    which = "cta" if want_cta else "mra"
                    batch = _next_from(which) or _next_from("mra" if which == "cta" else "cta")
                    which = which if batch is not None else ("mra" if which == "cta" else "cta")
                elif only_cta:
                    which, batch = "cta", _next_from("cta")
                else:
                    which, batch = "mra", _next_from("mra")

                if batch is None:
                    break

                out = self.train_step(batch)
                train_outputs.append(out)

                try:
                    last_step_loss = float(out.get("loss", float("nan")))
                except Exception:
                    last_step_loss = float("nan")

                if pbar is not None:
                    if (it % POSTFIX_EVERY == 0):
                        pbar.set_postfix_str(f"{which}|loss={last_step_loss:.4f}")
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

            self.on_train_epoch_end(train_outputs)

            # ========================================================
            # Validation iterations (optional alternate CTA/MRA)
            # ========================================================
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []

                # Fallback: use self.dataloader_val from parent
                if (val_cta is None) or (val_mra is None):
                    it_val = iter(self.dataloader_val)
                    for _ in range(self.num_val_iterations_per_epoch):
                        val_outputs.append(self.validation_step(next(it_val)))
                else:
                    itv_cta = iter(val_cta)
                    itv_mra = iter(val_mra)

                    def _next_val(which: str):
                        """
                        Same cycling logic for validation loaders.
                        """
                        nonlocal itv_cta, itv_mra
                        if which == "cta":
                            try:
                                return next(itv_cta)
                            except StopIteration:
                                itv_cta = iter(val_cta)
                                return next(itv_cta)
                        else:
                            try:
                                return next(itv_mra)
                            except StopIteration:
                                itv_mra = iter(val_mra)
                                return next(itv_mra)

                    want_cta_first_val = (val_alt_first == "cta")

                    for itv in range(self.num_val_iterations_per_epoch):
                        want_cta = (itv % 2 == 0) if want_cta_first_val else (itv % 2 == 1)
                        which = "cta" if want_cta else "mra"
                        batch = _next_val(which)
                        val_outputs.append(self.validation_step(batch))

                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()
