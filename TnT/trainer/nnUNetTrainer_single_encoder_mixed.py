# nnunetv2/training/nnUNetTrainer/nnUNetTrainer_single_encoder_mixed.py

import os
import torch
from typing import Dict
import numpy as np

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
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

from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import RemoveRandomConnectedComponentFromOneHotEncodingTransform

# Reuse your baseline logic + modality inference
from TnT.trainer.nnUNetTrainer_single_encoder_baseline import (
    nnUNetTrainer_single_encoder_baseline,
    _infer_mid_from_identifier,
)

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None
    
def find_dir():
    return os.path.dirname(os.path.realpath(__file__))

class ApplyTransformToChannels(BasicTransform):
    """
    Apply a transform only to selected channels of data_dict["data"].

    Assumes:
        data_dict["data"] shape is (C, X, Y, Z) or (C, X, Y) for a single sample.
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
            return data_dict

        if data.ndim < 2:
            return data_dict

        selected = data[self.channels].copy()

        tmp_dict = dict(data_dict)
        tmp_dict["data"] = selected
        tmp_dict = self.transform(**tmp_dict)

        out = data.copy()
        out[self.channels] = tmp_dict["data"]
        data_dict["data"] = out
        return data_dict

class nnUNetTrainer_single_encoder_mixed(nnUNetTrainer_single_encoder_baseline):
    """
    Mixed-modality mini-batch trainer.

    Compared to nnUNetTrainer_single_encoder_baseline (alternating per iteration),
    this trainer *always* merges half-batches from CTA and MRA into one batch:

        batch = cat(batch_cta, batch_mra) along dim=0

    So each optimizer step sees both modalities.
    """
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
        return nnUNetTrainer_single_encoder_mixed.get_training_transforms_image_only_aug(
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
            image_channels=(0,)
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

        # spatial transforms -> all channels
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

        # intensity transforms -> image channels only
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

        # mirror -> all channels
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

        return ComposeTransforms(transforms)
    # ------------------------------------------------------------
    # Dataloaders: CTA/MRA streams use half batch size
    # ------------------------------------------------------------
    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        if self.batch_size % 2 != 0:
            raise ValueError(
                f"{self.__class__.__name__} requires even batch_size, got {self.batch_size}"
            )
        per_stream_bs = self.batch_size // 2

        patch_size = self.configuration_manager.patch_size
        ds_scales = self._get_deep_supervision_scales()

        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

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

        keys_tr_all = list(dataset_tr.identifiers)
        keys_tr_cta = [k for k in keys_tr_all if _infer_mid_from_identifier(k) == 0]
        keys_tr_mra = [k for k in keys_tr_all if _infer_mid_from_identifier(k) == 1]

        # If split fails, fallback to parent
        if len(keys_tr_cta) == 0 or len(keys_tr_mra) == 0:
            return super().get_dataloaders()

        ds_cls = self.dataset_class

        dataset_cta = ds_cls(
            self.preprocessed_dataset_folder, keys_tr_cta,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
        )
        dataset_mra = ds_cls(
            self.preprocessed_dataset_folder, keys_tr_mra,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
        )

        dl_cta = nnUNetDataLoader(
            dataset_cta, per_stream_bs, initial_patch_size, self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling
        )
        dl_mra = nnUNetDataLoader(
            dataset_mra, per_stream_bs, initial_patch_size, self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling
        )

        # --- validation split (optional, used for patch-based val during training) ---
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

        # Full val loader (kept for compatibility / fallback)
        dl_val_full = nnUNetDataLoader(
            dataset_val, self.batch_size,
            self.configuration_manager.patch_size, self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling
        )

        # For mixed validation, we also use half batch size per stream by default
        dl_val_cta = nnUNetDataLoader(
            dataset_val_cta, per_stream_bs,
            self.configuration_manager.patch_size, self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling
        ) if dataset_val_cta is not None else None

        dl_val_mra = nnUNetDataLoader(
            dataset_val_mra, per_stream_bs,
            self.configuration_manager.patch_size, self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling
        ) if dataset_val_mra is not None else None

        # --- wrap with augmenters (same style as baseline) ---
        allowed = get_allowed_n_proc_DA()
        pin = (self.device.type == 'cuda')

        def _wrap(dl, np_):
            if dl is None:
                return None
            if np_ == 0:
                return SingleThreadedAugmenter(dl, None)
            return NonDetMultiThreadedAugmenter(
                dl, None, num_processes=np_, num_cached=max(6, np_ // 2),
                seeds=None, pin_memory=pin, wait_time=0.002
            )

        if allowed == 0:
            aug_cta = _wrap(dl_cta, 0)
            aug_mra = _wrap(dl_mra, 0)
            mt_val_full = _wrap(dl_val_full, 0)
            val_aug_cta = _wrap(dl_val_cta, 0)
            val_aug_mra = _wrap(dl_val_mra, 0)
        else:
            aug_cta = _wrap(dl_cta, allowed)
            aug_mra = _wrap(dl_mra, allowed)

            val_np = max(1, allowed // 2)
            mt_val_full = _wrap(dl_val_full, val_np)
            val_aug_cta = _wrap(dl_val_cta, val_np)
            val_aug_mra = _wrap(dl_val_mra, val_np)

        # Warm-up
        _ = next(iter(aug_cta)); _ = next(iter(aug_mra)); _ = next(iter(mt_val_full))
        if val_aug_cta is not None:
            _ = next(iter(val_aug_cta))
        if val_aug_mra is not None:
            _ = next(iter(val_aug_mra))

        self._train_aug_cta = aug_cta
        self._train_aug_mra = aug_mra
        self._val_aug_cta = val_aug_cta
        self._val_aug_mra = val_aug_mra

        # For mixed training, a natural epoch length is min(len(cta), len(mra))
        try:
            self.num_iterations_per_epoch = min(len(self._train_aug_cta), len(self._train_aug_mra))
        except Exception:
            pass

        # Return something for base compatibility + full val loader
        return self._train_aug_cta, mt_val_full

    # ------------------------------------------------------------
    # Mixed training loop
    # ------------------------------------------------------------
    @staticmethod
    def _merge_batches(b_ct: Dict, b_mr: Dict) -> Dict:
        """
        Merge two nnUNet batches by concatenating along batch dimension.

        Batch dict keys typically:
          - 'data': Tensor [B, C, ...]
          - 'target': Tensor or list of Tensors (deep supervision)
          - optional 'keys', 'properties'
        """
        out = {}
        out["data"] = torch.cat([b_ct["data"], b_mr["data"]], dim=0)

        t1, t2 = b_ct["target"], b_mr["target"]
        if isinstance(t1, list):
            assert isinstance(t2, list) and len(t1) == len(t2), "Target lists must match"
            out["target"] = [torch.cat([a, b], dim=0) for a, b in zip(t1, t2)]
        else:
            out["target"] = torch.cat([t1, t2], dim=0)

        if "keys" in b_ct and "keys" in b_mr:
            out["keys"] = list(b_ct["keys"]) + list(b_mr["keys"])
        if "properties" in b_ct and "properties" in b_mr:
            out["properties"] = list(b_ct["properties"]) + list(b_mr["properties"])
        return out

    def run_training(self):
        """
        Mixed-modality training:
          each step: batch = merge(next(CTA), next(MRA))

        Validation:
          default mixed as well (can switch to alternating via env MIXED_VAL_MODE=alt)
          - MIXED_VAL_MODE=mix (default): merge CTA+MRA for each val step
          - MIXED_VAL_MODE=alt: alternating CTA/MRA (baseline behavior)
        """
        self.on_train_start()

        steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH", str(getattr(self, "num_iterations_per_epoch", 250))))
        self.num_iterations_per_epoch = steps_per_epoch

        aug_cta = getattr(self, "_train_aug_cta", None)
        aug_mra = getattr(self, "_train_aug_mra", None)

        if (aug_cta is None) or (aug_mra is None):
            return super().run_training()

        show_pbar      = (tqdm is not None) and self._is_global_rank0() and bool(int(os.getenv("PROGRESS", "1")))
        POSTFIX_EVERY  = int(os.getenv("PBAR_POSTFIX_EVERY", "20"))
        MININTERVAL    = float(os.getenv("PBAR_MININTERVAL", "0.5"))
        MINITERS       = int(os.getenv("PBAR_MINITERS", "1"))
        SMOOTHING      = float(os.getenv("PBAR_SMOOTHING", "0.3"))

        # Validation loaders (optional)
        val_cta = getattr(self, "_val_aug_cta", None)
        val_mra = getattr(self, "_val_aug_mra", None)
        val_mode = os.getenv("MIXED_VAL_MODE", "mix").lower()  # 'mix' or 'alt'
        val_alt_first = os.getenv("VAL_ALT_FIRST", "cta").lower()

        for epoch in range(self.current_epoch, self.num_epochs):
            if self._is_global_rank0():
                self.print_to_log_file(
                    f"[Epoch {epoch+1}/{self.num_epochs}] MIXED CTA+MRA | steps={steps_per_epoch}",
                    also_print_to_console=True
                )

            it_cta = iter(aug_cta)
            it_mra = iter(aug_mra)

            self.on_epoch_start()
            self.on_train_epoch_start()

            pbar = tqdm(
                total=steps_per_epoch, desc=f"Epoch {epoch+1}/{self.num_epochs}",
                disable=not show_pbar, leave=True, dynamic_ncols=True,
                mininterval=MININTERVAL, miniters=MINITERS, smoothing=SMOOTHING, unit="it"
            ) if show_pbar else None

            train_outputs = []
            last_step_loss = float("nan")

            def _next_cta():
                nonlocal it_cta
                try:
                    return next(it_cta)
                except StopIteration:
                    it_cta = iter(aug_cta)
                    return next(it_cta)

            def _next_mra():
                nonlocal it_mra
                try:
                    return next(it_mra)
                except StopIteration:
                    it_mra = iter(aug_mra)
                    return next(it_mra)

            # ---- training steps ----
            for it in range(steps_per_epoch):
                b_ct = _next_cta()
                b_mr = _next_mra()
                batch = self._merge_batches(b_ct, b_mr)

                out = self.train_step(batch)
                train_outputs.append(out)

                try:
                    last_step_loss = float(out.get("loss", float("nan")))
                except Exception:
                    last_step_loss = float("nan")

                if pbar is not None:
                    if (it % POSTFIX_EVERY == 0):
                        pbar.set_postfix_str(f"mix|loss={last_step_loss:.4f}")
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

            self.on_train_epoch_end(train_outputs)

            # ---- validation ----
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []

                if (val_cta is None) or (val_mra is None):
                    # Fallback to default val loader
                    it_val = iter(self.dataloader_val)
                    for _ in range(self.num_val_iterations_per_epoch):
                        val_outputs.append(self.validation_step(next(it_val)))
                else:
                    itv_cta = iter(val_cta)
                    itv_mra = iter(val_mra)

                    def _next_val_cta():
                        nonlocal itv_cta
                        try:
                            return next(itv_cta)
                        except StopIteration:
                            itv_cta = iter(val_cta)
                            return next(itv_cta)

                    def _next_val_mra():
                        nonlocal itv_mra
                        try:
                            return next(itv_mra)
                        except StopIteration:
                            itv_mra = iter(val_mra)
                            return next(itv_mra)

                    if val_mode == "alt":
                        want_cta_first_val = (val_alt_first == "cta")
                        for itv in range(self.num_val_iterations_per_epoch):
                            want_cta = (itv % 2 == 0) if want_cta_first_val else (itv % 2 == 1)
                            batch = _next_val_cta() if want_cta else _next_val_mra()
                            val_outputs.append(self.validation_step(batch))
                    else:
                        # default: mixed validation
                        for _ in range(self.num_val_iterations_per_epoch):
                            b_ct = _next_val_cta()
                            b_mr = _next_val_mra()
                            batch = self._merge_batches(b_ct, b_mr)
                            val_outputs.append(self.validation_step(batch))

                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()