import numpy as np, SimpleITK as sitk 
from SimpleITK import GetArrayViewFromImage as ArrayView

def vs_single_label(*, gt: sitk.Image, pred: sitk.Image, label: int) -> float:
    """
    SimpleITK's GetVolumeSimilarity() computes the volume difference
        VS_sitk = 2(V1-V2)/(V1+V2)
    Convert to conventional VS with this formula:
        VS_conventional = 1 - |VS_sitk|/2
    Ref:
        Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool.
        BMC medical imaging. 2015 Aug 12;15(1):29.
    """
    gt_label_arr = sitk.GetArrayFromImage(gt == label)
    pred_label_arr = sitk.GetArrayFromImage(pred == label)

    # check if either gt or pred label_arr is all zero
    if (not np.any(gt_label_arr)) or (not np.any(pred_label_arr)):
        return 0

    pred.CopyInformation(gt)

    overlap_measures = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures.SetNumberOfThreads(1)
    overlap_measures.Execute(gt, pred)
    # SimpleITK's signed output: ranges from -2 to +2
    sitk_vs = overlap_measures.GetVolumeSimilarity(label)

    # Convert to conventional Volumetric Similarity: ranges from 0 to 1
    conventional_vs = 1.0 - (abs(sitk_vs) / 2.0)
    return conventional_vs

def dice_coefficient_single_label(
    *, gt: sitk.Image, pred: sitk.Image, label: int
) -> float:
    """
    use overlap measures filter with a single label
    for Dice Similarity Coefficient (DSC)

    NOTE: DSC in sitk.LabelOverlapMeasuresImageFilter
    does NOT use voxel spacing. It calculates the count
    of voxels without considering the area/volume.

    NOTE: two sitk.Images with DIFFERENT voxel spacings
    can still be calculated for overlap measures...(?!)
    """
    # print(f"\nfor label-{label}")

    # Check if label exists for both gt and pred
    # If not, DSC is automatically set to 0 due to FP or FN
    gt_label_arr = sitk.GetArrayFromImage(gt == label)
    pred_label_arr = sitk.GetArrayFromImage(pred == label)

    # check if either gt or pred label_arr is all zero
    if (not np.any(gt_label_arr)) or (not np.any(pred_label_arr)):
        # print(f"[!!Warning] label-{label} empty for gt or pred")
        return 0

    # NOTE: sometimes there are tiny differences in image Direction:
    # ITK ERROR: LabelOverlapMeasuresImageFilter(0x55bf9ad3e5e0):
    # Inputs do not occupy the same physical space!
    # Thus make sure they have the same metadata
    # Copies the Origin, Spacing, and Direction from the gt image
    pred.CopyInformation(gt)

    overlap_measures = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures.SetNumberOfThreads(1)
    overlap_measures.Execute(gt, pred)
    dice_score = overlap_measures.GetDiceCoefficient(label)
    # print("dice_score = ", dice_score)
    return dice_score



def arr_is_binary(arr: np.array) -> bool:
    """
    test if the numpy array is binary
    NOTE: all zeros or all ones are also binary!
    """
    return set(np.unique(arr)).issubset({0, 1})


def pad_sitk_image(image: sitk.Image) -> sitk.Image:
    # print("\nbefore padding, image:\n")
    # print(image.GetSize())
    # print(sitk.GetArrayFromImage(image))

    # Define the amount of padding to add to each side (x, y, z)
    # https://itk.org/Doxygen/html/classitk_1_1PadImageFilter.html
    dim = image.GetDimension()  # can be 2D or 3D
    pad_lower_bound = [1] * dim  # Padding to add at the beginning of each axis
    pad_upper_bound = [1] * dim  # Padding to add at the end of each axis

    # Pad the image with 0s
    constant = 0
    padded_image = sitk.ConstantPad(image, pad_lower_bound, pad_upper_bound, constant)

    # print("\nafter padding, padded_image:\n")
    # print(padded_image.GetSize())
    # print(sitk.GetArrayFromImage(padded_image))

    return padded_image

def _get_surface_distance(seg: sitk.Image) -> tuple[sitk.Image, sitk.Image, int]:
    """
    Code adapted from:
        ITK Forum:
            https://discourse.itk.org/t/computing-95-hausdorff-distance/3832/
        ITK tutorial surface_hausdorff_distance:
            NOTE: use the latest InsightSoftwareConsortium/SimpleITK-Notebooks repo
            https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/34_Segmentation_Evaluation.ipynb
            with fixes from:
            https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/commit/4a3967e5edeb6f746e4c79d53b416a2489ba8346
            https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/commit/0cb643655d9fc6f08cecffc1ffe1d0997d78dedb
        seg-metrics: a Python package to compute segmentation metrics
            https://github.com/Jingnan-Jia/segmentation_metrics
        ToothFairy1 Challenge:
            https://github.com/AImageLab-zip/ToothFairy/blob/main/ToothFairy/evaluation/evaluation.py

    NOTE: in bugfix https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/commit/4a3967e5edeb6f746e4c79d53b416a2489ba8346
    "BUG: Used segmentation distance maps and not surface distance maps.
        Distances between surfaces should use the surface distance maps and
        not distance maps based on the original segmentations."
    """

    # extract the contour outline for later masking
    seg_surface = sitk.LabelContour(
        seg,
        # set to fully connected
        fullyConnected=True,
    )

    # get map of the distance to boundary for input segmentation mask
    # use image spacing with Maurer distance transform
    seg_distance_map = sitk.Abs(
        sitk.SignedMaurerDistanceMap(
            seg_surface,  # fix in SimpleITK-Notebooks/commit/4a3967
            squaredDistance=False,
            useImageSpacing=True,
        )
    )

    # with np.printoptions(precision=1, suppress=True):
    #     print("seg_distance_map =\n", ArrayView(seg_distance_map))
    #     print("seg_distance_map.GetSize() =", seg_distance_map.GetSize())

    #     print("seg_surface =\n", ArrayView(seg_surface))
    #     print("seg_surface.GetSize() =", seg_surface.GetSize())

    # get the number of surface pixels for HD sorting later
    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(seg_surface)

    num_surface_pixels = int(statistics_image_filter.GetSum())
    # print("num_surface_pixels = ", num_surface_pixels)

    return seg_distance_map, seg_surface, num_surface_pixels

def hd95_single_label(*, gt: sitk.Image, pred: sitk.Image, label: int) -> list[float]:
    """
    Calculates the Hausdorff distance at 95% percentile

    NOTE: While there are many different implementations,
    packages, and even definitions(!) to calculate HD95,
    we decide to go with the definiton from
        Reinke, A., Tizabi, M.D., Baumgartner, M. et al.
        Understanding metric-related pitfalls in image analysis validation.
        Nat Methods 21, 182–194 (2024).
        See Fig. SN 3.63 and
        https://metrics-reloaded.dkfz.de/metric?id=hd95
    The implementation takes the max of two d_95:
        max(d_95(A,B), d_95(B,A))
    We verified this implementation with various Figs from
        Reinke et al., 2021
        Common Limitations of Image Processing Metrics: A Picture Story

    NOTE: in case of missing values (FP or FN), set the HD95
    to be roughly the maximum distance in ROI = <HD95_UPPER_BOUND> mm

    Parameters
    ----------
    gt:
        ground truth mask sitk image
    pred:
        predicted mask sitk image
    label:
        annotation label integer

    Returns
    ----------
    [float hd95_score, float hd100_score]
    The distance unit is the same as the voxelspacing,
        which is usually in mm.

    References:
        Reinke et al., 2024
            Metrics reloaded: recommendations for image analysis validation
        Reinke et al., 2021
            Common Limitations of Image Processing Metrics: A Picture Story
        ITK forum:
            https://discourse.itk.org/t/computing-95-hausdorff-distance/3832
        ITK tutorial surface_hausdorff_distance:
            NOTE: use the latest InsightSoftwareConsortium/SimpleITK-Notebooks repo
            https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/34_Segmentation_Evaluation.ipynb
            with fixes from:
            https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/commit/4a3967e5edeb6f746e4c79d53b416a2489ba8346
            https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/commit/0cb643655d9fc6f08cecffc1ffe1d0997d78dedb
        seg-metrics: a Python package to compute segmentation metrics
            https://github.com/Jingnan-Jia/segmentation_metrics
        ToothFairy1 Challenge:
            https://github.com/AImageLab-zip/ToothFairy/blob/main/ToothFairy/evaluation/evaluation.py
    """
    HD95_UPPER_BOUND = 290
    # print(f"\n--> hd95_single_label(label-{label})\n")

    # gt and pred should have the same shape
    assert gt.GetSize() == pred.GetSize(), "gt pred not matching shapes!"

    # img should be in 3D (allow for 2D for testing purposes)
    assert gt.GetDimension() in (2, 3), "sitk img in 2D|3D, only fo HD"

    # NOTE: need to pad the image in case it is completely filled
    # found that when the image is completely filled,
    # SignedMaurerDistanceMap does not work as it gives all 0 distance.
    # see issue: https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/issues/453
    # fix by pad the gt and pred

    gt = pad_sitk_image(gt)
    pred = pad_sitk_image(pred)

    # only need bool binary mask of the current label
    gt_label_img = gt == label
    pred_label_img = pred == label

    # gt_arr and pred_arr are from union of showed-up labels,
    # thus they will not be both all zeros
    # thus only FP and FN can happen
    # handle FP and FN with HD95_UPPER_BOUND

    gt_label_arr = sitk.GetArrayFromImage(gt_label_img)
    pred_label_arr = sitk.GetArrayFromImage(pred_label_img)

    # make sure the masks are binary
    assert arr_is_binary(gt_label_arr), "hd95_single_label expects binary gt_arr"
    assert arr_is_binary(pred_label_arr), "hd95_single_label expects binary pred_arr"

    # check if either gt or pred label_arr is all zero
    if (not np.any(gt_label_arr)) or (not np.any(pred_label_arr)):
        # print(f"[!!Warning] label-{label} empty for gt or pred")
        return [HD95_UPPER_BOUND, HD95_UPPER_BOUND]

    ##################################################################
    # Now the real HD95 implementation :)
    # -> max(d_95(A,B), d_95(B,A))
    ##################################################################

    # get the distance_map, surface, and number of surface pixels
    # for both gt/ref and pred
    (
        ref_distance_map,
        ref_surface,
        num_ref_surface_pixels,
    ) = _get_surface_distance(gt_label_img)
    (
        pred_distance_map,
        pred_surface,
        num_pred_surface_pixels,
    ) = _get_surface_distance(pred_label_img)

    # extract the distances of boundary_ref to boundary_pred
    # and vice versa for both directions
    # NOTE: SimpleITK MultiplyImageFilter requires
    # both input images to have the same pixel type
    # distance_map is float32, so need to cast surface to float
    ref2pred_distance_map = pred_distance_map * sitk.Cast(ref_surface, sitk.sitkFloat32)
    pred2ref_distance_map = ref_distance_map * sitk.Cast(pred_surface, sitk.sitkFloat32)

    # with np.printoptions(precision=1, suppress=True):
    #     print("ref2pred_distance_map =\n", ArrayView(ref2pred_distance_map))
    #     print("pred2ref_distance_map =\n", ArrayView(pred2ref_distance_map))

    # extract the non-zero distances from the distance_map
    ref2pred_distances = list(
        ArrayView(ref2pred_distance_map)[ArrayView(ref2pred_distance_map) != 0]
    )
    # create a list based on the number of surface pixels
    # populate the rest of the list with 0
    ref2pred_distances += [0] * (num_ref_surface_pixels - len(ref2pred_distances))

    # print("ref2pred_distances =\n", sorted(ref2pred_distances, reverse=True))
    # print("# ref2pred_distances =\n", len(ref2pred_distances))

    # do the same for ther other direction pred2ref
    pred2ref_distances = list(
        ArrayView(pred2ref_distance_map)[ArrayView(pred2ref_distance_map) != 0]
    )
    pred2ref_distances += [0] * (num_pred_surface_pixels - len(pred2ref_distances))

    # print("pred2ref_distances =\n", sorted(pred2ref_distances, reverse=True))
    # print("# pred2ref_distances =\n", len(pred2ref_distances))

    # use formula -> max(d_95(A,B), d_95(B,A))
    d_95_ref2pred = np.percentile(ref2pred_distances, 95)
    # print("d_95_ref2pred = ", d_95_ref2pred)
    d_95_pred2ref = np.percentile(pred2ref_distances, 95)
    # print("d_95_pred2ref = ", d_95_pred2ref)

    hd95_score = max(d_95_ref2pred, d_95_pred2ref)
    # print("hd95_score = ", hd95_score)

    # also keep track of HD max
    d_100_ref2pred = np.percentile(ref2pred_distances, 100)
    # print("d_100_ref2pred = ", d_100_ref2pred)
    d_100_pred2ref = np.percentile(pred2ref_distances, 100)
    # print("d_100_pred2ref = ", d_100_pred2ref)
    hd100_score = max(d_100_ref2pred, d_100_pred2ref)
    # print("hd100_score = ", hd100_score)

    return [hd95_score/HD95_UPPER_BOUND, hd100_score/HD95_UPPER_BOUND]