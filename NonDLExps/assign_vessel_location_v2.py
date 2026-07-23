import argparse
import csv
import sys
from typing import Optional

import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine
from scipy import ndimage
from scipy.spatial import cKDTree


#  vessel name 
VESSEL_LABELS: dict[int, str] = {
    1: "BA",
    2: "R-P1P2",
    3: "L-P1P2",
    4: "R-ICA-C6-C7",
    5: "R-M1",
    6: "L-ICA-C6-C7",
    7: "L-M1",
    8: "R-Pcom",
    9: "L-Pcom",
    10: "Acom",
    11: "R-A1A2",
    12: "L-A1A2",
    13: "R-A3",
    14: "L-A3",
    15: "3rd-A2",
    16: "3rd-A3",
    17: "R-M2",
    18: "R-M3",
    19: "L-M2",
    20: "L-M3",
    21: "R-P3P4",
    22: "L-P3P4",
    23: "R-VA",
    24: "L-VA",
    25: "R-SCA",
    26: "L-SCA",
    27: "R-AICA",
    28: "L-AICA",
    29: "R-PICA",
    30: "L-PICA",
    31: "R-AChA",
    32: "L-AChA",
    33: "R-OA",
    34: "L-OA",
    35: "R-ICA-C1-C5",
    36: "L-ICA-C1-C5",
}

LOCATION_LABELS: dict[str, int] = {
    "R-1.1 VA trunk": 1,
    "L-1.1 VA trunk": 2,
    "R-1.2 PICA trunk": 3,
    "L-1.2 PICA trunk": 4,
    "R-1.3 VA-PICA junction": 5,
    "L-1.3 VA-PICA junction": 6,
    "1.4 BA trunk": 7,
    "1.5 VA-BA junction": 8,
    "R-1.6 AICA trunk": 9,
    "L-1.6 AICA trunk": 10,
    "R-1.7 BA-AICA junction": 11,
    "L-1.7 BA-AICA junction": 12,
    "R-1.8 SCA trunk": 13,
    "L-1.8 SCA trunk": 14,
    "R-1.9 BA-SCA junction": 15,
    "L-1.9 BA-SCA junction": 16,
    "1.10 BA tip": 17,
    "R-2.1 P1P2": 18,
    "L-2.1 P1P2": 19,
    "R-2.2 P3P4": 20,
    "L-2.2 P3P4": 21,
    "R-3.1 ICA infraclinoid C1-C5": 22,
    "L-3.1 ICA infraclinoid C1-C5": 23,
    "R-3.2 ICA C6-OA-junction": 24,
    "L-3.2 ICA C6-OA-junction": 25,
    "R-3.3 ICA C6-nonOA": 26,
    "L-3.3 ICA C6-nonOA": 27,
    "R-3.4 ICA C7-Pcom-junction": 28,
    "L-3.4 ICA C7-Pcom-junction": 29,
    "R-3.5 ICA C7-AChA-junction": 30,
    "L-3.5 ICA C7-AChA-junction": 31,
    "R-3.6 ICA C7-nonBranch": 32,
    "L-3.6 ICA C7-nonBranch": 33,
    "R-3.7 ICA C7-terminus": 34,
    "L-3.7 ICA C7-terminus": 35,
    "4.1 Acom complex": 36,
    "R-4.2 A1": 37,
    "L-4.2 A1": 38,
    "R-4.3 A2": 39,
    "L-4.3 A2": 40,
    "R-4.4 A3": 41,
    "L-4.4 A3": 42,
    "R-4.5 Distal ACA branches": 43,
    "L-4.5 Distal ACA branches": 44,
    "R-5.1 M1 trunk": 45,
    "L-5.1 M1 trunk": 46,
    "R-5.2 M1-M2 junction": 47,
    "L-5.2 M1-M2 junction": 48,
    "R-5.3 Distal-M2M3": 49,
    "L-5.3 Distal-M2M3": 50,
}

VESSEL_TO_LOCATION: dict[str, str] = {
    "BA": "1.4 BA trunk",                       # todo: also 1.5/1.7/1.9/1.10
    "R-VA": "R-1.1 VA trunk",                   # todo: or 1.3 va-pica / 1.5 va-ba
    "L-VA": "L-1.1 VA trunk",                   # todo: or 1.3 va-pica / 1.5 va-ba
    "R-PICA": "R-1.2 PICA trunk",               # todo: or 1.3 va-pica
    "L-PICA": "L-1.2 PICA trunk",               # todo: or 1.3 va-pica
    "R-AICA": "R-1.6 AICA trunk",               # todo: or 1.7 ba-aica
    "L-AICA": "L-1.6 AICA trunk",               # todo: or 1.7 ba-aica
    "R-SCA": "R-1.8 SCA trunk",                 # todo: or 1.9 ba-sca
    "L-SCA": "L-1.8 SCA trunk",                 # todo: or 1.9 ba-sca
    "R-P1P2": "R-2.1 P1P2",
    "L-P1P2": "L-2.1 P1P2",
    "R-P3P4": "R-2.2 P3P4",
    "L-P3P4": "L-2.2 P3P4",
    "R-ICA-C1-C5": "R-3.1 ICA infraclinoid C1-C5",
    "L-ICA-C1-C5": "L-3.1 ICA infraclinoid C1-C5",
    "R-OA": "R-3.2 ICA C6-OA-junction",
    "L-OA": "L-3.2 ICA C6-OA-junction",
    "R-ICA-C6-C7": "R-3.3 ICA C6-nonOA",        # todo: c6-c7 spans 3.2-3.7
    "L-ICA-C6-C7": "L-3.3 ICA C6-nonOA",        # todo: c6-c7 spans 3.2-3.7
    "R-Pcom": "R-3.4 ICA C7-Pcom-junction",     # todo: pcom has no own class
    "L-Pcom": "L-3.4 ICA C7-Pcom-junction",     # todo: pcom has no own class
    "R-AChA": "R-3.5 ICA C7-AChA-junction",
    "L-AChA": "L-3.5 ICA C7-AChA-junction",
    "Acom": "4.1 Acom complex",
    "R-A1A2": "R-4.2 A1",                       # todo: spans 4.2 a1 and 4.3 a2
    "L-A1A2": "L-4.2 A1",                       # todo: spans 4.2 a1 and 4.3 a2
    "3rd-A2": "R-4.3 A2",                       # todo: azygos, no midline class
    "3rd-A3": "R-4.4 A3",                       # todo: azygos, no midline class
    "R-A3": "R-4.4 A3",
    "L-A3": "L-4.4 A3",
    "R-M1": "R-5.1 M1 trunk",                   # todo: or 5.2 m1-m2
    "L-M1": "L-5.1 M1 trunk",                   # todo: or 5.2 m1-m2
    "R-M2": "R-5.3 Distal-M2M3",
    "L-M2": "L-5.3 Distal-M2M3",
    "R-M3": "R-5.3 Distal-M2M3",
    "L-M3": "L-5.3 Distal-M2M3",
}

# if the aneurysm touches BOTH vessels of a pair, use the junction class.

JUNCTION_RULES: dict[frozenset[str], str] = {
    frozenset({"R-M1", "R-M2"}): "R-5.2 M1-M2 junction",
    frozenset({"L-M1", "L-M2"}): "L-5.2 M1-M2 junction",
    frozenset({"R-VA", "BA"}): "1.5 VA-BA junction",
    frozenset({"L-VA", "BA"}): "1.5 VA-BA junction",
    frozenset({"R-VA", "R-PICA"}): "R-1.3 VA-PICA junction",
    frozenset({"L-VA", "L-PICA"}): "L-1.3 VA-PICA junction",
    frozenset({"BA", "R-AICA"}): "R-1.7 BA-AICA junction",
    frozenset({"BA", "L-AICA"}): "L-1.7 BA-AICA junction",
    frozenset({"BA", "R-SCA"}): "R-1.9 BA-SCA junction",
    frozenset({"BA", "L-SCA"}): "L-1.9 BA-SCA junction",
    frozenset({"R-ICA-C6-C7", "R-OA"}): "R-3.2 ICA C6-OA-junction",
    frozenset({"L-ICA-C6-C7", "L-OA"}): "L-3.2 ICA C6-OA-junction",
    frozenset({"R-ICA-C6-C7", "R-Pcom"}): "R-3.4 ICA C7-Pcom-junction",
    frozenset({"L-ICA-C6-C7", "L-Pcom"}): "L-3.4 ICA C7-Pcom-junction",
    frozenset({"R-ICA-C6-C7", "R-AChA"}): "R-3.5 ICA C7-AChA-junction",
    frozenset({"L-ICA-C6-C7", "L-AChA"}): "L-3.5 ICA C7-AChA-junction",
    frozenset({"BA", "R-P1P2"}): "1.10 BA tip",   # todo: basilar tip, midline
    frozenset({"BA", "L-P1P2"}): "1.10 BA tip",   # todo: basilar tip, midline
    # fix : acom is not its own vessel; an acom aneurysm sits between both A1s
    # (or between an A1 and the acom), so touching two of them -> 4.1 acom complex.
    frozenset({"R-A1A2", "L-A1A2"}): "4.1 Acom complex",
    frozenset({"R-A1A2", "Acom"}): "4.1 Acom complex",
    frozenset({"L-A1A2", "Acom"}): "4.1 Acom complex",
}

_DIST_EPS_MM: float = 1e-3


def load_mask(path: str) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(path)
    labels = np.rint(np.asarray(img.dataobj)).astype(np.int64)
    return labels, img.affine


def check_grids(aneurysm_affine: np.ndarray, aneurysm_shape: tuple[int, ...],
                vessel_affine: np.ndarray, vessel_shape: tuple[int, ...]) -> None:
    if aneurysm_shape != vessel_shape:
        raise ValueError(
            f"shape mismatch: aneurysm {aneurysm_shape} vs vessels {vessel_shape}. "
            "no resampling, pass masks on the same grid."
        )
    if not np.allclose(aneurysm_affine, vessel_affine, atol=1e-4):
        raise ValueError(
            "affine mismatch between aneurysm and vessels. no resampling, pass "
            f"masks on the same grid.\naneurysm:\n{aneurysm_affine}\nvessels:\n{vessel_affine}"
        )


def voxel_world_coords(voxel_indices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return apply_affine(affine, voxel_indices.astype(np.float64))


def connected_components(mask: np.ndarray, min_voxels: int) -> tuple[np.ndarray, list[int]]:
    structure = np.ones((3, 3, 3), dtype=np.int8)  # 26-connectivity
    labelled, n = ndimage.label(mask > 0, structure=structure)
    kept = [c for c in range(1, n + 1)
            if int(np.count_nonzero(labelled == c)) >= min_voxels]
    return labelled, kept


def build_vessel_tree(vessel_arr: np.ndarray, affine: np.ndarray) -> tuple[cKDTree, np.ndarray]:
    voxel_indices = np.argwhere(vessel_arr > 0)
    if voxel_indices.size == 0:
        raise ValueError("vessel mask is empty, nothing to assign to.")
    world = voxel_world_coords(voxel_indices, affine)
    vessel_classes = vessel_arr[tuple(voxel_indices.T)].astype(np.int64)
    return cKDTree(world), vessel_classes


def laterality_from_label(name: str) -> str:
    """R/L from an R-/L- prefix (works for vessel and location names), else M."""
    if name.startswith("R-"):
        return "R"
    if name.startswith("L-"):
        return "L"
    return "M"


def rank_component_vessels(coords_mm: np.ndarray, tree: cKDTree,
                           vessel_classes: np.ndarray) -> tuple[list[dict[str, object]], float, float]:

    distances, nn_index = tree.query(coords_mm, k=1)
    distances = np.asarray(distances, dtype=np.float64)
    nn_classes = vessel_classes[np.asarray(nn_index)]
    weights = 1.0 / np.maximum(distances, _DIST_EPS_MM)

    ranked: list[dict[str, object]] = []
    for cls in np.unique(nn_classes):
        sel = nn_classes == cls
        value = int(cls)
        ranked.append({
            "value": value,
            "name": VESSEL_LABELS.get(value, f"UNKNOWN-{value}"),
            "weight": float(weights[sel].sum()),
            "min_dist_mm": float(distances[sel].min()),
        })
    ranked.sort(key=lambda r: r["weight"], reverse=True)
    return ranked, float(weights.sum()), float(distances.min())


def resolve_location(ranked: list[dict[str, object]], junction_tol_mm: float,
                     use_junctions: bool) -> tuple[str, Optional[int], str, str]:

    primary = ranked[0]["name"]

    if use_junctions:
        for cand in ranked[1:]:
            if float(cand["min_dist_mm"]) > junction_tol_mm:
                continue
            pair = frozenset({primary, str(cand["name"])})
            if pair in JUNCTION_RULES:
                loc = JUNCTION_RULES[pair]
                return loc, LOCATION_LABELS.get(loc), str(cand["name"]), "junction"

    loc = VESSEL_TO_LOCATION.get(primary)
    if loc is None:
        return "", None, "", "single"
    return loc, LOCATION_LABELS.get(loc), "", "single"


def assign_vessel_location(aneurysm_arr: np.ndarray, vessel_arr: np.ndarray,
                           affine: np.ndarray, min_voxels: int, max_dist_mm: float,
                           junction_tol_mm: float, use_junctions: bool) -> list[dict[str, object]]:
    tree, vessel_classes = build_vessel_tree(vessel_arr, affine)
    labelled, kept = connected_components(aneurysm_arr, min_voxels)

    rows: list[dict[str, object]] = []
    unmapped: set[str] = set()

    for component_id, comp_label in enumerate(kept, start=1):
        voxel_indices = np.argwhere(labelled == comp_label)
        coords_mm = voxel_world_coords(voxel_indices, affine)
        centroid = coords_mm.mean(axis=0)

        ranked, total_weight, min_dist = rank_component_vessels(coords_mm, tree, vessel_classes)
        primary = ranked[0]
        vessel_name = str(primary["name"])
        vote_confidence = float(primary["weight"]) / total_weight if total_weight > 0 else 0.0

        location_name, location_value, second_name, resolved_by = resolve_location(
            ranked, junction_tol_mm, use_junctions)
        if location_name == "":
            unmapped.add(vessel_name)

        lat_source = location_name if resolved_by == "junction" else vessel_name
        laterality = laterality_from_label(lat_source)

        rows.append({
            "component_id": component_id,
            "n_voxels": int(voxel_indices.shape[0]),
            "centroid_mm": f"{centroid[0]:.3f};{centroid[1]:.3f};{centroid[2]:.3f}",
            "nearest_vessel_label": vessel_name,
            "nearest_vessel_value": int(primary["value"]),
            "second_vessel_label": second_name,
            "resolved_by": resolved_by,
            "location_class": location_name,
            "location_value": "" if location_value is None else location_value,
            "laterality": laterality,
            "min_dist_mm": round(min_dist, 4),
            "vote_confidence": round(vote_confidence, 4),
            "flagged": bool(min_dist > max_dist_mm),
        })

    if unmapped:
        print("warning: vessels not mapped in VESSEL_TO_LOCATION (empty location cell): "
              + ", ".join(sorted(unmapped)), file=sys.stderr)

    return rows


CSV_FIELDS = [
    "component_id", "n_voxels", "centroid_mm",
    "nearest_vessel_label", "nearest_vessel_value",
    "second_vessel_label", "resolved_by",
    "location_class", "location_value",
    "laterality", "min_dist_mm", "vote_confidence", "flagged",
]


def write_csv(rows: list[dict[str, object]], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="assign each aneurysm the location of its nearest vessel, using "
                    "two-vessel junction rules (topaneu 2026 stage 2 baseline v2).")
    parser.add_argument("--aneurysm", required=True, help="aneurysm mask nifti.")
    parser.add_argument("--vessels", required=True, help="multiclass vessel mask nifti.")
    parser.add_argument("--out", required=True, help="output csv.")
    parser.add_argument("--min-voxels", type=int, default=5,
                        help="drop components smaller than this (default: 5).")
    parser.add_argument("--max-dist-mm", type=float, default=5.0,
                        help="flag components whose neck is farther than this (default: 5.0 mm).")
    parser.add_argument("--junction-dist-mm", type=float, default=3.0,
                        help="a second vessel this close (mm) can trigger a junction (default: 3.0).")
    parser.add_argument("--no-junctions", action="store_true",
                        help="disable fix A and behave like v1 (single nearest vessel).")
    args = parser.parse_args(argv)

    aneurysm_arr, aneurysm_affine = load_mask(args.aneurysm)
    vessel_arr, vessel_affine = load_mask(args.vessels)
    check_grids(aneurysm_affine, aneurysm_arr.shape, vessel_affine, vessel_arr.shape)

    rows = assign_vessel_location(aneurysm_arr, vessel_arr, aneurysm_affine,
                                  min_voxels=args.min_voxels, max_dist_mm=args.max_dist_mm,
                                  junction_tol_mm=args.junction_dist_mm,
                                  use_junctions=not args.no_junctions)
    write_csv(rows, args.out)

    n_flagged = sum(1 for r in rows if r["flagged"])
    n_junction = sum(1 for r in rows if r["resolved_by"] == "junction")
    print(f"wrote {len(rows)} row(s) to {args.out} ({n_junction} junction, {n_flagged} flagged).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
 python assign_vessel_location_v2.py `
   --aneurysm location_masks\topaneu_center2_mr_002.nii.gz `
   --vessels  vessel_masks\topaneu_center2_mr_002.nii.gz `
   --out out_002_v2.csv
"""
