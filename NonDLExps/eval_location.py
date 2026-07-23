import glob
import os
import sys
import numpy as np
import NonDLExps.assign_vessel_location_v2 as m

INV_LOCATION = {v: k for k, v in m.LOCATION_LABELS.items()}
MIN_VOXELS = 5
JUNCTION_TOL_MM = 3.0


def group_of(loc_name: str) -> str:
    name = loc_name[2:] if loc_name[:2] in ("R-", "L-") else loc_name
    return name.split(".", 1)[0].strip()


def eval_case(loc_path: str, ves_path: str, use_junctions: bool) -> list[dict]:
    gt_arr, gt_aff = m.load_mask(loc_path)
    ves_arr, ves_aff = m.load_mask(ves_path)
    m.check_grids(gt_aff, gt_arr.shape, ves_aff, ves_arr.shape)

    tree, vclasses = m.build_vessel_tree(ves_arr, gt_aff)
    labelled, kept = m.connected_components(gt_arr, MIN_VOXELS)

    out = []
    for comp in kept:
        idx = np.argwhere(labelled == comp)
        coords = m.voxel_world_coords(idx, gt_aff)

        vals = gt_arr[tuple(idx.T)]
        vals = vals[vals > 0]
        gt_val = int(np.bincount(vals).argmax())   # majority true label in the component
        gt_name = INV_LOCATION.get(gt_val, str(gt_val))

        ranked, tw, mind = m.rank_component_vessels(coords, tree, vclasses)
        loc_name, loc_val, second, how = m.resolve_location(ranked, JUNCTION_TOL_MM, use_junctions)

        out.append({
            "gt_val": gt_val, "gt_name": gt_name,
            "pred_val": loc_val, "pred_name": loc_name, "how": how,
        })
    return out


def score(records: list[dict]) -> None:
    n = len(records)
    if n == 0:
        print("  no components")
        return
    exact = side = group = 0
    for r in records:
        gt, pred = r["gt_name"], r["pred_name"]
        if r["pred_val"] == r["gt_val"]:
            exact += 1
        if pred and m.laterality_from_label(pred) == m.laterality_from_label(gt):
            side += 1
        if pred and group_of(pred) == group_of(gt):
            group += 1
    print(f"  components: {n}")
    print(f"  exact class : {exact:4d} / {n}  = {exact / n:6.1%}")
    print(f"  territory   : {group:4d} / {n}  = {group / n:6.1%}   (group 1-5)")
    print(f"  laterality  : {side:4d} / {n}  = {side / n:6.1%}")


def top_confusions(records: list[dict], k: int = 12) -> None:
    from collections import Counter
    c = Counter((r["gt_name"], r["pred_name"]) for r in records if r["pred_val"] != r["gt_val"])
    print(f"  top {k} confusions (true -> predicted):")
    for (gt, pred), cnt in c.most_common(k):
        print(f"    {cnt:3d}x  {gt}  ->  {pred or '(unmapped)'}")


def run(loc_dir: str, ves_dir: str) -> None:
    loc_files = sorted(glob.glob(os.path.join(loc_dir, "*.nii.gz")))
    print(f"cases found: {len(loc_files)}\n")

    for use_junctions in (False, True):
        records = []
        skipped = []
        for loc_path in loc_files:
            name = os.path.basename(loc_path)
            ves_path = os.path.join(ves_dir, name)
            if not os.path.exists(ves_path):
                skipped.append(name)
                continue
            records.extend(eval_case(loc_path, ves_path, use_junctions))

        tag = "WITH junctions (fix A)" if use_junctions else "WITHOUT junctions (v1)"
        print("=" * 60)
        print(tag)
        print("=" * 60)
        score(records)
        n_junc = sum(1 for r in records if r["how"] == "junction")
        if use_junctions:
            print(f"  resolved by junction rule: {n_junc}")
        top_confusions(records)
        if skipped:
            print(f"  skipped (no vessel mask): {len(skipped)}")
        print()


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    run(os.path.join(here, "location_masks"), os.path.join(here, "vessel_masks"))
