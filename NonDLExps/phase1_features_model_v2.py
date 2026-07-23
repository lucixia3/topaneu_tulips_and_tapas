import glob
import os
import numpy as np
from nibabel.affines import apply_affine
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

import NonDLExps.assign_vessel_location_v2 as m

INV_LOCATION = {v: k for k, v in m.LOCATION_LABELS.items()}
VESSEL_VALUES = sorted(m.VESSEL_LABELS)          # 1..36, fixed column order
MIN_VOXELS = 5
JUNCTION_TOL_MM = 3.0
N_SPLITS = 5
_OFFSETS = [(dx, dy, dz)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
            if not (dx == dy == dz == 0)]


def base_name(loc_name: str) -> str:
    """strip an R-/L- prefix: 'R-5.1 M1 trunk' -> '5.1 M1 trunk'."""
    return loc_name[2:] if loc_name[:2] in ("R-", "L-") else loc_name


BASE_LIST = sorted({base_name(n) for n in m.LOCATION_LABELS})   # 27 base classes
BASE_TO_IDX = {b: i for i, b in enumerate(BASE_LIST)}
BASE_OF_VAL = {v: base_name(n) for n, v in m.LOCATION_LABELS.items()}


def is_lateralized(base: str) -> bool:
    return ("R-" + base) in m.LOCATION_LABELS


def combine(base: str, lat: str) -> int:
    if is_lateralized(base):
        side = lat if lat in ("R", "L") else "R"     
        return m.LOCATION_LABELS[f"{side}-{base}"]
    return m.LOCATION_LABELS[base]                    # midline single

def skeleton_indices(vidx: np.ndarray) -> np.ndarray:
    mins = vidx.min(axis=0)
    shape = tuple((vidx.max(axis=0) - mins + 1).tolist())
    vol = np.zeros(shape, dtype=bool)
    vol[tuple((vidx - mins).T)] = True
    sk = np.argwhere(skeletonize(vol))
    if sk.shape[0] < 2:
        return vidx
    return sk + mins


def prepare_vessel_path(vidx: np.ndarray, affine: np.ndarray) -> dict | None:
    """skeletonize the vessel, build a graph, find the longest path, and return geodesic distances."""
    nodes = skeleton_indices(vidx)
    nodes_world = apply_affine(affine, nodes.astype(np.float64))
    n = len(nodes)
    if n < 2:
        return None

    idxmap = {(int(p[0]), int(p[1]), int(p[2])): i for i, p in enumerate(nodes)}
    rows, cols, data = [], [], []
    for i, p in enumerate(nodes):
        pi = (int(p[0]), int(p[1]), int(p[2]))
        for off in _OFFSETS:
            j = idxmap.get((pi[0] + off[0], pi[1] + off[1], pi[2] + off[2]))
            if j is not None and j > i:
                rows.append(i)
                cols.append(j)
                data.append(float(np.linalg.norm(nodes_world[i] - nodes_world[j])))
    if not data:
        return None

    graph = csr_matrix((data + data, (rows + cols, cols + rows)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    keep = np.where(labels == np.bincount(labels).argmax())[0]   # largest component
    if len(keep) < 2:
        return None

    # endpoints of the longest path: farthest node from an arbitrary start (A),
    # then farthest from A (B); dA = geodesic distance from A along the vessel.
    d0 = dijkstra(graph, indices=int(keep[0]), directed=False)
    a = int(keep[np.argmax(d0[keep])])
    dA = dijkstra(graph, indices=a, directed=False)
    total = float(dA[keep].max())
    if not np.isfinite(total) or total < 1e-6:
        return None
    return {"nodes_world": nodes_world, "keep": keep, "dA": dA, "total": total}


def along_from_path(path: dict, centroid: np.ndarray) -> float:
    """position of the aneurysm along the vessel centerline, normalized to [0,1]."""
    keep, nw = path["keep"], path["nodes_world"]
    p = int(keep[np.argmin(np.linalg.norm(nw[keep] - centroid, axis=1))])
    return float(np.clip(path["dA"][p] / path["total"], 0.0, 1.0))


def along_pca(vessel_world: np.ndarray, centroid: np.ndarray) -> float:
    c = vessel_world.mean(axis=0)
    x = vessel_world - c
    if x.shape[0] < 2:
        return 0.5
    axis = np.linalg.svd(x, full_matrices=False)[2][0]
    proj_v = x @ axis
    lo, hi = float(proj_v.min()), float(proj_v.max())
    if hi - lo < 1e-6:
        return 0.5
    return float(np.clip((float((centroid - c) @ axis) - lo) / (hi - lo), 0.0, 1.0))


#helpers
def modality_bit(case_name: str) -> int:
    """0 = MR, 1 = CT, from topaneu_{center}_{modality}_{id}."""
    parts = case_name.split("_")
    mod = parts[2].lower() if len(parts) > 2 else ""
    return 1 if mod.startswith("ct") else 0


def case_vessel_geometry(ves_arr: np.ndarray, affine: np.ndarray):
    trees, coords, vidx, all_world = {}, {}, {}, []
    for v in VESSEL_VALUES:
        idx = np.argwhere(ves_arr == v)
        if idx.size == 0:
            continue
        w = m.voxel_world_coords(idx, affine)
        trees[v] = cKDTree(w)
        coords[v] = w
        vidx[v] = idx
        all_world.append(w)
    world = np.concatenate(all_world, axis=0)
    return trees, coords, vidx, world.min(axis=0), world.max(axis=0)


def laterality_geom(ranked: list[dict]) -> str:
    for r in ranked:
        lat = m.laterality_from_label(str(r["name"]))
        if lat in ("R", "L"):
            return lat
    return "M"



def extract_case(loc_path: str, ves_path: str) -> list[dict]:
    gt_arr, gt_aff = m.load_mask(loc_path)
    ves_arr, ves_aff = m.load_mask(ves_path)
    m.check_grids(gt_aff, gt_arr.shape, ves_aff, ves_arr.shape)

    tree, vclasses = m.build_vessel_tree(ves_arr, gt_aff)          # for the baseline
    trees, vcoords, vidx, bb_lo, bb_hi = case_vessel_geometry(ves_arr, gt_aff)
    labelled, kept = m.connected_components(gt_arr, MIN_VOXELS)
    case_name = os.path.basename(loc_path).replace(".nii.gz", "")
    mod = modality_bit(case_name)
    span = np.maximum(bb_hi - bb_lo, 1e-6)
    voxel_vol = abs(np.linalg.det(gt_aff[:3, :3]))                 # mm^3 per voxel
    vessel_paths: dict[int, dict | None] = {}                      # cache per vessel

    rows = []
    for comp in kept:
        idx = np.argwhere(labelled == comp)
        coords = m.voxel_world_coords(idx, gt_aff)
        centroid = coords.mean(axis=0)

        dists = np.full(len(VESSEL_VALUES), np.nan)
        for j, v in enumerate(VESSEL_VALUES):
            if v in trees:
                dists[j] = float(trees[v].query(coords, k=1)[0].min())

        d_sorted = np.sort(dists)                                  # nan sorts to the end
        d1, d2, d3 = d_sorted[0], d_sorted[1], d_sorted[2]
        r21 = d2 / d1 if d1 > 1e-6 else np.nan
        r31 = d3 / d1 if d1 > 1e-6 else np.nan

        # geodesic along-vessel position on the nearest present vessel
        if np.all(np.isnan(dists)):
            along = 0.5
        else:
            nv = VESSEL_VALUES[int(np.nanargmin(dists))]
            if nv not in vessel_paths:
                vessel_paths[nv] = prepare_vessel_path(vidx[nv], gt_aff)
            path = vessel_paths[nv]
            along = along_from_path(path, centroid) if path else along_pca(vcoords[nv], centroid)

        # aneurysm size 
        n_vox = float(idx.shape[0])
        volume = n_vox * voxel_vol
        diameter = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))

        norm_xyz = (centroid - bb_lo) / span
        feats = np.concatenate([
            dists, [d1, d2, d3, r21, r31], [n_vox, volume, diameter],
            norm_xyz, [along, float(mod)],
        ])

        vals = gt_arr[tuple(idx.T)]
        vals = vals[vals > 0]
        gt_val = int(np.bincount(vals).argmax())

        ranked, tw, mind = m.rank_component_vessels(coords, tree, vclasses)
        _, base_val, _, _ = m.resolve_location(ranked, JUNCTION_TOL_MM, True)

        rows.append({
            "case": case_name, "feats": feats, "gt_val": gt_val,
            "geom_lat": laterality_geom(ranked),
            "base_val": base_val if base_val is not None else -1,
        })
    return rows



def name_of(val: int) -> str:
    return INV_LOCATION.get(int(val), "")


def group_of(val: int) -> str:
    name = name_of(val)
    return base_name(name).split(".", 1)[0].strip() if name else "?"


def three_layers(gt: np.ndarray, pred: np.ndarray, tag: str) -> None:
    n = len(gt)
    exact = int((gt == pred).sum())
    side = sum(1 for g, p in zip(gt, pred)
               if name_of(p) and m.laterality_from_label(name_of(p)) == m.laterality_from_label(name_of(g)))
    terr = sum(1 for g, p in zip(gt, pred) if name_of(p) and group_of(p) == group_of(g))
    print(f"{tag}")
    print(f"  exact class : {exact:4d} / {n} = {exact / n:6.1%}")
    print(f"  territory   : {terr:4d} / {n} = {terr / n:6.1%}")
    print(f"  laterality  : {side:4d} / {n} = {side / n:6.1%}")


def confusions(gt: np.ndarray, pred: np.ndarray, k: int = 12) -> None:
    from collections import Counter
    c = Counter((name_of(g), name_of(p)) for g, p in zip(gt, pred) if g != p)
    print(f"  top {k} confusions (true -> predicted):")
    for (g, p), cnt in c.most_common(k):
        print(f"    {cnt:3d}x  {g}  ->  {p or '(none)'}")


def new_gbm() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=3,
        l2_regularization=1.0, class_weight="balanced", random_state=0)


# main

def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    loc_dir = os.path.join(here, "location_masks")
    ves_dir = os.path.join(here, "vessel_masks")

    loc_files = sorted(glob.glob(os.path.join(loc_dir, "*.nii.gz")))
    records = []
    print(f"extracting features from {len(loc_files)} cases...", flush=True)
    for i, loc_path in enumerate(loc_files, 1):
        ves_path = os.path.join(ves_dir, os.path.basename(loc_path))
        if os.path.exists(ves_path):
            records.extend(extract_case(loc_path, ves_path))
        if i % 20 == 0 or i == len(loc_files):
            print(f"  {i}/{len(loc_files)} cases  ({len(records)} aneurysms)", flush=True)

    X = np.vstack([r["feats"] for r in records])
    y = np.array([r["gt_val"] for r in records])
    y_base = np.array([BASE_TO_IDX[BASE_OF_VAL[v]] for v in y])
    geom_lat = np.array([r["geom_lat"] for r in records])
    groups = np.array([r["case"] for r in records])
    base = np.array([r["base_val"] for r in records])
    print(f"\naneurysms: {len(y)}   cases: {len(set(groups))}   features: {X.shape[1]}   "
          f"base classes: {len(BASE_LIST)}\n", flush=True)

    oof_flat = np.full(len(y), -1)    
    oof_two = np.full(len(y), -1)      
    gkf = GroupKFold(n_splits=N_SPLITS)
    print(f"training {N_SPLITS} folds (GroupKFold by case)...", flush=True)
    for k, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        flat = new_gbm().fit(X[tr], y[tr])
        oof_flat[te] = flat.predict(X[te])

        head = new_gbm().fit(X[tr], y_base[tr])
        pred_base = head.predict(X[te])
        oof_two[te] = [combine(BASE_LIST[b], geom_lat[i]) for b, i in zip(pred_base, te)]

        acc_base = float((pred_base == y_base[te]).mean())
        acc_two = float((oof_two[te] == y[te]).mean())
        print(f"  fold {k}/{N_SPLITS}: base_exact={acc_base:.1%}  two-head_exact={acc_two:.1%}", flush=True)

    print("\n" + "=" * 56)
    three_layers(y, base, "BASELINE (rules v2, junctions on)")
    print("-" * 56)
    three_layers(y, oof_flat, "PHASE 1 FLAT GBM (50 classes)")
    print("-" * 56)
    three_layers(y, oof_two, "PHASE 1 TWO-HEAD (geom laterality + GBM base-27)")
    print("=" * 56 + "\n")
    confusions(y, oof_two)


if __name__ == "__main__":
    main()
