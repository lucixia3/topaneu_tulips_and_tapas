import argparse
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

import assign_vessel_location_v2 as m

INV_LOCATION = {v: k for k, v in m.LOCATION_LABELS.items()}
VESSEL_VALUES = sorted(m.VESSEL_LABELS)          # 1..36, fixed column order
MIN_VOXELS = 5
JUNCTION_TOL_MM = 3.0
N_SPLITS = 5
_OFFSETS = [(dx, dy, dz)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
            if not (dx == dy == dz == 0)]


ORIGIN_RULES = ("dijkstra_raw", "inferior_z", "proximal_bbox")
DEFAULT_ORIGIN_RULE = "dijkstra_raw"   # keep the shipped baseline unchanged
DEFAULT_VESSEL_MODE = "fine"

EXTRACT_VERSION = "v3"
ALONG_IDX = 47                          # position of `along` inside the 49-vector


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


# ---------------------------------------------------------------------------
# vessel centerline graph + geodesic frame
# ---------------------------------------------------------------------------
def build_skeleton_graph(vidx: np.ndarray, affine: np.ndarray):
    """Skeletonize a voxel set and build the weighted 26-connectivity graph of
    its largest connected component.
    """
    nodes = skeleton_indices(vidx)
    n = len(nodes)
    if n < 2:
        return None
    nodes_world = apply_affine(affine, nodes.astype(np.float64))

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
    return graph, nodes_world, keep


def _dijkstra_from(frame: dict, node: int) -> tuple[np.ndarray, np.ndarray]:
    """Geodesic distances and predecessors from `node`, cached inside `frame`."""
    node = int(node)
    cache = frame.setdefault("_dk", {})
    if node not in cache:
        dist, pred = dijkstra(frame["graph"], indices=node,
                              directed=False, return_predecessors=True)
        cache[node] = (dist, pred)
    return cache[node]


def geodesic_frame(vidx: np.ndarray, affine: np.ndarray) -> dict | None:

    built = build_skeleton_graph(vidx, affine)
    if built is None:
        return None
    graph, nodes_world, keep = built
    frame = {"graph": graph, "nodes_world": nodes_world, "keep": keep}

    d0, _ = _dijkstra_from(frame, int(keep[0]))
    a = int(keep[np.argmax(d0[keep])])
    dA, _ = _dijkstra_from(frame, a)
    b = int(keep[np.argmax(dA[keep])])
    total = float(dA[b])
    if not np.isfinite(total) or total < 1e-6:
        return None
    frame.update(a=a, b=b, total=total)
    return frame


def select_origin(frame: dict, rule: str, bbox_center: np.ndarray) -> int:
    """Pick the origin endpoint among the two centerline ends per `rule`."""
    a, b, nw = frame["a"], frame["b"], frame["nodes_world"]
    if rule == "dijkstra_raw":
        return a
    if rule == "inferior_z":
        return a if nw[a, 2] <= nw[b, 2] else b
    if rule == "proximal_bbox":
        da = float(np.linalg.norm(nw[a] - bbox_center))
        db = float(np.linalg.norm(nw[b] - bbox_center))
        return a if da <= db else b
    raise ValueError(f"unknown origin rule: {rule!r}")


def _nearest_node(frame: dict, centroid: np.ndarray) -> int:
    keep, nw = frame["keep"], frame["nodes_world"]
    return int(keep[np.argmin(np.linalg.norm(nw[keep] - centroid, axis=1))])


def along_on_frame(frame: dict, centroid: np.ndarray, rule: str,
                   bbox_center: np.ndarray) -> float:
    """Position of the aneurysm along the centerline in [0,1], measured from the
    anchored origin endpoint (task 1)."""
    q = _nearest_node(frame, centroid)
    origin = select_origin(frame, rule, bbox_center)
    d_origin, _ = _dijkstra_from(frame, origin)
    return float(np.clip(d_origin[q] / frame["total"], 0.0, 1.0))


def n_bifurcations_on_frame(frame: dict, centroid: np.ndarray, rule: str,
                            bbox_center: np.ndarray) -> float:
    """Number of skeleton bifurcations (degree >= 3 nodes) strictly between the
    aneurysm node and the anchored origin, along their shortest path (task 2)."""
    q = _nearest_node(frame, centroid)
    origin = select_origin(frame, rule, bbox_center)
    degrees = np.diff(frame["graph"].indptr)              # neighbours per node
    _, pred = _dijkstra_from(frame, origin)
    count = 0
    node = q
    guard = 0
    limit = len(pred)
    while node != origin and node >= 0 and guard < limit:
        if node != q and degrees[node] >= 3:
            count += 1
        node = int(pred[node])
        guard += 1
    return float(count)


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


# helpers
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
    bbox_center = (bb_lo + bb_hi) / 2.0
    voxel_vol = abs(np.linalg.det(gt_aff[:3, :3]))                 # mm^3 per voxel

    fine_frames: dict[int, dict | None] = {}                      # per-vessel cache
    binary_frame_built = False
    binary_frame: dict | None = None

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

        # ---- along-vessel on the nearest FINE vessel (all origin rules) ----
        along_fine = {}
        if np.all(np.isnan(dists)):
            for rule in ORIGIN_RULES:
                along_fine[rule] = 0.5
        else:
            nv = VESSEL_VALUES[int(np.nanargmin(dists))]
            if nv not in fine_frames:
                fine_frames[nv] = geodesic_frame(vidx[nv], gt_aff)
            frame = fine_frames[nv]
            if frame is None:
                fb = along_pca(vcoords[nv], centroid)
                for rule in ORIGIN_RULES:
                    along_fine[rule] = fb
            else:
                for rule in ORIGIN_RULES:
                    along_fine[rule] = along_on_frame(frame, centroid, rule, bbox_center)

        # ---- along-vessel + bifurcations on the BINARIZED network (task 2) ----
        if not binary_frame_built:
            all_vidx = np.argwhere(ves_arr > 0)
            binary_frame = geodesic_frame(all_vidx, gt_aff)
            binary_frame_built = True
        along_bin, nbif = {}, {}
        if binary_frame is None:
            for rule in ORIGIN_RULES:
                along_bin[rule] = 0.5
                nbif[rule] = 0.0
        else:
            for rule in ORIGIN_RULES:
                along_bin[rule] = along_on_frame(binary_frame, centroid, rule, bbox_center)
                nbif[rule] = n_bifurcations_on_frame(binary_frame, centroid, rule, bbox_center)

        # aneurysm size
        n_vox = float(idx.shape[0])
        volume = n_vox * voxel_vol
        diameter = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))

        norm_xyz = (centroid - bb_lo) / span
        feats_core = np.concatenate([
            dists, [d1, d2, d3, r21, r31], [n_vox, volume, diameter],
            norm_xyz, [along_fine["dijkstra_raw"], float(mod)],
        ])

        vals = gt_arr[tuple(idx.T)]
        vals = vals[vals > 0]
        gt_val = int(np.bincount(vals).argmax())

        ranked, tw, mind = m.rank_component_vessels(coords, tree, vclasses)
        _, base_val, _, _ = m.resolve_location(ranked, JUNCTION_TOL_MM, True)

        rows.append({
            "case": case_name, "feats_core": feats_core, "gt_val": gt_val,
            "geom_lat": laterality_geom(ranked),
            "base_val": base_val if base_val is not None else -1,
            "along_fine": along_fine, "along_bin": along_bin, "nbif": nbif,
        })
    return rows


# ---------------------------------------------------------------------------
# feature block layout (for assembly + permutation importance)
# ---------------------------------------------------------------------------
VESSEL_NAMES = [m.VESSEL_LABELS[v] for v in VESSEL_VALUES]
FEATURE_BLOCKS_FINE = [
    ("36_distances", [f"dist_{n}" for n in VESSEL_NAMES]),
    ("topk_ratios", ["d1", "d2", "d3", "r21", "r31"]),
    ("size", ["n_vox", "volume", "diameter"]),
    ("coords", ["norm_x", "norm_y", "norm_z"]),
    ("along_vessel", ["along"]),
    ("modality", ["mod"]),
]
FEATURE_BLOCKS_BINARY = FEATURE_BLOCKS_FINE + [("bifurcations", ["n_bifurcations"])]


def feature_layout(vessel_mode: str) -> tuple[list[str], list[tuple[str, list[int]]]]:
    """Flat feature names and (block_name, [col_idx...]) groups for a mode."""
    blocks = FEATURE_BLOCKS_BINARY if vessel_mode == "binary" else FEATURE_BLOCKS_FINE
    names, groups, k = [], [], 0
    for block, cols in blocks:
        idxs = list(range(k, k + len(cols)))
        groups.append((block, idxs))
        names.extend(cols)
        k += len(cols)
    return names, groups


def assemble_matrix(records: list[dict], vessel_mode: str, origin_rule: str) -> np.ndarray:
    """Build the design matrix for a given vessel-mode / origin-rule from the
    cached records (cheap column selection, no re-extraction)."""
    out = []
    for r in records:
        core = r["feats_core"].copy()
        if vessel_mode == "binary":
            core[ALONG_IDX] = r["along_bin"][origin_rule]
            out.append(np.append(core, r["nbif"][origin_rule]))
        else:
            core[ALONG_IDX] = r["along_fine"][origin_rule]
            out.append(core)
    return np.vstack(out)


# ---------------------------------------------------------------------------
# extraction cache
# ---------------------------------------------------------------------------
def extract_all(loc_dir: str, ves_dir: str) -> list[dict]:
    loc_files = sorted(glob.glob(os.path.join(loc_dir, "*.nii.gz")))
    records = []
    print(f"extracting features from {len(loc_files)} cases...", flush=True)
    for i, loc_path in enumerate(loc_files, 1):
        ves_path = os.path.join(ves_dir, os.path.basename(loc_path))
        if os.path.exists(ves_path):
            records.extend(extract_case(loc_path, ves_path))
        if i % 20 == 0 or i == len(loc_files):
            print(f"  {i}/{len(loc_files)} cases  ({len(records)} aneurysms)", flush=True)
    return records


def save_records(records: list[dict], path: str) -> None:
    np.savez_compressed(
        path,
        version=EXTRACT_VERSION,
        rules=np.array(ORIGIN_RULES),
        feats_core=np.vstack([r["feats_core"] for r in records]),
        along_fine=np.array([[r["along_fine"][k] for k in ORIGIN_RULES] for r in records]),
        along_bin=np.array([[r["along_bin"][k] for k in ORIGIN_RULES] for r in records]),
        nbif=np.array([[r["nbif"][k] for k in ORIGIN_RULES] for r in records]),
        gt_val=np.array([r["gt_val"] for r in records]),
        base_val=np.array([r["base_val"] for r in records]),
        geom_lat=np.array([r["geom_lat"] for r in records]),
        case=np.array([r["case"] for r in records]),
    )


def load_records(path: str) -> list[dict] | None:
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=False)
    if str(d["version"]) != EXTRACT_VERSION or list(d["rules"]) != list(ORIGIN_RULES):
        return None
    records = []
    for i in range(len(d["gt_val"])):
        records.append({
            "case": str(d["case"][i]),
            "feats_core": d["feats_core"][i],
            "gt_val": int(d["gt_val"][i]),
            "base_val": int(d["base_val"][i]),
            "geom_lat": str(d["geom_lat"][i]),
            "along_fine": {k: float(d["along_fine"][i, j]) for j, k in enumerate(ORIGIN_RULES)},
            "along_bin": {k: float(d["along_bin"][i, j]) for j, k in enumerate(ORIGIN_RULES)},
            "nbif": {k: float(d["nbif"][i, j]) for j, k in enumerate(ORIGIN_RULES)},
        })
    return records


def get_records(loc_dir: str, ves_dir: str, cache: str | None) -> list[dict]:
    if cache:
        cached = load_records(cache)
        if cached is not None:
            print(f"loaded {len(cached)} aneurysm records from cache {cache}", flush=True)
            return cached
    records = extract_all(loc_dir, ves_dir)
    if cache:
        save_records(records, cache)
        print(f"cached {len(records)} records to {cache}", flush=True)
    return records


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def name_of(val: int) -> str:
    return INV_LOCATION.get(int(val), "")


def group_of(val: int) -> str:
    name = name_of(val)
    return base_name(name).split(".", 1)[0].strip() if name else "?"


def layer_scores(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    """(exact, territory, laterality) accuracies as fractions."""
    n = len(gt)
    exact = int((gt == pred).sum())
    side = sum(1 for g, p in zip(gt, pred)
               if name_of(p) and m.laterality_from_label(name_of(p)) == m.laterality_from_label(name_of(g)))
    terr = sum(1 for g, p in zip(gt, pred) if name_of(p) and group_of(p) == group_of(g))
    return exact / n, terr / n, side / n


def three_layers(gt: np.ndarray, pred: np.ndarray, tag: str) -> None:
    exact, terr, side = layer_scores(gt, pred)
    n = len(gt)
    print(f"{tag}")
    print(f"  exact class : {exact * n:4.0f} / {n} = {exact:6.1%}")
    print(f"  territory   : {terr * n:4.0f} / {n} = {terr:6.1%}")
    print(f"  laterality  : {side * n:4.0f} / {n} = {side:6.1%}")


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


def make_splitter(seed: int | None) -> GroupKFold:
    """Deterministic GroupKFold when seed is None (baseline); shuffled otherwise."""
    if seed is None:
        return GroupKFold(n_splits=N_SPLITS)
    return GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)


# ---------------------------------------------------------------------------
# training / evaluation
# ---------------------------------------------------------------------------
def run_fold_models(X, y, y_base, geom_lat, groups, seed):
    """Fit the flat and two-head models over one GroupKFold split and return
    out-of-fold predictions for both."""
    oof_flat = np.full(len(y), -1)
    oof_two = np.full(len(y), -1)
    for tr, te in make_splitter(seed).split(X, y, groups):
        flat = new_gbm().fit(X[tr], y[tr])
        oof_flat[te] = flat.predict(X[te])

        head = new_gbm().fit(X[tr], y_base[tr])
        pred_base = head.predict(X[te])
        oof_two[te] = [combine(BASE_LIST[b], geom_lat[i]) for b, i in zip(pred_base, te)]
    return oof_flat, oof_two


def evaluate_seeds(records, vessel_mode, origin_rule, seeds, verbose=True):
    """Train the two-head + flat GBMs over one or more seeded GroupKFolds and
    report exact/territory/laterality as mean +/- std across seeds."""
    X = assemble_matrix(records, vessel_mode, origin_rule)
    y = np.array([r["gt_val"] for r in records])
    y_base = np.array([BASE_TO_IDX[BASE_OF_VAL[v]] for v in y])
    geom_lat = np.array([r["geom_lat"] for r in records])
    groups = np.array([r["case"] for r in records])
    base = np.array([r["base_val"] for r in records])

    print(f"\naneurysms: {len(y)}   cases: {len(set(groups))}   features: {X.shape[1]}   "
          f"mode={vessel_mode}   origin_rule={origin_rule}   base classes: {len(BASE_LIST)}", flush=True)

    seed_list = [None] if not seeds else list(range(seeds))
    flat_scores, two_scores = [], []
    for s in seed_list:
        oof_flat, oof_two = run_fold_models(X, y, y_base, geom_lat, groups, s)
        flat_scores.append(layer_scores(y, oof_flat))
        two_scores.append(layer_scores(y, oof_two))
        if verbose and (len(seed_list) == 1 or s == 0):
            # keep one detailed dump; the rest are folded into mean/std below.
            print("\n" + "=" * 60)
            three_layers(y, base, "BASELINE (rules v2, junctions on)")
            print("-" * 60)
            three_layers(y, oof_flat, f"PHASE 1 FLAT GBM (50 classes)  [seed={s}]")
            print("-" * 60)
            three_layers(y, oof_two, f"PHASE 1 TWO-HEAD (geom lat + GBM base-27)  [seed={s}]")
            print("=" * 60)

    base_scores = layer_scores(y, base)
    _report_meanstd("BASELINE (rules v2)", [base_scores], seed_list)
    _report_meanstd("FLAT GBM", flat_scores, seed_list)
    _report_meanstd("TWO-HEAD", two_scores, seed_list)

    if verbose and len(seed_list) == 1:
        confusions(y, run_fold_models(X, y, y_base, geom_lat, groups, seed_list[0])[1])
    return {"flat": flat_scores, "two": two_scores, "base": base_scores}


def _report_meanstd(tag, scores, seed_list):
    arr = np.array(scores, dtype=float)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    n = len(scores)
    seeds_txt = "deterministic" if seed_list == [None] else f"{len(seed_list)} seeds"
    print(f"\n{tag}  ({seeds_txt})")
    for label, i in (("exact", 0), ("territory", 1), ("laterality", 2)):
        if n > 1:
            print(f"  {label:11s}: {mean[i]:6.1%}  +/- {std[i]:.1%}")
        else:
            print(f"  {label:11s}: {mean[i]:6.1%}")


# ---------------------------------------------------------------------------
# permutation importance (task 3)
# ---------------------------------------------------------------------------
def permutation_importance_grouped(records, vessel_mode, origin_rule,
                                   seeds, n_repeats):
    """Permutation importance of every feature on the two-head model's *base
    head* (the learned GBM), respecting GroupKFold by case.

    For each fold we fit on train, score base-class accuracy on the untouched
    test fold, then permute each feature *within the test fold only* and measure
    the accuracy drop, averaged over `n_repeats` shuffles, all folds and all
    seeds. Larger drop = more important.
    """
    X = assemble_matrix(records, vessel_mode, origin_rule)
    y = np.array([r["gt_val"] for r in records])
    y_base = np.array([BASE_TO_IDX[BASE_OF_VAL[v]] for v in y])
    groups = np.array([r["case"] for r in records])
    names, blocks = feature_layout(vessel_mode)
    n_feat = X.shape[1]

    seed_list = list(range(seeds)) if seeds else [0]
    drops = [[] for _ in range(n_feat)]        # per-feature list of accuracy drops
    for s in seed_list:
        rng = np.random.default_rng(1000 + s)
        for fi, (tr, te) in enumerate(make_splitter(s).split(X, y_base, groups)):
            model = new_gbm().fit(X[tr], y_base[tr])
            yte = y_base[te]
            base_acc = float((model.predict(X[te]) == yte).mean())
            Xte = X[te]
            t = len(te)
            # Batch every (feature x repeat) permutation into a single predict:
            # for a test fold of size t we stack n_feat*n_repeats copies of Xte,
            # each with one column reshuffled, and predict once. This turns
            # n_feat*n_repeats predict calls into one per fold (~100x fewer),
            # which is what makes the run tractable.
            big = np.repeat(Xte[None, :, :], n_feat * n_repeats, axis=0)  # (F*R, t, D)
            k = 0
            for f in range(n_feat):
                col = Xte[:, f]
                for _ in range(n_repeats):
                    big[k, :, f] = rng.permutation(col)
                    k += 1
            preds = model.predict(big.reshape(-1, n_feat)).reshape(n_feat, n_repeats, t)
            acc = (preds == yte[None, None, :]).mean(axis=2)             # (n_feat, n_repeats)
            for f in range(n_feat):
                drops[f].extend((base_acc - acc[f]).tolist())
        print(f"  permutation seed {s} done", flush=True)

    imp_mean = np.array([np.mean(d) for d in drops])
    imp_std = np.array([np.std(d) for d in drops])

    order = np.argsort(imp_mean)[::-1]
    print("\n" + "=" * 64)
    print(f"PERMUTATION IMPORTANCE (base head)  mode={vessel_mode}  rule={origin_rule}")
    print(f"  {len(seed_list)} seed(s) x {N_SPLITS} folds x {n_repeats} repeats; drop in base-class accuracy")
    print("=" * 64)
    print(f"{'rank':>4}  {'feature':22s} {'importance':>12s}  {'std':>8s}")
    for rank, f in enumerate(order, 1):
        print(f"{rank:>4}  {names[f]:22s} {imp_mean[f]:12.4f}  {imp_std[f]:8.4f}")

    print("\n" + "-" * 64)
    print("BY BLOCK (summed importance, share of total):")
    total = float(imp_mean.clip(min=0).sum()) or 1.0
    block_rows = []
    for block, idxs in blocks:
        s_imp = float(imp_mean[idxs].clip(min=0).sum())
        block_rows.append((s_imp, block, len(idxs)))
    for s_imp, block, nfeat in sorted(block_rows, reverse=True):
        print(f"  {block:16s} ({nfeat:2d} feats)  sum={s_imp:8.4f}   {s_imp / total:6.1%}")
    print("=" * 64)
    return imp_mean, imp_std, names, blocks


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        description="phase 1 geometric features + two-head GBM for topaneu 2026 "
                    "aneurysm localization. Default flags reproduce the shipped baseline.")
    p.add_argument("--loc-dir", default=os.path.join(here, "location_masks"),
                   help="directory of multiclass location (GT) masks.")
    p.add_argument("--ves-dir", default=os.path.join(here, "vessel_masks"),
                   help="directory of multiclass vessel masks.")
    p.add_argument("--vessel-mode", choices=("fine", "binary"), default=DEFAULT_VESSEL_MODE,
                   help="'fine': along on the nearest of 36 vessel classes (baseline). "
                        "'binary': along + bifurcation count on the binarized network.")
    p.add_argument("--origin-rule", choices=ORIGIN_RULES, default=DEFAULT_ORIGIN_RULE,
                   help="how the along-vessel origin endpoint is anchored (task 1). "
                        "'dijkstra_raw' = legacy/baseline; 'inferior_z' recommended.")
    p.add_argument("--seeds", type=int, default=0,
                   help="0 = single deterministic GroupKFold (baseline output); "
                        "N>0 = N shuffled seeds, reported as mean +/- std.")
    p.add_argument("--permutation-importance", action="store_true",
                   help="run permutation importance instead of the normal report.")
    p.add_argument("--n-repeats", type=int, default=10,
                   help="permutation shuffles per feature per fold (default 10).")
    p.add_argument("--cache", default=None,
                   help="npz path to cache/reuse extracted features across runs.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    records = get_records(args.loc_dir, args.ves_dir, args.cache)

    if args.permutation_importance:
        permutation_importance_grouped(records, args.vessel_mode, args.origin_rule,
                                       args.seeds, args.n_repeats)
        return

    evaluate_seeds(records, args.vessel_mode, args.origin_rule, args.seeds)


if __name__ == "__main__":
    main()
