
import numpy as np
import phase1_features_model_v2 as p
TOL = 1e-9

def _z_line_vidx(x: int = 5, y: int = 5, z0: int = 2, z1: int = 20) -> np.ndarray:
    """A one-voxel-wide straight vessel running along the z axis."""
    return np.array([[x, y, k] for k in range(z0, z1 + 1)], dtype=np.int64)


def _flip_z_affine(z_shift: float) -> np.ndarray:
    """Affine that negates voxel z (world_z = -k + z_shift), so a volume can be
    presented with reversed raster order while keeping identical world geometry."""
    aff = np.eye(4)
    aff[2, 2] = -1.0
    aff[2, 3] = z_shift
    return aff


def test_voxel_order_reversal_invariant() -> None:
    """Reversing the order of the voxel list must not change `along` at all
    (the volume is rebuilt from a scatter, so order is irrelevant)."""
    vidx = _z_line_vidx()
    aff = np.eye(4)
    centroid = np.array([5.0, 5.0, 4.0])
    bbc = np.array([5.0, 5.0, 11.0])

    f1 = p.geodesic_frame(vidx, aff)
    f2 = p.geodesic_frame(vidx[::-1].copy(), aff)
    assert f1 is not None and f2 is not None
    for rule in p.ORIGIN_RULES:
        a1 = p.along_on_frame(f1, centroid, rule, bbc)
        a2 = p.along_on_frame(f2, centroid, rule, bbc)
        assert abs(a1 - a2) < TOL, f"{rule}: {a1} != {a2}"


def test_anchored_along_invariant_to_reversed_geometry() -> None:
 
    vidx = _z_line_vidx(z0=2, z1=20)

    affA = np.eye(4)

    affB = _flip_z_affine(22.0)

    centroid_world = np.array([5.0, 5.0, 4.0])      # near the inferior (z=2) end
    bbc = np.array([5.0, 5.0, 11.0])

    fA = p.geodesic_frame(vidx, affA)
    fB = p.geodesic_frame(vidx, affB)
    assert fA is not None and fB is not None

    zA = fA["nodes_world"][:, 2]
    zB = fB["nodes_world"][:, 2]
    assert abs(zA.min() - zB.min()) < 1e-6 and abs(zA.max() - zB.max()) < 1e-6

    along_A = p.along_on_frame(fA, centroid_world, "inferior_z", bbc)
    along_B = p.along_on_frame(fB, centroid_world, "inferior_z", bbc)
    assert abs(along_A - along_B) < 1e-6, (
        f"anchored along not invariant: {along_A} vs {along_B}")

    # the aneurysm is near the inferior end, so anchored along must be small.
    assert along_A < 0.25, f"expected small along near inferior end, got {along_A}"

    # teeth: the legacy rule flips between the two presentations.
    raw_A = p.along_on_frame(fA, centroid_world, "dijkstra_raw", bbc)
    raw_B = p.along_on_frame(fB, centroid_world, "dijkstra_raw", bbc)
    assert abs(raw_A - raw_B) > 0.5, (
        f"expected legacy along to flip, got {raw_A} vs {raw_B}")


def test_bifurcation_count_on_hand_built_graph() -> None:
    """Count degree>=3 nodes strictly between the aneurysm and the origin.

    Topology (node:neighbours)

    """
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6)]
    n = 7
    rows, cols, data = [], [], []
    for i, j in edges:
        rows += [i, j]
        cols += [j, i]
        data += [1.0, 1.0]
    from scipy.sparse import csr_matrix
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))

    # place nodes on a line for 0..4, branch 5,6 offset in y.
    nw = np.array([
        [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0],
        [2, 1, 0], [2, 2, 0],
    ], dtype=float)
    frame = {"graph": graph, "nodes_world": nw, "keep": np.arange(n),
             "a": 0, "b": 4, "total": 4.0}
    bbc = nw.mean(axis=0)

    nb = p.n_bifurcations_on_frame(frame, nw[6], "dijkstra_raw", bbc)
    assert nb == 1.0, f"expected 1 bifurcation between node 6 and origin, got {nb}"

    nb2 = p.n_bifurcations_on_frame(frame, nw[1], "dijkstra_raw", bbc)
    assert nb2 == 0.0, f"expected 0 bifurcations between node 1 and origin, got {nb2}"


def _run_all() -> int:
    tests = [
        test_voxel_order_reversal_invariant,
        test_anchored_along_invariant_to_reversed_geometry,
        test_bifurcation_count_on_hand_built_graph,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    import sys
    sys.exit(_run_all())
