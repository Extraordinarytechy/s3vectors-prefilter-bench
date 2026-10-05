import numpy as np

from src import ground_truth as gt
from tests.conftest import small_config
from src import dataset as ds_mod


def _naive(vectors, q, member_idx, k):
    rows = []
    for i in member_idx:
        x = [float(v) for v in vectors[i]]
        qq = [float(v) for v in q]
        dot = sum(a * b for a, b in zip(x, qq))
        nx = sum(a * a for a in x) ** 0.5
        nq = sum(b * b for b in qq) ** 0.5
        rows.append((1.0 - dot / (nx * nq), i))
    rows.sort()
    return [i for _, i in rows[:k]]


def test_matches_naive_reference():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(40, 6)).astype(np.float32)
    q = rng.normal(size=6).astype(np.float32)
    m = np.zeros(40, dtype=bool)
    m[::3] = True
    eng = gt.DistanceEngine(v)
    res = gt.exact_topk(eng.distances(q), m, 5, [5])
    assert list(res["top"]) == _naive(v, q, np.flatnonzero(m), 5)
    assert res["matching"] == 14


def test_tiebreak_by_key_and_tie_flag():
    d = np.array([0.5, 0.1, 0.1, 0.3, 0.1])
    m = np.ones(5, dtype=bool)
    res = gt.exact_topk(d, m, 5, [2, 3, 4])
    assert list(res["top"]) == [1, 2, 4, 3, 0]
    assert res["tie_flags"] == {"2": True, "3": False, "4": False}
    small = gt.exact_topk(d, np.array([True, True, False, False, False]), 5, [2, 3])
    assert small["tie_flags"] == {"2": False, "3": False}


def test_gt_lengths_for_0_1_3_matches_and_prefixes():
    cfg = small_config()
    data = ds_mod.build(cfg["dataset"])
    rows = gt.index_rows(gt.compute(data, cfg))
    q = data.query_ids[0]
    assert len(gt.gt_for_k(rows[(q, "FZERO")], 10)) == 0
    assert len(gt.gt_for_k(rows[(q, "FONE")], 10)) == 1
    assert len(gt.gt_for_k(rows[(q, "FFEW")], 10)) == 3
    f50 = rows[(q, "F50")]
    assert gt.gt_for_k(f50, 5) == f50["gt_keys"][:5]
    assert len(f50["gt_keys"]) == 10
    assert rows[(q, "NOFILTER")]["matching"] == 1000
    # distances use the norms of the stored float32 vectors
    eng = gt.DistanceEngine(data.vectors)
    d = eng.distances(data.queries[0])
    i = data.key_index[f50["gt_keys"][0]]
    x = data.vectors[i].astype(np.float64)
    qq = data.queries[0].astype(np.float64)
    assert abs(d[i] - (1 - x @ qq / (np.linalg.norm(x) * np.linalg.norm(qq)))) < 1e-15
