"""Exact filtered top-K by brute force, cosine distance.

d = 1 - (x·q) / (‖x‖·‖q‖), computed in float64 from the stored float32 values, so
float32 normalization error (~1e-7) cannot reorder neighbors near the 1e-6 tie threshold.
Ties are broken by key; keys are `v-%05d` in row order, so key order equals row order.
"""
from __future__ import annotations

import numpy as np

from . import filters as flt
from .dataset import Dataset

TIE_EPS = 1e-6


class DistanceEngine:
    """Caches float64 vectors and norms so each query costs one matrix-vector product."""

    def __init__(self, vectors: np.ndarray):
        self.v64 = vectors.astype(np.float64)
        self.norms = np.linalg.norm(self.v64, axis=1)

    def distances(self, q: np.ndarray) -> np.ndarray:
        q64 = np.asarray(q, dtype=np.float64)
        return 1.0 - (self.v64 @ q64) / (self.norms * np.linalg.norm(q64))


def sorted_members(d: np.ndarray, members: np.ndarray) -> np.ndarray:
    """Members ordered by (distance, row index) ascending."""
    members = np.asarray(members, dtype=np.int64)
    return members[np.lexsort((members, d[members]))]


def exact_topk(d: np.ndarray, m: np.ndarray, kmax: int, k_values: list[int]) -> dict:
    members = np.flatnonzero(m)
    order = sorted_members(d, members)
    top = order[:kmax]
    boundary = d[order[: kmax + 1]]
    ties = {}
    for k in k_values:
        ties[str(k)] = bool(len(members) > k and abs(boundary[k - 1] - boundary[k]) < TIE_EPS)
    return {"matching": int(len(members)), "top": top, "top_d": d[top], "tie_flags": ties}


def compute(dataset: Dataset, cfg: dict) -> list[dict]:
    """GT rows for every query × (scored filters + NOFILTER), top max(K) stored once."""
    k_values = cfg["k_values"]
    kmax = max(k_values)
    engine = DistanceEngine(dataset.vectors)
    masks = {f["id"]: flt.mask(f["filter"], dataset.columns, dataset.n) for f in flt.scored_filters(cfg)}
    rows = []
    for qi, qid in enumerate(dataset.query_ids):
        d = engine.distances(dataset.queries[qi])
        for f in flt.scored_filters(cfg):
            gt = exact_topk(d, masks[f["id"]], kmax, k_values)
            rows.append({"query_id": qid, "filter_id": f["id"], "matching": gt["matching"],
                         "gt_keys": [dataset.keys[i] for i in gt["top"]],
                         "gt_distances": [float(x) for x in gt["top_d"]],
                         "tie_flags": gt["tie_flags"]})
    return rows


def index_rows(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(r["query_id"], r["filter_id"]): r for r in rows}


def gt_for_k(row: dict, k: int) -> list[str]:
    """GT_K is the first min(K, matching) keys; smaller K values are prefixes of the top 50."""
    return row["gt_keys"][: min(k, row["matching"])]
