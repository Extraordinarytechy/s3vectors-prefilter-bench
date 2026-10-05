"""SIMULATED CLASSIC vs ENHANCED mechanics on a numpy IVF-flat index.

This is a mechanism illustration, not a model of AWS internals: AWS does not document its ANN
index structure, candidate budgets, or how CLASSIC interleaves filtering. Nothing here is
calibrated to AWS's "up to 5x more of the matching vectors" claim, and no timing is recorded.
"""
from __future__ import annotations

import numpy as np

from . import filters as flt
from .dataset import Dataset, stream
from .ground_truth import DistanceEngine, sorted_members
from .state import ConfigError

LIMITS = [
    "AWS's ANN structure and CLASSIC's filter interleaving are undocumented. Real CLASSIC may use "
    "adaptive budgets, graph indexes, or retries that remove or change the underfill.",
    "Simulator parameters (nlist, C) are illustrative, not fitted to AWS.",
    "Metadata is independent of vector position; correlated filters could behave differently.",
    "Simulator results show a mechanism, not a prediction of AWS numbers.",
    "No timing.",
]
POSTFILTER_PAIRS_NOTE = ("SIMULATED post-filter runs only for (B, C) pairs with B <= C, because at most C "
                         "candidates are seen. B = 10,000 is not simulated.")


class IVF:
    """Spherical k-means IVF-flat. Lists are visited in ascending centroid distance."""

    def __init__(self, vectors: np.ndarray, nlist: int, iters: int, seed: int):
        x = vectors.astype(np.float32)
        x64 = x.astype(np.float64)
        n = len(x)
        nlist = min(nlist, n)
        rng = np.random.default_rng(stream(seed, "simulator"))
        cent = x[np.sort(rng.choice(n, nlist, replace=False))].astype(np.float64)
        for _ in range(iters):
            assign, best = self._assign(x, cent)
            sums = np.zeros_like(cent)
            by_list = np.argsort(assign, kind="stable")
            present, starts = np.unique(assign[by_list], return_index=True)
            sums[present] = np.add.reduceat(x64[by_list], starts, axis=0)
            counts = np.bincount(assign, minlength=nlist)
            empty = np.flatnonzero(counts == 0)
            if len(empty):
                # Deterministic re-seed: the worst-assigned points, in ascending similarity order.
                worst = np.lexsort((np.arange(n), best))[: len(empty)]
                sums[empty] = x64[worst]
                counts[empty] = 1
            cent = sums / np.linalg.norm(sums, axis=1, keepdims=True)
        assign, _ = self._assign(x, cent)
        self.centroids = cent
        self.lists = [np.flatnonzero(assign == j) for j in range(nlist)]

    @staticmethod
    def _assign(x: np.ndarray, cent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sims = x @ cent.astype(np.float32).T
        assign = np.argmax(sims, axis=1)
        return assign, sims[np.arange(len(x)), assign]

    def visit_order(self, q: np.ndarray) -> np.ndarray:
        cs = self.centroids @ np.asarray(q, dtype=np.float64)
        order = np.lexsort((np.arange(len(cs)), -cs))
        return np.concatenate([self.lists[j] for j in order])


def classic(order, m, d, budget_c):
    """Filter during search: every visited vector consumes budget; only matches are scored."""
    seen = order[:budget_c]
    matched = seen[m[seen]]
    return sorted_members(d, matched), len(seen), len(matched)


def enhanced(order, m, d, budget_c):
    """Filter first: only members of M are scored and consume budget."""
    seen = order[m[order]][:budget_c]
    return sorted_members(d, seen), len(seen), len(seen)


def postfilter(order, m, d, budget_c, budget_b):
    """Unfiltered search over C candidates, keep top-B, then filter locally."""
    seen = order[:budget_c]
    top_b = sorted_members(d, seen)[:budget_b]
    return top_b[m[top_b]], len(seen), len(seen)


def validate(sim: dict) -> None:
    for b, c in sim["postfilter_pairs"]:
        if b > c:
            raise ConfigError(f"post-filter pair B={b} > C={c} is impossible")


def run(dataset: Dataset, cfg: dict) -> list[dict]:
    """Deterministic simulator requests (no scoring); one row per (model, budget, filter, q, K)."""
    sim = cfg["simulator"]
    validate(sim)
    k_values = cfg["k_values"]
    ivf = IVF(dataset.vectors, sim["nlist"], sim["kmeans_iters"], cfg["dataset"]["seed"])
    engine = DistanceEngine(dataset.vectors)
    fl = cfg["filters"]
    masks = {f["id"]: flt.mask(f["filter"], dataset.columns, dataset.n) for f in fl}
    per_query = []
    for qi in range(len(dataset.queries)):
        q = dataset.queries[qi]
        per_query.append((ivf.visit_order(q), engine.distances(q)))
    plans = ([("sim_classic", c, None) for c in sim["budgets_c"]]
             + [("sim_enhanced", c, None) for c in sim["budgets_c"]]
             + [("sim_postfilter", c, b) for b, c in sim["postfilter_pairs"]])
    rows, seq = [], 0
    for cls, c, b in plans:
        for f in fl:
            m = masks[f["id"]]
            for qi, qid in enumerate(dataset.query_ids):
                order, d = per_query[qi]
                if cls == "sim_classic":
                    ranked, units, scored = classic(order, m, d, c)
                elif cls == "sim_enhanced":
                    ranked, units, scored = enhanced(order, m, d, c)
                else:
                    ranked, units, scored = postfilter(order, m, d, c, b)
                for k in k_values:
                    seq += 1
                    rows.append({"request_id": f"r{seq:06d}", "evidence_class": cls, "budget_c": c,
                                 "budget_b": b, "filter_id": f["id"], "k": k, "query_id": qid,
                                 "returned_keys": [dataset.keys[i] for i in ranked[:k]],
                                 "budget_units_used": int(units), "candidates_scored": int(scored)})
    return rows
