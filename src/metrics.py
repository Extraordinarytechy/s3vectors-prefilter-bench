"""Recall@K, completeness, precision violations, consistency, aggregation, bootstrap CI,
plus the client-side post-filter baseline math and the pagination guard.
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Iterable

import numpy as np

from .dataset import stream
from .ground_truth import gt_for_k
from .state import IntegrityError

BOOTSTRAP_RESAMPLES = 1000
RESULTS_PER_PAGE = 100


class PaginationError(IntegrityError):
    """Pagination integrity error for one request: the request is a failed request."""


def dedupe(keys: Iterable[str]) -> tuple[list[str], int]:
    """Distinct keys in first-seen order, and the number of keys returned more than once."""
    seen: dict[str, int] = {}
    for k in keys:
        seen[k] = seen.get(k, 0) + 1
    return list(seen), sum(1 for c in seen.values() if c > 1)


def score_request(returned_keys: list[str], gt_row: dict, k: int, match_mask: np.ndarray,
                  key_index: dict[str, int], returned_distances: list[float] | None = None,
                  local_distances: np.ndarray | None = None) -> dict:
    """Per-request metrics against GT_K. Zero-match: recall/completeness are null."""
    distinct, duplicates = dedupe(returned_keys)
    unknown = [x for x in distinct if x not in key_index]
    if unknown:
        raise IntegrityError(f"{len(unknown)} returned keys are not in the dataset")
    r = distinct[:k]
    gt = set(gt_for_k(gt_row, k))
    matching = gt_row["matching"]
    denom = min(k, matching)
    correct = sum(1 for x in r if x in gt)
    violations = sum(1 for x in r if not match_mask[key_index[x]])
    out = {
        "matching": matching, "k": k, "returned": len(r), "correct": correct,
        "denominator": denom, "precision_violations": violations, "duplicate_keys": duplicates,
        "tie_flag": bool(gt_row["tie_flags"][str(k)]),
        "recall_at_k": correct / denom if denom else None,
        "completeness": len(r) / denom if denom else None,
        "matching_completeness": (len(r) - violations) / denom if denom else None,
        "zero_match_correct": (len(r) == 0) if matching == 0 else None,
        "distance_check": None,
    }
    if returned_distances is not None and local_distances is not None and r:
        by_key = {}
        for key, dist in zip(returned_keys, returned_distances):
            by_key.setdefault(key, dist)
        diffs = [abs(float(by_key[x]) - float(local_distances[key_index[x]])) for x in r]
        if not all(math.isfinite(v) for v in diffs):
            raise IntegrityError("non-finite distance returned")
        out["distance_check"] = max(diffs)
    return out


# ---------------------------------------------------------------- post-filter baseline

def check_pagination(pages: list[dict], top_k: int) -> None:
    """Raise PaginationError on too many pages, a repeated token, > topK results, or a key on two pages."""
    if len(pages) > math.ceil(top_k / RESULTS_PER_PAGE) + 1:
        raise PaginationError("page count exceeds ceil(topK/100)+1")
    tokens = [p.get("next_token") for p in pages if p.get("next_token")]
    if len(tokens) != len(set(tokens)):
        raise PaginationError("repeated nextToken")
    total = sum(len(p["keys"]) for p in pages)
    if total > top_k:
        raise PaginationError("more results than topK")
    if len(pages) > 1:
        owner: dict[str, int] = {}
        for i, p in enumerate(pages):
            for key in p["keys"]:
                if key in owner and owner[key] != i:
                    raise PaginationError("key repeated across pages")
                owner[key] = i


def merge_pages(pages: list[dict]) -> list[tuple[float, str]]:
    """Union of all pages ordered by (returned distance, key)."""
    pairs = [(float(d), key) for p in pages for key, d in zip(p["keys"], p["distances"])]
    return sorted(pairs)


def postfilter_select(merged: list[tuple[float, str]], match_mask: np.ndarray,
                      key_index: dict[str, int], k: int) -> tuple[list[str], list[float]]:
    """Keep keys that pass the filter under canonical metadata; take the first K."""
    keys, dists = [], []
    for dist, key in merged:
        if key not in key_index:
            raise IntegrityError("returned key is not in the dataset")
        if match_mask[key_index[key]]:
            keys.append(key)
            dists.append(dist)
            if len(keys) == k:
                break
    return keys, dists


# ---------------------------------------------------------------- aggregation

GROUP_FIELDS = ("evidence_class", "filter_id", "k", "budget_b", "budget_c")


def group_key(rec: dict) -> tuple:
    return tuple(rec.get(f) for f in GROUP_FIELDS)


def sort_key(key: tuple) -> tuple:
    return tuple((0, "") if v is None else ((1, v) if isinstance(v, str) else (2, v)) for v in key)


def bootstrap_ci(values: list[float], seed: int, ordinal: int) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(stream(seed, "bootstrap", ordinal))
    idx = rng.integers(0, len(arr), (BOOTSTRAP_RESAMPLES, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def consistency(keysets: dict[str, list[list[str]]]) -> dict:
    """identical_rate over cells with >= 2 successful repeats, and mean pairwise Jaccard."""
    cells = [v for v in keysets.values() if len(v) >= 2]
    if not cells:
        return {"identical_rate": None, "mean_jaccard": None, "consistency_cells": 0}
    identical = sum(1 for v in cells if all(s == v[0] for s in v[1:]))
    jac = []
    for v in cells:
        for a, b in combinations(v, 2):
            sa, sb = set(a), set(b)
            jac.append(1.0 if not sa and not sb else len(sa & sb) / len(sa | sb))
    return {"identical_rate": identical / len(cells), "mean_jaccard": float(np.mean(jac)),
            "consistency_cells": len(cells)}


def aggregate(records: list[dict], seed: int) -> list[dict]:
    """One row per (evidence_class, filter_id, k, budget_b, budget_c), query vector as the unit.

    Each record: the GROUP_FIELDS, query_id, ok, and (when ok) the score_request fields.
    Optional `keys` (returned key list) feeds the repeat-level consistency metric.
    """
    groups: dict[tuple, list[dict]] = {}
    for rec in records:
        groups.setdefault(group_key(rec), []).append(rec)
    rows = []
    for ordinal, key in enumerate(sorted(groups, key=sort_key)):
        recs = groups[key]
        ok = [r for r in recs if r.get("ok")]
        per_q: dict[str, list[dict]] = {}
        for r in ok:
            per_q.setdefault(r["query_id"], []).append(r)
        recall_q, compl_q, mcompl_q = [], [], []
        for qid in sorted(per_q):
            rs = per_q[qid]
            rec_vals = [r["recall_at_k"] for r in rs if r["recall_at_k"] is not None]
            if rec_vals:
                recall_q.append(float(np.mean(rec_vals)))
            c_vals = [r["completeness"] for r in rs if r["completeness"] is not None]
            if c_vals:
                compl_q.append(float(np.mean(c_vals)))
            m_vals = [r["matching_completeness"] for r in rs if r["matching_completeness"] is not None]
            if m_vals:
                mcompl_q.append(float(np.mean(m_vals)))
        zm = [r["zero_match_correct"] for r in ok if r.get("zero_match_correct") is not None]
        lo, hi = bootstrap_ci(recall_q, seed, ordinal)
        keysets: dict[str, list[list[str]]] = {}
        for r in ok:
            if r.get("keys") is not None:
                keysets.setdefault(r["query_id"], []).append(r["keys"])
        row = dict(zip(GROUP_FIELDS, key))
        row.update({
            "n_queries": len(per_q), "n_requests": len(ok), "failed_requests": len(recs) - len(ok),
            "matching": ok[0]["matching"] if ok else None,
            "recall_mean": _mean(recall_q),
            "recall_median": float(np.median(recall_q)) if recall_q else None,
            "recall_min": float(np.min(recall_q)) if recall_q else None,
            "recall_frac_1": (sum(1 for v in recall_q if v == 1.0) / len(recall_q)) if recall_q else None,
            "recall_ci_low": lo, "recall_ci_high": hi,
            "completeness_mean": _mean(compl_q),
            "matching_completeness_mean": _mean(mcompl_q),
            "returned_mean": _mean([r["returned"] for r in ok]),
            "precision_violations": sum(r["precision_violations"] for r in ok),
            "duplicate_keys": sum(r["duplicate_keys"] for r in ok),
            "ties": sum(1 for r in ok if r["tie_flag"]),
            "zero_match_correct_rate": (sum(zm) / len(zm)) if zm else None,
        })
        row.update(consistency(keysets))
        rows.append(row)
    return rows
