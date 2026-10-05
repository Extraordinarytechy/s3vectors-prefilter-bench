import numpy as np
import pytest

from src import metrics
from src.state import IntegrityError

KEYS = [f"v-{i:05d}" for i in range(20)]
KI = {k: i for i, k in enumerate(KEYS)}


def gt_row(gt_keys, matching, ties=None):
    return {"gt_keys": gt_keys, "matching": matching,
            "tie_flags": ties or {"5": False, "10": False}}


def mask_of(keys):
    m = np.zeros(20, dtype=bool)
    for k in keys:
        m[KI[k]] = True
    return m


def test_perfect_and_partial_recall():
    gt = KEYS[:10]
    m = mask_of(KEYS[:15])
    r = metrics.score_request(KEYS[:5], gt_row(gt, 15), 5, m, KI)
    assert r["recall_at_k"] == 1.0 and r["completeness"] == 1.0
    r = metrics.score_request([KEYS[0], KEYS[1], KEYS[12], KEYS[13], KEYS[14]], gt_row(gt, 15), 5, m, KI)
    assert r["recall_at_k"] == pytest.approx(0.4) and r["correct"] == 2


def test_fewer_than_k_matching():
    gt = KEYS[:3]
    r = metrics.score_request(KEYS[:3], gt_row(gt, 3), 10, mask_of(gt), KI)
    assert r["denominator"] == 3 and r["recall_at_k"] == 1.0 and r["completeness"] == 1.0


def test_zero_match_null_recall_and_correctness():
    r = metrics.score_request([], gt_row([], 0), 10, mask_of([]), KI)
    assert r["recall_at_k"] is None and r["completeness"] is None and r["zero_match_correct"] is True
    r = metrics.score_request([KEYS[0]], gt_row([], 0), 10, mask_of([]), KI)
    assert r["zero_match_correct"] is False and r["precision_violations"] == 1


def test_completeness_unclipped_with_precision_violations():
    gt = KEYS[:3]
    r = metrics.score_request(KEYS[:5], gt_row(gt, 3), 10, mask_of(gt), KI)
    assert r["completeness"] == pytest.approx(5 / 3)
    assert r["matching_completeness"] == 1.0 and r["precision_violations"] == 2


def test_unknown_keys_raise():
    with pytest.raises(IntegrityError):
        metrics.score_request(["nope"], gt_row(KEYS[:5], 5), 5, mask_of(KEYS[:5]), KI)


def test_single_page_duplicates_counted_distinct():
    gt = KEYS[:5]
    r = metrics.score_request([KEYS[0], KEYS[0], KEYS[1]], gt_row(gt, 5), 5, mask_of(gt), KI)
    assert r["returned"] == 2 and r["duplicate_keys"] == 1


def _rec(q, recall, ok=True, cls="aws_enhanced", b=None, c=None, keys=None):
    base = {"evidence_class": cls, "filter_id": "F", "k": 5, "budget_b": b, "budget_c": c, "query_id": q, "ok": ok}
    if not ok:
        return base
    return {**base, "recall_at_k": recall, "completeness": recall, "matching_completeness": recall, "returned": 5,
            "matching": 10, "precision_violations": 0, "duplicate_keys": 0, "tie_flag": False,
            "zero_match_correct": None, "keys": keys}


def test_aggregation_per_query_first():
    # q1 repeats: 1, 1, 0.4 -> mean 0.8; q2 repeats: 1 -> 1.0 (two failed)
    recs = [_rec("q1", 1.0), _rec("q1", 1.0), _rec("q1", 0.4), _rec("q2", 1.0), _rec("q2", None, ok=False),
            _rec("q2", None, ok=False), _rec("q3", None, ok=False)]
    row = metrics.aggregate(recs, seed=1)[0]
    assert row["recall_mean"] == pytest.approx(0.9)  # per-request mean would be 0.85
    assert row["recall_min"] == pytest.approx(0.8)
    assert row["recall_frac_1"] == pytest.approx(0.5)
    assert row["n_queries"] == 2 and row["n_requests"] == 4 and row["failed_requests"] == 3


def test_summary_key_keeps_budget_c_distinct():
    recs = [_rec("q1", 1.0, cls="sim_postfilter", b=100, c=500), _rec("q1", 0.5, cls="sim_postfilter", b=100, c=2000)]
    rows = metrics.aggregate(recs, seed=1)
    assert len(rows) == 2 and {r["budget_c"] for r in rows} == {500, 2000}


def test_bootstrap_deterministic():
    vals = [0.1, 0.5, 0.9, 1.0]
    assert metrics.bootstrap_ci(vals, 7, 3) == metrics.bootstrap_ci(vals, 7, 3)
    lo, hi = metrics.bootstrap_ci(vals, 7, 3)
    assert lo <= np.mean(vals) <= hi


def test_consistency():
    c = metrics.consistency({"q1": [["a", "b"], ["a", "b"]], "q2": [["a", "b"], ["a", "c"]]})
    assert c["identical_rate"] == 0.5
    assert c["mean_jaccard"] == pytest.approx((1.0 + 1 / 3) / 2)


def test_zero_match_rate_in_aggregate():
    recs = [{"evidence_class": "aws_enhanced", "filter_id": "FZERO", "k": 5, "budget_b": None, "budget_c": None,
             "query_id": q, "ok": True, "recall_at_k": None, "completeness": None, "matching_completeness": None,
             "returned": r, "matching": 0, "precision_violations": r, "duplicate_keys": 0, "tie_flag": False,
             "zero_match_correct": r == 0} for q, r in (("q1", 0), ("q2", 1))]
    row = metrics.aggregate(recs, seed=1)[0]
    assert row["recall_mean"] is None and row["zero_match_correct_rate"] == 0.5
