import numpy as np
import pytest

from src import metrics

KEYS = [f"v-{i:05d}" for i in range(10)]
KI = {k: i for i, k in enumerate(KEYS)}


def test_merge_orders_by_distance_then_key_across_pages():
    pages = [{"keys": [KEYS[3], KEYS[1]], "distances": [0.3, 0.1]},
             {"keys": [KEYS[2], KEYS[0]], "distances": [0.1, 0.05]}]
    assert [k for _, k in metrics.merge_pages(pages)] == [KEYS[0], KEYS[1], KEYS[2], KEYS[3]]


def test_local_filter_first_k_and_underfill():
    merged = [(0.1 * i, KEYS[i]) for i in range(10)]
    m = np.zeros(10, dtype=bool)
    m[[2, 5, 7]] = True
    keys, dists = metrics.postfilter_select(merged, m, KI, 2)
    assert keys == [KEYS[2], KEYS[5]] and dists == pytest.approx([0.2, 0.5])
    keys, _ = metrics.postfilter_select(merged, m, KI, 5)  # only 3 matches in B -> underfill
    assert keys == [KEYS[2], KEYS[5], KEYS[7]]


def test_baseline_scored_against_gt():
    merged = [(0.1 * i, KEYS[i]) for i in range(6)]
    m = np.zeros(10, dtype=bool)
    m[[1, 4, 8, 9]] = True  # 8 and 9 are matches the unfiltered top-B never saw
    keys, _ = metrics.postfilter_select(merged, m, KI, 3)
    gt = {"gt_keys": [KEYS[1], KEYS[8], KEYS[4], KEYS[9]], "matching": 4, "tie_flags": {"3": False}}
    r = metrics.score_request(keys, gt, 3, m, KI)
    assert r["returned"] == 2 and r["correct"] == 2
    assert r["recall_at_k"] == pytest.approx(2 / 3) and r["completeness"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("pages,top_k", [
    ([{"keys": ["a"], "next_token": "t"}, {"keys": ["b"], "next_token": "t"}, {"keys": ["c"]}], 300),
    ([{"keys": ["a"], "next_token": f"t{i}"} for i in range(4)], 200),
    ([{"keys": ["a"], "next_token": "t1"}, {"keys": ["a"]}], 200),
    ([{"keys": ["a", "b", "c"]}], 2),
])
def test_pagination_guard(pages, top_k):
    with pytest.raises(metrics.PaginationError):
        metrics.check_pagination(pages, top_k)


def test_pagination_ok():
    metrics.check_pagination([{"keys": ["a"], "next_token": "t1"}, {"keys": ["b"]}], 200)
