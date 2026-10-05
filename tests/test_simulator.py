import numpy as np
import pytest

from src import dataset as ds_mod
from src import filters as flt
from src import metrics, simulator
from src.ground_truth import DistanceEngine
from src.state import ConfigError
from tests.conftest import small_config


@pytest.fixture(scope="module")
def sim():
    cfg = small_config()
    data = ds_mod.build(cfg["dataset"])
    return cfg, data, simulator.run(data, cfg)


def test_deterministic(sim):
    cfg, data, rows = sim
    assert simulator.run(data, cfg) == rows


def test_budget_invariants(sim):
    cfg, data, rows = sim
    masks = {f["id"]: flt.mask(f["filter"], data.columns, data.n) for f in cfg["filters"]}
    for r in rows:
        m = int(masks[r["filter_id"]].sum())
        if r["evidence_class"] in ("sim_classic", "sim_postfilter"):
            assert r["budget_units_used"] == min(r["budget_c"], data.n)
        else:
            assert r["budget_units_used"] == min(r["budget_c"], m)
            assert r["candidates_scored"] == r["budget_units_used"]
        if r["evidence_class"] == "sim_enhanced":
            assert len(r["returned_keys"]) == min(r["k"], m)


def test_classic_counts_matching_visited_and_underfills():
    rng = np.random.default_rng(0)
    order = rng.permutation(1000)
    m = np.zeros(1000, dtype=bool)
    m[order[::100]] = True  # 1% selectivity, evenly spread over the visit order
    d = rng.random(1000)
    ranked, units, scored = simulator.classic(order, m, d, 200)
    assert units == 200 and scored == 2 and len(ranked) == 2  # s*C = 2 < K = 10 -> underfill
    ranked, units, scored = simulator.classic(order, m, d, 1000)
    assert scored == 10 and len(ranked) == 10  # s*C >> ... fills
    ranked, units, scored = simulator.enhanced(order, m, d, 200)
    assert units == scored == 10 and len(ranked) == 10


def test_postfilter_matches_postfilter_math():
    rng = np.random.default_rng(3)
    order = rng.permutation(300)
    m = rng.random(300) < 0.2
    d = rng.random(300)
    keys = [f"v-{i:05d}" for i in range(300)]
    ki = {k: i for i, k in enumerate(keys)}
    out, _, _ = simulator.postfilter(order, m, d, 150, 60)
    seen = order[:150]
    merged = sorted((float(d[i]), keys[i]) for i in seen)[:60]
    selected, _ = metrics.postfilter_select(merged, m, ki, 10)
    assert [keys[i] for i in out[:10]] == selected


def test_b_greater_than_c_rejected():
    cfg = small_config()
    cfg["simulator"]["postfilter_pairs"] = [[500, 200]]
    with pytest.raises(ConfigError):
        simulator.validate(cfg["simulator"])


def test_ivf_engine_consistency(sim):
    cfg, data, rows = sim
    ivf = simulator.IVF(data.vectors, 16, 5, cfg["dataset"]["seed"])
    order = ivf.visit_order(data.queries[0])
    assert sorted(order.tolist()) == list(range(data.n))
    assert DistanceEngine(data.vectors).distances(data.queries[0]).shape == (data.n,)
