import numpy as np
import pytest

from src import filters as flt
from src.runner import load_config
from src.state import ConfigError
from tests.conftest import small_config

META = [{"key": "a", "tenant": "t1", "category": "c1", "year": 2021},
        {"key": "b", "tenant": "t2", "category": "c1", "year": 2019},
        {"key": "c", "tenant": "t1", "category": "c2", "year": 2020}]
COLS = {k: np.array([m[k] for m in META]) for k in ("tenant", "category", "year")}

CASES = [
    ({"tenant": "t1"}, [True, False, True]),
    ({"tenant": {"$eq": "t2"}}, [False, True, False]),
    ({"tenant": {"$in": ["t2", "t9"]}}, [False, True, False]),
    ({"year": {"$gte": 2020}}, [True, False, True]),
    ({"$and": [{"tenant": "t1"}, {"category": "c1"}]}, [True, False, False]),
    ({"$and": [{"tenant": "t1"}, {"category": "c2"}, {"year": {"$gte": 2020}}]}, [False, False, True]),
    ({"missing": "x"}, [False, False, False]),
    (None, [True, True, True]),
]


@pytest.mark.parametrize("f,expected", CASES)
def test_evaluator_scalar_and_vectorized_agree(f, expected):
    assert [flt.matches(f, m) for m in META] == expected
    assert list(flt.mask(f, COLS, 3)) == expected


@pytest.mark.parametrize("f", [{"tenant": {"$ne": "t1"}}, {"$or": [{"tenant": "t1"}]}, {"t": {"$startsWith": "x"}},
                               {"$and": []}])
def test_unsupported_operators_raise(f):
    with pytest.raises(ConfigError):
        flt.matches(f, META[0])
    with pytest.raises(ConfigError):
        flt.mask(f, COLS, 3)
    with pytest.raises(ConfigError):
        flt.validate(f)


def test_constraint_counter_documented_examples():
    assert flt.count_constraints({"genre": "drama"}) == 1
    assert flt.count_constraints({"genre": {"$in": ["a", "b", "c"]}}) == 3
    assert flt.count_constraints({"$and": [{"genre": "drama"}, {"year": {"$lte": 2020}}]}) == 2


def test_c100_c101_counts():
    cfg = small_config()
    cfg["constraint_probe"]["in_count"] = 10
    c100, c101 = flt.constraint_probe_filters(cfg)
    assert flt.count_constraints(c100["filter"]) == 10
    assert flt.count_constraints(c101["filter"]) == 11


def test_full_config_c100_c101():
    from tests.conftest import REPO
    import json
    cfg = json.loads((REPO / "benchmark" / "experiment.json").read_text(encoding="utf-8"))
    c100, c101 = flt.constraint_probe_filters(cfg)
    assert flt.count_constraints(c100["filter"]) == 100
    assert flt.count_constraints(c101["filter"]) == 101


def test_config_rejects_over_100_constraints(tmp_path):
    import json
    cfg = small_config()
    cfg["filters"].append({"id": "BIG", "filter": {"tenant": {"$in": [f"x{i}" for i in range(101)]}}})
    path = tmp_path / "e.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_config_rejects_wrong_budgets(tmp_path):
    import json
    cfg = small_config()
    cfg["budgets"] = [100, 1000]
    path = tmp_path / "e.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)
