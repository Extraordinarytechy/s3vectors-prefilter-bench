import json

import numpy as np
import pytest

from src import dataset as ds_mod
from src import filters as flt
from src.state import IntegrityError
from tests.conftest import REPO, small_config


def _hashes(cfg):
    d = ds_mod.build(cfg["dataset"])
    return (ds_mod.sha256_bytes(d.vectors.tobytes()), ds_mod.sha256_bytes(d.queries.tobytes()),
            ds_mod.sha256_bytes(ds_mod.metadata_lines(d.metadata)))


def test_same_seed_identical_and_different_seed_differs():
    cfg = small_config()
    assert _hashes(cfg) == _hashes(cfg)
    other = small_config()
    other["dataset"]["seed"] += 1
    h1, h2 = _hashes(cfg), _hashes(other)
    assert all(a != b for a, b in zip(h1, h2))


def test_streams_are_spawn_children():
    seed = 20260930
    children = np.random.SeedSequence(seed).spawn(6)
    for i, name in enumerate(ds_mod.STREAM_NAMES):
        a = np.random.default_rng(ds_mod.stream(seed, name)).integers(0, 2**31, 4)
        b = np.random.default_rng(children[i]).integers(0, 2**31, 4)
        assert (a == b).all()


def test_small_vectors_unit_norm_and_counts():
    cfg = small_config()
    d = ds_mod.build(cfg["dataset"])
    assert d.vectors.dtype == np.float32 and d.vectors.shape == (1000, 16)
    assert d.queries.shape == (5, 16)
    assert np.all(np.isfinite(d.vectors))
    assert np.allclose(np.linalg.norm(d.vectors.astype(np.float64), axis=1), 1.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(d.queries.astype(np.float64), axis=1), 1.0, atol=1e-6)
    tenants = d.columns["tenant"]
    assert (tenants == "tenant-s50").sum() == 500
    assert (tenants == "tenant-one").sum() == 1
    assert len({t for t in tenants if t.startswith("tenant-bg-")}) == 10


def test_field_label_count_mismatch_raises():
    field = {"name": "x", "values": ["a", "b"], "count_each": 3}
    with pytest.raises(IntegrityError):
        ds_mod.field_labels(field, 7)


def test_full_dataset_shape_counts_and_metadata_size():
    cfg = json.loads((REPO / "benchmark" / "experiment.json").read_text(encoding="utf-8"))
    d = ds_mod.build(cfg["dataset"])
    assert d.vectors.shape == (50000, 384) and d.vectors.dtype == np.float32
    assert np.all(np.isfinite(d.vectors))
    assert np.allclose(np.linalg.norm(d.vectors.astype(np.float64), axis=1), 1.0, atol=1e-5)
    expected = {"F50": 25000, "F10": 5000, "F1": 500, "F01": 50, "F001": 5, "FFEW": 3, "FONE": 1, "FZERO": 0}
    for f in cfg["filters"]:
        count = int(flt.mask(f["filter"], d.columns, d.n).sum())
        if f["id"] in expected:
            assert count == expected[f["id"]], f["id"]
        else:
            assert 0 < count < 5000, f["id"]
    assert all((d.columns["category"] == c).sum() == 12500 for c in ("c0", "c1", "c2", "c3"))
    assert all((d.columns["year"] == y).sum() == 5000 for y in range(2015, 2025))
    bg = [t for t in d.columns["tenant"] if t.startswith("tenant-bg-")]
    assert len(bg) == 19441 and len(set(bg)) == 100
    assert max(ds_mod.filterable_metadata_bytes(m) for m in d.metadata) < 2048
    probes = {f["id"]: int(flt.mask(f["filter"], d.columns, d.n).sum()) for f in flt.constraint_probe_filters(cfg)}
    assert probes == {"C100": 19441, "C101": 19941}


def test_write_load_roundtrip_and_tamper(tmp_path):
    cfg = small_config()
    d = ds_mod.build(cfg["dataset"])
    ds_mod.write(d, cfg["dataset"], tmp_path, {"F": 1}, "code")
    loaded, manifest = ds_mod.load(tmp_path)
    assert loaded.keys == d.keys and (loaded.vectors == d.vectors).all()
    assert manifest["filter_match_counts"] == {"F": 1}
    with open(tmp_path / "metadata.jsonl", "ab") as fh:
        fh.write(b"\n")
    with pytest.raises(IntegrityError):
        ds_mod.load(tmp_path)
