import filecmp
import json

import pandas as pd
import pytest

from src import reporting, runner
from src.state import ConfigError
from tests.conftest import make_root


def test_summary_json_csv_roundtrip_and_labels(offline_root):
    summary = json.loads((offline_root / "results/processed/summary.json").read_text())
    assert summary["aws_results_present"] is False
    rows = summary["rows"]
    assert rows and {r["evidence_class"] for r in rows} == {"sim_classic", "sim_enhanced", "sim_postfilter"}
    for r in rows:
        assert r["label"] == reporting.label(r["evidence_class"], r["budget_b"], r["budget_c"])
        assert r["label"].startswith("SIMULATED")
    frame = pd.read_csv(offline_root / "results/processed/summary.csv")
    assert len(frame) == len(rows)
    assert set(frame.columns) == set().union(*(r.keys() for r in rows))
    assert frame["recall_mean"].dropna().tolist() == pytest.approx([r["recall_mean"] for r in rows
                                                                    if r["recall_mean"] is not None])
    assert not (offline_root / "figures/recall_vs_selectivity_aws.png").exists()
    assert (offline_root / "figures/recall_vs_selectivity_simulated.png").exists()
    assert (offline_root / "figures/flow_filter_candidate_search.png").exists()


def test_unknown_class_raises_and_excluded_classes_dropped():
    manifest = {"n": 10, "filter_match_counts": {"F": 1}}
    base = {"filter_id": "F", "k": 5, "budget_b": None, "budget_c": None}
    with pytest.raises(ConfigError):
        reporting.summary_rows([{**base, "evidence_class": "mystery"}], manifest)
    rows = reporting.summary_rows([{**base, "evidence_class": c} for c in ("ingest", "readiness", "capture")],
                                  manifest)
    assert rows == []
    assert reporting.label("aws_postfilter_baseline", b=10000).endswith("B=10000")
    assert "CLASSIC" not in reporting.label("aws_postfilter_baseline", b=100)


def test_report_never_reads_aborted_or_previous(tmp_path, offline_root):
    import shutil
    root = tmp_path / "r"
    shutil.copytree(offline_root, root)
    bad = {"aggregates": [{"evidence_class": "mystery", "filter_id": "F", "k": 5, "budget_b": None,
                           "budget_c": None}]}
    for sub in ("aborted/x", "previous/y"):
        (root / "results/aws" / sub).mkdir(parents=True)
        (root / "results/aws" / sub / "metrics.json").write_text(json.dumps(bad))
    assert runner.main(["report"], root=root) == 0


def test_real_and_simulated_never_share_a_figure(offline_root, monkeypatch):
    seen = []
    real = reporting._plot

    def spy(ax, pts, metric, i, name, ci=False):
        seen.append((id(ax), name))
        return real(ax, pts, metric, i, name, ci)

    monkeypatch.setattr(reporting, "_plot", spy)
    rows = json.loads((offline_root / "results/processed/summary.json").read_text())["rows"]
    aws_rows = [{**r, "evidence_class": "aws_enhanced", "budget_c": None, "label": "x"} for r in rows
                if r["evidence_class"] == "sim_enhanced" and r["budget_c"] == 200]
    cfg = runner.load_config(offline_root / "benchmark/experiment.json")
    out = offline_root.parent / "figs-test"
    reporting.figure_sim(rows + aws_rows, cfg, out / "s.png")
    reporting.figure_aws(rows + aws_rows, cfg, "recall_mean", "Recall@10", out / "a.png")
    axes = {}
    for ax, name in seen:
        axes.setdefault(ax, set()).add(name.startswith("SIMULATED"))
    assert all(len(kinds) == 1 for kinds in axes.values())


def test_latency_uses_first_attempt_requests_only(tmp_path):
    rows = [{"request_id": "r1", "evidence_class": "aws_enhanced", "filter_id": "F1", "budget": None},
            {"request_id": "r2", "evidence_class": "aws_enhanced", "filter_id": "F1", "budget": None}]
    (tmp_path / "queries.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    timings = [{"request_id": "r1", "page": 1, "attempt": 1, "rtt_ns": 2_000_000, "phase": "first"},
               {"request_id": "r2", "page": 1, "attempt": 1, "rtt_ns": 9_000_000, "phase": "first"},
               {"request_id": "r2", "page": 1, "attempt": 2, "rtt_ns": 1_000_000, "phase": "first"}]
    (tmp_path / "timings.jsonl").write_text("".join(json.dumps(t) + "\n" for t in timings))
    lrows, info = reporting.latency_rows(tmp_path)
    assert info == {"requests": 2, "excluded_retried": 1, "excluded_share": 0.5}
    assert lrows[0]["n"] == 1 and lrows[0]["median_ms"] == pytest.approx(2.0)


def test_latency_figure_separates_phases_and_never_uses_simulated(tmp_path, monkeypatch):
    lrows = [{"evidence_class": c, "filter_id": f, "budget": b, "phase": ph, "n": 3, "median_ms": ms, "p90_ms": ms}
             for ph in ("first", "warm")
             for c, f, b, ms in (("aws_enhanced", "F1", None, 330.0), ("aws_enhanced", "FZERO", None, 331.0),
                                 ("aws_ann_reference", "NOFILTER", None, 329.0),
                                 ("aws_postfilter_baseline", None, 100, 335.0),
                                 ("aws_postfilter_baseline", None, 10000, 33500.0))]
    manifest = {"n": 100, "filter_match_counts": {"F1": 1, "FZERO": 0, "NOFILTER": 100}}
    names = []
    real_axhline = reporting._plt().Axes.axhline

    def spy(self, *a, **kw):
        names.append(kw.get("label", ""))
        return real_axhline(self, *a, **kw)

    monkeypatch.setattr(reporting._plt().Axes, "axhline", spy)
    path = tmp_path / "lat.png"
    reporting.figure_latency(lrows, {"requests": 10, "excluded_retried": 1}, manifest, path)
    assert path.exists() and path.stat().st_size > 0
    assert len(names) == 6 and all(n.startswith("REAL AWS") for n in names)
    assert not any("CLASSIC" in n for n in names)


def _tree_files(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def test_pipeline_twice_is_byte_identical(tmp_path):
    root = make_root(tmp_path / "root")
    assert runner.main(["offline-all", "--out-root", "run1"], root=root) == 0
    assert runner.main(["offline-all", "--out-root", "run2"], root=root) == 0
    a, b = root / "run1", root / "run2"
    files = _tree_files(a)
    assert files == _tree_files(b) and len(files) >= 12
    _, mismatch, errors = filecmp.cmpfiles(a, b, files, shallow=False)
    assert mismatch == [] and errors == []
