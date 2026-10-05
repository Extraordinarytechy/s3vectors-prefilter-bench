import json

import pytest

from src import cost_model as cm
from src import runner
from src.state import BudgetExceeded, ConfigError
from tests.conftest import FIXTURE_PRICING

P = FIXTURE_PRICING  # $1 per unit everywhere: fixture values, not real prices


def test_storage_put_other():
    assert cm.storage_usd(1000, 1000, 730 / 24, P) == pytest.approx(1e6 / 1e9)
    assert cm.put_usd(1000, P) == pytest.approx(131072 / 1e9)  # 128 KB minimum
    assert cm.put_usd(2_000_000, P) == pytest.approx(2e6 / 1e9)
    assert cm.other_usd(500, P) == pytest.approx(0.5)


def test_query_page_and_data_returned_rules():
    page = cm.query_page_lines(50_000, 1600, P)
    assert page["query_requests"] == pytest.approx(0.001)
    assert page["data_processed"] == pytest.approx(50_000 * 1600 / 1e12)
    assert cm.data_returned_usd(100, 11, P) == 0.0  # 100 x 256 B < 512 KB free
    assert cm.data_returned_usd(10_000, 11, P) == pytest.approx((10_000 * 256 - 524288) / 1e9)
    assert cm.data_returned_usd(10, 1000, P) == 0.0
    assert cm.worst_page_data_returned_usd(11, P) == pytest.approx(100 * 256 / 1e9)


def test_plan_counts_pages_and_polling(small_cfg):
    manifest = {"bytes": {"vector_data": 64, "key_mean": 7, "filterable_metadata_mean": 50}}
    plan = cm.request_plan(small_cfg, manifest, "own")
    lines = {ln["line"]: ln for ln in plan["lines"]}
    assert lines["(b) post-filter baseline pages"]["count"] == 5 * 1 * 1  # 5 q x ceil(100/100) x 1 repeat
    assert lines["(a) ENHANCED filtered queries"]["count"] == 10 * 5 * 2 * 2
    assert lines["readiness ListVectors polling (worst case)"]["count"] == 3 * 1
    assert any("cleanup" in k for k in lines) and lines["capture queries"]["count"] == 3


def test_full_plan_request_total():
    cfg = json.loads((runner.REPO_ROOT / "benchmark" / "experiment.json").read_text())
    manifest = {"bytes": {"vector_data": 1536, "key_mean": 7, "filterable_metadata_mean": 55}}
    plan = cm.request_plan(cfg, manifest, "own")
    q = {ln["line"]: ln["count"] for ln in plan["lines"] if ln["phase"] == "aws-query"}
    assert sum(q.values()) == 17_703


def test_cap_arithmetic_and_cumulative_refusal():
    caps = cm.cap_values(0.20, 0.0)
    assert caps["approved_usd"] == pytest.approx(0.40) and caps["hard_cap_usd"] == pytest.approx(0.40)
    assert cm.cap_values(3.0, 0.0)["approvable"] is False
    # $4.80 already spent by earlier runs, base $0.20 -> 2 x 0.20 > 0.20 remaining -> refused
    refused = cm.cap_values(0.20, 4.80)
    assert refused["approvable"] is False and refused["hard_cap_usd"] == pytest.approx(0.20)
    assert cm.cap_values(0.05, 4.80)["hard_cap_usd"] == pytest.approx(0.10)


def test_estimate_refuses_with_prior_spend(tmp_path, offline_root):
    import shutil
    root = tmp_path / "r"
    shutil.copytree(offline_root, root)
    assert runner.main(["estimate"], root=root) == 0
    base = cm.parse_estimate_md((root / "results/aws/cost_estimate.md").read_text())["base_usd"]
    prev = root / "results/aws/previous/20261001t0000-aaaa"
    prev.mkdir(parents=True)
    prev.joinpath("run_metadata.json").write_text(json.dumps(
        {"run_id": "20261001t0000-aaaa", "spend_tally_usd": 5.0 - base, "created_resources": []}))
    (root / "results/aws/cost_estimate.md").unlink()
    assert runner.main(["estimate"], root=root) == 2
    assert not (root / "results/aws/cost_estimate.md").exists()


def test_plan_hash_includes_prior_spend(small_cfg):
    plan = {"lines": []}
    assert cm.plan_sha256(small_cfg, P, "own", plan, 0.0) != cm.plan_sha256(small_cfg, P, "own", plan, 0.01)


def test_parse_estimate_fixed_lines():
    text = "x\nplan_mode: own\nbase_usd: 0.1\napproved_usd: 0.2\nprior_runs_spend_usd: 0.0\nhard_cap_usd: 0.2\n" \
           "plan_sha256: abc\n"
    out = cm.parse_estimate_md(text)
    assert out["hard_cap_usd"] == 0.2 and out["plan_sha256"] == "abc"
    with pytest.raises(ConfigError):
        cm.parse_estimate_md("base_usd: 1\n")


@pytest.mark.parametrize("mx,expected", [(100, [100]), (1000, [100, 1000]), (10000, [100, 1000, 10000]),
                                         (5000, [100, 1000, 5000])])
def test_budget_rule(mx, expected):
    assert runner.budget_rule(mx) == expected


def test_pricing_validation():
    bad = json.loads(json.dumps(P))
    bad["rates"]["put_usd_per_gb"] = -1
    with pytest.raises(ConfigError):
        cm.validate_pricing(bad)


def test_presend_check_uses_worst_case_page(tmp_path):
    """The guarded client refuses before sending when tally + page cost + worst-case data returned > cap."""
    import logging
    from src.aws_client import GuardContext, GuardedS3Vectors
    from src.state import RunState
    from tests.fake_s3vectors import FakeS3Vectors

    worst = cm.worst_page_data_returned_usd(11, P)
    rs = RunState(tmp_path / "m.json", {"run_id": "20261004t1530-a1b2", "hard_cap_usd": 0.01 + worst / 2,
                                        "spend_tally_usd": 0.0, "created_resources": [], "next_request_seq": 1})
    fake = FakeS3Vectors()
    g = GuardedS3Vectors(fake, rs, GuardContext("20261004t1530-a1b2", "aws-query"), logging.getLogger("t"),
                         sleep=lambda s: None)
    with pytest.raises(BudgetExceeded):
        g.call("query_vectors", {"vectorBucketName": "b", "indexName": "i"}, cost_lines={"q": 0.01},
               worst_extra_usd=worst)
    assert fake.calls == [] and rs.spend == 0.0
