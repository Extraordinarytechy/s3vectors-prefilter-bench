"""Full AWS phase sequence against the in-memory fake (offline). Covers the CLASSIC decision,
AccessDenied handling in probe steps and log redaction at the runner level."""
import json
import re

import pytest

from src import runner
from src.state import read_jsonl
from tests.fake_s3vectors import ACCOUNT, FakeS3Vectors

AWS = ["--aws", "--profile", "default", "--region", "us-east-1"]
TWELVE = re.compile(r"(?<!\d)\d{12}(?!\d)")
EVIDENCE = ["run_metadata.json", "queries.jsonl", "responses.jsonl", "timings.jsonl", "ground_truth.jsonl",
            "metrics.json", "probe_classic.json", "cost_estimate.md", "cleanup.json"]


def run(root, fake, *args):
    return runner.main(list(args), root=root, client_factory=lambda profile, region: fake, sleep=lambda s: None)


def meta(root):
    return json.loads((root / "results/aws/run_metadata.json").read_text())


def rid(root):
    return meta(root)["run_id"]


def last_log_line(root):
    lines = (root / "logs" / f"run-{rid(root)}.log").read_text().strip().splitlines()
    return lines[-1]


def probe_steps(root):
    return {r["step"]: r for r in json.loads((root / "results/aws/probe_classic.json").read_text())["records"]}


def own_until_query(root, fake):
    assert run(root, fake, "aws-probe", *AWS, "--confirm-cost") == 0
    r = rid(root)
    assert run(root, fake, "aws-ingest", *AWS, "--confirm-cost", "--run-id", r) == 0
    assert run(root, fake, "aws-query", *AWS, "--confirm-cost", "--run-id", r) == 0
    return r


def test_full_own_flow(aws_root):
    root, fake = aws_root, FakeS3Vectors()
    r = own_until_query(root, fake)
    assert run(root, fake, "aws-capture", *AWS, "--run-id", r) == 0
    assert run(root, fake, "aws-cleanup", *AWS, "--run-id", r) == 0
    for name in EVIDENCE:
        assert (root / "results/aws" / name).exists(), name
    steps = probe_steps(root)
    assert list(steps) == ["1", "2", "3", "4", "5", "6", "7", "7a", "8", "9", "10", "11", "12", "13-main-index"]
    assert steps["8"]["ok"] is False and steps["10"]["ok"] is False and steps["13-main-index"]["ok"] is False
    assert steps["11"]["ok"] is True and steps["7a"]["probe_index_ready"] is True
    m = meta(root)
    assert m["classic_index_available"] is False and m["stopped_reason"] is None
    assert m["phases_completed"] == ["aws-probe", "aws-ingest", "aws-query", "aws-capture", "aws-cleanup"]
    assert m["spend_tally_usd"] <= m["hard_cap_usd"]
    metrics = json.loads((root / "results/aws/metrics.json").read_text())
    classes = {a["evidence_class"] for a in metrics["aggregates"]}
    assert classes == {"aws_enhanced", "aws_ann_reference", "aws_postfilter_baseline"}
    f1 = [a for a in metrics["aggregates"] if a["evidence_class"] == "aws_enhanced" and a["filter_id"] == "F1"]
    assert all(a["recall_mean"] == 1.0 for a in f1)  # the fake is exact filter-first
    assert {c["filter_id"] for c in metrics["constraint_probe"]} == {"C100", "C101"}
    queries = read_jsonl(root / "results/aws/queries.jsonl")
    assert any(q["evidence_class"] == "readiness" for q in queries)
    assert all(pr["evidence_class"] not in ("readiness", "ingest", "capture") for pr in metrics["per_request"])
    ids = [q["request_id"] for q in queries]
    assert len(ids) == len(set(ids))
    assert all("attempt" in t for t in read_jsonl(root / "results/aws/timings.jsonl"))
    names = fake.api_names()
    assert "delete_vectors" not in names
    upd = [c for c in fake.calls if c[0] == "update_index_mode"]
    assert len(upd) == 1 and upd[0][1]["indexName"].endswith("-probe")
    pdm = [c for c in fake.calls if c[0] == "put_vector_bucket_default_index_mode"]
    assert len(pdm) == 1 and pdm[0][1]["vectorBucketName"].endswith("-probe")
    assert fake.buckets == {}
    cleanup = json.loads((root / "results/aws/cleanup.json").read_text())
    assert cleanup["all_owned_gone"] is True
    shots = root / "results/aws/captures"
    assert (shots / "terminal_get_index.txt").exists()
    assert "REJECTED" in (shots / "terminal_classic_probe_records.txt").read_text()
    assert (root / "results/aws/SHOT_LIST.md").exists()
    assert last_log_line(root).endswith("PHASE COMPLETE aws-cleanup")
    assert runner.main(["report"], root=root) == 0
    summary = json.loads((root / "results/processed/summary.json").read_text())
    assert summary["aws_results_present"] is True
    assert (root / "figures/recall_vs_selectivity_aws.png").exists()
    assert runner.main(["verify-redaction"], root=root) == 0
    assert ACCOUNT not in (root / "results/aws/probe_classic.json").read_text()


@pytest.mark.parametrize("knobs,source", [
    ({"default_mode_classic": "accept"}, "bucket_default"),
    ({"update_classic": "accept"}, "update_index_mode"),
    ({"classic_query_on_enhanced": "accept"}, "query_mode_on_enhanced"),
])
def test_probe_stops_when_classic_obtainable(aws_root, knobs, source):
    """Steps 6, 9 or an accepted step-10 queryMode=CLASSIC all stop the AWS phase."""
    root, fake = aws_root, FakeS3Vectors(**knobs)
    assert run(root, fake, "aws-probe", *AWS, "--confirm-cost") == 3
    m = meta(root)
    assert m["classic_index_available"] is True and m["classic_source"] == source
    assert m["stopped_reason"] == "CLASSIC_OBTAINABLE" and "aws-probe" in m["phases_completed"]
    assert last_log_line(root).endswith("PHASE STOPPED aws-probe: CLASSIC_OBTAINABLE")
    n = len(fake.calls)
    assert run(root, fake, "aws-ingest", *AWS, "--confirm-cost", "--run-id", m["run_id"]) == 3
    assert len(fake.calls) == n
    assert run(root, fake, "aws-cleanup", *AWS, "--run-id", m["run_id"]) == 0
    assert fake.buckets == {}


def test_classic_accepted_on_main_finishes_grid_and_flags(aws_root):
    """An accepted step 13 sets the flag, warns, and lets the ENHANCED grid finish."""
    root, fake = aws_root, FakeS3Vectors(classic_query_on_enhanced="accept_main")
    own_until_query(root, fake)
    m = meta(root)
    assert m["classic_query_accepted_on_main"] is True and "aws-query" in m["phases_completed"]
    assert probe_steps(root)["13-main-index"]["ok"] is True and probe_steps(root)["10"]["ok"] is False
    log = (root / "logs" / f"run-{m['run_id']}.log").read_text()
    assert "CLASSIC_QUERY_ACCEPTED_ON_MAIN" in log
    assert last_log_line(root).endswith("PHASE COMPLETE aws-query")
    assert json.loads((root / "results/aws/metrics.json").read_text())["classic_query_accepted_on_main"] is True


@pytest.mark.parametrize("knobs,step", [({"update_classic": "deny"}, "8"), ({"default_mode_classic": "deny"}, "3")])
def test_access_denied_in_probe_is_recorded_and_continues(aws_root, knobs, step):
    """AccessDeniedException in a probe step is a recorded result; later steps still run."""
    root, fake = aws_root, FakeS3Vectors(**knobs)
    assert run(root, fake, "aws-probe", *AWS, "--confirm-cost") == 0
    steps = probe_steps(root)
    assert steps[step]["ok"] is False and steps[step]["error_code"] == "AccessDeniedException"
    assert steps[step]["http_status"] == 403
    for later in ("9", "10", "11", "12"):
        assert "skipped_reason" not in steps[later] and steps[later]["ok"] is not None
    assert ACCOUNT not in json.dumps(steps)


def test_access_denied_on_step1_is_fatal(aws_root):
    root, fake = aws_root, FakeS3Vectors(fail={"create_vector_bucket": ["AccessDeniedException"]})
    assert run(root, fake, "aws-probe", *AWS, "--confirm-cost") == 1
    assert last_log_line(root).endswith("PHASE FAILED aws-probe: AwsFatal")


def test_uncaught_client_error_traceback_is_redacted(aws_root, capsys):
    """A ClientError carrying a caller ARN + account id never reaches logs unredacted."""
    root, fake = aws_root, FakeS3Vectors()
    assert run(root, fake, "aws-probe", *AWS, "--confirm-cost") == 0
    fake.fail = {"create_vector_bucket": ["AccessDeniedException"]}
    assert run(root, fake, "aws-ingest", *AWS, "--confirm-cost", "--run-id", rid(root)) == 1
    log = (root / "logs" / f"run-{rid(root)}.log").read_text()
    err = capsys.readouterr().err
    assert "Traceback" in log and "AccessDeniedException" in log
    assert not TWELVE.search(log) and not TWELVE.search(err)
    assert "arn:aws:iam" not in log and "arn:aws:iam" not in err
    assert runner.main(["verify-redaction"], root=root) == 0


def test_lifecycle_refusals(aws_root, monkeypatch):
    root, fake = aws_root, FakeS3Vectors()
    monkeypatch.setenv("BENCH_REFUSED_PROFILES", "blocked-profile, other-blocked")
    assert run(root, fake, "aws-probe", *AWS, "--confirm-cost", "--run-id", "20261004t1530-a1b2") == 2
    assert run(root, fake, "aws-probe", *AWS) == 2  # no --confirm-cost
    assert run(root, fake, "aws-probe", "--aws", "--profile", "blocked-profile", "--region", "us-east-1",
               "--confirm-cost") == 2
    assert run(root, fake, "generate", "--borrowed") == 2
    assert fake.calls == []
    assert run(root, fake, "aws-probe", *AWS, "--confirm-cost") == 0
    r = rid(root)
    assert run(root, fake, "aws-capture", *AWS, "--run-id", r, "--confirm-cost") == 2
    assert run(root, fake, "aws-query", *AWS, "--confirm-cost", "--run-id", r) == 2  # before aws-ingest
    assert run(root, fake, "aws-probe", *AWS, "--confirm-cost") == 2  # live owned resources: new-run guard
    assert rid(root) == r and not (root / "results/aws/previous").exists()
    assert run(root, fake, "aws-ingest", *AWS, "--confirm-cost", "--run-id", r) == 0
    assert run(root, fake, "aws-ingest", *AWS, "--confirm-cost", "--run-id", r) == 2  # already completed


def test_plan_hash_mismatch_refused(aws_root):
    root, fake = aws_root, FakeS3Vectors()
    cfg = json.loads((root / "benchmark/experiment.json").read_text())
    cfg["repeats"] = 3
    (root / "benchmark/experiment.json").write_text(json.dumps(cfg))
    assert run(root, fake, "aws-probe", *AWS, "--confirm-cost") == 2
    assert fake.calls == []


def test_cleanup_resolves_pending_missing_and_never_touches_preexisting(aws_root):
    root, fake = aws_root, FakeS3Vectors()
    assert run(root, fake, "aws-probe", *AWS, "--confirm-cost") == 0
    r = rid(root)
    main = f"s3vectors-prefilter-bench-{r}"
    fake.buckets[main] = {"default": "ENHANCED", "indexes": {}, "tags": {}}  # someone else's, same name
    assert run(root, fake, "aws-ingest", *AWS, "--confirm-cost", "--run-id", r) == 1
    m = meta(root)
    conflict = [e for e in m["created_resources"] if e["bucket"] == main]
    assert conflict[0]["status"] == "create_failed" and conflict[0]["error_code"] == "ConflictException"
    assert not any(c[0] == "create_vector_bucket" and c[1]["vectorBucketName"] == main for c in fake.calls)
    # a pending probe index that exists, and a pending entry whose resource is missing
    for e in m["created_resources"]:
        if e["role"] == "probe_index":
            e["status"] = "pending"
    m["created_resources"].append({"type": "index", "role": "probe_index", "owned": True,
                                   "bucket": f"{main}-probe", "index": f"{main}-probe-gone",
                                   "arn_resource": "x", "status": "pending", "error_code": None,
                                   "created_utc": "", "updated_utc": ""})
    (root / "results/aws/run_metadata.json").write_text(json.dumps(m))
    before = len(fake.calls)
    assert run(root, fake, "aws-cleanup", *AWS, "--run-id", r) == 0
    touched = [c for c in fake.calls[before:] if c[1].get("vectorBucketName") == main]
    assert touched == []
    assert main in fake.buckets and f"{main}-probe" not in fake.buckets
    final = {(e["bucket"], e["index"]): e["status"] for e in meta(root)["created_resources"]}
    assert final[(f"{main}-probe", f"{main}-probe")] == "deleted"
    assert final[(f"{main}-probe", f"{main}-probe-gone")] == "not_found"
    actions = json.loads((root / "results/aws/cleanup.json").read_text())["actions"]
    assert any(a["action"] == "skipped: preexisting (ConflictException)" for a in actions)


def test_interrupted_query_rerun_archives_partial(aws_root):
    root, fake = aws_root, FakeS3Vectors()
    assert run(root, fake, "aws-probe", *AWS, "--confirm-cost") == 0
    r = rid(root)
    assert run(root, fake, "aws-ingest", *AWS, "--confirm-cost", "--run-id", r) == 0
    fake.fail = {"query_vectors": ["ValidationException"] * 5}  # step 13 + 4 grid failures > 1% of the plan
    assert run(root, fake, "aws-query", *AWS, "--confirm-cost", "--run-id", r) == 1
    assert (root / "results/aws/_partial").exists()
    assert run(root, fake, "aws-query", *AWS, "--confirm-cost", "--run-id", r) == 0
    aborted = list((root / "results/aws/aborted").iterdir())
    assert len(aborted) == 1 and (aborted[0] / "_partial" / "queries.jsonl").exists()
    assert not (root / "results/aws/_partial").exists()
    final_ids = [q["request_id"] for q in read_jsonl(root / "results/aws/queries.jsonl")]
    old_ids = [q["request_id"] for q in read_jsonl(aborted[0] / "_partial" / "queries.jsonl")]
    assert len(final_ids) == len(set(final_ids)) and not set(final_ids) & set(old_ids)
    steps = [s["step"] for s in json.loads((root / "results/aws/probe_classic.json").read_text())["records"]]
    assert steps.count("13-main-index") == 1
    assert (aborted[0] / "probe_classic.json").exists()
    m = meta(root)
    assert m["spend_tally_usd"] <= m["hard_cap_usd"]


def test_retried_requests_write_one_timing_row_per_attempt(aws_root):
    """Runner level: two 429s on step 13 give three timing rows and three charges."""
    root, fake = aws_root, FakeS3Vectors()
    assert run(root, fake, "aws-probe", *AWS, "--confirm-cost") == 0
    r = rid(root)
    assert run(root, fake, "aws-ingest", *AWS, "--confirm-cost", "--run-id", r) == 0
    fake.fail = {"query_vectors": ["TooManyRequestsException"] * 2}
    n_before = len([c for c in fake.calls if c[0] == "query_vectors"])
    assert run(root, fake, "aws-query", *AWS, "--confirm-cost", "--run-id", r) == 0
    step13 = probe_steps(root)["13-main-index"]
    rows = [t for t in read_jsonl(root / "results/aws/timings.jsonl")
            if t["request_id"] == step13["request"]["request_id"]]
    assert [t["attempt"] for t in rows] == [1, 2, 3]
    assert step13["attempts"] == 3
    sent = len([c for c in fake.calls if c[0] == "query_vectors"]) - n_before
    assert sent == len(read_jsonl(root / "results/aws/timings.jsonl")) - len(
        [t for t in read_jsonl(root / "results/aws/timings.jsonl") if t["evidence_class"] == "readiness"])


def test_capture_twice_appends_distinct_rows(aws_root):
    root, fake = aws_root, FakeS3Vectors()
    r = own_until_query(root, fake)
    assert run(root, fake, "aws-capture", *AWS, "--run-id", r) == 0
    assert run(root, fake, "aws-capture", *AWS, "--run-id", r) == 0
    caps = [q for q in read_jsonl(root / "results/aws/queries.jsonl") if q["evidence_class"] == "capture"]
    assert len(caps) == 4 and len({q["request_id"] for q in caps}) == 4


def _borrowed(aws_root, mode="CLASSIC"):
    root = aws_root
    fake = FakeS3Vectors(preexisting=["someone-elses-bucket"], classic_query_underfills=True)
    fake.buckets["someone-elses-bucket"]["default"] = mode
    assert runner.main(["estimate", "--borrowed"], root=root) == 0
    assert run(root, fake, "aws-ingest", *AWS, "--confirm-cost", "--borrowed-bucket", "someone-elses-bucket") == 2
    assert fake.api_names() == ["get_vector_bucket"]
    r = rid(root)
    assert meta(root)["mode"] == "borrowed"
    assert run(root, fake, "aws-ingest", *AWS, "--confirm-cost", "--borrowed-bucket", "someone-elses-bucket",
               "--run-id", r, "--confirm-index-name", "wrong-name") == 2
    assert fake.api_names() == ["get_vector_bucket"]
    return root, fake, r


def test_borrowed_flow(aws_root):
    root, fake, r = _borrowed(aws_root)
    name = f"s3vectors-prefilter-bench-{r}"
    assert run(root, fake, "aws-ingest", *AWS, "--confirm-cost", "--borrowed-bucket", "someone-elses-bucket",
               "--run-id", r, "--confirm-index-name", name) == 0
    ready = [q for q in read_jsonl(root / "results/aws/queries.jsonl") if q["evidence_class"] == "readiness"]
    assert ready and all(q["query_mode"] == "ENHANCED" for q in ready)
    assert run(root, fake, "aws-query", *AWS, "--confirm-cost", "--run-id", r) == 0
    metrics = json.loads((root / "results/aws/metrics.json").read_text())
    classes = {a["evidence_class"] for a in metrics["aggregates"]}
    assert classes == {"aws_classic", "aws_classic_index_enhanced_query", "aws_ann_reference",
                       "aws_postfilter_baseline"}
    a_one = [a for a in metrics["aggregates"] if a["evidence_class"] == "aws_classic" and a["filter_id"] == "FONE"]
    assert all(a["completeness_mean"] == 0.0 for a in a_one)  # the fake's CLASSIC underfill
    assert metrics["constraint_probe"] == []
    assert run(root, fake, "aws-capture", *AWS, "--run-id", r) == 0
    assert run(root, fake, "aws-cleanup", *AWS, "--run-id", r) == 0
    names = fake.api_names()
    for forbidden in ("update_index_mode", "put_vector_bucket_default_index_mode", "delete_vector_bucket",
                      "create_vector_bucket", "tag_resource", "delete_vectors"):
        assert forbidden not in names
    assert "someone-elses-bucket" in fake.buckets and fake.buckets["someone-elses-bucket"]["indexes"] == {}
    actions = json.loads((root / "results/aws/cleanup.json").read_text())["actions"]
    assert any(a["action"] == "skipped: borrowed" for a in actions)
    assert not (root / "results/aws/probe_classic.json").exists()


def test_borrowed_aborts_on_non_classic_index(aws_root):
    root, fake, r = _borrowed(aws_root, mode="ENHANCED")
    name = f"s3vectors-prefilter-bench-{r}"
    assert run(root, fake, "aws-ingest", *AWS, "--confirm-cost", "--borrowed-bucket", "someone-elses-bucket",
               "--run-id", r, "--confirm-index-name", name) == 1
    assert "put_vectors" not in fake.api_names()
    assert fake.buckets["someone-elses-bucket"]["indexes"] == {}
    entry = [e for e in meta(root)["created_resources"] if e["role"] == "borrowed_index"][0]
    assert entry["status"] == "deleted"
