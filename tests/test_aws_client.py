import io
import json
import logging
import re
import sys

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from src import aws_client as ac
from src.aws_client import GuardContext, GuardedS3Vectors, guard_check
from src.state import AwsFatal, OwnershipError, RunState
from tests.conftest import stubbed_client
from tests.fake_s3vectors import FakeS3Vectors

RID = "20261004t1530-a1b2"
PFX = f"s3vectors-prefilter-bench-{RID}"
ACCT = "1234" * 3  # a planted 12-digit account id, built at runtime
TWELVE = re.compile(r"(?<!\d)\d{12}(?!\d)")


def entry(role, bucket, index=None, owned=True, status="created", error_code=None):
    return {"type": "index" if index else "bucket", "role": role, "owned": owned, "bucket": bucket, "index": index,
            "status": status, "error_code": error_code}


MANIFEST = [
    entry("main_bucket", PFX), entry("main_index", PFX, PFX),
    entry("probe_bucket", f"{PFX}-probe"), entry("probe_index", f"{PFX}-probe", f"{PFX}-probe"),
    entry("borrowed_bucket", "someone-elses-bucket", owned=False), entry("borrowed_index", "someone-elses-bucket", PFX),
]


def arn(bucket, index=None):
    return f"arn:aws:s3vectors:us-east-1:{ACCT}:bucket/{bucket}" + (f"/index/{index}" if index else "")


def ctx(phase="aws-probe", **kw):
    return GuardContext(run_id=RID, phase=phase, confirm_cost=True, **kw)


B, BI = {"vectorBucketName": PFX}, {"vectorBucketName": PFX, "indexName": PFX}
PB, PI = {"vectorBucketName": f"{PFX}-probe"}, {"vectorBucketName": f"{PFX}-probe", "indexName": f"{PFX}-probe"}
BB, BBI = {"vectorBucketName": "someone-elses-bucket"}, {"vectorBucketName": "someone-elses-bucket", "indexName": PFX}

ALLOWED = [
    ("put_vector_bucket_default_index_mode", {**PB, "defaultIndexMode": "CLASSIC"}, "aws-probe"),
    ("update_index_mode", {**PI, "indexMode": "CLASSIC"}, "aws-probe"),
    ("put_vectors", {**BI, "vectors": []}, "aws-ingest"),
    ("put_vectors", {**PI, "vectors": []}, "aws-probe"),
    ("put_vectors", {**BBI, "vectors": []}, "aws-ingest"),
    ("delete_index", BI, "aws-cleanup"),
    ("delete_index", BBI, "aws-cleanup"),
    ("delete_vector_bucket", B, "aws-cleanup"),
    ("delete_vector_bucket", PB, "aws-cleanup"),
    ("tag_resource", {"resourceArn": arn(PFX, PFX), "tags": {}}, "aws-ingest"),
    ("get_index", BBI, "aws-query"),
    ("query_vectors", {"vectorBucketName": "anything", "indexName": "x"}, "aws-query"),
]
REFUSED = [
    ("put_vector_bucket_default_index_mode", {**B, "defaultIndexMode": "CLASSIC"}, "aws-probe"),
    ("put_vector_bucket_default_index_mode", {**BB, "defaultIndexMode": "CLASSIC"}, "aws-probe"),
    ("update_index_mode", {**BI, "indexMode": "CLASSIC"}, "aws-probe"),
    ("update_index_mode", {**BBI, "indexMode": "CLASSIC"}, "aws-probe"),
    ("update_index_mode", {**PI, "indexMode": "ENHANCED"}, "aws-probe"),
    ("update_index_mode", {**PI, "indexMode": "CLASSIC"}, "aws-query"),
    ("delete_vectors", {**BI, "keys": ["v-00000"]}, "aws-cleanup"),
    ("delete_index", BI, "aws-query"),
    ("delete_vector_bucket", BB, "aws-cleanup"),
    ("delete_vector_bucket", B, "aws-query"),
    ("tag_resource", {"resourceArn": arn("someone-elses-bucket"), "tags": {}}, "aws-ingest"),
    ("tag_resource", {"resourceArn": "not-an-arn", "tags": {}}, "aws-ingest"),
    ("put_vectors", {"vectorBucketName": "other-bucket", "indexName": "idx", "vectors": []}, "aws-ingest"),
    ("put_vectors", {"vectorBucketName": "s3vectors-prefilter-bench-20260101t0000-ffff", "indexName": "x",
                     "vectors": []}, "aws-ingest"),
    ("put_vectors", {"indexArn": arn(PFX, PFX), "vectors": []}, "aws-ingest"),
    ("delete_vector_bucket_policy", B, "aws-cleanup"),
]


@pytest.mark.parametrize("api,params,phase", ALLOWED)
def test_guard_allows(api, params, phase):
    guard_check(api, params, MANIFEST, ctx(phase))


@pytest.mark.parametrize("api,params,phase", REFUSED)
def test_guard_refuses(api, params, phase):
    with pytest.raises(OwnershipError):
        guard_check(api, params, MANIFEST, ctx(phase))


def test_create_requires_precheck_pending_prefix_and_confirm():
    m = [entry("main_bucket", PFX, status="pending")]
    with pytest.raises(OwnershipError, match="pre-check"):
        guard_check("create_vector_bucket", B, m, ctx("aws-ingest"))
    c = ctx("aws-ingest", prechecked_not_found={(PFX, None)})
    guard_check("create_vector_bucket", B, m, c)
    with pytest.raises(OwnershipError, match="confirm-cost"):
        guard_check("create_vector_bucket", B, m, GuardContext(RID, "aws-ingest", False,
                                                                prechecked_not_found={(PFX, None)}))
    with pytest.raises(OwnershipError, match="pending"):
        guard_check("create_vector_bucket", B, [], c)
    with pytest.raises(OwnershipError, match="prefix"):
        guard_check("create_vector_bucket", {"vectorBucketName": "other"}, m, c)


def test_borrowed_bucket_accepts_only_borrowed_index():
    m = [entry("borrowed_bucket", "bb", owned=False), entry("main_index", "bb", PFX, status="pending")]
    c = ctx("aws-ingest", prechecked_not_found={("bb", PFX)})
    with pytest.raises(OwnershipError):
        guard_check("create_index", {"vectorBucketName": "bb", "indexName": PFX}, m, c)
    m[1]["role"] = "borrowed_index"
    guard_check("create_index", {"vectorBucketName": "bb", "indexName": PFX}, m, c)


def test_cleanup_exception_for_pending_but_never_conflict():
    pending = [entry("main_bucket", PFX), entry("main_index", PFX, PFX, status="pending")]
    guard_check("delete_index", BI, pending, ctx("aws-cleanup"))
    conflict = [entry("main_bucket", PFX), entry("main_index", PFX, PFX, status="create_failed",
                                                  error_code="ConflictException")]
    with pytest.raises(OwnershipError):
        guard_check("delete_index", BI, conflict, ctx("aws-cleanup"))
    with pytest.raises(OwnershipError):
        guard_check("delete_index", BI, pending, ctx("aws-query"))


def test_borrowed_abort_path_deletes_only_borrowed_index():
    guard_check("delete_index", BBI, MANIFEST, ctx("aws-ingest", borrowed_abort=True))
    with pytest.raises(OwnershipError):
        guard_check("delete_index", BI, MANIFEST, ctx("aws-ingest", borrowed_abort=True))


# ---------------------------------------------------------------- retries, charging, error classes

def run_state(tmp_path, cap=10.0, spend=0.0):
    return RunState(tmp_path / "run_metadata.json", {"run_id": RID, "hard_cap_usd": cap, "spend_tally_usd": spend,
                                                     "created_resources": [], "next_request_seq": 1})


QV = {"vectorBucketName": PFX, "indexName": PFX, "topK": 5, "queryVector": {"float32": [0.1, 0.2]}}


def test_two_429_then_success_three_attempts_timings_charges(tmp_path):
    """One timing per attempt and every sent attempt charged."""
    client, stub = stubbed_client()
    for _ in range(2):
        stub.add_client_error("query_vectors", service_error_code="TooManyRequestsException", http_status_code=429)
    stub.add_response("query_vectors", {"vectors": [{"key": "v-00001", "distance": 0.1}], "distanceMetric": "cosine"})
    rs = run_state(tmp_path)
    sleeps = []
    with stub:
        g = GuardedS3Vectors(client, rs, ctx("aws-query"), logging.getLogger("t"), sleep=sleeps.append)
        res = g.call("query_vectors", QV, cost_lines={"query_requests": 0.5})
    assert res.ok and res.attempts == 3 and len(res.timings) == 3
    assert [t["attempt"] for t in res.timings] == [1, 2, 3]
    assert rs.spend == pytest.approx(1.5) and len(sleeps) == 2
    assert json.loads((tmp_path / "run_metadata.json").read_text())["spend_tally_usd"] == pytest.approx(1.5)


def test_boto3_client_has_builtin_retries_disabled(dummy_profile):
    client = ac.make_boto3_client(dummy_profile, "us-east-1")
    assert client.meta.config.retries == {"total_max_attempts": 1, "mode": "standard"}
    assert client.meta.config.connect_timeout == 10 and client.meta.config.read_timeout == 60


def test_retries_exhausted_returns_error(tmp_path):
    client, stub = stubbed_client()
    for _ in range(ac.MAX_ATTEMPTS):
        stub.add_client_error("query_vectors", service_error_code="ServiceUnavailableException", http_status_code=503)
    rs = run_state(tmp_path)
    with stub:
        res = GuardedS3Vectors(client, rs, ctx("aws-query"), logging.getLogger("t"),
                               sleep=lambda s: None).call("query_vectors", QV, cost_lines={"q": 0.01})
    assert not res.ok and res.attempts == ac.MAX_ATTEMPTS and res.error_code == "ServiceUnavailableException"


def test_access_denied_fatal_by_default_recorded_when_tolerated(tmp_path):
    """AccessDenied is a recorded result in probe steps, fatal elsewhere."""
    client, stub = stubbed_client()
    msg = f"User: arn:aws:iam::{ACCT}:user/x is not authorized"
    stub.add_client_error("get_index", service_error_code="AccessDeniedException", service_message=msg,
                          http_status_code=403)
    stub.add_client_error("get_index", service_error_code="AccessDeniedException", service_message=msg,
                          http_status_code=403)
    rs = run_state(tmp_path)
    with stub:
        g = GuardedS3Vectors(client, rs, ctx("aws-probe"), logging.getLogger("t"), sleep=lambda s: None)
        with pytest.raises(AwsFatal):
            g.call("get_index", BI, cost_lines={"o": 0.0})
        res = g.call("get_index", BI, cost_lines={"o": 0.0}, tolerate_errors=True)
    assert not res.ok and res.error_code == "AccessDeniedException" and res.http_status == 403
    assert ACCT not in res.error_message


class _Raises:
    _bench_test_double = True

    def __init__(self, exc):
        self.exc = exc

    def get_index(self, **_):
        raise self.exc


@pytest.mark.parametrize("exc", [
    NoCredentialsError(),
    ClientError({"Error": {"Code": "ExpiredTokenException", "Message": "expired"}}, "GetIndex"),
    ClientError({"Error": {"Code": "UnrecognizedClientException", "Message": "bad"}}, "GetIndex"),
])
def test_credential_failures_always_fatal(tmp_path, exc):
    g = GuardedS3Vectors(_Raises(exc), run_state(tmp_path), ctx("aws-probe"), logging.getLogger("t"),
                         sleep=lambda s: None)
    with pytest.raises(AwsFatal, match="credential"):
        g.call("get_index", BI, cost_lines={}, tolerate_errors=True)


def test_programming_errors_are_not_swallowed(tmp_path):
    g = GuardedS3Vectors(_Raises(KeyError("bug")), run_state(tmp_path), ctx("aws-probe"), logging.getLogger("t"))
    with pytest.raises(KeyError):
        g.call("get_index", BI, cost_lines={}, tolerate_errors=True)


def test_tally_persists_across_instances_and_raises_at_cap(tmp_path):
    rs = run_state(tmp_path, cap=1.0)
    fake = FakeS3Vectors()
    fake.buckets[PFX] = {"default": "ENHANCED", "indexes": {}, "tags": {}}
    GuardedS3Vectors(fake, rs, ctx("aws-query"), logging.getLogger("t")).call(
        "get_vector_bucket", B, cost_lines={"o": 0.6})
    rs2 = RunState.load(tmp_path / "run_metadata.json")
    g2 = GuardedS3Vectors(fake, rs2, ctx("aws-query"), logging.getLogger("t"))
    with pytest.raises(ac.BudgetExceeded):
        g2.call("get_vector_bucket", B, cost_lines={"o": 0.6})
    assert len(fake.calls) == 1


# ---------------------------------------------------------------- redaction

def test_redaction_rules():
    obj = {"a": f"acct {ACCT} end", "arn": arn("b"), "run": RID, "bucket": PFX, "sha": "ab" * 32,
           "rid": "r000123", "f": 0.123456789012, "rtt_ns": 123456789012, "frac": "0.123456789012",
           "nested": [f":{ACCT}:"]}
    out = ac.redact(obj)
    assert out["a"] == "acct ************ end" and out["arn"] == "arn:REDACTED"
    assert out["run"] == RID and out["bucket"] == PFX and out["sha"] == "ab" * 32 and out["rid"] == "r000123"
    assert out["f"] == 0.123456789012 and out["rtt_ns"] == 123456789012 and out["frac"] == "0.123456789012"
    assert out["nested"] == [":************:"]
    assert obj["a"].endswith("end") and ACCT in obj["a"]  # input untouched


def test_redact_returns_copy_and_keeps_next_token():
    resp = {"nextToken": f"tok-{ACCT}-x", "vectors": []}
    copy = ac.redact(resp)
    assert resp["nextToken"] == f"tok-{ACCT}-x" and copy is not resp


def test_redacting_formatter_covers_tracebacks():
    stream = io.StringIO()
    log = logging.getLogger("redact-test")
    log.handlers.clear()
    h = logging.StreamHandler(stream)
    h.setFormatter(ac.RedactingFormatter("%(levelname)s %(message)s"))
    log.addHandler(h)
    log.propagate = False
    try:
        raise ClientError({"Error": {"Code": "AccessDeniedException",
                                     "Message": f"User: arn:aws:iam::{ACCT}:user/me is not authorized"}}, "GetIndex")
    except ClientError:
        log.exception("failed for %s", ACCT)
    text = stream.getvalue()
    assert "Traceback" in text and "AccessDeniedException" in text
    assert not TWELVE.search(text) and "arn:aws:iam" not in text


def test_excepthook_redacts_uncaught(monkeypatch):
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    stream = io.StringIO()
    ac.install_redacting_excepthook(stream)
    try:
        raise ClientError({"Error": {"Code": "AccessDenied", "Message": f"arn:aws:iam::{ACCT}:user/me"}}, "Op")
    except ClientError as exc:
        sys.excepthook(type(exc), exc, exc.__traceback__)
    assert "Traceback" in stream.getvalue() and not TWELVE.search(stream.getvalue())


def test_verify_redaction_scans_results_logs_docs_readme(tmp_path):
    for d in ("results/aws/captures", "figures", "logs", "docs"):
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / "results/aws/ok.json").write_text(json.dumps({"rid": RID, "f": 123456789012}))
    (tmp_path / "docs/ok.md").write_text("pattern arn:aws[^\"\\s]* and run " + RID + "\n")
    assert ac.verify_redaction(tmp_path) == []
    (tmp_path / "results/aws/bad.jsonl").write_text(json.dumps({"x": "ok"}) + "\n" + json.dumps({"s": ACCT}) + "\n")
    (tmp_path / "results/aws/captures/bad.txt").write_text(f"line\naccount {ACCT}\n")
    (tmp_path / "logs/run-x.log").write_text(f"Traceback ... arn:aws:iam::{ACCT}:user/me\n")
    (tmp_path / "docs/bad.md").write_text(f"{ACCT}\n")
    (tmp_path / "README.md").write_text(f"x\n\n{ACCT}\n")
    hits = ac.verify_redaction(tmp_path)
    assert ("results/aws/bad.jsonl", 2) in hits and ("results/aws/captures/bad.txt", 2) in hits
    assert ("logs/run-x.log", 1) in hits and ("docs/bad.md", 1) in hits and ("README.md", 3) in hits
