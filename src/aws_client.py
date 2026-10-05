"""The only boto3 user: guarded client, ownership enforcement, retries, redaction, cost tally.

boto3 is imported lazily inside `make_boto3_client`, so offline phases never load it.

Error handling, redaction and retries:
- `call(..., tolerate_errors=True)` (probe steps, step 13, C100/C101) returns every service
  error, AccessDeniedException included, as a recorded result. Credential failures stay fatal.
- `RedactingFormatter` redacts the fully formatted record (tracebacks and stack_info
  included); `install_redacting_excepthook` covers uncaught exceptions; `verify_redaction` scans
  results/, figures/, logs/*.log, docs/*.md and README.md.
- The boto3 client is built with `total_max_attempts=1`, so this module's loop is the only
  retry layer; every sent attempt is charged and timed separately.
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
import logging
import random
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .state import AwsFatal, OwnershipError, BudgetExceeded, RunState, utc_now

# ---------------------------------------------------------------- redaction

ACCOUNT_RE = re.compile(r"(?<![0-9A-Za-z.])\d{12}(?![0-9A-Za-z.])")
ARN_RE = re.compile(r'arn:aws[^"\s]*')
# Raw-text verification flags ARNs that carry an account segment. Prose in docs/ legitimately
# names the redaction pattern itself ("arn:aws[^...]"), which is not an ARN.
ARN_WITH_ACCOUNT_RE = re.compile(r"arn:aws[\w-]*:[\w-]*:[\w-]*:\d{12}")
ACCOUNT_MASK = "************"


def redact_text(s: str) -> str:
    return ACCOUNT_RE.sub(ACCOUNT_MASK, ARN_RE.sub("arn:REDACTED", s))


def redact(obj: Any) -> Any:
    """Deep copy with string leaves redacted. Numbers are never touched; the input is never mutated."""
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, (bytes, bytearray)):
        return redact_text(bytes(obj).decode("utf-8", "replace"))
    return copy.deepcopy(obj)


class RedactingFormatter(logging.Formatter):
    """Redacts the complete formatted output, including exc_info tracebacks and stack_info."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_text(super().format(record))


def install_redacting_excepthook(stream=None) -> None:
    def hook(exc_type, exc, tb):
        out = stream if stream is not None else sys.stderr
        out.write(redact_text("".join(traceback.format_exception(exc_type, exc, tb))))
        out.flush()
    sys.excepthook = hook


def verify_redaction(root: Path) -> list[tuple[str, int]]:
    """Return (relative file, line) of every hit; never the matched value."""
    root = Path(root)
    files: list[Path] = []
    for sub in ("results", "figures"):
        base = root / sub
        if base.exists():
            files += [p for p in sorted(base.rglob("*"))
                      if p.is_file() and p.suffix in (".json", ".jsonl", ".md", ".csv", ".txt")]
    logs = root / "logs"
    if logs.exists():
        files += sorted(logs.glob("*.log"))
    if (root / "docs").exists():
        files += sorted((root / "docs").glob("*.md"))
    if (root / "README.md").exists():
        files.append(root / "README.md")
    hits: list[tuple[str, int]] = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix in (".json", ".jsonl"):
            if path.suffix == ".json":
                docs = [(1, text)]
            else:
                docs = [(i, line) for i, line in enumerate(text.splitlines(), 1) if line.strip()]
            for lineno, doc in docs:
                try:
                    parsed = json.loads(doc)
                except json.JSONDecodeError:
                    if ACCOUNT_RE.search(doc) or ARN_RE.search(doc):
                        hits.append((rel, lineno))
                    continue
                if any(ACCOUNT_RE.search(s) or ARN_RE.search(s) for s in _string_leaves(parsed)):
                    hits.append((rel, lineno))
        else:
            for lineno, line in enumerate(text.splitlines(), 1):
                if ACCOUNT_RE.search(line) or ARN_WITH_ACCOUNT_RE.search(line):
                    hits.append((rel, lineno))
    return hits


def _string_leaves(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _string_leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _string_leaves(v)


# ---------------------------------------------------------------- ownership guard

READ_ONLY = {"get_vector_bucket", "get_index", "list_vector_buckets", "list_indexes", "list_vectors",
             "query_vectors"}
CREATE = {"create_vector_bucket", "create_index"}
MUTATING = {"put_vectors", "delete_vectors", "update_index_mode", "delete_index", "delete_vector_bucket",
            "put_vector_bucket_default_index_mode", "tag_resource"}
INDEX_ROLES = {"main_index", "probe_index", "borrowed_index"}
BUCKET_ROLES = {"main_bucket", "probe_bucket", "borrowed_bucket"}
ARN_RESOURCE_RE = re.compile(r"^arn:aws[\w-]*:s3vectors:[a-z0-9-]+:\d{12}:bucket/([a-z0-9][a-z0-9-]*[a-z0-9])"
                             r"(?:/index/([a-z0-9][a-z0-9.-]*[a-z0-9]))?$")


def resource_prefix(run_id: str) -> str:
    return f"s3vectors-prefilter-bench-{run_id}"


def parse_arn_resource(arn: str) -> tuple[str, str | None]:
    m = ARN_RESOURCE_RE.match(arn or "")
    if not m:
        raise OwnershipError("unparseable resource ARN")
    return m.group(1), m.group(2)


@dataclass
class GuardContext:
    run_id: str
    phase: str
    confirm_cost: bool = False
    borrowed_abort: bool = False
    prechecked_not_found: set = field(default_factory=set)


def _target(api: str, params: dict) -> tuple[str, str | None]:
    if api == "tag_resource":
        return parse_arn_resource(params.get("resourceArn", ""))
    bucket = params.get("vectorBucketName")
    if not bucket:
        raise OwnershipError(f"{api}: calls must name the bucket explicitly")
    return bucket, params.get("indexName")


def guard_check(api: str, params: dict, manifest: list[dict], ctx: GuardContext) -> None:
    """Table-driven ownership guard. Raises OwnershipError; returns None when allowed."""
    if api in READ_ONLY:
        return
    prefix = resource_prefix(ctx.run_id)
    if api not in CREATE and api not in MUTATING:
        raise OwnershipError(f"{api} is not an allowed API")
    bucket, index = _target(api, params)
    name = index if index else bucket

    def find(b, i):
        return next((e for e in manifest if e["bucket"] == b and e.get("index") == i), None)

    if api in CREATE:
        if not name.startswith(prefix):
            raise OwnershipError(f"{api}: name lacks the run prefix")
        if not ctx.confirm_cost:
            raise OwnershipError(f"{api}: --confirm-cost not set")
        if (bucket, index) not in ctx.prechecked_not_found:
            raise OwnershipError(f"{api}: no NotFound pre-check for this name")
        entry = find(bucket, index)
        if entry is None or entry["status"] != "pending" or not entry["owned"]:
            raise OwnershipError(f"{api}: no pending manifest entry")
        if api == "create_index":
            parent = find(bucket, None)
            if parent is None or parent["status"] != "created":
                raise OwnershipError("create_index: parent bucket not in manifest")
            if parent["role"] == "borrowed_bucket" and entry["role"] != "borrowed_index":
                raise OwnershipError("create_index: only a borrowed_index may be created in a borrowed bucket")
            if parent["role"] != "borrowed_bucket" and not parent["owned"]:
                raise OwnershipError("create_index: parent bucket not owned")
        return

    entry = find(bucket, index)
    if entry is None:
        raise OwnershipError(f"{api}: target not in this run's manifest")
    cleanup_exception = (
        ctx.phase == "aws-cleanup" and api in ("delete_index", "delete_vector_bucket")
        and entry["owned"] and entry["status"] in ("pending", "create_failed")
        and entry.get("error_code") != "ConflictException" and entry["role"] != "borrowed_bucket")
    if not cleanup_exception and not (entry["owned"] and entry["status"] == "created"):
        raise OwnershipError(f"{api}: target is not an owned, created resource")
    if not name.startswith(prefix):
        raise OwnershipError(f"{api}: target name lacks the run prefix")
    role = entry["role"]
    allowed = {
        "put_vector_bucket_default_index_mode": role == "probe_bucket",
        "update_index_mode": (role == "probe_index" and params.get("indexMode") == "CLASSIC"
                              and ctx.phase == "aws-probe"),
        "put_vectors": role in INDEX_ROLES,
        "delete_vectors": False,
        "delete_index": role in INDEX_ROLES and (ctx.phase == "aws-cleanup"
                                                 or (ctx.borrowed_abort and role == "borrowed_index")),
        "delete_vector_bucket": role in ("main_bucket", "probe_bucket") and ctx.phase == "aws-cleanup",
        "tag_resource": role != "borrowed_bucket",
    }[api]
    if not allowed:
        raise OwnershipError(f"{api} refused for role {role} in phase {ctx.phase}")


# ---------------------------------------------------------------- error classification

RETRYABLE_CODES = {"ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException",
                   "InternalServerException", "RequestTimeoutException", "SlowDown"}
CREDENTIAL_CODES = {"ExpiredTokenException", "ExpiredToken", "UnrecognizedClientException",
                    "InvalidClientTokenId", "InvalidAccessKeyId", "SignatureDoesNotMatch"}
CREDENTIAL_CLASSES = {"NoCredentialsError", "PartialCredentialsError", "CredentialRetrievalError",
                      "ProfileNotFound", "TokenRetrievalError", "UnauthorizedSSOTokenError",
                      "SSOTokenLoadError"}
ACCESS_DENIED_CODES = {"AccessDeniedException", "AccessDenied"}
CONNECTION_CLASSES = {"EndpointConnectionError", "ConnectionClosedError", "ReadTimeoutError",
                      "ConnectTimeoutError", "ConnectionError", "HTTPClientError"}
CLIENT_SIDE_CLASSES = {"ParamValidationError"}
MAX_ATTEMPTS = 8
BACKOFF_BASE_S = 0.5
BACKOFF_CAP_S = 20.0


def error_info(exc: BaseException) -> tuple[str, str, int | None]:
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict) and "Error" in resp:
        err = resp.get("Error", {})
        status = resp.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return str(err.get("Code", "")), str(err.get("Message", "")), status
    return type(exc).__name__, str(exc), None


@dataclass
class CallResult:
    api: str
    ok: bool
    response: dict | None
    error_code: str | None
    error_message: str | None
    http_status: int | None
    attempts: int
    timings: list = field(default_factory=list)  # [{"attempt", "rtt_ns", "utc"}]


REGISTRY: list = []  # every client wrapped by GuardedS3Vectors in this process (test_no_real_aws)


class GuardedS3Vectors:
    """Single choke point for AWS calls: guard -> cap check -> charge -> send, per attempt."""

    def __init__(self, client, run: RunState, ctx: GuardContext, logger: logging.Logger,
                 enforce_cap: bool = True, sleep: Callable[[float], None] = time.sleep,
                 rng: random.Random | None = None, clock: Callable[[], int] = time.perf_counter_ns):
        self.client = client
        self.run = run
        self.ctx = ctx
        self.log = logger
        self.enforce_cap = enforce_cap
        self.sleep = sleep
        self.rng = rng or random.Random()
        self.clock = clock
        self.attempts_sent = 0
        REGISTRY.append(client)

    def _backoff(self, attempt: int) -> float:
        return self.rng.uniform(0.0, min(BACKOFF_CAP_S, BACKOFF_BASE_S * (2 ** (attempt - 1))))

    def call(self, api: str, params: dict, *, cost_lines: dict[str, float], worst_extra_usd: float = 0.0,
             tolerate_errors: bool = False) -> CallResult:
        """Send one logical call with retries. Every attempt is charged before it is sent.

        Returns a CallResult for service errors (ok=False). Raises AwsFatal for credential
        failures, and for AccessDenied unless `tolerate_errors` (probe steps).
        """
        guard_check(api, params, self.run.manifest, self.ctx)
        fn = getattr(self.client, api)
        timings = []
        attempt = 0
        while True:
            attempt += 1
            cost = sum(cost_lines.values())
            if self.enforce_cap and self.run.spend + cost + worst_extra_usd > self.run.hard_cap + 1e-12:
                raise BudgetExceeded(f"{api}: tally {self.run.spend:.6f} + {cost + worst_extra_usd:.6f} "
                                     f"would exceed hard cap {self.run.hard_cap:.6f}")
            self.run.charge(cost_lines)
            self.attempts_sent += 1
            utc = utc_now()
            t0 = self.clock()
            try:
                resp = fn(**params)
            except Exception as exc:  # classified below; nothing is swallowed silently
                rtt = self.clock() - t0
                timings.append({"attempt": attempt, "rtt_ns": int(rtt), "utc": utc})
                code, message, status = error_info(exc)
                cls = type(exc).__name__
                if is_unclassified(exc) and cls not in CREDENTIAL_CLASSES:
                    raise  # a bug, not an AWS outcome
                if cls in CREDENTIAL_CLASSES or code in CREDENTIAL_CODES:
                    raise AwsFatal(f"credential failure on {api}: {code}") from exc
                if code in ACCESS_DENIED_CODES and not tolerate_errors:
                    raise AwsFatal(f"access denied on {api}: {redact_text(message)}") from exc
                retryable = (code in RETRYABLE_CODES or cls in CONNECTION_CLASSES
                             or (status is not None and (status == 429 or status >= 500)))
                if retryable and attempt < MAX_ATTEMPTS:
                    self.log.warning("retry %s attempt %d after %s", api, attempt, code)
                    self.sleep(self._backoff(attempt))
                    continue
                return CallResult(api, False, None, code, redact_text(message), status, attempt, timings)
            rtt = self.clock() - t0
            timings.append({"attempt": attempt, "rtt_ns": int(rtt), "utc": utc})
            status = (resp or {}).get("ResponseMetadata", {}).get("HTTPStatusCode")
            return CallResult(api, True, resp, None, None, status, attempt, timings)

    def charge(self, lines: dict[str, float]) -> None:
        """Post-call corrections, e.g. data returned once per logical query."""
        if any(v > 0 for v in lines.values()):
            self.run.charge(lines)


def is_unclassified(exc: BaseException) -> bool:
    """True for exceptions that are neither AWS service errors nor known botocore client-side errors."""
    resp = getattr(exc, "response", None)
    name = type(exc).__name__
    return (not (isinstance(resp, dict) and "Error" in resp)
            and name not in CLIENT_SIDE_CLASSES and name not in CONNECTION_CLASSES)


def clean_response(resp: dict | None) -> dict | None:
    """Response for serialization: ResponseMetadata reduced to status and request id; then redacted."""
    if resp is None:
        return None
    out = {k: v for k, v in resp.items() if k != "ResponseMetadata"}
    md = resp.get("ResponseMetadata", {})
    out["ResponseMetadata"] = {"HTTPStatusCode": md.get("HTTPStatusCode"), "RequestId": md.get("RequestId")}
    return redact(out)


def make_boto3_client(profile: str, region: str):
    """The only place boto3 is imported. botocore retries are disabled."""
    import boto3
    from botocore.config import Config

    session = boto3.session.Session(profile_name=profile, region_name=region)
    cfg = Config(retries={"total_max_attempts": 1, "mode": "standard"}, connect_timeout=10, read_timeout=60)
    return session.client("s3vectors", config=cfg)


def boto3_versions() -> dict:
    import boto3
    import botocore
    return {"boto3": boto3.__version__, "botocore": botocore.__version__}
