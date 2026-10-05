"""Pure cost model over a pricing snapshot and a request plan. No AWS dependency.

Conservative unit choices: GB = 1e9 bytes and TB = 1e12 bytes (more units than binary GiB/TiB),
the 128 KB PUT minimum as 128 KiB, every QueryVectors page billed as a full query, and the
first (highest) data-processed tier for every query.
"""
from __future__ import annotations

import hashlib
import math
import re

from .state import ConfigError, canonical_json, read_json

GB = 1e9
TB = 1e12
CAP_USD = 5.00
RATE_KEYS = (
    "storage_usd_per_gb_month", "put_usd_per_gb", "put_min_bytes", "other_requests_usd_per_1000",
    "query_requests_usd_per_1000", "data_processed_usd_per_tb_first_100k", "data_returned_usd_per_gb",
    "data_returned_min_bytes_per_result", "data_returned_free_bytes_per_query",
)
HOURS_PER_MONTH = 730.0
PROBE_VECTORS = 10
PROBE_DIM = 8


def load_pricing(path) -> dict:
    p = read_json(path)
    validate_pricing(p)
    return p


def validate_pricing(p: dict) -> None:
    if not p.get("source_url") or not p.get("fetched_utc"):
        raise ConfigError("pricing needs source_url and fetched_utc")
    rates = p.get("rates", {})
    for key in RATE_KEYS:
        v = rates.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
            raise ConfigError(f"pricing rate {key} missing or invalid")


# ---------------------------------------------------------------- per-request costs

def storage_usd(n_vectors: int, bpv: int, retention_days: float, p: dict) -> float:
    gb = n_vectors * bpv / GB
    return gb * p["rates"]["storage_usd_per_gb_month"] * (retention_days * 24.0 / HOURS_PER_MONTH)


def put_usd(batch_bytes: int, p: dict) -> float:
    r = p["rates"]
    return max(batch_bytes, r["put_min_bytes"]) / GB * r["put_usd_per_gb"]


def other_usd(count: int, p: dict) -> float:
    return count * p["rates"]["other_requests_usd_per_1000"] / 1000.0


def query_page_lines(index_vectors: int, bpv: int, p: dict) -> dict:
    """Cost lines of one QueryVectors page: request fee + data processed (index size, not matches)."""
    r = p["rates"]
    return {"query_requests": r["query_requests_usd_per_1000"] / 1000.0,
            "data_processed": index_vectors * bpv / TB * r["data_processed_usd_per_tb_first_100k"]}


def data_returned_usd(results: int, result_bytes: int, p: dict) -> float:
    """Per logical query: max(0, results * max(256 B, result bytes) - 512 KB) at $/GB."""
    r = p["rates"]
    billed = results * max(r["data_returned_min_bytes_per_result"], result_bytes)
    return max(0.0, billed - r["data_returned_free_bytes_per_query"]) / GB * r["data_returned_usd_per_gb"]


def worst_page_data_returned_usd(result_bytes: int, p: dict) -> float:
    """Pre-send check value: 100 results per page at the per-result minimum, no free allowance."""
    r = p["rates"]
    return 100 * max(r["data_returned_min_bytes_per_result"], result_bytes) / GB * r["data_returned_usd_per_gb"]


def result_bytes(key_bytes: float) -> int:
    """Bytes of one returned result with returnDistance=true, returnMetadata=false: key + float32."""
    return int(math.ceil(key_bytes)) + 4


# ---------------------------------------------------------------- request plan

def postfilter_pages(budgets: list[int], per_page: int) -> int:
    return sum(math.ceil(b / per_page) for b in budgets)


def request_plan(cfg: dict, manifest: dict, mode: str) -> dict:
    """Counts of every billed request the run may issue, grouped into priced lines."""
    if mode not in ("own", "borrowed"):
        raise ConfigError(f"unknown plan mode {mode}")
    ds, aws = cfg["dataset"], cfg["aws"]
    n, q = ds["n"], ds["n_queries"]
    nk = len(cfg["k_values"])
    nf = len(cfg["filters"])
    reps, preps = cfg["repeats"], cfg["postfilter_repeats"]
    per_page = cfg["topk_max_documented"]["results_per_page"]
    bpv = int(math.ceil(manifest["bytes"]["vector_data"] + manifest["bytes"]["key_mean"]
                        + manifest["bytes"]["filterable_metadata_mean"]))
    probe_bpv = PROBE_DIM * 4 + len("p-00") + len('{"tenant":"p0"}')
    rbytes = result_bytes(manifest["bytes"]["key_mean"])
    batch = aws["put_batch_size"]
    batches = math.ceil(n / batch)
    batch_bytes = [min(batch, n - i * batch) * bpv for i in range(batches)]
    list_pages = math.ceil(n / aws["list_vectors_max_results"])
    polls = aws["readiness_max_polls"]
    grid = nf * q * nk * reps
    ann = q * nk * reps
    pf_pages = q * postfilter_pages(cfg["budgets"], per_page) * preps
    pf_logical = [{"results": b, "count": q * preps} for b in cfg["budgets"]]
    captures = aws["capture_invocations"]

    lines = []

    def add(line, kind, count, index="main", phase=None, **extra):
        lines.append({"line": line, "kind": kind, "count": int(count), "index": index, "phase": phase, **extra})

    if mode == "own":
        add("storage main index (7-day retention, pre-charged)", "storage", 1)
        add("storage probe index", "storage", 1, index="probe")
        add("PutVectors main index (128 KB minimum per PUT)", "put", batches, batch_bytes=batch_bytes)
        add("PutVectors probe index", "put", 1, index="probe", batch_bytes=[PROBE_VECTORS * probe_bpv])
        add("(a) ENHANCED filtered queries", "query", grid, phase="aws-query")
        add("ANN reference (unfiltered)", "query", ann, phase="aws-query")
        add("(b) post-filter baseline pages", "query", pf_pages, phase="aws-query")
        add("probe step 13 + C100/C101", "query", 3, phase="aws-query")
        add("probe steps 10-12", "query", 3, index="probe", phase="aws-probe")
        add("readiness queries (worst case)", "query", polls, phase="aws-ingest")
        add("capture queries", "query", captures, phase="aws-capture")
        add("data returned (b) baseline", "data_returned", 0, logical=pf_logical, result_bytes=rbytes)
        probe_other = 2 + 1 + 2 + 1 + 1 + 1 + 2 + 1 + aws["probe_list_max_polls"] + 1
        add("probe other requests (pre-checks, create, tag, get, put-default, list polls, update)",
            "other", probe_other, phase="aws-probe")
        add("ingest other requests (pre-checks, create, tag, GetIndex x2, PutVectors)", "other",
            2 + 2 + 2 + 2 + batches, phase="aws-ingest")
        add("readiness ListVectors polling (worst case)", "other", polls * list_pages, phase="aws-ingest")
        add("capture GetIndex", "other", captures, phase="aws-capture")
        entries = 4
    else:
        add("storage borrowed-mode index (7-day retention, pre-charged)", "storage", 1)
        add("PutVectors borrowed-mode index", "put", batches, batch_bytes=batch_bytes)
        add("Stage A queryMode=CLASSIC grid", "query", grid, phase="aws-query")
        add("Stage B queryMode=ENHANCED grid", "query", grid, phase="aws-query")
        add("ANN reference (unfiltered)", "query", ann, phase="aws-query")
        add("(b) post-filter baseline pages", "query", pf_pages, phase="aws-query")
        add("readiness queries (worst case)", "query", polls, phase="aws-ingest")
        add("capture queries", "query", captures, phase="aws-capture")
        add("data returned (b) baseline", "data_returned", 0, logical=pf_logical, result_bytes=rbytes)
        add("ingest other requests (GetVectorBucket, pre-check, create, tag, GetIndex x2, PutVectors)",
            "other", 1 + 1 + 1 + 1 + 2 + batches, phase="aws-ingest")
        add("readiness ListVectors polling (worst case)", "other", polls * list_pages, phase="aws-ingest")
        add("capture GetIndex", "other", captures, phase="aws-capture")
        entries = 1
    cleanup = entries * 2 + entries + aws["cleanup_list_bucket_pages"] + entries
    add("cleanup (deletes, Get* per uncertain entry, verification lists)", "other", cleanup, phase="aws-cleanup")
    return {"mode": mode, "bytes_per_vector": bpv, "probe_bytes_per_vector": probe_bpv,
            "result_bytes": rbytes, "index_vectors": {"main": n, "probe": PROBE_VECTORS},
            "retention_days": aws["retention_days"], "lines": lines}


def phase_request_totals(plan: dict) -> dict:
    """Planned request count per phase, for the progress file."""
    totals: dict[str, int] = {}
    for ln in plan["lines"]:
        if ln["kind"] in ("query", "other") and ln["phase"]:
            totals[ln["phase"]] = totals.get(ln["phase"], 0) + ln["count"]
    return totals


def estimate(plan: dict, p: dict) -> dict:
    validate_pricing(p)
    out = []
    for ln in plan["lines"]:
        idx_n = plan["index_vectors"][ln["index"]]
        bpv = plan["bytes_per_vector"] if ln["index"] == "main" else plan["probe_bytes_per_vector"]
        if ln["kind"] == "storage":
            usd = storage_usd(idx_n, bpv, plan["retention_days"], p) * ln["count"]
        elif ln["kind"] == "put":
            usd = sum(put_usd(b, p) for b in ln["batch_bytes"])
        elif ln["kind"] == "query":
            page = query_page_lines(idx_n, bpv, p)
            usd = ln["count"] * (page["query_requests"] + page["data_processed"])
        elif ln["kind"] == "data_returned":
            usd = sum(g["count"] * data_returned_usd(g["results"], ln["result_bytes"], p) for g in ln["logical"])
        elif ln["kind"] == "other":
            usd = other_usd(ln["count"], p)
        else:
            raise ConfigError(f"unknown line kind {ln['kind']}")
        out.append({**{k: v for k, v in ln.items() if k != "batch_bytes"}, "usd": usd})
    base = sum(x["usd"] for x in out)
    return {"lines": out, "base_usd": base}


def cap_values(base_usd: float, prior_runs_spend_usd: float) -> dict:
    """approved = 2 x base; hard cap = min(approved, 5.00 - prior); refuse if 2 x base > 5.00 - prior."""
    remaining = CAP_USD - prior_runs_spend_usd
    approved = 2.0 * base_usd
    approvable = approved <= remaining
    return {"base_usd": round(base_usd, 6), "approved_usd": round(approved, 6),
            "prior_runs_spend_usd": round(prior_runs_spend_usd, 6),
            "remaining_project_budget_usd": round(remaining, 6),
            "hard_cap_usd": round(min(approved, remaining), 6), "approvable": approvable}


def plan_sha256(cfg: dict, pricing: dict, mode: str, plan: dict, prior_runs_spend_usd: float) -> str:
    blob = canonical_json({"experiment": cfg, "pricing": pricing, "mode": mode, "plan": plan,
                           "prior_runs_spend_usd": round(prior_runs_spend_usd, 6)})
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


FIXED_LINE = re.compile(r"^(\w+): (\S+)$")
FIXED_KEYS = ("base_usd", "approved_usd", "prior_runs_spend_usd", "hard_cap_usd", "plan_sha256", "plan_mode")


def parse_estimate_md(text: str) -> dict:
    """Parse the fixed-format `key: value` lines of cost_estimate.md."""
    out = {}
    for line in text.splitlines():
        m = FIXED_LINE.match(line)
        if m and m.group(1) in FIXED_KEYS:
            out[m.group(1)] = m.group(2)
    missing = [k for k in FIXED_KEYS if k not in out]
    if missing:
        raise ConfigError(f"cost_estimate.md lacks fixed lines: {missing}")
    for k in ("base_usd", "approved_usd", "prior_runs_spend_usd", "hard_cap_usd"):
        out[k] = float(out[k])
    return out
