"""CLI and phase runner. Run as `python -u -m src.runner <phase> ...`.

The BLAS thread variables are set before anything can import numpy (determinism setting).
Offline phases never import boto3; only aws_client.make_boto3_client does.
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import secrets  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from . import cost_model, metrics, reporting, simulator  # noqa: E402
from . import dataset as ds_mod  # noqa: E402
from . import filters as flt  # noqa: E402
from . import ground_truth as gt_mod  # noqa: E402
from .aws_client import (CallResult, GuardContext, GuardedS3Vectors, RedactingFormatter, clean_response,  # noqa: E402
                         install_redacting_excepthook, redact, resource_prefix, verify_redaction)
from .state import (AwsFatal, ConfigError, IntegrityError, ModeMismatch, PhaseLock, PhaseStopped,  # noqa: E402
                    RunState, UsageError, append_jsonl, atomic_write_json, atomic_write_jsonl,
                    atomic_write_text, dumps, new_run_guard, prior_runs_spend_usd,
                    read_json, read_jsonl, round_sig, utc_now, write_progress)

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFLINE_PHASES = ("generate", "ground-truth", "simulate", "report", "offline-all")
AWS_PHASES = ("aws-probe", "aws-ingest", "aws-query", "aws-capture", "aws-cleanup")
ALL_PHASES = OFFLINE_PHASES + ("estimate", "verify-redaction") + AWS_PHASES
ALLOWED_FLAGS = {
    **{p: {"out_root"} for p in OFFLINE_PHASES},
    "estimate": {"borrowed"},
    "verify-redaction": set(),
    "aws-probe": {"aws", "profile", "region", "confirm_cost"},
    "aws-ingest": {"aws", "profile", "region", "confirm_cost", "run_id", "borrowed_bucket", "confirm_index_name"},
    "aws-query": {"aws", "profile", "region", "confirm_cost", "run_id"},
    "aws-capture": {"aws", "profile", "region", "run_id"},
    "aws-cleanup": {"aws", "profile", "region", "run_id"},
}
FLAG_NAMES = ("aws", "profile", "region", "confirm_cost", "run_id", "borrowed_bucket", "confirm_index_name",
              "borrowed", "out_root")
REGION_RE = re.compile(r"^[a-z]{2}(-[a-z]+)+-\d$")
RUN_ID_RE = re.compile(r"^\d{8}t\d{4}-[0-9a-f]{4}$")
BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$")
TAG_PROJECT = "s3vectors-prefilter-bench"
EXIT_FAIL, EXIT_USAGE, EXIT_STOPPED = 1, 2, 3
PROBE_DIM = cost_model.PROBE_DIM
PROBE_VECTORS = cost_model.PROBE_VECTORS


# ================================================================ config

def load_config(path: Path) -> dict:
    cfg = read_json(path)
    ds = cfg["dataset"]
    if not (isinstance(ds["n"], int) and ds["n"] >= 1):
        raise ConfigError("dataset.n must be >= 1")
    if not (1 <= ds["dim"] <= 4096):
        raise ConfigError("dataset.dim must be in 1..4096")
    for field in ds["metadata_fields"]:
        ds_mod.field_labels(field, ds["n"])  # raises on count mismatch
    ks = cfg["k_values"]
    if not ks or any(not isinstance(k, int) or k < 1 or k > 100 for k in ks):
        raise ConfigError("k_values must be positive integers <= 100")
    tk = cfg["topk_max_documented"]
    if not (isinstance(tk.get("value"), int) and tk["value"] > 0 and tk.get("source_url") and tk.get("fetched_utc")):
        raise ConfigError("topk_max_documented needs value, source_url, fetched_utc")
    if not (isinstance(tk.get("results_per_page"), int) and tk["results_per_page"] > 0):
        raise ConfigError("topk_max_documented.results_per_page must be a positive integer")
    if cfg["budgets"] != budget_rule(tk["value"]):
        raise ConfigError(f"budgets must equal {budget_rule(tk['value'])} (budget rule)")
    if not (isinstance(cfg["aws"]["list_vectors_max_results"], int) and cfg["aws"]["list_vectors_max_results"] > 0):
        raise ConfigError("aws.list_vectors_max_results must be a positive integer")
    if cfg["repeats"] < 1 or cfg["postfilter_repeats"] < 1:
        raise ConfigError("repeats must be >= 1")
    ids = [f["id"] for f in cfg["filters"]]
    if len(ids) != len(set(ids)) or "NOFILTER" in ids or {"C100", "C101"} & set(ids):
        raise ConfigError("filter ids must be unique and not reserved")
    for f in cfg["filters"]:
        flt.validate(f["filter"])
        if flt.count_constraints(f["filter"]) > flt.MAX_CONSTRAINTS:
            raise ConfigError(f"filter {f['id']} has more than {flt.MAX_CONSTRAINTS} constraints")
    simulator.validate(cfg["simulator"])
    return cfg


def budget_rule(max_topk: int) -> list[int]:
    return sorted({b for b in (100, 1000, 10000) if b <= max_topk} | {max_topk})


# ================================================================ paths and logging

class Paths:
    def __init__(self, root: Path, out_root: Path | None = None):
        self.root = Path(root)
        out = Path(out_root) if out_root else self.root
        if not out.is_absolute():
            out = self.root / out
        self.out = out
        self.config = self.root / "benchmark" / "experiment.json"
        self.pricing = self.root / "benchmark" / "pricing_us-east-1.json"
        self.data = out / "data"
        self.sim = out / "results" / "simulator"
        self.gt = out / "results" / "aws" / "ground_truth.jsonl"
        self.processed = out / "results" / "processed"
        self.figures = out / "figures"
        self.aws = self.root / "results" / "aws"
        self.logs = self.root / "logs"
        self.screens = self.aws / "captures"
        self.shot_list = self.aws / "SHOT_LIST.md"
        self.estimate_md = self.aws / "cost_estimate.md"
        self.meta = self.aws / "run_metadata.json"
        self.probe = self.aws / "probe_classic.json"
        self.partial = self.aws / "_partial"
        self.lock = self.aws / ".lock"


def make_logger(log_file: Path | None = None) -> logging.Logger:
    log = logging.getLogger("bench")
    log.setLevel(logging.INFO)
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h)
        h.close()
    fmt = RedactingFormatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if log_file is not None:
        add_file_handler(log, log_file)
    return log


def add_file_handler(log: logging.Logger, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)


def code_sha256() -> str:
    return hashlib.sha256((Path(__file__).parent / "dataset.py").read_bytes()).hexdigest()


def git_commit(root: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def package_versions() -> dict:
    from importlib import metadata
    out = {"python": sys.version.split()[0]}
    for pkg in ("boto3", "botocore", "numpy", "pandas", "matplotlib", "pytest"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


# ================================================================ offline phases

def phase_generate(p: Paths, cfg: dict, log) -> None:
    data = ds_mod.build(cfg["dataset"])
    counts = {}
    for f in flt.scored_filters(cfg) + flt.constraint_probe_filters(cfg):
        counts[f["id"]] = int(flt.mask(f["filter"], data.columns, data.n).sum())
    ds_mod.write(data, cfg["dataset"], p.data, counts, code_sha256())
    log.info("generate: wrote %s (N=%d, dim=%d)", p.data, data.n, cfg["dataset"]["dim"])


def phase_ground_truth(p: Paths, cfg: dict, log) -> None:
    data, _ = ds_mod.load(p.data)
    rows = gt_mod.compute(data, cfg)
    p.gt.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(p.gt, [round_sig(r) for r in rows])
    log.info("ground-truth: %d rows", len(rows))


def score_sim_rows(rows: list[dict], data, gt_index, masks) -> list[dict]:
    out = []
    for r in rows:
        m = metrics.score_request(r["returned_keys"], gt_index[(r["query_id"], r["filter_id"])], r["k"],
                                  masks[r["filter_id"]], data.key_index)
        out.append({**r, **m, "ok": True})
    return out


def phase_simulate(p: Paths, cfg: dict, log) -> None:
    data, manifest = ds_mod.load(p.data)
    gt_index = gt_mod.index_rows(read_jsonl(p.gt))
    rows = simulator.run(data, cfg)
    masks = {f["id"]: flt.mask(f["filter"], data.columns, data.n) for f in cfg["filters"]}
    scored = score_sim_rows(rows, data, gt_index, masks)
    agg_input = [{**r, "keys": r["returned_keys"]} for r in scored]
    aggregates = metrics.aggregate(agg_input, cfg["dataset"]["seed"])
    counters: dict[tuple, list] = {}
    for r in scored:
        counters.setdefault(metrics.group_key(r), []).append((r["candidates_scored"], r["budget_units_used"]))
    for a in aggregates:
        vals = counters[metrics.group_key(a)]
        a["candidates_scored_mean"] = float(np.mean([v[0] for v in vals]))
        a["budget_units_used_mean"] = float(np.mean([v[1] for v in vals]))
    p.sim.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(p.sim / "results.jsonl", [round_sig(r) for r in scored])
    atomic_write_json(p.sim / "metrics.json", round_sig({"evidence": "SIMULATED", "aggregates": aggregates}))
    atomic_write_json(p.sim / "run_metadata.json", round_sig({
        "evidence": "SIMULATED", "params": cfg["simulator"], "k_values": cfg["k_values"],
        "seed": cfg["dataset"]["seed"], "dataset_sha256": manifest["sha256"],
        "limits": simulator.LIMITS, "postfilter_pairs_note": simulator.POSTFILTER_PAIRS_NOTE,
        "n_requests": len(scored)}))
    log.info("simulate: %d SIMULATED requests", len(scored))


def phase_report(p: Paths, cfg: dict, log) -> None:
    reporting.report(p, cfg)
    log.info("report: wrote %s and %s", p.processed, p.figures)


def phase_estimate(p: Paths, cfg: dict, log, borrowed: bool) -> int:
    pricing = cost_model.load_pricing(p.pricing)
    manifest = read_json(p.data / "manifest.json")
    mode = "borrowed" if borrowed else "own"
    plan = cost_model.request_plan(cfg, manifest, mode)
    est = cost_model.estimate(plan, pricing)
    prior = prior_runs_spend_usd(p.aws, None)
    caps = cost_model.cap_values(est["base_usd"], prior)
    if not caps["approvable"]:
        log.error("estimate refused: 2 x base_usd %.6f exceeds the remaining project budget %.6f "
                  "(5.00 - prior_runs_spend_usd %.6f). Reduce the plan.", est["base_usd"],
                  caps["remaining_project_budget_usd"], prior)
        return EXIT_USAGE
    digest = cost_model.plan_sha256(cfg, pricing, mode, plan, prior)
    atomic_write_text(p.estimate_md, reporting.cost_estimate_md(cfg, pricing, plan, est, caps, digest, mode))
    log.info("estimate: base_usd=%.6f hard_cap_usd=%.6f -> %s (PAUSE for user approval)",
             caps["base_usd"], caps["hard_cap_usd"], p.estimate_md)
    return 0


# ================================================================ CLI validation

def validate_flags(phase: str, args) -> None:
    given = {n for n in FLAG_NAMES if getattr(args, n) not in (None, False)}
    extra = given - ALLOWED_FLAGS[phase]
    if extra:
        raise UsageError(f"flag(s) not allowed on {phase}: {sorted('--' + e.replace('_', '-') for e in extra)}")
    if phase not in AWS_PHASES:
        return
    if not (args.aws and args.profile and args.region):
        raise UsageError(f"{phase} requires --aws --profile --region")
    refused = {p.strip() for p in os.environ.get("BENCH_REFUSED_PROFILES", "").split(",") if p.strip()}
    if args.profile in refused:
        raise UsageError(f"profile {args.profile} is refused (BENCH_REFUSED_PROFILES)")
    if not REGION_RE.match(args.region):
        raise UsageError("invalid --region")
    if phase in ("aws-probe", "aws-ingest", "aws-query") and not args.confirm_cost:
        raise UsageError(f"{phase} requires --confirm-cost (only after explicit user approval)")
    borrowed_inv1 = phase == "aws-ingest" and args.borrowed_bucket and not args.run_id
    if phase == "aws-probe" or borrowed_inv1:
        if args.run_id:
            raise UsageError("--run-id is not allowed on an id-generating invocation")
    elif not args.run_id:
        raise UsageError(f"{phase} requires --run-id")
    if args.run_id and not RUN_ID_RE.match(args.run_id):
        raise UsageError("invalid --run-id format")
    if args.borrowed_bucket and not (3 <= len(args.borrowed_bucket) <= 63 and BUCKET_RE.match(args.borrowed_bucket)):
        raise UsageError("invalid --borrowed-bucket")
    if args.confirm_index_name and not (args.borrowed_bucket and args.run_id):
        raise UsageError("--confirm-index-name is only allowed on borrowed aws-ingest invocation 2")
    if phase == "aws-ingest" and args.borrowed_bucket and args.run_id and not args.confirm_index_name:
        raise UsageError("borrowed aws-ingest invocation 2 requires --confirm-index-name")


# ================================================================ AWS phase machinery

class Throttle:
    def __init__(self, rate_per_s: float, sleep, clock=time.monotonic):
        self.interval = 1.0 / rate_per_s
        self.sleep = sleep
        self.clock = clock
        self.next_t = 0.0

    def wait(self) -> None:
        now = self.clock()
        if now < self.next_t:
            self.sleep(self.next_t - now)
            now = self.next_t
        self.next_t = now + self.interval


class Recorder:
    """Appends queries/responses/timings rows (redacted) into one directory."""

    def __init__(self, directory: Path, run_id: str):
        self.dir = Path(directory)
        self.run_id = run_id

    def _row(self, name: str, row: dict) -> None:
        append_jsonl(self.dir / name, [redact({"run_id": self.run_id, **row})])

    def query(self, row: dict) -> None:
        self._row("queries.jsonl", row)

    def response(self, row: dict) -> None:
        self._row("responses.jsonl", row)

    def timings(self, request_id: str, page: int, timing_phase: str, evidence_class: str, timings: list) -> None:
        for t in timings:
            self._row("timings.jsonl", {"request_id": request_id, "page": page, "attempt": t["attempt"],
                                        "rtt_ns": t["rtt_ns"], "phase": timing_phase, "utc": t["utc"],
                                        "evidence_class": evidence_class})


class AwsPhase:
    def __init__(self, phase: str, p: Paths, cfg: dict, args, log, client_factory, sleep):
        self.phase = phase
        self.p = p
        self.cfg = cfg
        self.args = args
        self.log = log
        self.client_factory = client_factory
        self.sleep = sleep
        self.run: RunState | None = None
        self.client: GuardedS3Vectors | None = None
        self.started = time.monotonic()
        self.done = 0
        self.total = 0
        self.pricing = cost_model.load_pricing(p.pricing)
        self.manifest_data = read_json(p.data / "manifest.json")
        self.plan = None
        self._versions = None
        self.last_request_id = None

    # ---------------- run state and estimate

    def estimate_fixed(self) -> dict:
        if not self.p.estimate_md.exists():
            raise UsageError("results/aws/cost_estimate.md is missing; run estimate and get approval")
        return cost_model.parse_estimate_md(self.p.estimate_md.read_text(encoding="utf-8"))

    def check_plan(self, mode: str, current_run_id: str) -> dict:
        fixed = self.estimate_fixed()
        if fixed["plan_mode"] != mode:
            raise UsageError(f"cost_estimate.md is for plan mode {fixed['plan_mode']}, run is {mode}")
        plan = cost_model.request_plan(self.cfg, self.manifest_data, mode)
        prior = prior_runs_spend_usd(self.p.aws, current_run_id)
        digest = cost_model.plan_sha256(self.cfg, self.pricing, mode, plan, prior)
        if digest != fixed["plan_sha256"]:
            raise UsageError("plan_sha256 mismatch: config, pricing, or prior-run spend changed since the "
                             "approved estimate; re-run estimate and get a new approval")
        self.plan = plan
        return fixed

    def new_run(self, mode: str) -> RunState:
        archived = new_run_guard(self.p.aws)
        if archived:
            self.log.info("archived previous run %s to results/aws/previous/", archived)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M") + "-" + secrets.token_hex(2)
        fixed = self.check_plan(mode, run_id)
        meta = {
            "run_id": run_id, "mode": mode, "region": self.args.region, "profile": self.args.profile,
            "start_utc": utc_now(), "end_utc": None, "git_commit": git_commit(self.p.root),
            "package_versions": package_versions(),
            "dataset_manifest_sha256": hashlib.sha256((self.p.data / "manifest.json").read_bytes()).hexdigest(),
            "plan_sha256": fixed["plan_sha256"], "topk_max_documented": self.cfg["topk_max_documented"],
            "created_resources": [], "next_request_seq": 1, "phases_completed": [],
            "spend_tally_usd": 0.0, "spend_by_line_usd": {}, "base_usd": fixed["base_usd"],
            "approved_usd": fixed["approved_usd"], "prior_runs_spend_usd": fixed["prior_runs_spend_usd"],
            "hard_cap_usd": fixed["hard_cap_usd"], "classic_index_available": None, "classic_source": None,
            "classic_query_accepted_on_main": None, "stopped_reason": None, "phase_durations_s": {},
            "attempts_sent": 0,
        }
        self.run = RunState(self.p.meta, meta, self.sleep)
        self.run.save()
        add_file_handler(self.log, self.p.logs / f"run-{run_id}.log")
        self.log.info("new run id %s (mode %s, hard_cap_usd %.6f)", run_id, mode, meta["hard_cap_usd"])
        print(f"RUN_ID {run_id}", flush=True)
        return self.run

    def load_run(self) -> RunState:
        if not self.p.meta.exists():
            raise UsageError("results/aws/run_metadata.json is missing")
        self.run = RunState.load(self.p.meta)
        self.run.sleep = self.sleep
        if self.run.run_id != self.args.run_id:
            raise UsageError("--run-id does not match run_metadata.json")
        add_file_handler(self.log, self.p.logs / f"run-{self.run.run_id}.log")
        return self.run

    def connect(self, enforce_cap: bool = True, borrowed_abort: bool = False) -> None:
        raw = self.client_factory(self.args.profile, self.args.region)
        ctx = GuardContext(run_id=self.run.run_id, phase=self.phase, confirm_cost=bool(self.args.confirm_cost),
                           borrowed_abort=borrowed_abort)
        self.client = GuardedS3Vectors(raw, self.run, ctx, self.log, enforce_cap=enforce_cap, sleep=self.sleep)

    def complete(self) -> None:
        pc = self.run.meta.setdefault("phases_completed", [])
        if self.phase not in pc:
            pc.append(self.phase)
        self.finish_durations()

    def finish_durations(self) -> None:
        d = self.run.meta.setdefault("phase_durations_s", {})
        d[self.phase] = round(d.get(self.phase, 0.0) + time.monotonic() - self.started, 1)
        self.run.meta["end_utc"] = utc_now()
        self.run.meta["attempts_sent"] = self.run.meta.get("attempts_sent", 0) + (
            self.client.attempts_sent if self.client else 0)
        self.run.save()

    def progress(self, status: str = "running") -> None:
        if self.run is None:
            return
        write_progress(self.p.logs / f"progress-{self.run.run_id}.json", phase=self.phase, done=self.done,
                       total=self.total, elapsed_s=time.monotonic() - self.started, spend=round(self.run.spend, 6),
                       cap=self.run.hard_cap, status=status)

    def tick(self) -> None:
        self.done += 1
        if self.done % 100 == 0:
            self.progress()

    def versions(self) -> dict:
        if self._versions is None:
            try:
                from .aws_client import boto3_versions
                self._versions = boto3_versions()
            except ImportError:
                self._versions = {"boto3": None, "botocore": None}
        return self._versions

    # ---------------- cost helpers

    def bpv(self, index_kind: str) -> int:
        return self.plan["bytes_per_vector"] if index_kind == "main" else self.plan["probe_bytes_per_vector"]

    def n_index(self, index_kind: str) -> int:
        return self.plan["index_vectors"][index_kind]

    def other_cost(self) -> dict:
        return {"other_requests": cost_model.other_usd(1, self.pricing)}

    def page_cost(self, index_kind: str) -> dict:
        return cost_model.query_page_lines(self.n_index(index_kind), self.bpv(index_kind), self.pricing)

    def worst_page(self) -> float:
        return cost_model.worst_page_data_returned_usd(self.plan["result_bytes"], self.pricing)

    def other(self, api: str, params: dict, tolerate: bool = False):
        self.last_request_id = self.run.reserve_request_id()
        res = self.client.call(api, params, cost_lines=self.other_cost(), tolerate_errors=tolerate)
        self.tick()
        return res

    # ---------------- resources

    def tags(self) -> dict:
        return {"project": TAG_PROJECT, "run_id": self.run.run_id}

    def _get(self, bucket: str, index: str | None):
        if index:
            return self.other("get_index", {"vectorBucketName": bucket, "indexName": index}, tolerate=True)
        return self.other("get_vector_bucket", {"vectorBucketName": bucket}, tolerate=True)

    def ensure_resource(self, rtype: str, role: str, bucket: str, index: str | None, extra_params: dict,
                        index_kind: str | None = None):
        """Create (or reuse on rerun) one owned resource with the pre-check and pending entry.

        Returns the CallResult of the create, or None when an existing `created` entry is reused.
        """
        entry = self.run.find(bucket, index)
        if entry is not None and entry["status"] == "created":
            return None
        if entry is not None and entry["status"] == "pending":
            res = self._get(bucket, index)
            if res.ok:
                self.run.upsert(rtype=rtype, role=role, owned=True, bucket=bucket, index=index, status="created")
                return None
            if res.error_code != "NotFoundException":
                raise AwsFatal(f"cannot resolve pending {role}: {res.error_code}")
        if entry is not None and entry["status"] == "create_failed":
            raise AwsFatal(f"{role} previously failed to create ({entry.get('error_code')}); start a new run")
        pre = self._get(bucket, index)
        if pre.ok:
            self.run.upsert(rtype=rtype, role=role, owned=True, bucket=bucket, index=index,
                            status="create_failed", error_code="ConflictException")
            raise AwsFatal(f"{role} name already exists before this run; it is never adopted")
        if pre.error_code != "NotFoundException":
            raise AwsFatal(f"pre-check for {role} failed: {pre.error_code}")
        self.client.ctx.prechecked_not_found.add((bucket, index))
        self.run.upsert(rtype=rtype, role=role, owned=True, bucket=bucket, index=index, status="pending")
        api = "create_index" if index else "create_vector_bucket"
        params = {"vectorBucketName": bucket, **({"indexName": index} if index else {}), **extra_params,
                  "tags": self.tags()}
        storage = 0.0
        if index:
            storage = cost_model.storage_usd(self.n_index(index_kind), self.bpv(index_kind),
                                             self.plan["retention_days"], self.pricing)
        self.run.reserve_request_id()
        res = self.client.call(api, params, cost_lines=self.other_cost(), worst_extra_usd=storage,
                               tolerate_errors=self.phase == "aws-probe")
        self.tick()
        if res.ok:
            self.run.upsert(rtype=rtype, role=role, owned=True, bucket=bucket, index=index, status="created")
            if storage:
                self.client.charge({"storage": storage})
        else:
            self.run.upsert(rtype=rtype, role=role, owned=True, bucket=bucket, index=index,
                            status="create_failed", error_code=res.error_code)
        return res

    def put_batches(self, bucket: str, index: str, vectors: np.ndarray, metadata: list[dict], recorder: Recorder,
                    index_kind: str, batch: int, rate: float, evidence_class: str = "ingest") -> None:
        throttle = Throttle(rate, self.sleep)
        for i in range(0, len(metadata), batch):
            chunk = metadata[i:i + batch]
            vecs = vectors[i:i + batch]
            if not np.all(np.isfinite(vecs)) or np.any(np.linalg.norm(vecs, axis=1) == 0):
                raise IntegrityError("non-finite or zero vector in upload batch")
            payload = [{"key": m["key"], "data": {"float32": [float(x) for x in v]},
                        "metadata": {k: val for k, val in m.items() if k != "key"}} for m, v in zip(chunk, vecs)]
            nbytes = len(chunk) * self.bpv(index_kind)
            rid = self.run.reserve_request_id()
            throttle.wait()
            res = self.client.call("put_vectors", {"vectorBucketName": bucket, "indexName": index,
                                                   "vectors": payload},
                                   cost_lines={"put": cost_model.put_usd(nbytes, self.pricing)})
            self.tick()
            recorder.query({"request_id": rid, "evidence_class": evidence_class, "api": "PutVectors",
                            "batch_index": i // batch, "key_first": chunk[0]["key"], "key_last": chunk[-1]["key"],
                            "count": len(chunk)})
            recorder.response({"request_id": rid, "page": 1, "ok": res.ok, "error_code": res.error_code,
                               "error_message": res.error_message, "attempts": res.attempts,
                               "http_status": res.http_status})
            if not res.ok:
                raise AwsFatal(f"PutVectors batch {i // batch} failed: {res.error_code}")

    # ---------------- query helper

    def query(self, *, recorder: Recorder, bucket: str, index: str, index_kind: str, evidence_class: str,
              vector, top_k: int, filt, filter_id: str | None, query_id: str, k=None, budget=None, repeat=None,
              query_mode=None, order_index=None, timing_phase="other", tolerate=False, throttle=None,
              key_index=None) -> dict:
        rid = self.run.reserve_request_id()
        params = {"vectorBucketName": bucket, "indexName": index, "topK": int(top_k),
                  "queryVector": {"float32": [float(x) for x in vector]},
                  "returnDistance": True, "returnMetadata": False}
        if filt is not None:
            params["filter"] = filt
        if query_mode:
            params["queryMode"] = query_mode
        recorder.query({"request_id": rid, "evidence_class": evidence_class, "query_id": query_id,
                        "filter_id": filter_id, "filter": filt, "k": k, "top_k": int(top_k), "budget": budget,
                        "repeat": repeat, "query_mode": query_mode, "order_index": order_index})
        per_page = self.cfg["topk_max_documented"]["results_per_page"]
        max_pages = math.ceil(top_k / per_page) + 1
        pages, token, seen_tokens = [], None, set()
        out = {"request_id": rid, "ok": False, "pages": pages, "error_code": None, "error_message": None,
               "attempts": [], "http_status": None, "response": None}
        while True:
            if len(pages) >= max_pages:
                out.update(error_code="PaginationIntegrityError", error_message="page cap exceeded")
                break
            pp = dict(params)
            if token:
                pp["nextToken"] = token
            if throttle:
                throttle.wait()
            res = self.client.call("query_vectors", pp, cost_lines=self.page_cost(index_kind),
                                   worst_extra_usd=self.worst_page(), tolerate_errors=tolerate)
            self.tick()
            page_no = len(pages) + 1
            recorder.timings(rid, page_no, timing_phase, evidence_class, res.timings)
            out["attempts"].append(res.attempts)
            out["http_status"] = res.http_status
            if not res.ok:
                recorder.response({"request_id": rid, "page": page_no, "ok": False, "error_code": res.error_code,
                                   "error_message": res.error_message, "attempts": res.attempts,
                                   "http_status": res.http_status, "keys": [], "distances": [],
                                   "next_token_present": False})
                out.update(error_code=res.error_code, error_message=res.error_message)
                return out
            body = res.response
            keys = [v["key"] for v in body.get("vectors", [])]
            dists = [v.get("distance") for v in body.get("vectors", [])]
            if key_index is not None:
                if any(kk not in key_index for kk in keys):
                    raise IntegrityError(f"{rid}: returned key not in the dataset")
                if any(d is None or not math.isfinite(float(d)) for d in dists):
                    raise IntegrityError(f"{rid}: non-finite or missing distance")
            token = body.get("nextToken")
            recorder.response({"request_id": rid, "page": page_no, "ok": True, "error_code": None,
                               "error_message": None, "attempts": res.attempts, "http_status": res.http_status,
                               "keys": keys, "distances": dists, "next_token_present": bool(token)})
            pages.append({"keys": keys, "distances": dists, "next_token": token})
            out["response"] = clean_response(body)
            if not token:
                break
            if token in seen_tokens:
                out.update(error_code="PaginationIntegrityError", error_message="repeated nextToken")
                break
            seen_tokens.add(token)
        results = sum(len(pg["keys"]) for pg in pages)
        self.client.charge({"data_returned": cost_model.data_returned_usd(results, self.plan["result_bytes"],
                                                                          self.pricing)})
        if out["error_code"]:
            recorder.response({"request_id": rid, "page": None, "ok": False, "error_code": out["error_code"],
                               "error_message": out["error_message"], "attempts": None})
            return out
        try:
            metrics.check_pagination(pages, top_k)
        except metrics.PaginationError as exc:
            out.update(error_code="PaginationIntegrityError", error_message=str(exc))
            recorder.response({"request_id": rid, "page": None, "ok": False, "error_code": out["error_code"],
                               "error_message": out["error_message"], "attempts": None})
            return out
        out["ok"] = True
        return out

    # ---------------- probe records

    def probe_load(self) -> dict:
        if not self.p.probe.exists():
            raise IntegrityError("probe_classic.json is missing")
        doc = read_json(self.p.probe)
        if doc.get("run_id") != self.run.run_id:
            raise IntegrityError("probe_classic.json run_id mismatch")
        return doc

    def probe_append(self, record: dict) -> None:
        doc = self.probe_load()
        doc["records"].append(record)
        atomic_write_json(self.p.probe, doc, self.sleep)

    def probe_record(self, step: str, api: str, request: dict, res, expectation: str | None, source: str | None,
                     **extra) -> dict:
        return redact({"step": step, "api": api, "request": request, "ok": res.ok, "http_status": res.http_status,
                       "error_code": res.error_code, "error_message": res.error_message,
                       "response": clean_response(res.response), "attempts": res.attempts,
                       "timestamp_utc": utc_now(), "boto3_version": self.versions()["boto3"],
                       "documented_expectation": expectation, "expectation_source": source, **extra})

    @staticmethod
    def skipped(step: str, api: str, reason: str) -> dict:
        return {"step": step, "api": api, "ok": None, "skipped_reason": reason, "timestamp_utc": utc_now()}


# ================================================================ aws-probe

def probe_vectors(seed: int) -> np.ndarray:
    rng = np.random.default_rng(ds_mod.stream(seed, "queries", PROBE_DIM))
    x = rng.normal(0.0, 1.0, (PROBE_VECTORS, PROBE_DIM))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def classic_decision(records: list[dict]) -> tuple[bool, str | None]:
    """CLASSIC is obtainable via bucket default, UpdateIndexMode, or an accepted queryMode."""
    by = {r["step"]: r for r in records}

    def mode(step):
        r = by.get(step) or {}
        return ((r.get("response") or {}).get("index") or {}).get("indexMode") if r.get("ok") else None

    m6, m9 = mode("6"), mode("9")
    if m6 == "CLASSIC":
        return True, "bucket_default"
    if m9 == "CLASSIC":
        return True, "update_index_mode"
    if (by.get("10") or {}).get("ok") is True:
        return True, "query_mode_on_enhanced"
    return False, None


def run_probe(ph: AwsPhase) -> None:
    run = ph.run
    if run.meta["mode"] != "own":
        raise UsageError("aws-probe runs only in own mode")
    ph.connect()
    ph.total = cost_model.phase_request_totals(ph.plan).get("aws-probe", 0)
    ph.progress()
    prefix = resource_prefix(run.run_id)
    bucket = index = f"{prefix}-probe"
    if ph.p.probe.exists():
        old = read_json(ph.p.probe)
        if old.get("run_id") == run.run_id and old.get("records"):
            dest = ph.p.aws / "aborted" / datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S")
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ph.p.probe), str(dest / "probe_classic.json"))
    atomic_write_json(ph.p.probe, {"run_id": run.run_id, "records": []}, ph.sleep)
    EXP = {
        "3": ("not documented for a new bucket; the probe exists to find out", "DOC-MODE"),
        "6": ("not documented for a new bucket; the probe exists to find out", "DOC-MODE"),
        "8": ("rejected: CLASSIC only for indexes in buckets created before 2026-09-30", "BOTO-update_index_mode"),
        "10": ("rejected: CLASSIC can't be specified for an ENHANCED index", "BOTO-query_vectors"),
    }

    def add(rec):
        ph.probe_append(rec)

    # Step 1: CreateVectorBucket. AccessDenied and credential failures here stay fatal.
    res1 = ph.ensure_resource("bucket", "probe_bucket", bucket, None, {})
    entry = run.find(bucket, None)
    if res1 is None:
        add({"step": "1", "api": "CreateVectorBucket", "ok": True, "reused": True, "timestamp_utc": utc_now()})
    else:
        if not res1.ok and res1.error_code in ("AccessDeniedException", "AccessDenied"):
            raise AwsFatal("access denied on step 1 CreateVectorBucket")
        add(ph.probe_record("1", "CreateVectorBucket", {"vectorBucketName": bucket, "tags": ph.tags()}, res1,
                            None, None))
    bucket_ok = entry["status"] == "created"
    b = {"vectorBucketName": bucket}
    for st, api, fn, params in (("2", "GetVectorBucket", "get_vector_bucket", b),
                                ("3", "PutVectorBucketDefaultIndexMode", "put_vector_bucket_default_index_mode",
                                 {**b, "defaultIndexMode": "CLASSIC"}),
                                ("4", "GetVectorBucket", "get_vector_bucket", b)):
        if not bucket_ok:
            add(AwsPhase.skipped(st, api, "probe bucket not created"))
            continue
        res = ph.other(fn, params, tolerate=True)
        exp, src = EXP.get(st, (None, None))
        add(ph.probe_record(st, api, params, res, exp, src))

    # Step 5: CreateIndex (dim 8, cosine)
    index_ok = False
    if bucket_ok:
        res5 = ph.ensure_resource("index", "probe_index", bucket, index,
                                  {"dataType": "float32", "dimension": PROBE_DIM, "distanceMetric": "cosine"},
                                  index_kind="probe")
        if res5 is None:
            add({"step": "5", "api": "CreateIndex", "ok": True, "reused": True, "timestamp_utc": utc_now()})
        else:
            add(ph.probe_record("5", "CreateIndex", {**b, "indexName": index, "dimension": PROBE_DIM,
                                                     "distanceMetric": "cosine"}, res5, None, None))
        index_ok = run.find(bucket, index)["status"] == "created"
    else:
        add(AwsPhase.skipped("5", "CreateIndex", "probe bucket not created"))
    ix = {**b, "indexName": index}
    no_ix = "probe index not created"
    if index_ok:
        res6 = ph.other("get_index", ix, tolerate=True)
        exp, src = EXP["6"]
        add(ph.probe_record("6", "GetIndex", ix, res6, exp, src))
    else:
        add(AwsPhase.skipped("6", "GetIndex", no_ix))

    # Step 7: PutVectors (10 deterministic unit vectors), step 7a: ListVectors polling
    pv = probe_vectors(ph.cfg["dataset"]["seed"])
    pmeta = [{"key": f"p-{i:02d}", "tenant": f"p{i % 2}"} for i in range(PROBE_VECTORS)]
    ready = False
    if index_ok:
        payload = [{"key": m["key"], "data": {"float32": [float(x) for x in v]}, "metadata": {"tenant": m["tenant"]}}
                   for m, v in zip(pmeta, pv)]
        ph.run.reserve_request_id()
        res7 = ph.client.call("put_vectors", {**ix, "vectors": payload},
                              cost_lines={"put": cost_model.put_usd(PROBE_VECTORS * ph.bpv("probe"), ph.pricing)},
                              tolerate_errors=True)
        ph.tick()
        add(ph.probe_record("7", "PutVectors", {**ix, "vector_count": PROBE_VECTORS, "metadata": pmeta}, res7,
                            None, None))
        polls, listed, last = 0, 0, None
        if res7.ok:
            aws = ph.cfg["aws"]
            while polls < aws["probe_list_max_polls"]:
                polls += 1
                last = ph.other("list_vectors", {**ix, "maxResults": aws["list_vectors_max_results"]}, tolerate=True)
                listed = len((last.response or {}).get("vectors", [])) if last.ok else 0
                if listed >= PROBE_VECTORS:
                    break
                ph.sleep(aws["probe_list_poll_s"])
            ready = listed >= PROBE_VECTORS
            rec = ph.probe_record("7a", "ListVectors", {**ix, "polls": polls}, last, None, None,
                                  polls=polls, listed=listed, probe_index_ready=ready)
            add(rec)
        else:
            add(AwsPhase.skipped("7a", "ListVectors", "PutVectors failed"))
    else:
        add(AwsPhase.skipped("7", "PutVectors", no_ix))
        add(AwsPhase.skipped("7a", "ListVectors", no_ix))

    # Step 8: UpdateIndexMode CLASSIC; step 9: GetIndex
    for st, api, fn, params in (("8", "UpdateIndexMode", "update_index_mode", {**ix, "indexMode": "CLASSIC"}),
                                ("9", "GetIndex", "get_index", ix)):
        if not index_ok:
            add(AwsPhase.skipped(st, api, no_ix))
            continue
        res = ph.other(fn, params, tolerate=True)
        exp, src = EXP.get(st, (None, None))
        add(ph.probe_record(st, api, params, res, exp, src))

    # Steps 10-12: QueryVectors CLASSIC / ENHANCED / no queryMode, filter tenant=p0, K=3
    for st, mode in (("10", "CLASSIC"), ("11", "ENHANCED"), ("12", None)):
        if not index_ok:
            add(AwsPhase.skipped(st, "QueryVectors", no_ix))
            continue
        params = {**ix, "topK": 3, "queryVector": {"float32": [float(x) for x in pv[0]]},
                  "filter": {"tenant": "p0"}, "returnDistance": True}
        if mode:
            params["queryMode"] = mode
        ph.run.reserve_request_id()
        res = ph.client.call("query_vectors", params, cost_lines=ph.page_cost("probe"),
                             worst_extra_usd=ph.worst_page(), tolerate_errors=True)
        ph.tick()
        exp, src = EXP.get(st, (None, None))
        add(ph.probe_record(st, "QueryVectors", params, res, exp, src, query_mode=mode, probe_index_ready=ready))

    records = ph.probe_load()["records"]
    available, source = classic_decision(records)
    run.meta["classic_index_available"] = available
    run.meta["classic_source"] = source
    ph.complete()
    if available:
        run.meta["stopped_reason"] = "CLASSIC_OBTAINABLE"
        run.save()
        ph.log.warning("CLASSIC obtainable (classic_source=%s): AWS phase stops; only aws-cleanup may run next", source)
        raise PhaseStopped("CLASSIC_OBTAINABLE")
    run.save()


# ================================================================ aws-ingest

def readiness(ph: AwsPhase, rec: Recorder, bucket: str, index: str, data, expected_mode: str) -> None:
    aws = ph.cfg["aws"]
    t0 = time.monotonic()
    res = ph.other("get_index", {"vectorBucketName": bucket, "indexName": index})
    if not res.ok:
        raise AwsFatal(f"GetIndex failed: {res.error_code}")
    mode = res.response["index"].get("indexMode")
    ph.run.meta.setdefault("get_index", {})[index] = clean_response(res.response)
    if mode != expected_mode:
        raise ModeMismatch(f"index mode {mode}, expected {expected_mode}")
    count, polls = 0, 0
    while polls < aws["readiness_max_polls"]:
        polls += 1
        count, token = 0, None
        while True:
            params = {"vectorBucketName": bucket, "indexName": index, "maxResults": aws["list_vectors_max_results"]}
            if token:
                params["nextToken"] = token
            r = ph.other("list_vectors", params)
            if not r.ok:
                break
            count += len(r.response.get("vectors", []))
            token = r.response.get("nextToken")
            if not token:
                break
        if count >= data.n:
            break
        ph.sleep(aws["readiness_poll_s"])
    if count < data.n:
        raise AwsFatal("not ready: ListVectors count below N")
    fone = next(f for f in ph.cfg["filters"] if f["id"] == "FONE")
    expected = [data.keys[i] for i in np.flatnonzero(flt.mask(fone["filter"], data.columns, data.n))]
    qmode = "ENHANCED" if mode == "CLASSIC" else None
    ok = False
    for _ in range(aws["readiness_max_polls"]):
        out = ph.query(recorder=rec, bucket=bucket, index=index, index_kind="main", evidence_class="readiness",
                       vector=data.queries[0], top_k=5, filt=fone["filter"], filter_id="FONE",
                       query_id=data.query_ids[0], k=5, query_mode=qmode, key_index=data.key_index)
        if out["ok"] and [kk for pg in out["pages"] for kk in pg["keys"]] == expected:
            ok = True
            break
        ph.sleep(aws["readiness_poll_s"])
    if not ok:
        raise AwsFatal("not ready: readiness query did not return the tenant-one key")
    ph.run.meta.setdefault("readiness_s", {})[index] = round(time.monotonic() - t0, 1)
    ph.run.save()


def run_ingest_own(ph: AwsPhase, data) -> None:
    run = ph.run
    if "aws-probe" not in run.meta["phases_completed"]:
        raise UsageError("aws-ingest requires aws-probe to be completed")
    ph.connect()
    ph.total = cost_model.phase_request_totals(ph.plan).get("aws-ingest", 0)
    ph.progress()
    name = resource_prefix(run.run_id)
    rec = Recorder(ph.p.aws, run.run_id)
    res = ph.ensure_resource("bucket", "main_bucket", name, None, {})
    if res is not None and not res.ok:
        raise AwsFatal(f"CreateVectorBucket failed: {res.error_code}")
    ds = ph.cfg["dataset"]
    res = ph.ensure_resource("index", "main_index", name, name,
                             {"dataType": "float32", "dimension": ds["dim"], "distanceMetric": ds["metric"]},
                             index_kind="main")
    if res is not None and not res.ok:
        raise AwsFatal(f"CreateIndex failed: {res.error_code}")
    g = ph.other("get_index", {"vectorBucketName": name, "indexName": name})  # before any PutVectors
    if not g.ok:
        raise AwsFatal(f"GetIndex failed: {g.error_code}")
    if g.response["index"].get("indexMode") != "ENHANCED":
        raise ModeMismatch(f"main index mode {g.response['index'].get('indexMode')}, expected ENHANCED")
    ph.put_batches(name, name, data.vectors, data.metadata, rec, "main", ph.cfg["aws"]["put_batch_size"],
                   ph.cfg["aws"]["ingest_rate_per_s"])
    readiness(ph, rec, name, name, data, "ENHANCED")
    ph.complete()


def run_ingest_borrowed_start(ph: AwsPhase) -> None:
    """Invocation 1: read-only. Generates the id, records the borrowed bucket, and stops (exit 2)."""
    run = ph.run
    ph.connect()
    bucket = ph.args.borrowed_bucket
    res = ph.other("get_vector_bucket", {"vectorBucketName": bucket}, tolerate=True)
    if not res.ok:
        raise AwsFatal(f"borrowed bucket not readable: {res.error_code}")
    run.upsert(rtype="bucket", role="borrowed_bucket", owned=False, bucket=bucket, index=None, status="created")
    planned = resource_prefix(run.run_id)
    ph.finish_durations()
    raise UsageError(f"planned index {planned}; re-run with --run-id {run.run_id} --confirm-index-name {planned}")


def run_ingest_borrowed(ph: AwsPhase, data) -> None:
    run = ph.run
    if run.meta["mode"] != "borrowed":
        raise UsageError("--borrowed-bucket given but the run is not in borrowed mode")
    planned = resource_prefix(run.run_id)
    bentry = run.find_role("borrowed_bucket")
    if ph.args.confirm_index_name != planned or bentry is None or bentry["bucket"] != ph.args.borrowed_bucket:
        raise UsageError("--confirm-index-name or --borrowed-bucket does not match the recorded plan")
    ph.connect()
    ph.total = cost_model.phase_request_totals(ph.plan).get("aws-ingest", 0)
    bucket = bentry["bucket"]
    rec = Recorder(ph.p.aws, run.run_id)
    ds = ph.cfg["dataset"]
    res = ph.ensure_resource("index", "borrowed_index", bucket, planned,
                             {"dataType": "float32", "dimension": ds["dim"], "distanceMetric": ds["metric"]},
                             index_kind="main")
    if res is not None and not res.ok:
        raise AwsFatal(f"CreateIndex failed: {res.error_code}")
    g = ph.other("get_index", {"vectorBucketName": bucket, "indexName": planned})
    mode = g.response["index"].get("indexMode") if g.ok else None
    if mode != "CLASSIC":
        ph.log.error("borrowed index mode %s is not CLASSIC; deleting only the new index", mode)
        ph.client.ctx.borrowed_abort = True
        d = ph.other("delete_index", {"vectorBucketName": bucket, "indexName": planned}, tolerate=True)
        run.upsert(rtype="index", role="borrowed_index", owned=True, bucket=bucket, index=planned,
                   status="deleted" if d.ok else "delete_failed", error_code=d.error_code)
        raise ModeMismatch(f"borrowed index mode {mode}, expected CLASSIC")
    ph.put_batches(bucket, planned, data.vectors, data.metadata, rec, "main", ph.cfg["aws"]["put_batch_size"],
                   ph.cfg["aws"]["ingest_rate_per_s"])
    readiness(ph, rec, bucket, planned, data, "CLASSIC")
    ph.complete()


# ================================================================ aws-query

def plan_passes(cfg: dict, data, mode: str) -> list[list[dict]]:
    grid_classes = [("aws_enhanced", None)] if mode == "own" else [
        ("aws_classic", "CLASSIC"), ("aws_classic_index_enhanced_query", "ENHANCED")]
    passes = []
    rng = np.random.default_rng(ds_mod.stream(cfg["dataset"]["seed"], "order"))
    for r in range(1, cfg["repeats"] + 1):
        items = []
        for cls, qm in grid_classes:
            for f in cfg["filters"]:
                for qid in data.query_ids:
                    for k in cfg["k_values"]:
                        items.append({"cls": cls, "query_mode": qm, "filter_id": f["id"], "filter": f["filter"],
                                      "query_id": qid, "k": k, "top_k": k, "budget": None, "repeat": r})
        for qid in data.query_ids:
            for k in cfg["k_values"]:
                items.append({"cls": "aws_ann_reference", "query_mode": None, "filter_id": "NOFILTER",
                              "filter": None, "query_id": qid, "k": k, "top_k": k, "budget": None, "repeat": r})
        if r <= cfg["postfilter_repeats"]:
            for qid in data.query_ids:
                for b in cfg["budgets"]:
                    items.append({"cls": "aws_postfilter_baseline", "query_mode": None, "filter_id": None,
                                  "filter": None, "query_id": qid, "k": None, "top_k": b, "budget": b, "repeat": r})
        perm = rng.permutation(len(items))
        passes.append([items[i] for i in perm])
    return passes


def archive_partial(p: Paths, run_id: str) -> None:
    """Move an interrupted aws-query's _partial/ and its step-13 record into aborted/<utc>/."""
    has_partial = p.partial.exists()
    stale_13 = False
    if p.probe.exists():
        doc = read_json(p.probe)
        stale_13 = doc.get("run_id") == run_id and any(r["step"] == "13-main-index" for r in doc["records"])
    if not (has_partial or stale_13):
        return
    dest = p.aws / "aborted" / datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%f")
    dest.mkdir(parents=True, exist_ok=True)
    if has_partial:
        shutil.move(str(p.partial), str(dest / "_partial"))
    if stale_13:
        atomic_write_json(dest / "probe_classic.json", doc)
        doc["records"] = [r for r in doc["records"] if r["step"] != "13-main-index"]
        atomic_write_json(p.probe, doc)


def merge_partial(p: Paths) -> None:
    for name in ("queries.jsonl", "responses.jsonl", "timings.jsonl"):
        rows = read_jsonl(p.aws / name) + read_jsonl(p.partial / name)
        atomic_write_jsonl(p.aws / name, rows)
    os.replace(p.partial / "metrics.json", p.aws / "metrics.json")
    shutil.rmtree(p.partial)


def run_query(ph: AwsPhase, data, gt_rows) -> None:
    run = ph.run
    if "aws-ingest" not in run.meta["phases_completed"]:
        raise UsageError("aws-query requires aws-ingest to be completed")
    mode = run.meta["mode"]
    role = "main_index" if mode == "own" else "borrowed_index"
    entry = run.find_role(role)
    bucket, index = entry["bucket"], entry["index"]
    archive_partial(ph.p, run.run_id)
    ph.p.partial.mkdir(parents=True, exist_ok=True)
    rec = Recorder(ph.p.partial, run.run_id)
    ph.connect()
    ph.total = cost_model.phase_request_totals(ph.plan).get("aws-query", 0)
    ph.progress()
    cfg = ph.cfg
    qidx = {qid: i for i, qid in enumerate(data.query_ids)}
    fmap = {f["id"]: f for f in cfg["filters"]}
    masks = {f["id"]: flt.mask(f["filter"], data.columns, data.n) for f in flt.scored_filters(cfg)}
    gt_index = gt_mod.index_rows(gt_rows)
    engine = gt_mod.DistanceEngine(data.vectors)
    local_d = {}
    throttle = Throttle(cfg["aws"]["query_rate_per_s"], ph.sleep)

    def qkw(item, order_index=None):
        return dict(recorder=rec, bucket=bucket, index=index, index_kind="main", throttle=throttle,
                    key_index=data.key_index, order_index=order_index)

    # Probe step 13 (own mode): queryMode=CLASSIC on the main index, F1, K=10, q-00
    if mode == "own":
        f1 = fmap["F1"]
        out = ph.query(**qkw(None), evidence_class="probe_classic", vector=data.queries[0], top_k=10,
                       filt=f1["filter"], filter_id="F1", query_id=data.query_ids[0], k=10, query_mode="CLASSIC",
                       tolerate=True)
        res_like = CallResult("query_vectors", out["ok"], None, out["error_code"], out["error_message"],
                              out["http_status"], sum(out["attempts"]))
        record = ph.probe_record("13-main-index", "QueryVectors",
                                 {"vectorBucketName": bucket, "indexName": index, "topK": 10, "filter": f1["filter"],
                                  "queryMode": "CLASSIC", "query_id": data.query_ids[0], "request_id": out["request_id"]},
                                 res_like, "rejected: CLASSIC can't be specified for an ENHANCED index",
                                 "BOTO-query_vectors")
        record["response"] = out["response"]
        ph.probe_append(record)
        run.meta["classic_query_accepted_on_main"] = bool(out["ok"])
        run.save()
        if out["ok"]:
            ph.log.warning("probe step 13-main-index: queryMode=CLASSIC was ACCEPTED on the ENHANCED main index; "
                           "continuing the ENHANCED grid (flag classic_query_accepted_on_main=true); escalate")

    records, failed = [], 0
    planned_total = sum(len(ps) for ps in plan_passes(cfg, data, mode))
    abort_at = cfg["aws"]["failed_request_abort_fraction"] * planned_total
    order_index = 0
    for pass_items in plan_passes(cfg, data, mode):
        for item in pass_items:
            order_index += 1
            qi = qidx[item["query_id"]]
            out = ph.query(**qkw(item, order_index), evidence_class=item["cls"], vector=data.queries[qi],
                           top_k=item["top_k"], filt=item["filter"], filter_id=item["filter_id"],
                           query_id=item["query_id"], k=item["k"], budget=item["budget"], repeat=item["repeat"],
                           query_mode=item["query_mode"], timing_phase="first" if item["repeat"] == 1 else "warm")
            if qi not in local_d:
                local_d[qi] = engine.distances(data.queries[qi])
            base = {"evidence_class": item["cls"], "query_id": item["query_id"], "repeat": item["repeat"],
                    "request_id": out["request_id"], "budget_c": None}
            if item["cls"] == "aws_postfilter_baseline":
                targets = [(f["id"], k) for f in cfg["filters"] for k in cfg["k_values"]]
            else:
                targets = [(item["filter_id"], item["k"])]
            if not out["ok"]:
                failed += 1
                for fid, k in targets:
                    records.append({**base, "filter_id": fid, "k": k, "budget_b": item["budget"], "ok": False,
                                    "error_code": out["error_code"]})
                if failed > abort_at:
                    raise AwsFatal(f"failed requests {failed} exceed {cfg['aws']['failed_request_abort_fraction']:.0%}"
                                   " of the plan")
                continue
            if item["cls"] == "aws_postfilter_baseline":
                merged = metrics.merge_pages(out["pages"])
                for fid, k in targets:
                    keys, dists = metrics.postfilter_select(merged, masks[fid], data.key_index, k)
                    m = metrics.score_request(keys, gt_index[(item["query_id"], fid)], k, masks[fid], data.key_index,
                                              dists, local_d[qi])
                    records.append({**base, "filter_id": fid, "k": k, "budget_b": item["budget"], "ok": True,
                                    "keys": keys, **m})
            else:
                keys = [kk for pg in out["pages"] for kk in pg["keys"]]
                dists = [d for pg in out["pages"] for d in pg["distances"]]
                fid, k = targets[0]
                m = metrics.score_request(keys, gt_index[(item["query_id"], fid)], k, masks[fid], data.key_index,
                                          dists, local_d[qi])
                records.append({**base, "filter_id": fid, "k": k, "budget_b": None, "ok": True,
                                "keys": metrics.dedupe(keys)[0][:k], **m})

    constraint = []
    if mode == "own":
        spec = cfg["constraint_probe"]
        qi = qidx[spec["query_id"]]
        for cf in flt.constraint_probe_filters(cfg):
            out = ph.query(**qkw(None), evidence_class="probe_constraint_limit", vector=data.queries[qi],
                           top_k=spec["k"], filt=cf["filter"], filter_id=cf["id"], query_id=spec["query_id"],
                           k=spec["k"], tolerate=True)
            constraint.append({"filter_id": cf["id"], "constraints": flt.count_constraints(cf["filter"]),
                               "ok": out["ok"], "error_code": out["error_code"], "error_message": out["error_message"],
                               "returned": sum(len(pg["keys"]) for pg in out["pages"]) if out["ok"] else None,
                               "request_id": out["request_id"]})

    aggregates = metrics.aggregate(records, cfg["dataset"]["seed"])
    per_request = [{k: v for k, v in r.items() if k != "keys"} for r in records]
    atomic_write_json(ph.p.partial / "metrics.json", round_sig({
        "run_id": run.run_id, "mode": mode, "per_request": per_request, "aggregates": aggregates,
        "failed_requests": failed, "constraint_probe": constraint,
        "classic_query_accepted_on_main": run.meta.get("classic_query_accepted_on_main")}))
    merge_partial(ph.p)
    ph.complete()
    if mode == "own":
        atomic_write_text(ph.p.shot_list, reporting.shot_list_md(run.run_id))
        if run.meta.get("classic_query_accepted_on_main"):
            ph.log.warning("CLASSIC_QUERY_ACCEPTED_ON_MAIN: ENHANCED grid finished; review probe record "
                           "13-main-index before trusting CLASSIC conclusions")


# ================================================================ aws-capture

def run_capture(ph: AwsPhase, data) -> None:
    run = ph.run
    if "aws-query" not in run.meta["phases_completed"]:
        raise UsageError("aws-capture requires aws-query to be completed")
    if run.meta["mode"] == "own":
        role = "main_index"
    else:
        role = "borrowed_index"
    entry = run.find_role(role)
    if entry is None or entry["status"] != "created":
        raise UsageError("capture needs the index to exist")
    bucket, index = entry["bucket"], entry["index"]
    ph.plan = cost_model.request_plan(ph.cfg, ph.manifest_data, run.meta["mode"])
    ph.connect()
    tmp = ph.p.aws / "_capture"
    if tmp.exists():
        shutil.rmtree(tmp)
    rec = Recorder(tmp, run.run_id)
    g = ph.other("get_index", {"vectorBucketName": bucket, "indexName": index})
    rid = ph.last_request_id
    rec.query({"request_id": rid, "evidence_class": "capture", "api": "GetIndex"})
    rec.response({"request_id": rid, "page": 1, "ok": g.ok, "error_code": g.error_code,
                  "response": clean_response(g.response)})
    ph.p.screens.mkdir(parents=True, exist_ok=True)
    atomic_write_text(ph.p.screens / "terminal_get_index.txt",
                      "$ aws s3vectors get-index (via boto3 get_index)\n"
                      + dumps(redact({"vectorBucketName": bucket, "indexName": index}), indent=2) + "\n"
                      + dumps(clean_response(g.response) if g.ok else {"error_code": g.error_code}, indent=2) + "\n")
    f01 = next(f for f in ph.cfg["filters"] if f["id"] == "F01")
    out = ph.query(recorder=rec, bucket=bucket, index=index, index_kind="main", evidence_class="capture",
                   vector=data.queries[0], top_k=10, filt=f01["filter"], filter_id="F01", query_id=data.query_ids[0],
                   k=10, key_index=data.key_index)
    vec = [float(x) for x in data.queries[0]]
    req = {"vectorBucketName": bucket, "indexName": index, "topK": 10, "filter": f01["filter"],
           "returnDistance": True, "returnMetadata": False,
           "queryVector": {"float32": [round(v, 6) for v in vec[:8]] + [f"... {len(vec)} float32 values total"]}}
    atomic_write_text(ph.p.screens / "terminal_query_request_response.txt",
                      "$ QueryVectors request (query q-00, filter F01, K=10)\n" + dumps(redact(req), indent=2)
                      + "\n\n$ QueryVectors response\n"
                      + dumps(out["response"] if out["ok"] else {"error_code": out["error_code"]}, indent=2) + "\n")
    for name in ("queries.jsonl", "responses.jsonl", "timings.jsonl"):
        atomic_write_jsonl(ph.p.aws / name, read_jsonl(ph.p.aws / name) + read_jsonl(tmp / name))
    shutil.rmtree(tmp)
    missing = False
    if run.meta["mode"] == "own":
        doc = ph.probe_load()
        by = {r["step"]: r for r in doc["records"]}
        parts = []
        for st in ("8", "10", "13-main-index"):
            r = by.get(st)
            if r is None:
                parts.append(f"MISSING: step {st}\n")
                missing = True
                continue
            if r.get("skipped_reason"):
                head = f"=== step {st}: SKIPPED ({r['skipped_reason']})"
            else:
                head = f"=== step {st}: {'ACCEPTED' if r.get('ok') else 'REJECTED'}"
            parts.append(head + "\n" + dumps(r, indent=2) + "\n")
        atomic_write_text(ph.p.screens / "terminal_classic_probe_records.txt", "\n".join(parts))
    ph.complete()
    if missing:
        raise IntegrityError("probe records missing for capture")


# ================================================================ aws-cleanup

def run_cleanup(ph: AwsPhase) -> None:
    run = ph.run
    ph.plan = cost_model.request_plan(ph.cfg, ph.manifest_data, run.meta["mode"])
    ph.connect(enforce_cap=False)
    actions = []
    to_delete = []
    for e in run.manifest:
        name = e["index"] or e["bucket"]
        if not e["owned"]:
            actions.append({"role": e["role"], "name": name, "action": "skipped: borrowed", "final_status": e["status"],
                            "utc": utc_now()})
            continue
        if e["status"] in ("deleted", "not_found"):
            continue
        if e["status"] == "create_failed" and e.get("error_code") == "ConflictException":
            ph.log.warning("pre-existing resource %s is not ours and is never touched", name)
            actions.append({"role": e["role"], "name": name, "action": "skipped: preexisting (ConflictException)",
                            "final_status": e["status"], "utc": utc_now()})
            continue
        if e["status"] in ("pending", "create_failed"):
            r = ph._get(e["bucket"], e["index"])
            if r.ok:
                to_delete.append(e)
            elif r.error_code == "NotFoundException":
                e["status"] = "not_found"
                actions.append({"role": e["role"], "name": name, "action": "resolved: not found",
                                "final_status": "not_found", "utc": utc_now()})
            else:
                e["status"] = "delete_failed"
                actions.append({"role": e["role"], "name": name, "action": "resolve failed",
                                "error_code": r.error_code, "final_status": "delete_failed", "utc": utc_now()})
            run.save()
            continue
        to_delete.append(e)
    for e in sorted(to_delete, key=lambda x: 0 if x["index"] else 1):
        if e["index"]:
            api, params = "delete_index", {"vectorBucketName": e["bucket"], "indexName": e["index"]}
        else:
            api, params = "delete_vector_bucket", {"vectorBucketName": e["bucket"]}
        r = ph.other(api, params, tolerate=True)
        status = "deleted" if r.ok else ("not_found" if r.error_code == "NotFoundException" else "delete_failed")
        e["status"] = status
        e["updated_utc"] = utc_now()
        run.save()
        actions.append({"role": e["role"], "name": e["index"] or e["bucket"], "action": api, "ok": r.ok,
                        "error_code": r.error_code, "error_message": r.error_message, "final_status": status,
                        "utc": utc_now()})
    verification = []
    prefix = resource_prefix(run.run_id)
    for e in run.manifest:
        if e["index"] and e["owned"] and not (e["status"] == "create_failed" and e.get("error_code") == "ConflictException"):
            r = ph.other("get_index", {"vectorBucketName": e["bucket"], "indexName": e["index"]}, tolerate=True)
            verification.append({"check": "GetIndex", "name": e["index"], "gone": r.error_code == "NotFoundException"})
    for e in run.manifest:
        conflict = e["status"] == "create_failed" and e.get("error_code") == "ConflictException"
        if not e["index"] and not conflict and (not e["owned"] or e["status"] not in ("deleted", "not_found")):
            r = ph.other("list_indexes", {"vectorBucketName": e["bucket"], "prefix": prefix}, tolerate=True)
            left = [i["indexName"] for i in (r.response or {}).get("indexes", [])] if r.ok else None
            verification.append({"check": "ListIndexes", "bucket_role": e["role"], "run_prefixed_indexes": left})
    preexisting = {e["bucket"] for e in run.manifest if not e["index"] and e["status"] == "create_failed"
                   and e.get("error_code") == "ConflictException"}
    names, token = [], None
    while True:
        params = {"prefix": prefix, **({"nextToken": token} if token else {})}
        r = ph.other("list_vector_buckets", params, tolerate=True)
        if not r.ok:
            names = None
            break
        names += [b["vectorBucketName"] for b in r.response.get("vectorBuckets", [])]
        token = r.response.get("nextToken")
        if not token:
            break
    leftovers = None if names is None else sorted(set(names) - preexisting)
    verification.append({"check": "ListVectorBuckets", "prefix": prefix, "leftover_buckets": leftovers,
                         "preexisting_not_ours": sorted(preexisting)})
    failed = [e for e in run.manifest if e["owned"] and e["status"] == "delete_failed"]
    atomic_write_json(ph.p.aws / "cleanup.json", redact({
        "run_id": run.run_id, "utc": utc_now(), "actions": actions, "verification": verification,
        "manifest_final": run.manifest, "all_owned_gone": not failed and leftovers == [] and all(
            v.get("gone", True) for v in verification if v["check"] == "GetIndex")}))
    ph.complete()
    if failed:
        raise AwsFatal("delete failed for: " + ", ".join(e["index"] or e["bucket"] for e in failed))


# ================================================================ AWS dispatcher

def run_aws(phase: str, p: Paths, cfg: dict, args, log, client_factory, sleep) -> int:
    ph = AwsPhase(phase, p, cfg, args, log, client_factory, sleep)
    lock = PhaseLock(p.lock, phase, args.run_id, log)
    status, code, marker = "failed", EXIT_FAIL, None
    try:
        lock.acquire()
        borrowed_inv1 = phase == "aws-ingest" and args.borrowed_bucket and not args.run_id
        if phase == "aws-probe":
            ph.new_run("own")
        elif borrowed_inv1:
            ph.new_run("borrowed")
        else:
            run = ph.load_run()
            if run.meta.get("stopped_reason") and phase != "aws-cleanup":
                raise PhaseStopped(run.meta["stopped_reason"])
            done = run.meta.get("phases_completed", [])
            if phase in done and phase not in ("aws-cleanup", "aws-capture"):
                raise UsageError(f"{phase} already completed for run {run.run_id}")
            if phase in ("aws-ingest", "aws-query"):
                ph.check_plan(run.meta["mode"], run.run_id)
            if phase == "aws-ingest" and run.meta["mode"] == "own" and args.borrowed_bucket:
                raise UsageError("--borrowed-bucket given but the run is in own mode")
        if phase == "aws-probe":
            run_probe(ph)
        elif phase == "aws-ingest":
            if borrowed_inv1:
                run_ingest_borrowed_start(ph)
            else:
                data, _ = ds_mod.load(p.data)
                (run_ingest_borrowed if args.borrowed_bucket else run_ingest_own)(ph, data)
        elif phase == "aws-query":
            data, _ = ds_mod.load(p.data)
            run_query(ph, data, read_jsonl(p.gt))
        elif phase == "aws-capture":
            data, _ = ds_mod.load(p.data)
            run_capture(ph, data)
        elif phase == "aws-cleanup":
            run_cleanup(ph)
        status, code, marker = "complete", 0, f"PHASE COMPLETE {phase}"
    except PhaseStopped as exc:
        status, code, marker = "stopped", EXIT_STOPPED, f"PHASE STOPPED {phase}: {exc.reason}"
    except UsageError as exc:
        log.error("%s", exc)
        status, code, marker = "failed", EXIT_USAGE, f"PHASE FAILED {phase}: UsageError"
    except BaseException as exc:  # noqa: BLE001 - logged through the redacting formatter, then re-classified
        log.exception("phase %s failed: %s", phase, type(exc).__name__)
        status, code, marker = "failed", EXIT_FAIL, f"PHASE FAILED {phase}: {type(exc).__name__}"
        if ph.run is not None:
            try:
                ph.finish_durations()
            except BaseException:  # noqa: BLE001 - best effort; the original error is already logged
                pass
        if ph.run is not None:
            log.error("resources stay recorded; clean up with: aws-cleanup --aws --profile %s --region %s --run-id %s",
                      args.profile, args.region, ph.run.run_id)
    finally:
        try:
            ph.progress(status)
        finally:
            lock.release()
    if code == 0:
        log.info("%s", marker)
    elif code == EXIT_STOPPED:
        log.warning("%s", marker)
    else:
        log.error("%s", marker)
    return code


# ================================================================ main

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="src.runner", description="S3 Vectors pre-filtering benchmark")
    ap.add_argument("phase", choices=ALL_PHASES)
    ap.add_argument("--aws", action="store_true")
    ap.add_argument("--profile")
    ap.add_argument("--region")
    ap.add_argument("--confirm-cost", action="store_true")
    ap.add_argument("--run-id")
    ap.add_argument("--borrowed-bucket")
    ap.add_argument("--confirm-index-name")
    ap.add_argument("--borrowed", action="store_true")
    ap.add_argument("--out-root")
    return ap


def main(argv=None, *, root: Path | None = None, client_factory=None, sleep=time.sleep) -> int:
    install_redacting_excepthook()
    args = build_parser().parse_args(argv)
    p = Paths(root or REPO_ROOT, args.out_root)
    log = make_logger()
    phase = args.phase
    try:
        validate_flags(phase, args)
        if phase == "verify-redaction":
            hits = verify_redaction(p.root)
            for rel, line in hits:
                print(f"REDACTION HIT {rel}:{line}", flush=True)
            print(f"verify-redaction: {'FAIL' if hits else 'PASS'} ({len(hits)} hits)", flush=True)
            return 1 if hits else 0
        cfg = load_config(p.config)
        if phase == "estimate":
            return phase_estimate(p, cfg, log, args.borrowed)
        if phase in AWS_PHASES:
            if client_factory is None:
                from .aws_client import make_boto3_client
                client_factory = make_boto3_client
            return run_aws(phase, p, cfg, args, log, client_factory, sleep)
        steps = {"generate": [phase_generate], "ground-truth": [phase_ground_truth], "simulate": [phase_simulate],
                 "report": [phase_report],
                 "offline-all": [phase_generate, phase_ground_truth, phase_simulate, phase_report]}[phase]
        for fn in steps:
            fn(p, cfg, log)
        return 0
    except (UsageError, ConfigError) as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        return EXIT_USAGE
    except IntegrityError as exc:
        log.error("IntegrityError: %s", exc)
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
