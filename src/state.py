"""Run-state I/O with no AWS dependency.

Holds the shared exception types, canonical JSON serialization, the atomic state writer,
the single-instance phase lock, the progress file, the new-run guard, and the cumulative
prior-spend computation for the project-wide $5 cap.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


class BenchError(Exception):
    """Base class for benchmark errors."""


class ConfigError(BenchError):
    """Invalid configuration or pricing input (exit 2)."""


class IntegrityError(BenchError):
    """Data-integrity violation: hashes, unknown keys, malformed state."""


class AwsFatal(BenchError):
    """Unrecoverable AWS-phase error; the run stops and the manifest is kept."""


class OwnershipError(AwsFatal):
    """The ownership guard refused a call; nothing was sent."""


class BudgetExceeded(AwsFatal):
    """The next billed request would exceed hard_cap_usd; nothing was sent."""


class ModeMismatch(AwsFatal):
    """GetIndex reported an unexpected index mode."""


class UsageError(BenchError):
    """Refused CLI input or lifecycle rule (exit 2, nothing sent)."""


class PhaseStopped(BenchError):
    """Deliberate stop (exit 3), e.g. CLASSIC_OBTAINABLE."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------- serialization

def round_sig(obj: Any, digits: int = 9) -> Any:
    """Round every float leaf to `digits` significant digits (deterministic outputs)."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise IntegrityError("non-finite float in deterministic output")
        return float(f"{obj:.{digits}g}")
    if isinstance(obj, dict):
        return {k: round_sig(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_sig(v, digits) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar
        return round_sig(obj.item(), digits)
    raise TypeError(f"unserializable type {type(obj).__name__}")


def dumps(obj: Any, indent: int | None = None) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=indent,
                      separators=(",", ":") if indent is None else (",", ": "), allow_nan=False)


def canonical_json(obj: Any) -> str:
    """Canonical JSON for hashing: sort_keys, no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(dumps(row) + "\n")
        fh.flush()


# ---------------------------------------------------------------- atomic writes

REPLACE_RETRIES = 5
REPLACE_DELAY_S = 0.2


def _atomic_write_bytes(path: Path, data: bytes, sleep: Callable[[float], None] = time.sleep) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    for attempt in range(1, REPLACE_RETRIES + 1):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == REPLACE_RETRIES:
                # The temp file is left in place for manual recovery.
                raise AwsFatal("state write failed")
            sleep(REPLACE_DELAY_S)


def atomic_write_json(path: Path, obj: Any, sleep: Callable[[float], None] = time.sleep) -> None:
    _atomic_write_bytes(path, (dumps(obj, indent=2) + "\n").encode("utf-8"), sleep)


def atomic_write_text(path: Path, text: str, sleep: Callable[[float], None] = time.sleep) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"), sleep)


def atomic_write_jsonl(path: Path, rows: Iterable[dict], sleep: Callable[[float], None] = time.sleep) -> None:
    _atomic_write_bytes(path, "".join(dumps(r) + "\n" for r in rows).encode("utf-8"), sleep)


# ---------------------------------------------------------------- phase lock

def _pid_alive_runner(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return b"src.runner" in fh.read()
    except OSError:
        return False


class PhaseLock:
    """O_EXCL lock file at results/aws/.lock. Released in __exit__ even on exceptions."""

    def __init__(self, path: Path, phase: str, run_id: str | None, logger=None,
                 alive: Callable[[int], bool] = _pid_alive_runner):
        self.path = Path(path)
        self.phase = phase
        self.run_id = run_id
        self.logger = logger
        self.alive = alive
        self.held = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    info = read_json(self.path)
                    pid = int(info.get("pid", -1))
                except (ValueError, OSError, json.JSONDecodeError):
                    info, pid = {}, -1
                if pid > 0 and self.alive(pid):
                    raise UsageError(f"phase already running: {info.get('phase')} pid {pid}")
                if self.logger:
                    self.logger.warning("removing stale lock %s (pid %s)", self.path.name, pid)
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(dumps({"pid": os.getpid(), "phase": self.phase, "run_id": self.run_id,
                                "start_utc": utc_now()}))
            self.held = True
            return
        raise UsageError("could not acquire lock")

    def release(self) -> None:
        if self.held:
            self.path.unlink(missing_ok=True)
            self.held = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


# ---------------------------------------------------------------- progress file

def progress_record(phase: str, done: int, total: int, elapsed_s: float, spend: float,
                    cap: float, status: str) -> dict:
    percent = round(100.0 * done / total, 2) if total > 0 else 100.0
    return {"phase": phase, "done": done, "total": total, "percent": percent,
            "elapsed_s": round(elapsed_s, 1), "spend_tally_usd": spend, "hard_cap_usd": cap,
            "updated_utc": utc_now(), "status": status}


def write_progress(path: Path, **kwargs) -> None:
    atomic_write_json(path, progress_record(**kwargs))


# ---------------------------------------------------------------- run metadata

ARCHIVED_FILES = ["run_metadata.json", "queries.jsonl", "responses.jsonl", "timings.jsonl",
                  "metrics.json", "probe_classic.json", "cleanup.json", "_partial", "aborted"]
LIVE_STATUSES_EXCLUDED = {"deleted", "not_found"}


def live_owned_entries(meta: dict) -> list[dict]:
    """Owned manifest entries that may still exist. ConflictException create_failed never blocks."""
    out = []
    for e in meta.get("created_resources", []):
        if not e.get("owned"):
            continue
        if e.get("status") in LIVE_STATUSES_EXCLUDED:
            continue
        if e.get("status") == "create_failed" and e.get("error_code") == "ConflictException":
            continue
        out.append(e)
    return out


def new_run_guard(aws_dir: Path) -> str | None:
    """Refuse while an old run has live owned resources; otherwise archive its files.

    Returns the archived run id, or None when there was nothing to archive.
    """
    aws_dir = Path(aws_dir)
    meta_path = aws_dir / "run_metadata.json"
    if not meta_path.exists():
        return None
    meta = read_json(meta_path)
    old = meta.get("run_id")
    if not old:
        raise IntegrityError("run_metadata.json has no run_id")
    if live_owned_entries(meta):
        raise UsageError(f"previous run has live owned resources; run: aws-cleanup --run-id {old}")
    dest = aws_dir / "previous" / old
    dest.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVED_FILES:
        src = aws_dir / name
        if src.exists():
            shutil.move(str(src), str(dest / name))
    if (aws_dir / "cost_estimate.md").exists():
        shutil.copy2(aws_dir / "cost_estimate.md", dest / "cost_estimate.md")
    return old


class RunState:
    """run_metadata.json as the cross-phase run state: manifest, request ids, spend tally.

    Every mutation is persisted with atomic_write_json. A request id is reserved in memory and
    persisted by the pre-send charge that always follows it, so the id and the charge reach disk
    in one write before the request is sent.
    """

    def __init__(self, path: Path, meta: dict, sleep: Callable[[float], None] = time.sleep):
        self.path = Path(path)
        self.meta = meta
        self.sleep = sleep

    @classmethod
    def load(cls, path: Path) -> "RunState":
        return cls(path, read_json(path))

    def save(self) -> None:
        atomic_write_json(self.path, self.meta, self.sleep)

    @property
    def run_id(self) -> str:
        return self.meta["run_id"]

    @property
    def spend(self) -> float:
        return float(self.meta.get("spend_tally_usd", 0.0))

    @property
    def hard_cap(self) -> float:
        return float(self.meta["hard_cap_usd"])

    def reserve_request_id(self) -> str:
        seq = int(self.meta.get("next_request_seq", 1))
        self.meta["next_request_seq"] = seq + 1
        return f"r{seq:06d}"

    def charge(self, lines: dict[str, float]) -> None:
        by_line = self.meta.setdefault("spend_by_line_usd", {})
        total = 0.0
        for line, usd in lines.items():
            by_line[line] = by_line.get(line, 0.0) + usd
            total += usd
        self.meta["spend_tally_usd"] = self.spend + total
        self.save()

    # ---- manifest

    @property
    def manifest(self) -> list[dict]:
        return self.meta.setdefault("created_resources", [])

    def find(self, bucket: str, index: str | None) -> dict | None:
        for e in self.manifest:
            if e["bucket"] == bucket and e.get("index") == index:
                return e
        return None

    def find_role(self, role: str) -> dict | None:
        for e in self.manifest:
            if e["role"] == role:
                return e
        return None

    def upsert(self, *, rtype: str, role: str, owned: bool, bucket: str, index: str | None,
               status: str, error_code: str | None = None) -> dict:
        entry = self.find(bucket, index)
        now = utc_now()
        if entry is None:
            resource = f"bucket/{bucket}" + (f"/index/{index}" if index else "")
            entry = {"type": rtype, "role": role, "owned": owned, "bucket": bucket, "index": index,
                     "arn_resource": resource, "status": status, "error_code": error_code,
                     "created_utc": now, "updated_utc": now}
            self.manifest.append(entry)
        else:
            entry.update({"status": status, "error_code": error_code, "updated_utc": now})
        self.save()
        return entry


def prior_runs_spend_usd(aws_dir: Path, current_run_id: str | None) -> float:
    """Cumulative spend of every run other than `current_run_id`.

    Sums spend_tally_usd from results/aws/run_metadata.json (when it belongs to another run)
    and every run_metadata.json found anywhere under results/aws/previous/ and
    results/aws/aborted/. Each run id is counted once (its largest recorded tally), so a
    run whose files were archived more than once is not double-counted. aborted/ folders of
    the current run hold only partial request logs, whose spend is already in the current
    run's tally.
    """
    aws_dir = Path(aws_dir)
    per_run: dict[str, float] = {}
    candidates: list[Path] = []
    if (aws_dir / "run_metadata.json").exists():
        candidates.append(aws_dir / "run_metadata.json")
    for sub in ("previous", "aborted"):
        if (aws_dir / sub).exists():
            candidates.extend(sorted((aws_dir / sub).rglob("run_metadata.json")))
    for path in candidates:
        try:
            meta = read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            raise IntegrityError(f"unreadable run state {path.name}: {exc}") from exc
        rid = meta.get("run_id")
        if not rid or rid == current_run_id:
            continue
        spend = float(meta.get("spend_tally_usd", 0.0))
        per_run[rid] = max(per_run.get(rid, 0.0), spend)
    return round(sum(per_run.values()), 6)
