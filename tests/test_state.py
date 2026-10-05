import json
import logging
import os

import pytest

from src import state
from src.state import AwsFatal, PhaseLock, UsageError


def test_atomic_write_retries_then_succeeds(tmp_path, monkeypatch):
    real = os.replace
    calls = {"n": 0}

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError("locked")
        return real(a, b)

    monkeypatch.setattr(state.os, "replace", flaky)
    state.atomic_write_json(tmp_path / "x.json", {"a": 1}, sleep=lambda s: None)
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": 1}
    assert calls["n"] == 3


def test_atomic_write_fails_after_five_and_keeps_temp(tmp_path, monkeypatch):
    def always(a, b):
        raise PermissionError("locked")

    monkeypatch.setattr(state.os, "replace", always)
    with pytest.raises(AwsFatal, match="state write failed"):
        state.atomic_write_json(tmp_path / "x.json", {"a": 1}, sleep=lambda s: None)
    assert (tmp_path / "x.json.tmp").exists()


def test_lock_live_pid_refuses_and_stale_is_replaced(tmp_path, caplog):
    lock_path = tmp_path / ".lock"
    with PhaseLock(lock_path, "aws-query", "r", alive=lambda pid: True):
        with pytest.raises(UsageError, match="phase already running"):
            PhaseLock(lock_path, "aws-query", "r", alive=lambda pid: True).acquire()
    assert not lock_path.exists()
    lock_path.write_text(json.dumps({"pid": 999999, "phase": "aws-query"}))
    log = logging.getLogger("t")
    with caplog.at_level(logging.WARNING):
        with PhaseLock(lock_path, "aws-query", "r", logger=log, alive=lambda pid: False):
            assert json.loads(lock_path.read_text())["pid"] == os.getpid()
    assert "stale lock" in caplog.text


def test_lock_released_on_exception(tmp_path):
    lock_path = tmp_path / ".lock"
    with pytest.raises(RuntimeError):
        with PhaseLock(lock_path, "aws-probe", None):
            raise RuntimeError("boom")
    assert not lock_path.exists()


def test_progress_fields():
    rec = state.progress_record("aws-query", 50, 200, 12.34, 0.1, 0.4, "running")
    assert rec["percent"] == 25.0 and rec["status"] == "running"
    assert set(rec) == {"phase", "done", "total", "percent", "elapsed_s", "spend_tally_usd", "hard_cap_usd",
                        "updated_utc", "status"}


def _meta(entries, rid="20261004t1530-a1b2", spend=0.0):
    return {"run_id": rid, "created_resources": entries, "spend_tally_usd": spend}


@pytest.mark.parametrize("status", ["created", "pending", "delete_failed"])
def test_new_run_guard_refuses_live(tmp_path, status):
    (tmp_path / "run_metadata.json").write_text(json.dumps(_meta([{"owned": True, "status": status}])))
    with pytest.raises(UsageError, match="aws-cleanup --run-id"):
        state.new_run_guard(tmp_path)
    assert (tmp_path / "run_metadata.json").exists()


def test_new_run_guard_archives(tmp_path):
    entries = [{"owned": True, "status": "create_failed", "error_code": "ConflictException"},
               {"owned": True, "status": "deleted"}, {"owned": False, "status": "created"}]
    (tmp_path / "run_metadata.json").write_text(json.dumps(_meta(entries)))
    for name in ("queries.jsonl", "probe_classic.json", "ground_truth.jsonl", "cost_estimate.md"):
        (tmp_path / name).write_text("x")
    (tmp_path / "_partial").mkdir()
    assert state.new_run_guard(tmp_path) == "20261004t1530-a1b2"
    dest = tmp_path / "previous" / "20261004t1530-a1b2"
    assert (dest / "run_metadata.json").exists() and (dest / "queries.jsonl").exists() and (dest / "_partial").exists()
    assert (dest / "cost_estimate.md").exists()
    assert (tmp_path / "ground_truth.jsonl").exists() and (tmp_path / "cost_estimate.md").exists()
    assert not (tmp_path / "run_metadata.json").exists()


def test_prior_runs_spend_is_cumulative(tmp_path):
    """Spend of previous/aborted runs and of a different current run is summed."""
    (tmp_path / "previous" / "a").mkdir(parents=True)
    (tmp_path / "previous" / "a" / "run_metadata.json").write_text(json.dumps(_meta([], "a", 1.25)))
    (tmp_path / "previous" / "b" / "aborted" / "x").mkdir(parents=True)
    (tmp_path / "previous" / "b" / "run_metadata.json").write_text(json.dumps(_meta([], "b", 0.5)))
    (tmp_path / "aborted" / "y").mkdir(parents=True)
    (tmp_path / "aborted" / "y" / "run_metadata.json").write_text(json.dumps(_meta([], "a", 1.0)))  # same run a
    (tmp_path / "run_metadata.json").write_text(json.dumps(_meta([], "c", 2.0)))
    assert state.prior_runs_spend_usd(tmp_path, None) == pytest.approx(3.75)
    assert state.prior_runs_spend_usd(tmp_path, "c") == pytest.approx(1.75)


def test_run_state_reserve_and_charge_persist_together(tmp_path):
    rs = state.RunState(tmp_path / "m.json", {"run_id": "x", "next_request_seq": 7, "spend_tally_usd": 0.0,
                                              "hard_cap_usd": 1.0})
    assert rs.reserve_request_id() == "r000007"
    rs.charge({"query_requests": 0.25})
    on_disk = json.loads((tmp_path / "m.json").read_text())
    assert on_disk["next_request_seq"] == 8 and on_disk["spend_tally_usd"] == 0.25
    assert on_disk["spend_by_line_usd"] == {"query_requests": 0.25}
