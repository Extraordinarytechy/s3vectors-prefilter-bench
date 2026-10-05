"""Shared fixtures. Real AWS is unreachable from every test.

The BLAS thread variables are set before anything imports numpy.
"""
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import copy  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


class NetworkBlocked(RuntimeError):
    """Raised by any attempt to open a network connection inside a test."""


@pytest.fixture(autouse=True)
def no_real_aws(monkeypatch, tmp_path_factory):
    nowhere = tmp_path_factory.getbasetemp() / "no-aws-config"
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(nowhere / "config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(nowhere / "credentials"))
    monkeypatch.delenv("AWS_PROFILE", raising=False)

    def blocked(*_a, **_k):
        raise NetworkBlocked("network access is blocked in tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    from src import aws_client
    aws_client.REGISTRY.clear()
    yield
    for client in aws_client.REGISTRY:
        assert getattr(client, "_bench_test_double", False), "a GuardedS3Vectors wrapped a non-test client"
    aws_client.REGISTRY.clear()


def small_config() -> dict:
    """A small experiment with the same structure as benchmark/experiment.json (N=1000, dim=16)."""
    cfg = json.loads((REPO / "benchmark" / "experiment.json").read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["dataset"].update({"n": 1000, "dim": 16, "n_queries": 5})
    cfg["dataset"]["generator"] = {"n_clusters": 8, "latent_dim": 4, "latent_noise": 0.35, "ambient_noise": 0.05}
    cfg["dataset"]["metadata_fields"] = [
        {"name": "tenant", "counts": {"tenant-s50": 500, "tenant-s10": 100, "tenant-s1": 10, "tenant-s01": 5,
                                      "tenant-s001": 2, "tenant-few": 3, "tenant-one": 1},
         "background": {"prefix": "tenant-bg-", "count": 10, "total": 379}},
        {"name": "category", "values": ["c0", "c1", "c2", "c3"], "count_each": 250},
        {"name": "year", "values": list(range(2015, 2025)), "count_each": 100},
    ]
    cfg["constraint_probe"]["in_count"] = 10
    cfg["k_values"] = [5, 10]
    cfg["repeats"] = 2
    cfg["postfilter_repeats"] = 1
    cfg["topk_max_documented"]["value"] = 100
    cfg["budgets"] = [100]
    cfg["aws"]["put_batch_size"] = 500
    cfg["aws"]["readiness_max_polls"] = 3
    cfg["aws"]["probe_list_max_polls"] = 2
    cfg["simulator"] = {"nlist": 16, "kmeans_iters": 5, "budgets_c": [50, 200], "headline_c": 200,
                        "postfilter_pairs": [[20, 50], [100, 200]]}
    return cfg


FIXTURE_PRICING = {
    "region": "test", "source_url": "https://example.invalid/fixture", "fetched_utc": "fixture",
    "rates": {"storage_usd_per_gb_month": 1.0, "put_usd_per_gb": 1.0, "put_min_bytes": 131072,
              "other_requests_usd_per_1000": 1.0, "query_requests_usd_per_1000": 1.0,
              "data_processed_usd_per_tb_first_100k": 1.0, "data_returned_usd_per_gb": 1.0,
              "data_returned_min_bytes_per_result": 256, "data_returned_free_bytes_per_query": 524288},
}


def make_root(path: Path, cfg: dict | None = None, pricing: dict | None = None) -> Path:
    (path / "benchmark").mkdir(parents=True, exist_ok=True)
    (path / "benchmark" / "experiment.json").write_text(json.dumps(cfg or small_config()), encoding="utf-8")
    (path / "benchmark" / "pricing_us-east-1.json").write_text(json.dumps(pricing or FIXTURE_PRICING),
                                                               encoding="utf-8")
    return path


@pytest.fixture
def dummy_profile(tmp_path, monkeypatch):
    """A throwaway `default` profile with dummy keys, so the real boto3 construction path can be exercised."""
    cfg = tmp_path / "aws-config"
    creds = tmp_path / "aws-credentials"
    cfg.write_text("[default]\nregion = us-east-1\n", encoding="utf-8")
    creds.write_text("[default]\naws_access_key_id = testing\naws_secret_access_key = testing\n", encoding="utf-8")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(cfg))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(creds))
    return "default"


@pytest.fixture
def small_cfg():
    return small_config()


@pytest.fixture(scope="session")
def offline_root(tmp_path_factory):
    """A small root with generate -> ground-truth -> simulate -> report already run (shared, read-only)."""
    from src import runner

    root = make_root(tmp_path_factory.mktemp("offline"))
    assert runner.main(["offline-all"], root=root) == 0
    return root


@pytest.fixture
def aws_root(tmp_path, offline_root):
    """A fresh copy of the offline root with an approved estimate, ready for the AWS phases."""
    from src import runner

    root = tmp_path / "root"
    shutil.copytree(offline_root, root)
    assert runner.main(["estimate"], root=root) == 0
    return root


def stubbed_client():
    """A real botocore s3vectors client (dummy credentials) wrapped by a Stubber. No network is ever used."""
    import boto3
    from botocore.config import Config
    from botocore.stub import Stubber

    client = boto3.session.Session(aws_access_key_id="testing", aws_secret_access_key="testing",
                                   region_name="us-east-1").client(
        "s3vectors", config=Config(retries={"total_max_attempts": 1, "mode": "standard"}))
    client._bench_test_double = True
    stub = Stubber(client)
    return client, stub
