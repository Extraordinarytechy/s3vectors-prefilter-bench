"""Proof that no test path can call real AWS."""
import json
import os
import socket
import subprocess
import sys

import pytest

from src import aws_client
from tests.conftest import REPO, NetworkBlocked, make_root


def test_sockets_are_blocked():
    with pytest.raises(NetworkBlocked):
        socket.create_connection(("s3vectors.us-east-1.api.aws", 443), timeout=1)
    with pytest.raises(NetworkBlocked):
        socket.socket().connect(("127.0.0.1", 9))


def test_real_boto3_call_path_cannot_reach_aws(dummy_profile):
    """The production client factory, used for real, fails before any byte leaves the machine."""
    client = aws_client.make_boto3_client(dummy_profile, "us-east-1")
    with pytest.raises(Exception) as info:
        client.list_vector_buckets()
    chain, exc = [], info.value
    while exc is not None:
        chain.append(exc)
        exc = exc.__cause__ or exc.__context__
    assert any(isinstance(e, NetworkBlocked) or "NetworkBlocked" in repr(e) for e in chain), repr(info.value)


def test_missing_profile_is_a_credential_failure():
    from botocore.exceptions import ProfileNotFound
    with pytest.raises(ProfileNotFound):
        aws_client.make_boto3_client("default", "us-east-1")


def test_credentials_are_dummies():
    assert os.environ["AWS_ACCESS_KEY_ID"] == "testing"
    assert not os.path.exists(os.environ["AWS_SHARED_CREDENTIALS_FILE"])
    assert not os.path.exists(os.environ["AWS_CONFIG_FILE"])


def test_guarded_client_registry_only_holds_test_doubles(tmp_path):
    import logging
    from src.aws_client import GuardContext, GuardedS3Vectors
    from src.state import RunState
    from tests.fake_s3vectors import FakeS3Vectors

    rs = RunState(tmp_path / "m.json", {"run_id": "x", "hard_cap_usd": 1.0, "spend_tally_usd": 0.0,
                                        "created_resources": []})
    GuardedS3Vectors(FakeS3Vectors(), rs, GuardContext("x", "aws-query"), logging.getLogger("t"))
    assert all(getattr(c, "_bench_test_double", False) for c in aws_client.REGISTRY)
    # The autouse fixture re-checks this registry after every test in the suite.


def test_offline_runner_never_imports_boto3(tmp_path):
    root = make_root(tmp_path / "root")
    code = ("import sys; sys.path.insert(0, %r); from src import runner; from pathlib import Path; "
            "rc = runner.main(['generate'], root=Path(%r)); "
            "rc2 = runner.main(['ground-truth'], root=Path(%r)); "
            "print(json.dumps({'rc': [rc, rc2], 'boto3': 'boto3' in sys.modules, 'botocore': 'botocore' in sys.modules}))"
            % (str(REPO), str(root), str(root)))
    out = subprocess.run([sys.executable, "-c", "import json; " + code], capture_output=True, text=True,
                         timeout=300, env={**os.environ, "PYTHONPATH": str(REPO)})
    assert out.returncode == 0, out.stderr
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res == {"rc": [0, 0], "boto3": False, "botocore": False}
