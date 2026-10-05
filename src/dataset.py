"""Deterministic vectors, queries and metadata (DESIGN §4).

Draw order (round-3 NIT 7) in the `vectors` stream: centers, W, cluster ids, latent noise,
ambient noise. Queries draw cluster ids, latent noise and ambient noise in the same order
from the `queries` stream and reuse the dataset's centers and W. Every vector and every
query is L2-normalized with the same rule (NIT 6): normalize in float64, then cast to float32.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .state import IntegrityError, atomic_write_json, dumps, read_json

STREAM_NAMES = ("vectors", "queries", "metadata", "simulator", "bootstrap", "order")


def stream(seed: int, name: str, *extra: int) -> np.random.SeedSequence:
    """Child `name` of SeedSequence(seed).spawn(6); `extra` derives a grandchild deterministically."""
    idx = STREAM_NAMES.index(name)
    return np.random.SeedSequence(seed, spawn_key=(idx, *extra))


def key_for(i: int) -> str:
    return f"v-{i:05d}"


def query_id_for(i: int) -> str:
    return f"q-{i:02d}"


def _normalize(x64: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x64, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.all(np.isfinite(x64)):
        raise IntegrityError("zero or non-finite vector generated")
    return (x64 / norms).astype(np.float32)


def generate_vectors(ds: dict) -> tuple[np.ndarray, np.ndarray]:
    g = ds["generator"]
    n, dim, q = ds["n"], ds["dim"], ds["n_queries"]
    nc, ld = g["n_clusters"], g["latent_dim"]
    rng = np.random.default_rng(stream(ds["seed"], "vectors"))
    centers = rng.normal(0.0, 1.0, (nc, ld))
    w = rng.normal(0.0, 1.0 / np.sqrt(ld), (ld, dim))
    ids = rng.integers(0, nc, n)
    latent = centers[ids] + g["latent_noise"] * rng.normal(0.0, 1.0, (n, ld))
    x = latent @ w + g["ambient_noise"] * rng.normal(0.0, 1.0, (n, dim))
    rq = np.random.default_rng(stream(ds["seed"], "queries"))
    qids = rq.integers(0, nc, q)
    qlatent = centers[qids] + g["latent_noise"] * rq.normal(0.0, 1.0, (q, ld))
    xq = qlatent @ w + g["ambient_noise"] * rq.normal(0.0, 1.0, (q, dim))
    return _normalize(x), _normalize(xq)


def field_labels(field: dict, n: int) -> list:
    """Unshuffled label list with the exact counts of one metadata field."""
    labels: list = []
    if "counts" in field:
        for value, count in field["counts"].items():
            labels.extend([value] * count)
    if "background" in field:
        bg = field["background"]
        base, extra = divmod(bg["total"], bg["count"])
        width = len(str(bg["count"] - 1))
        for j in range(bg["count"]):
            labels.extend([f"{bg['prefix']}{j:0{width}d}"] * (base + (1 if j < extra else 0)))
    if "values" in field:
        for value in field["values"]:
            labels.extend([value] * field["count_each"])
    if len(labels) != n:
        raise IntegrityError(f"field {field['name']} has {len(labels)} labels, expected {n}")
    return labels


def generate_metadata(ds: dict) -> list[dict]:
    n = ds["n"]
    rng = np.random.default_rng(stream(ds["seed"], "metadata"))
    columns = {}
    for field in ds["metadata_fields"]:
        labels = field_labels(field, n)
        perm = rng.permutation(n)
        columns[field["name"]] = [labels[i] for i in perm]
    return [{"key": key_for(i), **{name: col[i] for name, col in columns.items()}} for i in range(n)]


def background_values(ds: dict, field_name: str) -> list[str]:
    for field in ds["metadata_fields"]:
        if field["name"] == field_name and "background" in field:
            bg = field["background"]
            width = len(str(bg["count"] - 1))
            return [f"{bg['prefix']}{j:0{width}d}" for j in range(bg["count"])]
    raise IntegrityError(f"no background values for {field_name}")


def metadata_lines(metadata: list[dict]) -> bytes:
    return "".join(dumps(m) + "\n" for m in metadata).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def filterable_metadata_bytes(meta: dict) -> int:
    """Bytes of the filterable metadata JSON as uploaded (the key is not metadata)."""
    return len(dumps({k: v for k, v in meta.items() if k != "key"}).encode("utf-8"))


class Dataset:
    """Canonical in-memory copy loaded from (or written to) data/."""

    def __init__(self, vectors: np.ndarray, queries: np.ndarray, metadata: list[dict]):
        self.vectors = vectors
        self.queries = queries
        self.metadata = metadata
        self.keys = [m["key"] for m in metadata]
        self.key_index = {k: i for i, k in enumerate(self.keys)}
        self.query_ids = [query_id_for(i) for i in range(len(queries))]
        self.columns = {name: np.array([m[name] for m in metadata])
                        for name in metadata[0] if name != "key"} if metadata else {}

    @property
    def n(self) -> int:
        return len(self.keys)


def build(ds: dict) -> Dataset:
    vectors, queries = generate_vectors(ds)
    return Dataset(vectors, queries, generate_metadata(ds))


def write(dataset: Dataset, ds: dict, data_dir: Path, filter_counts: dict, code_sha256: str) -> dict:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "vectors.npy", dataset.vectors, allow_pickle=False)
    np.save(data_dir / "queries.npy", dataset.queries, allow_pickle=False)
    meta_bytes = metadata_lines(dataset.metadata)
    with open(data_dir / "metadata.jsonl", "wb") as fh:
        fh.write(meta_bytes)
    distributions = {}
    for field in ds["metadata_fields"]:
        values, counts = np.unique(dataset.columns[field["name"]], return_counts=True)
        distributions[field["name"]] = {str(v): int(c) for v, c in zip(values, counts)}
    key_bytes = [len(k.encode("utf-8")) for k in dataset.keys]
    md_bytes = [filterable_metadata_bytes(m) for m in dataset.metadata]
    manifest = {
        "n": ds["n"], "dim": ds["dim"], "metric": ds["metric"], "seed": ds["seed"],
        "n_queries": ds["n_queries"], "dtype": "float32", "generator": ds["generator"],
        "seed_streams": list(STREAM_NAMES),
        "field_distributions": distributions,
        "filter_match_counts": filter_counts,
        "bytes": {"vector_data": ds["dim"] * 4, "key_mean": float(np.mean(key_bytes)),
                  "filterable_metadata_mean": float(np.mean(md_bytes)),
                  "filterable_metadata_max": int(np.max(md_bytes))},
        "sha256": {"vectors.npy_tobytes": sha256_bytes(dataset.vectors.tobytes()),
                   "queries.npy_tobytes": sha256_bytes(dataset.queries.tobytes()),
                   "metadata.jsonl": sha256_bytes(meta_bytes)},
        "generator_code_sha256": code_sha256,
    }
    atomic_write_json(data_dir / "manifest.json", manifest)
    return manifest


def load(data_dir: Path) -> tuple[Dataset, dict]:
    """Load the canonical files and verify their hashes against manifest.json (§13.2)."""
    data_dir = Path(data_dir)
    manifest = read_json(data_dir / "manifest.json")
    vectors = np.load(data_dir / "vectors.npy", allow_pickle=False)
    queries = np.load(data_dir / "queries.npy", allow_pickle=False)
    with open(data_dir / "metadata.jsonl", "rb") as fh:
        meta_bytes = fh.read()
    actual = {"vectors.npy_tobytes": sha256_bytes(vectors.tobytes()),
              "queries.npy_tobytes": sha256_bytes(queries.tobytes()),
              "metadata.jsonl": sha256_bytes(meta_bytes)}
    if actual != manifest["sha256"]:
        raise IntegrityError("canonical data sha256 does not match data/manifest.json")
    metadata = [json.loads(line) for line in meta_bytes.decode("utf-8").splitlines() if line]
    return Dataset(vectors, queries, metadata), manifest


def bytes_per_vector(manifest: dict) -> int:
    """Average logical bytes per vector (vector data + key + filterable metadata), rounded up."""
    b = manifest["bytes"]
    return int(np.ceil(b["vector_data"] + b["key_mean"] + b["filterable_metadata_mean"]))
