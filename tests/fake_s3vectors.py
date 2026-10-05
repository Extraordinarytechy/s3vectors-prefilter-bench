"""In-memory fake of the S3 Vectors client surface the benchmark uses (offline tests only).

ENHANCED queries are exact filter-first (filter, then brute-force cosine), paged by 100.
Behavior knobs let tests model AWS outcomes that are undocumented or unexpected:
- `default_mode_classic`: "reject" (ValidationException), "deny" (AccessDeniedException) or "accept"
- `update_classic`: "reject", "deny" or "accept"
- `classic_query_on_enhanced`: "reject", "accept", or "accept_main" (accepted on non-probe indexes only)
- `classic_query_underfills`: CLASSIC-mode queries on a CLASSIC index return no results
"""
from __future__ import annotations

import numpy as np
from botocore.exceptions import ClientError

from src import filters as flt

ACCOUNT = "000000000000"


def _err(code: str, message: str, status: int, op: str):
    return ClientError({"Error": {"Code": code, "Message": message},
                        "ResponseMetadata": {"HTTPStatusCode": status}}, op)


def _ok(**body):
    return {"ResponseMetadata": {"HTTPStatusCode": 200, "RequestId": "FAKE"}, **body}


class FakeS3Vectors:
    _bench_test_double = True

    def __init__(self, *, default_mode_classic="reject", update_classic="reject",
                 classic_query_on_enhanced="reject", classic_query_underfills=False,
                 bucket_default_mode="ENHANCED", preexisting=(), fail=None):
        self.buckets: dict[str, dict] = {}
        self.calls: list[tuple[str, dict]] = []
        self.default_mode_classic = default_mode_classic
        self.update_classic = update_classic
        self.classic_query_on_enhanced = classic_query_on_enhanced
        self.classic_query_underfills = classic_query_underfills
        self.bucket_default_mode = bucket_default_mode
        self.fail = fail or {}  # api -> list of error codes to raise first, in order
        for name in preexisting:
            self._new_bucket(name, "ENHANCED")

    # ---- helpers

    def _new_bucket(self, name, mode):
        self.buckets[name] = {"default": mode, "indexes": {}, "tags": {}}

    def _record(self, api, params):
        self.calls.append((api, params))
        queue = self.fail.get(api)
        if queue:
            code = queue.pop(0)
            status = {"TooManyRequestsException": 429, "AccessDeniedException": 403}.get(code, 400)
            raise _err(code, f"injected {code} for arn:aws:iam::{ACCOUNT}:user/x", status, api)

    def _bucket(self, name, op):
        if name not in self.buckets:
            raise _err("NotFoundException", "bucket not found", 404, op)
        return self.buckets[name]

    def _index(self, bucket, index, op):
        b = self._bucket(bucket, op)
        if index not in b["indexes"]:
            raise _err("NotFoundException", "index not found", 404, op)
        return b["indexes"][index]

    def api_names(self):
        return [c[0] for c in self.calls]

    # ---- buckets

    def create_vector_bucket(self, **p):
        self._record("create_vector_bucket", p)
        if p["vectorBucketName"] in self.buckets:
            raise _err("ConflictException", "exists", 409, "CreateVectorBucket")
        self._new_bucket(p["vectorBucketName"], self.bucket_default_mode)
        return _ok(vectorBucketArn=f"arn:aws:s3vectors:us-east-1:{ACCOUNT}:bucket/{p['vectorBucketName']}")

    def get_vector_bucket(self, **p):
        self._record("get_vector_bucket", p)
        b = self._bucket(p["vectorBucketName"], "GetVectorBucket")
        return _ok(vectorBucket={"vectorBucketName": p["vectorBucketName"], "defaultIndexMode": b["default"],
                                 "vectorBucketArn": f"arn:aws:s3vectors:us-east-1:{ACCOUNT}:bucket/"
                                                    f"{p['vectorBucketName']}"})

    def put_vector_bucket_default_index_mode(self, **p):
        self._record("put_vector_bucket_default_index_mode", p)
        b = self._bucket(p["vectorBucketName"], "PutVectorBucketDefaultIndexMode")
        if p["defaultIndexMode"] == "CLASSIC":
            if self.default_mode_classic == "reject":
                raise _err("ValidationException", "CLASSIC not allowed for this bucket", 400, "Put")
            if self.default_mode_classic == "deny":
                raise _err("AccessDeniedException", f"User: arn:aws:iam::{ACCOUNT}:user/x denied", 403, "Put")
        b["default"] = p["defaultIndexMode"]
        return _ok()

    def list_vector_buckets(self, **p):
        self._record("list_vector_buckets", p)
        names = sorted(n for n in self.buckets if n.startswith(p.get("prefix", "")))
        return _ok(vectorBuckets=[{"vectorBucketName": n} for n in names])

    def delete_vector_bucket(self, **p):
        self._record("delete_vector_bucket", p)
        b = self._bucket(p["vectorBucketName"], "DeleteVectorBucket")
        if b["indexes"]:
            raise _err("ConflictException", "bucket not empty", 409, "DeleteVectorBucket")
        del self.buckets[p["vectorBucketName"]]
        return _ok()

    def tag_resource(self, **p):
        self._record("tag_resource", p)
        return _ok()

    # ---- indexes

    def create_index(self, **p):
        self._record("create_index", p)
        b = self._bucket(p["vectorBucketName"], "CreateIndex")
        if p["indexName"] in b["indexes"]:
            raise _err("ConflictException", "exists", 409, "CreateIndex")
        b["indexes"][p["indexName"]] = {"mode": b["default"], "dim": p["dimension"],
                                        "metric": p["distanceMetric"], "vectors": {}}
        return _ok(indexArn=f"arn:aws:s3vectors:us-east-1:{ACCOUNT}:bucket/{p['vectorBucketName']}"
                            f"/index/{p['indexName']}")

    def get_index(self, **p):
        self._record("get_index", p)
        ix = self._index(p["vectorBucketName"], p["indexName"], "GetIndex")
        return _ok(index={"indexName": p["indexName"], "vectorBucketName": p["vectorBucketName"],
                          "dimension": ix["dim"], "distanceMetric": ix["metric"], "dataType": "float32",
                          "indexMode": ix["mode"],
                          "indexArn": f"arn:aws:s3vectors:us-east-1:{ACCOUNT}:bucket/x/index/y"})

    def list_indexes(self, **p):
        self._record("list_indexes", p)
        b = self._bucket(p["vectorBucketName"], "ListIndexes")
        names = sorted(n for n in b["indexes"] if n.startswith(p.get("prefix", "")))
        return _ok(indexes=[{"indexName": n} for n in names])

    def update_index_mode(self, **p):
        self._record("update_index_mode", p)
        ix = self._index(p["vectorBucketName"], p["indexName"], "UpdateIndexMode")
        if p["indexMode"] == "CLASSIC":
            if self.update_classic == "reject":
                raise _err("ValidationException", "CLASSIC only for pre-Sep-30 buckets", 400, "UpdateIndexMode")
            if self.update_classic == "deny":
                raise _err("AccessDeniedException", f"User: arn:aws:iam::{ACCOUNT}:user/x denied", 403,
                           "UpdateIndexMode")
        ix["mode"] = p["indexMode"]
        return _ok()

    def delete_index(self, **p):
        self._record("delete_index", p)
        b = self._bucket(p["vectorBucketName"], "DeleteIndex")
        if p["indexName"] not in b["indexes"]:
            raise _err("NotFoundException", "index not found", 404, "DeleteIndex")
        del b["indexes"][p["indexName"]]
        return _ok()

    # ---- vectors

    def put_vectors(self, **p):
        self._record("put_vectors", p)
        ix = self._index(p["vectorBucketName"], p["indexName"], "PutVectors")
        for v in p["vectors"]:
            ix["vectors"][v["key"]] = (np.asarray(v["data"]["float32"], dtype=np.float64), v.get("metadata", {}))
        return _ok()

    def list_vectors(self, **p):
        self._record("list_vectors", p)
        ix = self._index(p["vectorBucketName"], p["indexName"], "ListVectors")
        keys = sorted(ix["vectors"])
        start = int(p.get("nextToken", "0"))
        size = p.get("maxResults", 500)
        page = keys[start:start + size]
        out = {"vectors": [{"key": k} for k in page]}
        if start + size < len(keys):
            out["nextToken"] = str(start + size)
        return _ok(**out)

    def query_vectors(self, **p):
        self._record("query_vectors", p)
        ix = self._index(p["vectorBucketName"], p["indexName"], "QueryVectors")
        mode = p.get("queryMode")
        accept = self.classic_query_on_enhanced == "accept" or (
            self.classic_query_on_enhanced == "accept_main" and not p["indexName"].endswith("-probe"))
        if mode == "CLASSIC" and ix["mode"] == "ENHANCED" and not accept:
            raise _err("ValidationException", "You can't specify CLASSIC for an ENHANCED index", 400, "Query")
        flt_json = p.get("filter")
        if flt_json is not None and flt.count_constraints(flt_json) > 100 and mode != "CLASSIC" \
                and ix["mode"] == "ENHANCED":
            raise _err("ValidationException", "too many filter constraints", 400, "QueryVectors")
        effective = mode or ix["mode"]
        if effective == "CLASSIC" and ix["mode"] == "CLASSIC" and self.classic_query_underfills:
            return _ok(vectors=[], distanceMetric="cosine")
        q = np.asarray(p["queryVector"]["float32"], dtype=np.float64)
        rows = []
        for key, (vec, meta) in ix["vectors"].items():
            if flt_json is None or flt.matches(flt_json, meta):
                d = 1.0 - float(vec @ q) / (np.linalg.norm(vec) * np.linalg.norm(q))
                rows.append((d, key))
        rows.sort()
        rows = rows[: p["topK"]]
        start = int(p.get("nextToken", "0"))
        page = rows[start:start + 100]
        out = {"vectors": [{"key": k, "distance": d} for d, k in page], "distanceMetric": "cosine"}
        if start + 100 < len(rows):
            out["nextToken"] = str(start + 100)
        return _ok(**out)
