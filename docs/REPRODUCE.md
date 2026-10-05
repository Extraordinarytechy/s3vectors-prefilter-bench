# Reproduce the benchmark

Commands run from the repository root in a Linux or WSL bash shell with Python 3.10. In WSL the shell can return before a long command finishes, so redirect output to a log under `logs/` and poll it.

## 1. Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import boto3, botocore; print(boto3.__version__, botocore.__version__)"   # 1.43.107 1.43.107
.venv/bin/python -u -m pytest -q tests
```

`requirements.txt` pins exact versions (boto3/botocore 1.43.107, numpy 2.2.6, pandas 2.3.3, matplotlib 3.10.9, pytest 9.1.1). Older boto3 builds have no `s3vectors` client. The tests use a fake S3 Vectors client, and one test proves no test path calls real AWS.

## 2. Offline run (no AWS, no credentials)

```bash
.venv/bin/python -u -m src.runner offline-all     # generate -> ground-truth -> simulate -> report
.venv/bin/python -u -m src.runner estimate        # writes results/aws/cost_estimate.md
```

This regenerates `data/` (checked against `data/manifest.json` hashes), `results/aws/ground_truth.jsonl`, `results/simulator/`, `results/processed/` and `figures/`. With the committed `results/aws/` evidence in place, `report` rebuilds the REAL AWS tables and figures from the raw files.

Determinism check:

```bash
.venv/bin/python -u -m src.runner offline-all --out-root runs/run1
.venv/bin/python -u -m src.runner offline-all --out-root runs/run2
diff -r runs/run1 runs/run2 && echo IDENTICAL
```

## 3. Real AWS run (own account, billable)

This run cost $0.0731 by the runner's pricing model. Your cost depends on the current pricing page.

1. Run `estimate` and read `results/aws/cost_estimate.md`. Re-check the pricing page and the topK limit on the day; if either changed, edit `benchmark/` and rerun `estimate`.
2. Use only an account you control. The profile must be given explicitly.
3. Launch each phase detached and poll its log. The last line is `PHASE COMPLETE`, `PHASE STOPPED` or `PHASE FAILED`.

```bash
mkdir -p logs && setsid nohup .venv/bin/python -u -m src.runner aws-probe --aws --profile <profile> --region us-east-1 --confirm-cost > logs/aws-probe.log 2>&1 < /dev/null &
```

Phases in order (`aws-probe` prints `RUN_ID <id>`):

```bash
.venv/bin/python -u -m src.runner aws-probe   --aws --profile <profile> --region us-east-1 --confirm-cost
.venv/bin/python -u -m src.runner aws-ingest  --aws --profile <profile> --region us-east-1 --confirm-cost --run-id <id>
.venv/bin/python -u -m src.runner aws-query   --aws --profile <profile> --region us-east-1 --confirm-cost --run-id <id>
.venv/bin/python -u -m src.runner aws-capture --aws --profile <profile> --region us-east-1 --run-id <id>
.venv/bin/python -u -m src.runner aws-cleanup --aws --profile <profile> --region us-east-1 --run-id <id>
.venv/bin/python -u -m src.runner report
.venv/bin/python -u -m src.runner verify-redaction
```

- Exit codes: 0 complete, 1 failed, 2 refused input (nothing sent), 3 deliberate stop.
- `--confirm-cost` asserts that the approved estimate is the one on disk. The runner refuses if the config, pricing, or prior spend changed since then.
- `aws-probe` exits 3 with `CLASSIC_OBTAINABLE` if AWS unexpectedly lets the new bucket use CLASSIC. Only `aws-cleanup` may run after that.
- If you want Console screenshots, take them between `aws-query` and `aws-cleanup`, and hide the account ID, ARNs and email.
- `aws-cleanup` deletes only the resources recorded for that run, verifies with list calls, writes `results/aws/cleanup.json`, and can be rerun at any time. Run it even after a failure.

Expected outcome on a bucket created after Sep 30, 2026: the CLASSIC probe is rejected, the index is ENHANCED, and the recall numbers are close to `results/processed/tables.md` (the AWS service may change over time).

## 4. Borrowed-account mode (advanced): read this warning first

> **WARNING.** This mode runs inside another person's AWS account, in their vector bucket. It exists only to measure real CLASSIC, which needs a bucket created **before Sep 30, 2026**. It was not exercised in the committed run; it's covered by unit tests against a fake client only.
>
> - Get the bucket owner's explicit permission, and agree on cost (they pay) and duration before you start. Use credentials they issued for this purpose.
> - It creates exactly **one** new disposable index, `s3vectors-prefilter-bench-<id>`, in the borrowed bucket, tagged and recorded as owned. It never modifies, tags, re-defaults, or deletes the bucket or anything else in it.
> - It never calls `PutVectorBucketDefaultIndexMode` on the borrowed bucket, because that would change the mode of the owner's future indexes.
> - It never calls `UpdateIndexMode`, `PutVectors`, `DeleteVectors` or any delete on an index it didn't create. There's no Stage C (`UpdateIndexMode → ENHANCED`) in borrowed mode.
> - If the new index isn't CLASSIC, the run logs an error, deletes only that new index, and exits 1.
> - Cleanup deletes only that index. The bucket is never touched.
> - The CLASSIC probe, the constraint probe and Console screenshots are not run in this mode.
> - Never commit the owner's account ID, ARNs, or credentials. Run `verify-redaction` before sharing anything.

Steps:

```bash
.venv/bin/python -u -m src.runner estimate --borrowed
# invocation 1: read-only; generates the run id, records the bucket as not owned, and stops (exit 2)
.venv/bin/python -u -m src.runner aws-ingest --aws --profile <p> --region <r> --confirm-cost --borrowed-bucket <name>
# invocation 2: must repeat the exact planned index name
.venv/bin/python -u -m src.runner aws-ingest --aws --profile <p> --region <r> --confirm-cost --borrowed-bucket <name> --run-id <id> --confirm-index-name s3vectors-prefilter-bench-<id>
.venv/bin/python -u -m src.runner aws-query   --aws --profile <p> --region <r> --confirm-cost --run-id <id>
.venv/bin/python -u -m src.runner aws-capture --aws --profile <p> --region <r> --run-id <id>
.venv/bin/python -u -m src.runner aws-cleanup --aws --profile <p> --region <r> --run-id <id>
```

`aws-query` in this mode runs Stage A (`queryMode=CLASSIC`), Stage B (`queryMode=ENHANCED` on the same CLASSIC index), the ANN reference, and the post-filter baseline, on that one index.

## 5. Where to look

| Output | File |
|---|---|
| Run metadata, manifest, spend tally | `results/aws/run_metadata.json` |
| Requests, responses, timings | `results/aws/queries.jsonl`, `responses.jsonl`, `timings.jsonl` |
| Ground truth | `results/aws/ground_truth.jsonl` |
| CLASSIC probe | `results/aws/probe_classic.json` |
| Metrics, constraint probe, failed requests | `results/aws/metrics.json` |
| Cleanup verification | `results/aws/cleanup.json` |
| Summary tables | `results/processed/summary.json`, `summary.csv`, `tables.md`, `latency.json` |
| Figures | `figures/` |
| Terminal captures and Console shot list (written by a new own-account run) | `results/aws/captures/`, `results/aws/SHOT_LIST.md` |
| Run logs and progress | `logs/run-<id>.log`, `logs/progress-<id>.json` |
| Method | `docs/METHODOLOGY.md` |
