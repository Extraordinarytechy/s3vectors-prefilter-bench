# s3vectors-prefilter-bench

A reproducible benchmark of Amazon S3 Vectors metadata pre-filtering (ENHANCED index mode), scored by
ground-truth Recall@K against exact, locally computed nearest neighbors on a deterministic synthetic dataset
(50,000 x 384, cosine). The method is in `docs/METHODOLOGY.md`.

Evidence classes are kept separate everywhere: REAL AWS measurements (ENHANCED filtered queries, a client-side
post-filter baseline that is never called CLASSIC, the CLASSIC probe, client-observed round trips), the
SIMULATED CLASSIC/ENHANCED mechanism model, and AWS documented facts.

## Setup

Python 3.10, from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import boto3, botocore; print(boto3.__version__, botocore.__version__)"   # 1.43.107 1.43.107
.venv/bin/python -u -m pytest -q tests
```

`requirements.txt` pins exact versions. Every phase sets `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and
`MKL_NUM_THREADS` to 1 before numpy loads, so the offline outputs are byte-identical between runs.

## Simulator-only run (no AWS, no credentials, boto3 is never imported)

```bash
.venv/bin/python -u -m src.runner generate
.venv/bin/python -u -m src.runner ground-truth
.venv/bin/python -u -m src.runner simulate
.venv/bin/python -u -m src.runner report
.venv/bin/python -u -m src.runner offline-all [--out-root DIR]   # generate -> ground-truth -> simulate -> report
.venv/bin/python -u -m src.runner estimate                       # writes results/aws/cost_estimate.md
```

Outputs: `data/` (canonical dataset; the two large files regenerate from the seed and are checked against
`data/manifest.json`), `results/aws/ground_truth.jsonl`, `results/simulator/`, `results/processed/` and
`figures/`. Simulator results are labeled SIMULATED: they show a mechanism, not AWS internals or AWS numbers.

Double-run determinism check:

```bash
.venv/bin/python -u -m src.runner offline-all --out-root runs/run1
.venv/bin/python -u -m src.runner offline-all --out-root runs/run2
diff -r runs/run1 runs/run2 && echo IDENTICAL
```

## Opt-in real AWS run

Real AWS calls happen only in `aws-*` phases, and only with explicit flags. First run `estimate`, read
`results/aws/cost_estimate.md`, and get explicit approval. `--confirm-cost` is the operator's statement that the
approved estimate is the one on disk; the runner refuses if the config, pricing, or prior-run spend changed since
(plan hash). Each `aws-*` phase is launched detached, because the shell may return early:

```bash
mkdir -p logs && setsid nohup .venv/bin/python -u -m src.runner aws-query --aws --profile default --region us-east-1 --confirm-cost --run-id <id> > logs/aws-query-<id>.log 2>&1 < /dev/null &
```

```bash
.venv/bin/python -u -m src.runner aws-probe   --aws --profile default --region us-east-1 --confirm-cost
.venv/bin/python -u -m src.runner aws-ingest  --aws --profile default --region us-east-1 --confirm-cost --run-id <id>
.venv/bin/python -u -m src.runner aws-query   --aws --profile default --region us-east-1 --confirm-cost --run-id <id>
.venv/bin/python -u -m src.runner aws-capture --aws --profile default --region us-east-1 --run-id <id>
.venv/bin/python -u -m src.runner aws-cleanup --aws --profile default --region us-east-1 --run-id <id>
.venv/bin/python -u -m src.runner verify-redaction
```

`aws-probe` prints `RUN_ID <id>`; the id is also `run_id` in `results/aws/run_metadata.json`. Poll the log (its
last line is `PHASE COMPLETE`, `PHASE STOPPED` or `PHASE FAILED`) and `logs/progress-<id>.json`.

- Exit codes: 0 complete, 1 failed, 2 refused input or lifecycle rule (nothing sent), 3 deliberate stop.
- `aws-probe` exits 3 with `stopped_reason = CLASSIC_OBTAINABLE` if AWS lets this new bucket obtain CLASSIC (bucket
  default, `UpdateIndexMode`, or an accepted `queryMode=CLASSIC`). Only `aws-cleanup` may run after that.
- If `queryMode=CLASSIC` is accepted on the ENHANCED main index (probe step 13), `aws-query` finishes the ENHANCED
  grid, sets `classic_query_accepted_on_main = true`, and logs a `CLASSIC_QUERY_ACCEPTED_ON_MAIN` warning.
- The ownership guard refuses any mutating call on a resource this run did not create (manifest in
  `run_metadata.json`), and a create is sent only after a read-only check shows the name does not exist.

### Phase x mode

| Phase | Own account (default, the primary run) | Borrowed `--borrowed-bucket <name>` (advanced) |
|---|---|---|
| `estimate` | base plan | `estimate --borrowed`: one index ingest and polling, Stage A, Stage B, ANN reference, (b) baseline, capture, cleanup |
| `aws-probe` | CLASSIC probe steps 1–12. Exit 3 and stop if CLASSIC is obtainable | not run (exit 2 if the run is in borrowed mode) |
| `aws-ingest` | main bucket and index, ingest, readiness (expects ENHANCED) | invocation 1: read-only, generates the id and stops. Invocation 2 (confirmed): creates the `borrowed_index`, `GetIndex`, aborts unless CLASSIC, ingest, readiness |
| `aws-query` | step 13, then the (a), ANN-reference, and (b) passes, then C100/C101 | Stage A, Stage B, ANN reference, (b) baseline |
| `aws-capture` | `GetIndex` and one query on the main index, plus the probe-records capture | `GetIndex` and one query on the `borrowed_index` |
| `aws-cleanup` | all owned resources | the `borrowed_index` only; the bucket is never touched |

## Borrowed-account mode (advanced; read the warnings in `docs/REPRODUCE.md` first)

For someone else's vector bucket created before Sep 30, 2026. It creates exactly one new disposable index in
that bucket and never modifies, tags, re-defaults, or deletes the bucket or anything else in it.

```bash
.venv/bin/python -u -m src.runner estimate --borrowed
.venv/bin/python -u -m src.runner aws-ingest --aws --profile <p> --region <r> --confirm-cost --borrowed-bucket <name>
.venv/bin/python -u -m src.runner aws-ingest --aws --profile <p> --region <r> --confirm-cost --borrowed-bucket <name> --run-id <id> --confirm-index-name s3vectors-prefilter-bench-<id>
# then aws-query / aws-capture / aws-cleanup with --run-id <id>
```

## Cost

See the generated `results/aws/cost_estimate.md`. It is computed from `benchmark/pricing_us-east-1.json`, which
is transcribed from the official S3 pricing page; no price is hardcoded in code. The $5 cap is cumulative across
runs: the hard cap is `min(2 x base, 5.00 - spend of all earlier runs)`, and every sent attempt is charged before
it is sent.

## Cleanup

```bash
.venv/bin/python -u -m src.runner aws-cleanup --aws --profile default --region us-east-1 --run-id <id>
```

It deletes only resources recorded as owned by that run, verifies with list calls, writes
`results/aws/cleanup.json`, never needs `--confirm-cost`, and may be rerun at any time.
