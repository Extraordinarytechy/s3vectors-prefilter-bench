# Methodology

How the benchmark in run `20261005t0953-a9ba` (us-east-1, Oct 5, 2026) was run and measured. This file describes what was actually done. Every number cited here is in `results/processed/summary.json`, `latency.json`, or a file under `results/aws/`.

## Question

What changes when Amazon S3 Vectors filters with ENHANCED metadata pre-filtering, especially for highly selective filters? The primary metric is ground-truth Recall@K. Result count (completeness) is secondary.

## Evidence classes

Kept separate in data, tables, charts and prose:

1. **REAL AWS**
   - (a) ENHANCED filtered queries, plus an unfiltered ANN reference on the same index
   - (b) a client-side post-filter baseline: an unfiltered query at topK = B, then the filter applied locally. It is never called CLASSIC.
   - (c) the CLASSIC probe: responses and errors
   - (d) client-observed round-trip latency, including the network
2. **SIMULATED**: a numpy mechanism model of CLASSIC and ENHANCED. Not AWS.
3. **AWS documented**: facts and claims from primary AWS pages (`research/AWS_FACTS.md`, `research/SOURCE_LEDGER.md`).

## Dataset

- Deterministic and synthetic: N = 50,000, 384 dimensions, float32, cosine, master seed 20260930 (`numpy.random.SeedSequence` with independent streams for vectors, queries, metadata, simulator, bootstrap and query order).
- Vectors come from a 64-cluster Gaussian mixture in 32 latent dimensions, projected to 384 with small noise and L2-normalized, so ANN behavior isn't dominated by distance concentration.
- 50 query vectors from a separate stream (not dataset members).
- Metadata (all filterable, no non-filterable keys): `tenant` with exact counts, `category` c0–c3 (12,500 each), `year` 2015–2024 (5,000 each). Labels are assigned by exact counts and shuffled independently of vector position.
- The canonical local copy is `data/vectors.npy`, `data/queries.npy`, `data/metadata.jsonl`, with sha256 hashes in `data/manifest.json`. The upload was built from these files only.

## Filters

Ten filters, with counts measured on the canonical data: F50 25,000 (50%), F10 5,000 (10%), FAND3 656 (1.312%, 3-condition `$and` mixing string and number), F1 500 (1%), FAND2 135 (0.27%, 2-condition `$and`), F01 50 (0.1%), F001 5 (0.01%), FFEW 3, FONE 1, FZERO 0. They cover broad filters, fewer matches than K, exactly one match, and zero matches. K ∈ {5, 10, 20, 50}. `$startsWith` was excluded because it adds no recall insight on this tenant-equality filter set.

Two constraint-limit probe filters: C100 (`$in` with 100 values, 19,441 matches) and C101 (101 values).

## AWS resources

All disposable, named `s3vectors-prefilter-bench-<run-id>` (`-probe` for the probe), tagged `project=s3vectors-prefilter-bench` and `run_id=<id>`, and recorded in the run manifest:

- main bucket and main index (384-dim, cosine), ENHANCED by default because the bucket is new
- probe bucket and probe index (8-dim, cosine, 10 vectors)

An ownership guard in `src/aws_client.py` refuses any mutating call on a resource the run didn't create. The run had a hard spend cap of $0.374573 inside the project's $5 cap, and the operator approved the cost estimate before any mutating call.

## Ingestion and readiness

100 `PutVectors` calls of 500 vectors, throttled client-side. Then `GetIndex` (abort unless `indexMode` is ENHANCED), `ListVectors` until all 50,000 keys are listed, and a readiness query (FONE, K = 5) that must return the single matching key. Readiness queries are excluded from metrics.

## CLASSIC probe

On the probe bucket: `CreateVectorBucket`, `GetVectorBucket`, `PutVectorBucketDefaultIndexMode CLASSIC` (step 3), `CreateIndex`, `GetIndex`, `PutVectors`, `ListVectors` polling, `UpdateIndexMode CLASSIC` (step 8), `GetIndex` (step 9), and `QueryVectors` with `queryMode=CLASSIC` (step 10), `ENHANCED`, and no mode. Step 13 sent `queryMode=CLASSIC` to the main index. Steps 3, 8, 10 and 13 were rejected with `ValidationException`, and every `GetIndex` returned ENHANCED. Records are in `results/aws/probe_classic.json`. Because CLASSIC wasn't obtainable, no CLASSIC measurement exists, and the planned same-index A/B could not run.

## Query plan

Every query sets `returnDistance=true` and `returnMetadata=false`, and requests run sequentially.

| Class | Plan | Requests |
|---|---|---|
| ENHANCED filtered | 10 filters × 50 queries × 4 K × 3 repeats, filter sent, no `queryMode` | 6,000 |
| Unfiltered ANN reference | 50 × 4 K × 3 repeats | 600 |
| Post-filter baseline | 50 queries × B ∈ {100, 1,000, 10,000} × 2 repeats, unfiltered, topK = B, paged 100 per page | 300 logical, 11,100 pages |
| Constraint probe | C100, C101 on `q-00`, K = 10 | 2 |

Each repeat pass was shuffled with the seeded `order` stream, so time drift doesn't line up with any filter or class. B = 10,000 is the documented topK maximum, which is 20% of N; that makes the baseline far more generous than it would be on a production-size index. One unfiltered baseline request serves every filter and every K: pages are merged, ordered by (distance, key), filtered locally, and truncated to the first K matches.

## Ground truth and metrics

For each (query, filter): apply the filter to the canonical metadata, compute exact float64 cosine distance `1 − x·q` to every matching vector, sort by (distance, key), and take the first min(K, matching). Ties at the K boundary (gap < 1e-6) are flagged and reported, not excluded.

Per request:

- `recall = |returned ∩ GT_K| / min(K, matching)`
- `completeness = returned / min(K, matching)`, never clipped
- zero-match case: recall and completeness are null, and the request is scored by `zero_match_correct = (returned == 0)`
- precision violations (returned keys that fail the filter), duplicate keys, and unknown keys (a fatal integrity error)

Aggregation is per query vector: average over repeats first, then report mean, median, min, fraction with recall 1, mean completeness, and a 95% bootstrap CI over the 50 query vectors (1,000 resamples). Consistency across repeats is `identical_rate` (share of cells with identical key sequences) and mean pairwise Jaccard.

## Latency

`time.perf_counter_ns()` around each boto3 call, labeled "client-observed round trip incl. network". Repeat 1 is "first" (not claimed to be cold) and later repeats are "warm". Post-filter latency is the sum of a request's sequential pages. Server time is not separated from network time.

## Failures and exclusions

Transient errors are retried with exponential backoff. Exactly one measurement request failed: `r002032`, a B = 1,000 baseline request in repeat 1. The host PC slept between page 5 (10:49:29 UTC) and page 6 (11:05:25 UTC); the first attempt hit `ConnectionClosedError` and the retry got `ValidationException` "Invalid page token". It's recorded and excluded from metrics and latency, which is why the B = 1,000 rows have 99 requests. The probe rejections and C101 are expected outcomes, not failures.

## Simulator (SIMULATED, not AWS)

IVF-flat in numpy (nlist 256, spherical k-means) over the same canonical vectors, scored by the same metrics code against the same ground truth:

- CLASSIC model: filter during search; every visited vector uses one unit of a candidate budget C ∈ {500, 2,000, 8,000}
- ENHANCED model: filter first; only matching vectors are scored
- simulated post-filter, as a sanity bridge to the real baseline

Nothing is calibrated to AWS's "up to 5x" claim, there is no timing, and AWS's ANN internals are undocumented.

## Cost and cleanup

Cost is tallied before each call from `benchmark/pricing_us-east-1.json`, transcribed from the official S3 pricing page. The run total was $0.0731 (`run_metadata.json` `spend_tally_usd`). That's a model, not an AWS bill. After the Console screenshots, cleanup deleted only the manifest's resources and verified with `GetIndex` and `ListVectorBuckets` (`results/aws/cleanup.json`, `all_owned_gone: true`).

## Determinism and tests

Thread counts are pinned to 1, so the offline pipeline (generate → ground truth → simulate → report) produces byte-identical outputs across two runs (`diff -r`). The pytest suite (152 passed) covers determinism, selectivity counts, ground truth, recall including the fewer-than-K and zero-match cases, the simulator, post-filter math, aggregation and serialization, the ownership guard and borrowed mode, and a test that no test path calls real AWS.
