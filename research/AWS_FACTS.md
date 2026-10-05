# Amazon S3 Vectors metadata pre-filtering: verified fact sheet

Verified 2026-10-02 (UTC) against primary AWS sources only. All pages were fetched with HTTP 200 the same day. Each claim is traced in `SOURCE_LEDGER.md`. When a fact is not stated in any primary source, this sheet says **unknown**.

Source keys used below:
- [BLOG] AWS News Blog, "Amazon S3 Vectors now supports metadata pre-filtering for higher recall on filtered searches" (Daniel Abib, 30 SEP 2026)
- [WN] What's New, "Amazon S3 Vectors introduces metadata pre-filtering for up to 5x higher recall on filtered search"
- [DOC-FILTER] User Guide, Metadata filtering
- [DOC-BP] User Guide, S3 Vectors best practices
- [DOC-MODE] User Guide, Changing a vector index's mode
- [DOC-LIMITS] User Guide, Limitations and restrictions
- [DOC-REGIONS] User Guide, AWS Regions, endpoints, and quotas for S3 Vectors
- [DOC-CREATE] User Guide, Creating a vector index in a vector bucket
- [BOTO-*] Boto3 1.43.107 reference for `query_vectors`, `create_index`, `get_index`, `update_index_mode`, `put_vector_bucket_default_index_mode`, `put_vectors`
- [PRICING] https://aws.amazon.com/s3/pricing/ (S3 Vectors section), plus the price feed that page renders from (`b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/s3/USD/current/s3.json`, manifest `hawkFilePublicationDate` 2026-09-28T23:04:16Z)

## 1. Launch

- Announced 2026-09-30: [WN] "Posted on: Sep 30, 2026"; [BLOG] "30 SEP 2026".
- The rollout was still in progress at launch. [WN]: "We are in the process of deploying this change and plan to complete the deployment in the coming days." Any region could still have lacked the feature in the days right after launch.

## 2. What "metadata pre-filtering" means

- S3 Vectors evaluates the metadata filter before the vector search. It first finds the vectors that match the filter, then searches only those for the nearest neighbors [DOC-FILTER].
- Pre-filtering applies to indexes whose index mode is `ENHANCED` [DOC-FILTER].
- The `QueryVectors` filter syntax and the `PutVectors` write path are unchanged [WN], [BLOG]. See the caveat on the 100-constraint limit in section 11.

## 3. CLASSIC vs ENHANCED behavior

| | CLASSIC | ENHANCED |
|---|---|---|
| Filter evaluation | "in tandem" with vector search. Candidates are checked against the filter during the top-K search [DOC-FILTER], [BLOG] | Before vector search (pre-filter) [DOC-FILTER] |
| API wording | "Applies metadata filters during the vector search" [BOTO-query_vectors], [BOTO-get_index] | "Applies metadata filters before the vector search" (same) |
| Filtered result count | "may return fewer than top K results when the vector index contains very few matching results" [DOC-FILTER]. "a query with a selective filter can return fewer than top K results" [DOC-MODE] | "ensuring high recall even when filters match a small fraction of vectors" [DOC-FILTER], [DOC-BP], [DOC-MODE] |
| `$startsWith` | Only when the request sets `queryMode=ENHANCED` [DOC-FILTER], [DOC-MODE] | Available [DOC-FILTER] |
| 100-filter-constraint limit | Not applied ("This limit only applies to ENHANCED indexes") [DOC-FILTER] | Applied. Over the limit returns a validation error [DOC-MODE] |
| Who gets it by default | Indexes in vector buckets created before 2026-09-30 [DOC-FILTER], [DOC-MODE] | Indexes in vector buckets created on/after 2026-09-30 [DOC-FILTER], [BLOG] |

## 4. Selectivity effect

- Recall: the CLASSIC shortfall shows up when "the vector index contains very few matching results" [DOC-FILTER]. [BLOG] says narrow filters, such as one client in a firm-wide archive, "are where pre-filtering improves recall most." The 5x figure is stated only "on highly selective filters" [BLOG] or "when your filter is selective" [WN].
- Latency on ENHANCED: the work a filtered query does, and therefore its latency, grows with (a) index size, (b) "a filter that matches a larger share of the vectors in the index", and (c) the number of filter constraints [DOC-FILTER, "Query performance with filters"], [DOC-BP, "Writing filters for lower latency"]. AWS recommends "prefer selective filters and use the fewest constraints" [DOC-BP]. So on ENHANCED, less selective filters cost more latency.
- AWS publishes no numeric selectivity threshold, latency figures for ENHANCED vs CLASSIC, or benchmark dataset: **unknown**. [DOC-FILTER] says to test "with representative data and queries".

## 5. The "up to 5x" claim: what it measures

- [BLOG], exact sentence: "On highly selective filters, pre-filtering returns up to 5x more of the matching vectors than the same query returned before on CLASSIC indexes."
- [WN] body: "returning up to 5x more of the matching vectors when your filter is selective."
- [WN] headline: "...metadata pre-filtering for up to 5x higher recall on filtered search."

What 5x refers to:
- The quantity is the count of returned vectors that match the filter (result completeness). It is not defined as recall@k against exact ground truth.
- The baseline is the same query on a CLASSIC index.
- The condition is "highly selective" filters.
- "Up to" makes it a best-case upper bound, not a typical gain.
- On CLASSIC, every returned vector already passes the filter. The shortfall is fewer than K results [DOC-FILTER]. So "5x more matching vectors" is in effect a ratio of result counts (ENHANCED count / CLASSIC count). Recall against exact filtered top-k also depends on ANN quality inside the filtered subset, and AWS makes no claim about that.
- Only the WN headline turns this into "5x higher recall". The BLOG title says just "higher recall", with no multiplier. Discrepancy D1.
- No primary source discloses the methodology: dataset, index size, selectivity levels, K, and how "recall" was computed are all **unknown**.

## 6. Index mode at creation (no mode parameter on CreateIndex)

- `CreateIndex` has no index-mode parameter. The boto3 request syntax lists `vectorBucketName`, `vectorBucketArn`, `indexName`, `dataType`, `dimension`, `distanceMetric`, `metadataConfiguration`, `encryptionConfiguration`, `tags` [BOTO-create_index]. The CLI examples in [DOC-CREATE] and [BLOG] pass no mode either.
- A new index inherits the bucket's default index mode [DOC-MODE].
- `PutVectorBucketDefaultIndexMode` (`defaultIndexMode` = `CLASSIC` | `ENHANCED`, required) "applies to vector indexes that you create after the request succeeds. The operation doesn't change existing vector indexes." [BOTO-put_vector_bucket_default_index_mode], [DOC-MODE]
- Read the bucket default with `GetVectorBucket` or the console Properties tab [DOC-MODE]. The `GetVectorBucket` response field name was not verified in boto3 (**unknown**).
- Read an index's mode from `GetIndex` → `index.indexMode` (`CLASSIC` | `ENHANCED`) [BOTO-get_index].
- Buckets created on/after 2026-09-30 create ENHANCED indexes [DOC-FILTER], [BLOG]. Buckets created before that date use CLASSIC "including indexes you create in those buckets later" [DOC-MODE], or "until you set the bucket default" [BLOG].
- Permissions for the new actions: `s3vectors:UpdateIndexMode` [BOTO-update_index_mode] and `s3vectors:PutVectorBucketDefaultIndexMode` [BOTO-put_vector_bucket_default_index_mode]. [BLOG] tells readers to make sure their IAM policy "grants permissions for the new actions".

## 7. Migration and reversibility

- `UpdateIndexMode` changes an existing index in place: "no re-ingestion, no changes to your queries or your application, and no additional charge" [DOC-MODE]. [BLOG] also says the new operators are "available immediately".
- It changes only the target index, not the bucket default or other indexes [BOTO-update_index_mode].
- Before switching, confirm queries use 100 or fewer filter constraints, because over-limit queries return a validation error on ENHANCED [DOC-MODE].
- Console: "Enable enhanced index mode" on the index page [DOC-MODE].
- Reversibility: "You can set the mode to ENHANCED for any vector index. You can set the mode to CLASSIC only for a vector index in a vector bucket created before September 30, 2026." [BOTO-update_index_mode]
  - Reverting to CLASSIC works through the CLI, SDKs, or REST API. "You cannot do this from the Amazon S3 console." [DOC-MODE]
  - Indexes in buckets created on/after 2026-09-30 can never be CLASSIC.
- How long the mode transition takes, and whether queries are affected during it: **unknown** (not stated).

## 8. QueryVectors `queryMode`

- Parameter `queryMode` = `CLASSIC` | `ENHANCED`. If omitted, the index's current mode is used [BOTO-query_vectors].
- `CLASSIC`: "You can't specify CLASSIC for an ENHANCED index." [BOTO-query_vectors]
- `ENHANCED` can be set per query on a CLASSIC index ("the index stays as it is"). On an ENHANCED index it "has no additional effect" [DOC-MODE], [DOC-BP].
- `$startsWith` used under `queryMode=ENHANCED` on a CLASSIC index returns a validation error if `queryMode` is later removed [DOC-MODE].
- Whether the 100-constraint limit applies to a CLASSIC index queried with `queryMode=ENHANCED`: **unknown**. The docs only say the limit applies "on an ENHANCED index" [DOC-FILTER], [DOC-LIMITS].
- Benchmark implication: a same-index CLASSIC vs ENHANCED A/B is only possible on a CLASSIC index, which must live in a bucket created before 2026-09-30. Run it with and without `queryMode=ENHANCED`.

## 9. Filter operators and `$startsWith`

- Operators: `$eq`, `$ne` (string/number/boolean); `$gt`, `$gte`, `$lt`, `$lte` (number); `$startsWith` (string, prefix match); `$in`, `$nin` (non-empty array of primitives); `$exists` (boolean); `$and`, `$or` (non-empty array of filters). A bare key/value is implicit `$eq`. `$eq` against an array metadata value matches if any element matches [DOC-FILTER].
- `$startsWith` "requires ENHANCED query behavior": either an ENHANCED index, or a CLASSIC index with `queryMode=ENHANCED` [DOC-FILTER], [DOC-MODE].

## 10. Metadata limits [DOC-LIMITS] unless noted

- Filterable metadata per vector: up to 2 KB (also [BLOG]). Exceeding the limits makes `PutVectors` return 400 Bad Request [DOC-FILTER].
- Total metadata per vector (filterable + non-filterable): up to 40 KB.
- Total metadata keys per vector: up to 50.
- Non-filterable metadata keys per index: up to 10, each key name up to 63 characters. They are set at `CreateIndex` and cannot be changed later [DOC-FILTER], [DOC-CREATE].
- Supported filterable types: string, number, boolean, list [DOC-FILTER].
- Dimension: 1–4,096. Vectors per index: up to 2 billion. Top-K per `QueryVectors`: up to 10,000. Results per page in a `QueryVectors` response: up to 100, with pagination via `nextToken` [BOTO-query_vectors].

## 11. The 100 filter-constraint rule

- On an ENHANCED index, one query filter can use up to 100 filter constraints. "Each value the filter evaluates counts as one constraint." [DOC-FILTER], [DOC-LIMITS]. [BLOG] says "counted per value the filter evaluates".
- Worked counts from [DOC-FILTER]:
  - a bare equality = 1
  - `$in` with 3 values = 3
  - `$and` of an equality and a `$lte` = 2
  - The `$and`/`$or` wrapper itself is not counted in that example.
- Over the limit on ENHANCED returns a validation error [DOC-MODE].
- Remedies [DOC-BP], [BLOG]:
  - Consolidate long value lists into one grouping key (for example `caseId` instead of a 300-value `$in`).
  - Or split the filter across parallel queries and merge by distance, with `returnDistance=true`.
- [DOC-BP] says use CLASSIC only if a query can neither be simplified to ≤100 constraints nor split.
- How a multi-operator object on one field counts, such as `{"price":{"$gte":10,"$lte":50}}`, and how `$nin`/`$exists`/`$startsWith` count: **unknown**. Presumably one per evaluated value, but no example is given.

## 12. Pricing

- Pre-filtering itself has no extra charge: "available at no additional cost" [WN], [BLOG]. Changing an index to ENHANCED has "no additional charge" [DOC-MODE].
- Standard S3 Vectors charges apply [BLOG]: "storage, PUT requests, and queries". [PRICING] breaks query charges into three parts: a per-query request fee, a $/TB data-processed charge, and a $/GB data-returned charge.

US East (N. Virginia) and US West (Oregon) had identical rates, taken from the price feed behind [PRICING] (feed published 2026-09-28T23:04:16Z, fetched 2026-10-02). They match the worked example on the page:

| Dimension | Price |
|---|---|
| Vector storage | $0.06 per GB-month (logical vector data + key + metadata) |
| PUT | $0.20 per logical GB uploaded; minimum charge 128 KB per PUT |
| GET, LIST and all other requests | $0.055 per 1,000 requests |
| Query requests | $0.0025 per 1,000 requests ($2.50 per million) |
| Data processed, first 100K vectors | $0.004 per TB |
| Data processed, 100K–10M vectors | $0.002 per TB |
| Data processed, over 10M vectors | $0.0004 per TB |
| Data returned | $0.01 per GB; each result billed at ≥256 bytes; first 512 KB per query free (see D5) |

- Data processed is defined as the number of vectors in the index × average vector size (vector data + key + filterable metadata). Non-filterable metadata is excluded [PRICING].
  - As written, this depends on index size, not on how many vectors the filter matches.
  - Whether ENHANCED/pre-filtering changes the data-processed billing for a filtered query: **unknown**. No source says it does.
- Overwritten or deleted vectors may stay in the index's billed size for up to a day [PRICING].
- Other regions: not extracted. Re-fetch before quoting any cost figure for them.

## 13. Regions

- "All commercial AWS Regions where Amazon S3 Vectors is available, and in the AWS China Regions" [WN], [BLOG].
- [DOC-REGIONS] as fetched lists 34 endpoints: commercial regions plus AWS GovCloud (US-East/US-West) and AWS European Sovereign Cloud (Germany). It lists no China regions. See D4.
- Pre-filtering in GovCloud and the European Sovereign Cloud: **unknown**. Neither is named in the launch statements.

## 14. Throughput and permissions

- `PutVectors`: up to 500 vectors per call [DOC-LIMITS], [DOC-BP]. Request payload up to 20 MiB [DOC-LIMITS].
  - Over-capacity batches can be rejected with `ServiceUnavailableException` ("Currently unable to handle the request") [BOTO-put_vectors].
  - Vector data must be float32. Zero vectors are not allowed for cosine. NaN/Infinity are rejected [BOTO-put_vectors].
- Per index, combined `PutVectors` + `DeleteVectors`: up to 1,000 requests/s and up to 2,500 vectors/s, whichever is reached first [DOC-LIMITS], [DOC-BP]. For example, 5 × 500-vector batches/s [DOC-BP].
- `QueryVectors`/`GetVectors`/`ListVectors`: "hundreds of ... requests per second per S3 vector index" [DOC-BP]. No exact figure: **unknown**.
- Exceeding rates may return 429 `TooManyRequestsException`. AWS recommends a retry mechanism and lowering the request rate [DOC-BP].
- `GetVectors`: up to 100 vectors per call [DOC-LIMITS].
- IAM: `QueryVectors` needs `s3vectors:QueryVectors`. Setting a `filter` or `returnMetadata=true` also needs `s3vectors:GetVectors`, otherwise 403 Forbidden [BOTO-query_vectors].

## 15. Discrepancies and caveats found

- **D1: "5x higher recall" vs "5x more of the matching vectors".** The WN headline says recall. The WN body and the BLOG measure the number of matching vectors returned versus CLASSIC, on (highly) selective filters, "up to". Treat it as a best-case result-count ratio, not a recall@k multiplier.
- **D2: "no change to your queries" [BLOG], [WN] vs the 100-constraint limit [DOC-MODE].** On ENHANCED, queries with more than 100 constraints fail validation. So "no query changes" holds only for queries at or under the limit.
- **D3: bucket-default wording.**
  - [DOC-MODE]: pre-2026-09-30 buckets use CLASSIC "including indexes you create in those buckets later".
  - [BLOG]: "until you set the bucket default".
  - These are consistent: CLASSIC holds until `PutVectorBucketDefaultIndexMode` is called.
- **D4: China regions.** The launch statements say the feature is available in the AWS China Regions. The fetched [DOC-REGIONS] page lists no China endpoints (China docs may be published separately; not verified).
- **D5: data-returned free tier.** [PRICING] main text and footnote say "first 512KB of data returned per query is free". The narrative of Pricing example 1 on the same page says "a free tier of 500KB per query". Use 512 KB (stated twice, in normative text) and flag it.
- **D6: pricing wording.** [BLOG] lists "storage, PUT requests, and queries". [PRICING] also bills "GET, LIST and all other requests" and splits query cost into request + data processed + data returned.
- **D7: rollout.** [WN] says deployment was to complete "in the coming days" after 2026-09-30. Verify mode support per region via `GetIndex`/`indexMode` before benchmarking.
