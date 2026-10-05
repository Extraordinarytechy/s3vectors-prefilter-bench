# Source ledger

All sources were fetched on 2026-10-02 (UTC) with HTTP 200 and read in full as plain text. "Verified? = Yes" means the claim was checked against the source text itself, not taken from a summary. Boto3 pages are version 1.43.107.

URLs:
- BLOG = https://aws.amazon.com/blogs/aws/amazon-s3-vectors-now-supports-metadata-pre-filtering-for-higher-recall-on-filtered-searches/
- WN = https://aws.amazon.com/about-aws/whats-new/2026/09/s3-vectors-introduces-metadata-pre-filtering/
- DOC-FILTER = https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-metadata-filtering.html
- DOC-BP = https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-best-practices.html
- DOC-MODE = https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-index-mode.html
- DOC-LIMITS = https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-limitations.html
- DOC-REGIONS = https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-regions-quotas.html
- DOC-CREATE = https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-create-index.html
- BOTO-QV = https://docs.aws.amazon.com/boto3/latest/reference/services/s3vectors/client/query_vectors.html
- BOTO-CI = https://docs.aws.amazon.com/boto3/latest/reference/services/s3vectors/client/create_index.html
- BOTO-GI = https://docs.aws.amazon.com/boto3/latest/reference/services/s3vectors/client/get_index.html
- BOTO-UIM = https://docs.aws.amazon.com/boto3/latest/reference/services/s3vectors/client/update_index_mode.html
- BOTO-PDIM = https://docs.aws.amazon.com/boto3/latest/reference/services/s3vectors/client/put_vector_bucket_default_index_mode.html
- BOTO-PV = https://docs.aws.amazon.com/boto3/latest/reference/services/s3vectors/client/put_vectors.html
- PRICING = https://aws.amazon.com/s3/pricing/ (Vectors tab)
- PRICE-FEED = https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/s3/USD/current/s3.json (the data endpoint the PRICING page renders its `{priceOf!s3/s3!…}` tokens from; manifest hawkFilePublicationDate 2026-09-28T23:04:16Z)

| Claim | Source URL | Exact location | Verified? |
|---|---|---|---|
| Launch date 2026-09-30 | WN; BLOG | WN "Posted on: Sep 30, 2026"; BLOG byline "30 SEP 2026" | Yes |
| Rollout still in progress at launch ("coming days") | WN | Final paragraph | Yes |
| Pre-filtering = filter evaluated before vector search; only matching vectors searched | DOC-FILTER | Intro, para 2 ("a technique known as pre-filtering") | Yes |
| Pre-filtering applies to ENHANCED indexes | DOC-FILTER | Intro, para 3 | Yes |
| Buckets created on/after 2026-09-30 create ENHANCED indexes; older buckets have CLASSIC | DOC-FILTER; BLOG | DOC-FILTER intro para 3; BLOG "Things to know" bullet 1 | Yes |
| Older buckets stay CLASSIC "including indexes you create in those buckets later" | DOC-MODE | Intro, para 1 | Yes |
| CLASSIC = search and filter "in tandem"; may return fewer than top K | DOC-FILTER; BLOG | DOC-FILTER intro para 4; BLOG "How pre-filtering works" para 2 | Yes |
| ENHANCED ensures high recall even when filters match a small fraction | DOC-FILTER; DOC-BP | DOC-FILTER intro para 2; DOC-BP "Choosing an index mode" | Yes |
| "up to 5x more of the matching vectors" on highly selective filters vs same query on CLASSIC | BLOG | "How pre-filtering works", final sentence | Yes |
| "returning up to 5x more of the matching vectors when your filter is selective" | WN | Body, para 1 | Yes |
| Headline "up to 5x higher recall" | WN | Page title / headline | Yes |
| 5x methodology (dataset, K, selectivity, recall definition) | none | Not published | No (unknown) |
| Narrow filters are where pre-filtering improves recall most | BLOG | "Common use cases", Legal bullet | Yes |
| ENHANCED latency grows with index size, share matched, number of constraints | DOC-FILTER; DOC-BP | DOC-FILTER "Query performance with filters"; DOC-BP "Writing filters for lower latency" | Yes |
| `$startsWith` requires ENHANCED behavior (ENHANCED index, or CLASSIC + queryMode=ENHANCED) | DOC-FILTER; DOC-MODE | DOC-FILTER note after operator table; DOC-MODE "Test an index before you change it" | Yes |
| Removing queryMode on a `$startsWith` query against CLASSIC returns validation error | DOC-MODE | "Test an index before you change it", para 2 | Yes |
| Operator list ($eq…$or) and input types | DOC-FILTER | "Filterable metadata" operator table | Yes |
| Filterable metadata per vector up to 2 KB | DOC-LIMITS; BLOG | DOC-LIMITS list; BLOG intro para 1 | Yes |
| Total metadata up to 40 KB; up to 50 keys; up to 10 non-filterable keys | DOC-LIMITS | List items | Yes |
| Over-limit metadata gives PutVectors 400 Bad Request | DOC-FILTER | "Filterable metadata", para 2 | Yes |
| Non-filterable keys fixed at index creation; key name ≤63 chars | DOC-FILTER; DOC-CREATE | DOC-FILTER "Non-filterable metadata" para 2; DOC-CREATE top note | Yes |
| Up to 100 filter constraints per query on ENHANCED; each evaluated value counts; examples 1 / 3 / 2 | DOC-FILTER; DOC-LIMITS; BLOG | DOC-FILTER "Filter constraints per query"; DOC-LIMITS list; BLOG "Things to know" bullet 2 | Yes |
| Constraint limit applies only to ENHANCED indexes | DOC-FILTER | "Filter constraints per query", para 1 last sentence | Yes |
| Over 100 constraints on ENHANCED gives validation error | DOC-MODE | Intro, para after CLI example | Yes |
| Remedies: consolidate into grouping key, or split + merge by distance (returnDistance=true) | DOC-BP; BLOG | DOC-BP "Writing filters for lower latency"; BLOG "Things to know" bullet 2 | Yes |
| Use CLASSIC only if a query can't be simplified to ≤100 constraints or split | DOC-BP | "Choosing an index mode" | Yes |
| Constraint-limit applicability to CLASSIC index + queryMode=ENHANCED | none | Not stated | No (unknown) |
| CreateIndex has no index-mode parameter | BOTO-CI; DOC-CREATE | BOTO-CI "Request Syntax"; DOC-CREATE CLI/Python examples | Yes |
| GetIndex returns `indexMode` CLASSIC/ENHANCED | BOTO-GI | Response Syntax, `index.indexMode` | Yes |
| PutVectorBucketDefaultIndexMode (CLASSIC/ENHANCED) applies to indexes created after the call; doesn't change existing | BOTO-PDIM; DOC-MODE | BOTO-PDIM description + `defaultIndexMode`; DOC-MODE "Set the index mode for new indexes" | Yes |
| Bucket default readable via GetVectorBucket / console Properties | DOC-MODE | "Set the index mode for new indexes", para 1 | Yes (response field name not verified) |
| UpdateIndexMode in place: no re-ingestion, no query/app changes, no extra charge | DOC-MODE; BLOG | DOC-MODE intro para 2; BLOG "Turning on pre-filtering for existing indexes" | Yes |
| UpdateIndexMode affects only the target index | BOTO-UIM | Description | Yes |
| CLASSIC settable only for an index in a bucket created before 2026-09-30 | BOTO-UIM; DOC-MODE | BOTO-UIM description + `indexMode` values; DOC-MODE "Turn off the enhanced index mode" | Yes |
| Reverting to CLASSIC not possible from the console | DOC-MODE | "Turn off the enhanced index mode", para 1 | Yes |
| Mode transition duration / in-flight query behavior | none | Not stated | No (unknown) |
| queryMode CLASSIC/ENHANCED; default = index's mode | BOTO-QV | Parameters, `queryMode` | Yes |
| CLASSIC can't be specified for an ENHANCED index | BOTO-QV | Parameters, `queryMode`, CLASSIC value | Yes |
| ENHANCED per-query on CLASSIC index (index unchanged); no effect on ENHANCED index | DOC-MODE; DOC-BP | DOC-MODE "Test an index before you change it" para 1; DOC-BP "Choosing an index mode" | Yes |
| Filter or returnMetadata=true requires s3vectors:GetVectors, else 403 | BOTO-QV | "Permissions" | Yes |
| New IAM actions s3vectors:UpdateIndexMode, s3vectors:PutVectorBucketDefaultIndexMode | BOTO-UIM; BOTO-PDIM; BLOG | "Permissions" sections; BLOG "Getting started" para 1 | Yes |
| Top-K up to 10,000; up to 100 results per page; nextToken pagination | DOC-LIMITS; BOTO-QV | DOC-LIMITS list; BOTO-QV `nextToken` | Yes |
| No additional cost for pre-filtering | WN; BLOG; DOC-MODE | WN para 3; BLOG "Now available"; DOC-MODE intro para 2 | Yes |
| Standard S3 Vectors pricing applies (storage, PUT, queries) | BLOG | "Now available" | Yes |
| Query charge = request fee + $/TB data processed + $/GB data returned | PRICING | Vectors tab, "Query cost" | Yes |
| us-east-1 / us-west-2 rates: storage $0.06/GB-mo; PUT $0.20/GB; other requests $0.055/1K; query $0.0025/1K; processed $0.004 / $0.002 / $0.0004 per TB; returned $0.01/GB | PRICE-FEED; PRICING | PRICE-FEED regions "US East (N. Virginia)" and "US West (Oregon)", tokens referenced by PRICING tables; matches PRICING "Pricing example 1" | Yes |
| PUT minimum 128 KB per PUT | PRICING | Footnote under "S3 Vectors request pricing" | Yes |
| Data returned: ≥256 bytes per result; first 512 KB/query free | PRICING | "Query cost" para 3 and footnote *** | Yes |
| Data returned free tier "500KB" (conflicting) | PRICING | "Pricing example 1" narrative | Yes (discrepancy D5) |
| Data processed = vectors in index × avg size (data + key + filterable metadata) | PRICING | "Query cost" para 2; data processed row footnote | Yes |
| Whether pre-filtering changes data-processed billing | none | Not stated | No (unknown) |
| Available in all commercial regions with S3 Vectors + AWS China Regions | WN; BLOG | WN para 3; BLOG "Now available" | Yes |
| Regions page lists 34 endpoints incl. GovCloud and European Sovereign Cloud, no China | DOC-REGIONS | "S3 Vectors AWS Regions and endpoints" table | Yes (discrepancy D4) |
| Pre-filtering in GovCloud / European Sovereign Cloud | none | Not stated | No (unknown) |
| PutVectors max 500 vectors per call; payload ≤20 MiB | DOC-LIMITS; DOC-BP | DOC-LIMITS list; DOC-BP "Inserting and deleting vectors" | Yes |
| ≤1,000 Put+Delete req/s and ≤2,500 vectors/s per index; 429 TooManyRequestsException | DOC-LIMITS; DOC-BP | DOC-LIMITS list; DOC-BP "Inserting and deleting vectors" | Yes |
| Query/Get/List: "hundreds of" req/s per index; retry on 429 | DOC-BP | "Accessing and querying vectors" | Yes (exact number unknown) |
| Oversized PutVectors batch may give ServiceUnavailableException | BOTO-PV | Parameters, `vectors` | Yes |
| float32 only; no zero vectors (cosine); no NaN/Infinity | BOTO-PV | Note + `data` parameter | Yes |
