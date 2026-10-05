"""Raw evidence -> results/processed/*, tables, figures; plus the generated Markdown artifacts.

Reads only raw evidence files: results/simulator/metrics.json, results/aws/metrics.json (top level
only; aborted/ and previous/ are never read), results/aws/timings.jsonl and data/manifest.json.
Real-AWS and SIMULATED classes are never drawn on the same axes.
"""
from __future__ import annotations

from pathlib import Path

from . import metrics
from .state import ConfigError, atomic_write_json, atomic_write_text, read_json, read_jsonl, round_sig

LABELS = {
    "aws_enhanced": "REAL AWS: ENHANCED filtered query",
    "aws_ann_reference": "REAL AWS: unfiltered ANN reference",
    "aws_postfilter_baseline": "REAL AWS: client-side post-filter baseline (unfiltered query + local filter), B={b}",
    "probe_classic": "REAL AWS: CLASSIC probe response",
    "probe_constraint_limit": "REAL AWS: constraint-limit probe",
    "aws_classic": "REAL AWS (borrowed pre-Sep-30 bucket): CLASSIC query on CLASSIC index (Stage A)",
    "aws_classic_index_enhanced_query":
        "REAL AWS (borrowed pre-Sep-30 bucket): queryMode=ENHANCED on CLASSIC index (Stage B)",
    "sim_classic": "SIMULATED: CLASSIC model (filter during search, budget C={c})",
    "sim_enhanced": "SIMULATED: ENHANCED model (filter first, budget C={c})",
    "sim_postfilter": "SIMULATED: post-filter, B={b}, C={c}",
}
EXCLUDED_CLASSES = {"ingest", "readiness", "capture"}
AWS_CLASSES = {"aws_enhanced", "aws_ann_reference", "aws_postfilter_baseline", "aws_classic",
               "aws_classic_index_enhanced_query"}
SIM_CLASSES = {"sim_classic", "sim_enhanced", "sim_postfilter"}
FIG_K = 10
# Okabe-Ito colorblind-safe palette, paired with distinct markers so color is never the only cue.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
SUMMARY_FIRST = ["label", "evidence_class", "filter_id", "k", "budget_b", "budget_c", "measured_matching",
                 "selectivity_pct", "n_queries", "n_requests", "failed_requests", "recall_mean", "recall_ci_low",
                 "recall_ci_high", "recall_median", "recall_min", "recall_frac_1", "completeness_mean",
                 "matching_completeness_mean", "returned_mean", "zero_match_correct_rate", "precision_violations",
                 "duplicate_keys", "ties", "identical_rate", "mean_jaccard", "consistency_cells"]


def label(cls: str, b=None, c=None) -> str:
    if cls not in LABELS:
        raise ConfigError(f"unknown evidence_class {cls!r}; no row may reach a table without a label")
    return LABELS[cls].format(b=b, c=c)


def _fmt(v, digits=3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def summary_rows(aggregates: list[dict], manifest: dict) -> list[dict]:
    n = manifest["n"]
    counts = manifest["filter_match_counts"]
    rows = []
    for a in aggregates:
        cls = a["evidence_class"]
        if cls in EXCLUDED_CLASSES:
            continue
        lab = label(cls, a.get("budget_b"), a.get("budget_c"))
        m = counts.get(a["filter_id"])
        rows.append({**a, "label": lab, "measured_matching": m,
                     "selectivity_pct": (100.0 * m / n) if m is not None else None})
    rows.sort(key=lambda r: metrics.sort_key(metrics.group_key(r)))
    return rows


def write_summary(processed: Path, rows: list[dict], aws_present: bool, manifest: dict) -> None:
    import pandas as pd

    processed.mkdir(parents=True, exist_ok=True)
    rows = round_sig(rows)
    atomic_write_json(processed / "summary.json", {"aws_results_present": aws_present, "n": manifest["n"],
                                                   "rows": rows})
    keys = set().union(*(r.keys() for r in rows)) if rows else set()
    columns = [c for c in SUMMARY_FIRST if c in keys] + sorted(keys - set(SUMMARY_FIRST))
    frame = pd.DataFrame(rows, columns=columns)
    atomic_write_text(processed / "summary.csv", frame.to_csv(index=False, lineterminator="\n"))


def latency_rows(aws_dir: Path) -> tuple[list[dict], dict]:
    """Client-observed round trip incl. network, from requests whose every page succeeded on attempt 1."""
    import numpy as np

    queries = {q["request_id"]: q for q in read_jsonl(aws_dir / "queries.jsonl")}
    per_req: dict[str, list[dict]] = {}
    for t in read_jsonl(aws_dir / "timings.jsonl"):
        per_req.setdefault(t["request_id"], []).append(t)
    groups: dict[tuple, list[float]] = {}
    total, excluded = 0, 0
    for rid, ts in per_req.items():
        q = queries.get(rid, {})
        cls = q.get("evidence_class")
        if cls not in AWS_CLASSES:
            continue
        total += 1
        if any(t["attempt"] > 1 for t in ts):
            excluded += 1
            continue
        key = (cls, q.get("filter_id"), q.get("budget"), ts[0]["phase"])
        groups.setdefault(key, []).append(sum(t["rtt_ns"] for t in ts) / 1e6)
    rows = []
    for (cls, fid, b, phase), vals in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        rows.append({"evidence_class": cls, "filter_id": fid, "budget": b, "phase": phase, "n": len(vals),
                     "median_ms": float(np.median(vals)), "p90_ms": float(np.percentile(vals, 90))})
    return rows, {"requests": total, "excluded_retried": excluded,
                  "excluded_share": (excluded / total) if total else None}


def tables_md(rows: list[dict], manifest: dict, aws_metrics: dict | None, latency) -> str:
    out = ["# Generated result tables", "",
           "Generated by `src/reporting.py` from raw evidence files. Do not edit by hand.", "",
           "## Dataset and filters (measured on the canonical data)", "",
           f"N = {manifest['n']}, dim = {manifest['dim']}, metric = {manifest['metric']}, seed = {manifest['seed']}.",
           "", "| Filter | Matching vectors | Selectivity |", "|---|---|---|"]
    for fid, m in sorted(manifest["filter_match_counts"].items()):
        out.append(f"| {fid} | {m} | {100.0 * m / manifest['n']:.4f}% |")
    sections = [("REAL AWS", [r for r in rows if r["evidence_class"] in AWS_CLASSES]),
                ("SIMULATED (mechanism model, not AWS)", [r for r in rows if r["evidence_class"] in SIM_CLASSES])]
    for title, rs in sections:
        if not rs:
            continue
        out += ["", f"## {title}: Recall@K and completeness", "",
                "Recall@K = |returned ∩ exact top-K| / min(K, matching); completeness = returned / min(K, matching). "
                "Per-query means over repeats first, then statistics over query vectors. "
                "Zero-match rows have null recall and report zero_match_correct_rate.", "",
                "| Label | Filter | K | Matching | Recall mean [95% CI] | Recall min | Frac. recall=1 | "
                "Completeness | Returned | Zero-match correct | Prec. viol. | n_q | n_req |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rs:
            ci = (f" [{_fmt(r['recall_ci_low'])}, {_fmt(r['recall_ci_high'])}]"
                  if r["recall_ci_low"] is not None else "")
            out.append(f"| {r['label']} | {r['filter_id']} | {r['k']} | {r['measured_matching']} | "
                       f"{_fmt(r['recall_mean'])}{ci} | {_fmt(r['recall_min'])} | {_fmt(r['recall_frac_1'])} | "
                       f"{_fmt(r['completeness_mean'])} | {_fmt(r['returned_mean'], 2)} | "
                       f"{_fmt(r['zero_match_correct_rate'])} | {r['precision_violations']} | {r['n_queries']} | "
                       f"{r['n_requests']} |")
    sim = [r for r in rows if r["evidence_class"] in SIM_CLASSES]
    if sim:
        out += ["", "## SIMULATED cost proxy (candidates scored, not time)", "",
                "| Label | Filter | K | candidates_scored mean | budget_units_used mean |", "|---|---|---|---|---|"]
        for r in sim:
            if r["k"] == FIG_K:
                out.append(f"| {r['label']} | {r['filter_id']} | {r['k']} | {_fmt(r['candidates_scored_mean'], 1)} | "
                           f"{_fmt(r['budget_units_used_mean'], 1)} |")
    edge = [r for r in rows if r["filter_id"] in ("FFEW", "FONE", "FZERO")
            or (r["filter_id"] == "F001" and r["k"] > 5)]
    if edge:
        out += ["", "## Edge cases (fewer than K, exactly one, zero matches)", "",
                "| Label | Filter | K | Matching | Recall mean | Completeness | Zero-match correct | Duplicates |",
                "|---|---|---|---|---|---|---|---|"]
        for r in edge:
            out.append(f"| {r['label']} | {r['filter_id']} | {r['k']} | {r['measured_matching']} | "
                       f"{_fmt(r['recall_mean'])} | {_fmt(r['completeness_mean'])} | "
                       f"{_fmt(r['zero_match_correct_rate'])} | {r['duplicate_keys']} |")
    if aws_metrics and aws_metrics.get("constraint_probe"):
        out += ["", f"## {label('probe_constraint_limit')} (not scored)", "",
                "| Filter | Constraints | Accepted | Error code | Returned |", "|---|---|---|---|---|"]
        for c in aws_metrics["constraint_probe"]:
            out.append(f"| {c['filter_id']} | {c['constraints']} | {c['ok']} | {c['error_code'] or ''} | "
                       f"{_fmt(c['returned'])} |")
    if latency is not None:
        lrows, info = latency
        out += ["", "## REAL AWS: client-observed round trip incl. network (not AWS server latency)", "",
                f"Only requests whose every page succeeded on the first attempt are included; "
                f"{info['excluded_retried']} of {info['requests']} requests were excluded as retried "
                f"(share {_fmt(info['excluded_share'])}). 'first' is repeat 1 and is not claimed to be truly cold.", "",
                "| Class | Filter | B | Phase | n | Median ms | p90 ms |", "|---|---|---|---|---|---|---|"]
        for r in lrows:
            out.append(f"| {r['evidence_class']} | {r['filter_id']} | {_fmt(r['budget'])} | {r['phase']} | {r['n']} | "
                       f"{r['median_ms']:.1f} | {r['p90_ms']:.1f} |")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- figures

def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9})
    return plt


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, metadata={"Software": None})


def _series(rows, cls, metric, **match):
    pts = [r for r in rows if r["evidence_class"] == cls and r["k"] == FIG_K and r["measured_matching"]
           and r["filter_id"] != "NOFILTER" and all(r.get(k) == v for k, v in match.items())
           and r[metric] is not None]
    pts.sort(key=lambda r: r["selectivity_pct"])
    return pts


def _plot(ax, pts, metric, i, name, ci=False):
    xs = [r["selectivity_pct"] for r in pts]
    ys = [r[metric] for r in pts]
    kw = dict(color=PALETTE[i % len(PALETTE)], marker=MARKERS[i % len(MARKERS)], label=name, linewidth=1.2,
              markersize=5)
    if ci:
        lo = [max(0.0, y - r["recall_ci_low"]) for y, r in zip(ys, pts)]
        hi = [max(0.0, r["recall_ci_high"] - y) for y, r in zip(ys, pts)]
        ax.errorbar(xs, ys, yerr=[lo, hi], capsize=2, **kw)
    else:
        ax.plot(xs, ys, **kw)


def figure_aws(rows, cfg, metric, ylabel, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    i = 0
    for cls in ("aws_enhanced", "aws_classic", "aws_classic_index_enhanced_query"):
        pts = _series(rows, cls, metric)
        if pts:
            _plot(ax, pts, metric, i, label(cls), ci=metric == "recall_mean")
            i += 1
    for b in cfg["budgets"]:
        pts = _series(rows, "aws_postfilter_baseline", metric, budget_b=b)
        if pts:
            _plot(ax, pts, metric, i, label("aws_postfilter_baseline", b=b), ci=metric == "recall_mean")
            i += 1
    ax.set_xscale("log")
    ax.set_xlabel("Measured filter selectivity (% of N, log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"REAL AWS: {ylabel} vs selectivity, K={FIG_K}\nzero-match filter omitted; see tables", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    # Legend below the axes so it never hides the curve crossings.
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False)
    fig.set_size_inches(7, 5.4)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def figure_sim(rows, cfg, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    i = 0
    for c in cfg["simulator"]["budgets_c"]:
        pts = _series(rows, "sim_classic", "recall_mean", budget_c=c)
        _plot(ax, pts, "recall_mean", i, label("sim_classic", c=c), ci=True)
        i += 1
    c = cfg["simulator"]["headline_c"]
    _plot(ax, _series(rows, "sim_enhanced", "recall_mean", budget_c=c), "recall_mean", i, label("sim_enhanced", c=c),
          ci=True)
    ax.set_xscale("log")
    ax.set_xlabel("Measured filter selectivity (% of N, log scale)")
    ax.set_ylabel(f"Recall@{FIG_K}")
    ax.set_title(f"SIMULATED (numpy IVF model, not AWS): Recall@{FIG_K} vs selectivity\n"
                 "zero-match filter omitted; see tables", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def figure_latency(lrows: list[dict], info: dict, manifest: dict, path: Path) -> None:
    """Optional latency figure: REAL AWS client round trip, 'first' (repeat 1) and 'warm' in separate panels.

    ENHANCED medians are plotted against measured selectivity (all K pooled). The unfiltered ANN reference and the
    post-filter baseline do not depend on the filter, so they are drawn as horizontal reference lines.
    """
    plt = _plt()
    n, counts = manifest["n"], manifest["filter_match_counts"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.4), sharey=True)
    for ax, phase in zip(axes, ("first", "warm")):
        rs = [r for r in lrows if r["phase"] == phase]
        pts = sorted((100.0 * counts[r["filter_id"]] / n, r["median_ms"]) for r in rs
                     if r["evidence_class"] == "aws_enhanced" and counts.get(r["filter_id"]))
        if pts:
            ax.plot([x for x, _ in pts], [y for _, y in pts], color=PALETTE[0], marker=MARKERS[0], linewidth=1.2,
                    markersize=5, label=f"{label('aws_enhanced')} (median, all K)")
        refs = [r for r in rs if r["evidence_class"] in ("aws_ann_reference", "aws_postfilter_baseline")]
        refs.sort(key=lambda r: (r["evidence_class"], r["budget"] or 0))
        for i, r in enumerate(refs, 1):
            name = (label("aws_ann_reference") if r["evidence_class"] == "aws_ann_reference"
                    else label("aws_postfilter_baseline", b=r["budget"]))
            ax.axhline(r["median_ms"], color=PALETTE[i % len(PALETTE)], linestyle=["--", ":", "-."][i % 3],
                       linewidth=1.0, label=f"{name} (median, filter-independent)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Measured filter selectivity (% of N, log scale)")
        ax.set_title(f"phase = {phase}" + (" (repeat 1; not claimed truly cold)" if phase == "first" else
                                           " (repeats 2+)"), fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Client-observed round trip, ms (log scale)")
    handles, names = axes[1].get_legend_handles_labels()
    fig.legend(handles, names, fontsize=7, loc="lower center", ncol=1, frameon=False)
    fig.suptitle("REAL AWS: client-observed round trip incl. network (not AWS server latency)\n"
                 f"zero-match filter omitted; {info['excluded_retried']} of {info['requests']} requests excluded as "
                 "retried; post-filter = sum of its pages", fontsize=9)
    fig.tight_layout(rect=(0, 0.25, 1, 1))
    _save(fig, path)
    plt.close(fig)


def figure_flow(path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.set_axis_off()
    rows = [
        ("Filter first (ENHANCED, as AWS documents it)",
         ["metadata filter", "matching set M", "ANN search inside M", "top-K"]),
        ("Filter during search (CLASSIC: AWS documents filtering during search; the candidate bound is our assumption)",
         ["ANN search over all", "filter each candidate", "bounded candidates\n(assumed)", "<= K matches"]),
        ("Client-side post-filter baseline (not CLASSIC)",
         ["unfiltered top-B", "local filter", "first K matches", "<= K matches"]),
    ]
    for r, (title, boxes) in enumerate(rows):
        y = 2.4 - r * 1.1
        ax.text(0.0, y + 0.38, title, fontsize=8, fontweight="bold")
        for j, text in enumerate(boxes):
            x = 0.05 + j * 2.45
            ax.add_patch(plt.Rectangle((x, y - 0.25), 2.0, 0.5, fill=False, linewidth=1.0,
                                       edgecolor=PALETTE[r]))
            ax.text(x + 1.0, y, text, ha="center", va="center", fontsize=7.5)
            if j < len(boxes) - 1:
                ax.annotate("", xy=(x + 2.4, y), xytext=(x + 2.02, y),
                            arrowprops={"arrowstyle": "->", "color": "#000000", "lw": 0.8})
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.2, 3.0)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------- entry point

def report(p, cfg: dict) -> None:
    manifest = read_json(p.data / "manifest.json")
    aggregates = []
    sim_path = p.sim / "metrics.json"
    if sim_path.exists():
        aggregates += read_json(sim_path)["aggregates"]
    aws_path = p.out / "results" / "aws" / "metrics.json"
    aws_metrics = read_json(aws_path) if aws_path.exists() else None
    if aws_metrics:
        aggregates += aws_metrics["aggregates"]
    rows = summary_rows(aggregates, manifest)
    write_summary(p.processed, rows, aws_metrics is not None, manifest)
    latency = None
    if aws_metrics is not None and (p.out / "results" / "aws" / "timings.jsonl").exists():
        latency = latency_rows(p.out / "results" / "aws")
        atomic_write_json(p.processed / "latency.json", round_sig({"rows": latency[0], **latency[1],
                                                                  "label": "client-observed round trip incl. network"}))
    atomic_write_text(p.processed / "tables.md", tables_md(round_sig(rows), manifest, aws_metrics, latency))
    if any(r["evidence_class"] in SIM_CLASSES for r in rows):
        figure_sim(rows, cfg, p.figures / "recall_vs_selectivity_simulated.png")
    if aws_metrics is not None:
        figure_aws(rows, cfg, "recall_mean", f"Recall@{FIG_K}", p.figures / "recall_vs_selectivity_aws.png")
        figure_aws(rows, cfg, "completeness_mean", "Completeness", p.figures / "completeness_vs_selectivity_aws.png")
    if latency is not None and latency[0]:
        figure_latency(latency[0], latency[1], manifest, p.figures / "latency_vs_selectivity_aws.png")
    figure_flow(p.figures / "flow_filter_candidate_search.png")


# ---------------------------------------------------------------- generated Markdown artifacts

def cost_estimate_md(cfg, pricing, plan, est, caps, digest, mode) -> str:
    rates = pricing["rates"]
    lines = [
        "# Cost estimate: S3 Vectors pre-filtering benchmark", "",
        "Generated by `python -u -m src.runner estimate`. The fixed `key: value` lines below are parsed by the runner.",
        "", f"plan_mode: {mode}", f"base_usd: {caps['base_usd']:.6f}", f"approved_usd: {caps['approved_usd']:.6f}",
        f"prior_runs_spend_usd: {caps['prior_runs_spend_usd']:.6f}", f"hard_cap_usd: {caps['hard_cap_usd']:.6f}",
        f"plan_sha256: {digest}", "",
        "Cap arithmetic: approved_usd = 2 x base_usd; hard_cap_usd = min(approved_usd, 5.00 - prior_runs_spend_usd). "
        "prior_runs_spend_usd sums spend_tally_usd of every earlier run (results/aws/previous/*, results/aws/aborted/*, "
        "and a results/aws/run_metadata.json of a different run), so the $5 cap is cumulative across runs.", "",
        f"Pricing source: {pricing['source_url']} (fetched {pricing['fetched_utc']}). {pricing.get('provenance', '')}",
        "", "| Rate | Value | Page wording |", "|---|---|---|"]
    for k in sorted(rates):
        lines.append(f"| {k} | {rates[k]} | {pricing.get('wording', {}).get(k, '')} |")
    lines += ["", "## Resources (all named with the run id, tagged project=s3vectors-prefilter-bench and run_id)", ""]
    if mode == "own":
        lines += ["- main bucket `s3vectors-prefilter-bench-<run_id>` (role main_bucket)",
                  "- main index `s3vectors-prefilter-bench-<run_id>` in the main bucket (float32, dim "
                  f"{cfg['dataset']['dim']}, cosine; role main_index)",
                  "- probe bucket `s3vectors-prefilter-bench-<run_id>-probe` (role probe_bucket)",
                  "- probe index `s3vectors-prefilter-bench-<run_id>-probe` (dim 8, cosine, 10 vectors; role probe_index)"]
        lines += ["", "## CLASSIC probe plan", "",
                  "1 CreateVectorBucket (probe), 2 GetVectorBucket, 3 PutVectorBucketDefaultIndexMode CLASSIC, "
                  "4 GetVectorBucket, 5 CreateIndex (dim 8), 6 GetIndex, 7 PutVectors (10), 7a ListVectors polling "
                  f"(<= {cfg['aws']['probe_list_max_polls']}), 8 UpdateIndexMode CLASSIC, 9 GetIndex, "
                  "10 QueryVectors queryMode=CLASSIC, 11 queryMode=ENHANCED, 12 no queryMode; step 13 in aws-query: "
                  "queryMode=CLASSIC on the main index. Any accepted CLASSIC (steps 6, 9, 10) stops the AWS phase "
                  "with exit 3; an accepted step 13 is flagged and escalated."]
    else:
        lines += ["- borrowed bucket (read-only, never created, tagged, or deleted)",
                  "- borrowed-mode index `s3vectors-prefilter-bench-<run_id>` in the borrowed bucket (role borrowed_index)"]
    lines += ["", "## Request plan and line items", "", "| Line | Kind | Count | Phase | USD |", "|---|---|---|---|---|"]
    for ln in est["lines"]:
        lines.append(f"| {ln['line']} | {ln['kind']} | {ln['count']} | {ln['phase'] or ''} | {ln['usd']:.6f} |")
    lines += cost_arithmetic_md(rates, plan, est, caps)
    lines += ["", f"bytes per vector (vector + key + filterable metadata): {plan['bytes_per_vector']}; "
                  f"result bytes: {plan['result_bytes']}; retention: {plan['retention_days']} days.",
              "Conservative choices: every QueryVectors page billed as a query; data processed on index size; "
              "128 KB minimum per PUT; GB = 1e9 bytes, TB = 1e12 bytes; every retried attempt charged.", "",
              "**PAUSE: explicit user approval is required before any mutating AWS call.**", ""]
    return "\n".join(lines)


def cost_arithmetic_md(rates, plan, est, caps) -> list[str]:
    """Worked arithmetic behind the line items and the cap values (GB = 1e9 B, TB = 1e12 B, 730 h/month)."""
    n, bpv, days = plan["index_vectors"]["main"], plan["bytes_per_vector"], plan["retention_days"]
    idx_bytes = n * bpv
    q_fee = rates["query_requests_usd_per_1000"] / 1000.0
    dp = idx_bytes / 1e12 * rates["data_processed_usd_per_tb_first_100k"]
    sub: dict[str, float] = {}
    for ln in est["lines"]:
        sub[ln["kind"]] = sub.get(ln["kind"], 0.0) + ln["usd"]
    remaining = 5.00 - caps["prior_runs_spend_usd"]
    out = ["", "## Arithmetic", "",
           f"- main index size: {n} vectors x {bpv} B = {idx_bytes} B = {idx_bytes / 1e9:.5f} GB",
           f"- storage: {idx_bytes / 1e9:.5f} GB x ${rates['storage_usd_per_gb_month']}/GB-month x "
           f"({days} d x 24 h / 730 h) = ${idx_bytes / 1e9 * rates['storage_usd_per_gb_month'] * days * 24 / 730:.6f}",
           f"- one QueryVectors page on the main index: request ${q_fee:.7f} + data processed "
           f"{idx_bytes / 1e12:.8f} TB x ${rates['data_processed_usd_per_tb_first_100k']}/TB (${dp:.9f}) "
           f"= ${q_fee + dp:.9f}",
           f"- PUT: each 500-vector batch is billed at max(batch bytes, {rates['put_min_bytes']} B) x "
           f"${rates['put_usd_per_gb']}/GB",
           f"- other requests: count x ${rates['other_requests_usd_per_1000']}/1,000", "",
           "| Kind | Subtotal USD |", "|---|---|"]
    out += [f"| {k} | {v:.6f} |" for k, v in sub.items()]
    out += ["", f"- base_usd = {' + '.join(f'{v:.6f}' for v in sub.values())} = {caps['base_usd']:.6f} "
            "(sum of unrounded subtotals)",
            f"- approved_usd = 2 x {caps['base_usd']:.6f} = {caps['approved_usd']:.6f}",
            f"- remaining project budget = 5.00 - {caps['prior_runs_spend_usd']:.6f} = {remaining:.6f}",
            f"- hard_cap_usd = min({caps['approved_usd']:.6f}, {remaining:.6f}) = {caps['hard_cap_usd']:.6f} "
            f"({caps['hard_cap_usd'] / 5.00:.1%} of the $5 cap)"]
    return out


def shot_list_md(run_id: str) -> str:
    b = f"s3vectors-prefilter-bench-{run_id}"
    shots = [
        ("S3 -> Vector buckets (us-east-1)", f"the run's two buckets `{b}` and `{b}-probe`",
         "shot01_vector_buckets.png"),
        (f"S3 -> Vector buckets -> `{b}` -> Properties", "the bucket's default index mode", "shot02_bucket_properties.png"),
        (f"S3 -> Vector buckets -> `{b}` -> index `{b}` -> details",
         "index mode ENHANCED, dimension 384, distance metric cosine", "shot03_index_details.png"),
        (f"S3 -> Vector buckets -> `{b}-probe` -> index `{b}-probe` -> details",
         "the probe index mode after probe step 8 (UpdateIndexMode CLASSIC)", "shot04_probe_index_details.png"),
    ]
    out = ["# Screenshot shot list (PAUSE: resources exist; do not clean up yet)", "",
           "Region: us-east-1. Every shot: hide or crop the account ID, any ARN, and the email/account menu in the "
           "top-right corner. Save the Console screenshots locally; they are not part of the repository.", ""]
    for i, (path, visible, name) in enumerate(shots, 1):
        out += [f"{i}. Console path: {path}", f"   - Must be visible: {visible}",
                "   - Hide/crop: account ID, ARNs, email (top-right menu)", f"   - Filename: `{name}`", ""]
    out += ["Terminal captures (text, written to `results/aws/captures/` by `aws-capture --run-id " + run_id + "`):",
            "",
            "- `terminal_get_index.txt`: GetIndex on the main index",
            "- `terminal_query_request_response.txt`: one QueryVectors request and response (F01, K=10)",
            "- `terminal_classic_probe_records.txt`: probe steps 8, 10 and 13-main-index, each headed ACCEPTED or REJECTED",
            ""]
    return "\n".join(out)
