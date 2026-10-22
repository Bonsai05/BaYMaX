"""Benchmark harness for the AI/Memory module.

    python -m ai.eval.run                 # from the repo root
    python -m ai.eval.run --out ai/eval/results

Reports (all on the synthetic dataset, see synthetic.py for its limits):
  1. Retrieval ablation: lexical vs vector vs hybrid vs hybrid+rerank, overall and per
     query type, with recall@5, recall@10, MRR, nDCG@10 and latency.
  2. Safety: injected third-party documents in the top-5, and how many of them reach the
     prompt without an UNTRUSTED flag (must be zero).
  3. Admission: decision accuracy overall and per category, plus quarantine precision/recall.
  4. Router: accuracy against the golden cases, and stage usage.
  5. Cost of the pipeline: admission latency and indexing time.

Real models (needs `pip install fastembed` and internet once, to download weights):

    python -m ai.eval.run --embedder fastembed --reranker fastembed
    python -m ai.eval.run --compare ai/eval/results/hashing-coverage.json ai/eval/results/fastembed-fastembed.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from ..api import AIModule
from ..contracts import RetrievalRequest, RouteRequest
from .metrics import mean, mrr, ndcg_at_k, percentile, recall_at_k
from .synthetic import DAY, NOW, Dataset, build_dataset

MODES = ["lexical", "vector", "hybrid", "hybrid_rerank"]


def build_module(ds: Dataset, **kw) -> tuple[AIModule, list[float], float]:
    ai = AIModule(**kw)
    admit_ms = []
    for ev in ds.events:
        t = time.perf_counter()
        ai.admit(ev, index_now=False)
        admit_ms.append((time.perf_counter() - t) * 1000)
    t = time.perf_counter()
    ai.index_pending()
    index_s = time.perf_counter() - t
    return ai, admit_ms, index_s


def eval_retrieval(ai: AIModule, ds: Dataset, top_k: int = 10) -> dict:
    out: dict = {}
    for mode in MODES:
        per_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        lat, inj_top5, inj_unflagged, n_q = [], 0, 0, 0
        for q in ds.queries:
            req = RetrievalRequest(query=q["query"], top_k=top_k, mode=mode)
            if "since_days" in q:
                req.since = NOW - q["since_days"] * DAY
            res = ai.retriever.retrieve(req, now=NOW)
            ranked = [i.canonical_source for i in res.items]
            rel = set(q["relevant"])
            m = {"recall@5": recall_at_k(ranked, rel, 5), "recall@10": recall_at_k(ranked, rel, 10),
                 "mrr": mrr(ranked, rel), "ndcg@10": ndcg_at_k(ranked, rel, 10)}
            for key, val in m.items():
                per_type[q["type"]][key].append(val)
                per_type["ALL"][key].append(val)
            lat.append(res.timings_ms["total"])
            n_q += 1
            for it in res.items[:5]:
                if it.canonical_source in ds.injected_event_ids:
                    inj_top5 += 1
                    inj_unflagged += 0 if it.untrusted else 1
        out[mode] = {
            "by_type": {t: {k: mean(v) for k, v in d.items()} for t, d in per_type.items()},
            "latency_ms": {"median": percentile(lat, 50), "p95": percentile(lat, 95)},
            "injected_in_top5": inj_top5, "injected_unflagged": inj_unflagged, "queries": n_q,
        }
    return out


def eval_admission(ds: Dataset) -> dict:
    ai = AIModule()
    rows, by_cat = [], defaultdict(lambda: [0, 0])
    for case in ds.admission_cases:
        got = ai.admit(case["event"], index_now=False).decision.value
        ok = got == case["expected"]
        by_cat[case["category"]][0] += int(ok)
        by_cat[case["category"]][1] += 1
        rows.append((case["expected"], got, case["category"], ok))
    ai.close()
    tp = sum(1 for e, g, *_ in rows if e == "quarantine" and g == "quarantine")
    fp = sum(1 for e, g, *_ in rows if e != "quarantine" and g == "quarantine")
    fn = sum(1 for e, g, *_ in rows if e == "quarantine" and g != "quarantine")
    return {
        "accuracy": mean([1.0 if r[3] else 0.0 for r in rows]), "cases": len(rows),
        "quarantine_precision": tp / (tp + fp) if tp + fp else 0.0,
        "quarantine_recall": tp / (tp + fn) if tp + fn else 0.0,
        "by_category": {c: {"correct": a, "total": b} for c, (a, b) in sorted(by_cat.items())},
        "misses": [{"category": r[2], "expected": r[0], "got": r[1]} for r in rows if not r[3]],
    }


def eval_router(ai: AIModule, ds: Dataset) -> dict:
    correct, stage = 0, defaultdict(int)
    wrong = []
    for case in ds.router_cases:
        d = ai.route(RouteRequest(query=case["query"], available_tools=["SEARCH_FILES", "READ_FILE"]))
        stage[d.stage] += 1
        if d.route.value == case["expected"]:
            correct += 1
        else:
            wrong.append({"query": case["query"], "expected": case["expected"], "got": d.route.value})
    return {"accuracy": correct / len(ds.router_cases), "cases": len(ds.router_cases),
            "stage_usage": dict(stage), "wrong": wrong}


def to_markdown(r: dict) -> str:
    L: list[str] = ["# AI/Memory benchmark (synthetic dataset)", ""]
    L += [f"Corpus: {r['corpus_events']} events, {r['queries']} queries. Embedder: {r['embedder']}. Reranker: {r['reranker']}.", ""]
    L += ["## Retrieval ablation (all queries)", "",
          "| mode | recall@5 | recall@10 | MRR | nDCG@10 | median ms | p95 ms | injected in top-5 | unflagged |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for mode in MODES:
        m = r["retrieval"][mode]; a = m["by_type"]["ALL"]
        L.append(f"| {mode} | {a['recall@5']:.3f} | {a['recall@10']:.3f} | {a['mrr']:.3f} | {a['ndcg@10']:.3f} "
                 f"| {m['latency_ms']['median']:.2f} | {m['latency_ms']['p95']:.2f} "
                 f"| {m['injected_in_top5']} | {m['injected_unflagged']} |")
    types = [t for t in r["retrieval"]["hybrid"]["by_type"] if t != "ALL"]
    for metric in ("recall@5", "mrr"):
        L += ["", f"## {metric} by query type", "", "| mode | " + " | ".join(types) + " |",
              "| --- |" + " --- |" * len(types)]
        for mode in MODES:
            bt = r["retrieval"][mode]["by_type"]
            L.append(f"| {mode} | " + " | ".join(f"{bt[t][metric]:.3f}" for t in types) + " |")
    a = r["admission"]
    L += ["", "## Admission", "",
          f"Decision accuracy {a['accuracy']:.3f} over {a['cases']} cases. Quarantine precision "
          f"{a['quarantine_precision']:.3f}, recall {a['quarantine_recall']:.3f}.", "",
          "| category | correct | total |", "| --- | --- | --- |"]
    for c, v in a["by_category"].items():
        L.append(f"| {c} | {v['correct']} | {v['total']} |")
    rt = r["router"]
    L += ["", "## Router", "", f"Accuracy {rt['accuracy']:.3f} over {rt['cases']} golden cases. "
          f"Stage usage {rt['stage_usage']}."]
    for w in rt["wrong"]:
        L.append(f"- wrong: '{w['query']}' expected {w['expected']}, got {w['got']}")
    c = r["cost"]
    L += ["", "## Pipeline cost", "",
          f"Admission median {c['admit_median_ms']:.2f} ms, p95 {c['admit_p95_ms']:.2f} ms per event. "
          f"Indexing {c['index_seconds']:.2f} s for {r['corpus_events']} events.", ""]
    return "\n".join(L)


def make_components(args):
    """Build the embedder and reranker chosen on the command line."""
    embedder = reranker = None
    if args.embedder == "fastembed":
        from ..models.fastembed_adapters import DEFAULT_EMBED_MODEL, FastEmbedEmbedder
        embedder = FastEmbedEmbedder(args.embedder_model or DEFAULT_EMBED_MODEL)
    if args.reranker == "fastembed":
        from ..models.fastembed_adapters import DEFAULT_RERANK_MODEL, FastEmbedReranker
        reranker = FastEmbedReranker(args.reranker_model or DEFAULT_RERANK_MODEL)
    return embedder, reranker


def compare(path_a: str, path_b: str) -> str:
    """Side-by-side of two saved runs (for example hashing vs a real model)."""
    a, b = json.loads(Path(path_a).read_text()), json.loads(Path(path_b).read_text())
    L = [f"# Comparison: {a['embedder']} + {a['reranker']}  vs  {b['embedder']} + {b['reranker']}", "",
         "| mode | query type | recall@5 A | recall@5 B | delta | MRR A | MRR B | delta |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for mode in MODES:
        for qt in ("ALL", "exact", "paraphrase", "continue", "temporal"):
            x, y = a["retrieval"][mode]["by_type"][qt], b["retrieval"][mode]["by_type"][qt]
            L.append(f"| {mode} | {qt} | {x['recall@5']:.3f} | {y['recall@5']:.3f} | {y['recall@5']-x['recall@5']:+.3f} "
                     f"| {x['mrr']:.3f} | {y['mrr']:.3f} | {y['mrr']-x['mrr']:+.3f} |")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--embedder", choices=["hashing", "fastembed"], default="hashing")
    ap.add_argument("--embedder-model", default=None)
    ap.add_argument("--reranker", choices=["coverage", "fastembed"], default="coverage")
    ap.add_argument("--reranker-model", default=None)
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                    help="print a side-by-side of two saved runs and exit")
    args = ap.parse_args()
    if args.compare:
        print(compare(*args.compare))
        return

    embedder, reranker = make_components(args)
    ds = build_dataset(args.seed)
    ai, admit_ms, index_s = build_module(ds, embedder=embedder, reranker=reranker)
    reranker_id = getattr(ai.retriever.reranker, "model_id", "coverage-placeholder")
    result = {
        "corpus_events": len(ds.events), "queries": len(ds.queries), "embedder": ai.embedder.model_id,
        "reranker": reranker_id,
        "retrieval": eval_retrieval(ai, ds), "admission": eval_admission(ds),
        "router": eval_router(ai, ds),
        "cost": {"admit_median_ms": percentile(admit_ms, 50), "admit_p95_ms": percentile(admit_ms, 95),
                 "index_seconds": index_s},
    }
    ai.close()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.embedder}-{args.reranker}"
    md = to_markdown(result)
    for name in (tag, "latest"):
        (out / f"{name}.json").write_text(json.dumps(result, indent=2))
        (out / f"{name}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
