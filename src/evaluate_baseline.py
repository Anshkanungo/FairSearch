"""
evaluate_baseline.py — Stage 4: measure baseline retrieval quality.

Computes Precision@K and Recall@K for the Naive RAG retriever over a curated
set of neutral natural-phrase queries (see eval_queries.py). Relevance proxy
(no human labels, no external API):

    A candidate paper is RELEVANT to a query if
        (1) it is in the query's expected primary cs.* subcategory, AND
        (2) its abstract shares >= MIN_TERM_OVERLAP distinctive terms
            with the query.

Precision@K = relevant retrieved / K.
Recall@K    = relevant retrieved / total relevant in the corpus.
(Recall@K is structurally low because broad queries have large relevant sets
while K is small; we report it per the assignment and interpret accordingly.)

Run:
    python src/evaluate_baseline.py

Outputs:
    results/baseline_metrics.csv   — per-query precision/recall
    results/baseline_summary.json  — aggregate means
"""

import json
import re
import sys

import pandas as pd

import config
from eval_queries import EVAL_QUERIES

STOPWORDS = set("""
a an the of for and or to in on with via using from at by as is are be this
that these those we our their its it new towards toward into over under more
less most least can could should would may might will into within across
""".split())


def distinctive_terms(text: str) -> set:
    """Lowercase alpha tokens, length>=4, not a stopword."""
    tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


def is_relevant(query_terms: set, query_subcat: str, cand_meta: dict,
                cand_abstract: str) -> bool:
    """Relevant = same expected subcategory AND enough term overlap."""
    if cand_meta.get("primary_category") != query_subcat:
        return False
    overlap = query_terms & distinctive_terms(cand_abstract)
    return len(overlap) >= config.MIN_TERM_OVERLAP


def main():
    if not config.SAMPLE_PARQUET.exists():
        sys.exit("Run data_ingest.py and build_index.py first.")

    from rag_pipeline import NaiveRAG

    df = pd.read_parquet(config.SAMPLE_PARQUET).reset_index(drop=True)

    # Precompute distinctive terms per abstract once (recall denominator).
    all_abstract_terms = [distinctive_terms(a) for a in df["abstract"].tolist()]
    all_subcats = df["primary_category"].tolist()

    print(f"Evaluating on {len(EVAL_QUERIES)} curated neutral queries.\n")
    rag = NaiveRAG()
    per_query = []

    for query, q_subcat in EVAL_QUERIES:
        q_terms = distinctive_terms(query)
        hits = rag.retrieve(query, k=config.TOP_K)

        rel_retrieved = sum(
            is_relevant(q_terms, q_subcat, h["metadata"], h["abstract"])
            for h in hits
        )

        # Total relevant in corpus for this query (recall denominator).
        total_relevant = 0
        for terms, subcat in zip(all_abstract_terms, all_subcats):
            if subcat == q_subcat and len(q_terms & terms) >= config.MIN_TERM_OVERLAP:
                total_relevant += 1

        k = len(hits) if hits else 1
        precision = rel_retrieved / k
        recall = (rel_retrieved / total_relevant) if total_relevant else 0.0

        per_query.append({
            "query": query,
            "subcategory": q_subcat,
            "relevant_retrieved": rel_retrieved,
            "k": k,
            "total_relevant": total_relevant,
            "precision_at_k": round(precision, 4),
            "recall_at_k": round(recall, 4),
        })

    metrics = pd.DataFrame(per_query)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(config.RESULTS_DIR / "baseline_metrics.csv", index=False)

    summary = {
        "num_queries": len(metrics),
        "top_k": config.TOP_K,
        "mean_precision_at_k": round(metrics["precision_at_k"].mean(), 4),
        "mean_recall_at_k": round(metrics["recall_at_k"].mean(), 4),
        "embedding_model": config.EMBED_MODEL,
        "query_type": "curated neutral natural-phrase queries",
        "relevance_rule": (
            f"expected primary subcategory AND >= {config.MIN_TERM_OVERLAP} "
            "distinctive term overlap"
        ),
    }
    with open(config.RESULTS_DIR / "baseline_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("=" * 56)
    print(f"  Baseline Naive RAG — {config.EMBED_MODEL}")
    print("=" * 56)
    print(f"  Queries evaluated : {summary['num_queries']}")
    print(f"  Top-K             : {summary['top_k']}")
    print(f"  Mean Precision@{config.TOP_K} : {summary['mean_precision_at_k']:.4f}")
    print(f"  Mean Recall@{config.TOP_K}    : {summary['mean_recall_at_k']:.4f}")
    print("=" * 56)
    print("\nPer-query results:")
    print(metrics[["query", "subcategory", "precision_at_k", "recall_at_k"]]
          .to_string(index=False))
    print(f"\nWrote per-query CSV and summary JSON to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()