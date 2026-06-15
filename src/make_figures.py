"""
make_figures.py — turn baseline_metrics.csv into figures for the report/slides.

Produces:
    results/precision_recall_by_subcategory.png
    results/precision_recall_distribution.png

Run after evaluate_baseline.py:
    python src/make_figures.py
"""

import sys

import pandas as pd
import matplotlib.pyplot as plt

import config


def main():
    csv = config.RESULTS_DIR / "baseline_metrics.csv"
    if not csv.exists():
        sys.exit("Run evaluate_baseline.py first.")

    df = pd.read_csv(csv)

    # --- Figure 1: mean precision/recall per subcategory ------------------
    by_cat = df.groupby("subcategory")[
        ["precision_at_k", "recall_at_k"]
    ].mean().sort_values("precision_at_k", ascending=False)

    ax = by_cat.plot(kind="bar", figsize=(8, 4.5), width=0.8,
                     color=["#4A7BB7", "#C97B4B"])
    ax.set_ylabel(f"Score @{config.TOP_K}")
    ax.set_xlabel("Primary subcategory")
    ax.set_title("Baseline Naive RAG: Precision and Recall by subcategory")
    ax.legend(["Precision", "Recall"])
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out1 = config.RESULTS_DIR / "precision_recall_by_subcategory.png"
    plt.savefig(out1, dpi=200)
    plt.close()

    # --- Figure 2: distribution of per-query precision --------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["precision_at_k"], bins=20, color="#4A7BB7", alpha=0.85)
    ax.axvline(df["precision_at_k"].mean(), color="#C0392B", linestyle="--",
               label=f"mean = {df['precision_at_k'].mean():.3f}")
    ax.set_xlabel(f"Precision@{config.TOP_K}")
    ax.set_ylabel("Number of queries")
    ax.set_title("Distribution of per-query precision")
    ax.legend()
    plt.tight_layout()
    out2 = config.RESULTS_DIR / "precision_recall_distribution.png"
    plt.savefig(out2, dpi=200)
    plt.close()

    print(f"Wrote:\n  {out1}\n  {out2}")


if __name__ == "__main__":
    main()
