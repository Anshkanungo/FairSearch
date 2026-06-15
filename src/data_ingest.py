"""
data_ingest.py — Stage 1 of the pipeline.

Reads the raw arXiv Kaggle JSON dump (~2.5M records), filters to cs.*
papers in our year window, removes duplicates, performs light cleaning,
and writes a stratified ~50K sample to data/cs_sample.parquet.

Run:
    python src/data_ingest.py

The raw dump is line-delimited JSON (one record per line), so we stream it
line by line rather than loading 4GB into memory at once.
"""

import json
import re
import sys

import pandas as pd
from tqdm import tqdm

import config


def parse_primary_category(categories: str) -> str:
    """arXiv stores categories space-delimited; the first token is primary."""
    if not categories:
        return ""
    return categories.split()[0]


def extract_year(update_date: str) -> int:
    """update_date looks like '2021-04-12'. Return the year as int, or -1."""
    if not update_date:
        return -1
    m = re.match(r"(\d{4})", update_date)
    return int(m.group(1)) if m else -1


def clean_text(text: str) -> str:
    """Collapse whitespace/newlines in abstracts so embeddings are stable."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def stream_records(path):
    """Yield one parsed JSON record per line, skipping malformed lines."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # skip the rare malformed line


def main():
    if not config.RAW_ARXIV_JSON.exists():
        sys.exit(
            f"Raw dump not found at {config.RAW_ARXIV_JSON}.\n"
            "Download it from "
            "https://www.kaggle.com/datasets/Cornell-University/arxiv "
            "and place the .json in the data/ folder."
        )

    print("Streaming raw records and filtering to cs.* ...")
    rows = []
    for rec in tqdm(stream_records(config.RAW_ARXIV_JSON)):
        primary = parse_primary_category(rec.get("categories", ""))
        if not primary.startswith(config.CS_PREFIX):
            continue

        year = extract_year(rec.get("update_date", ""))
        if not (config.YEAR_MIN <= year <= config.YEAR_MAX):
            continue

        abstract = clean_text(rec.get("abstract", ""))
        if len(abstract) < 50:        # drop empty / stub abstracts
            continue

        rows.append({
            "id": rec.get("id", ""),
            "title": clean_text(rec.get("title", "")),
            "abstract": abstract,
            "authors": rec.get("authors", ""),
            "primary_category": primary,
            "categories": rec.get("categories", ""),
            "year": year,
            "doi": rec.get("doi", "") or "",
            "comments": rec.get("comments", "") or "",
            "submitter": rec.get("submitter", "") or "",
        })

    df = pd.DataFrame(rows)
    print(f"  kept {len(df):,} cs.* records in {config.YEAR_MIN}-{config.YEAR_MAX}")

    # --- Deduplicate on arXiv id (keep first occurrence) -------------------
    before = len(df)
    df = df.drop_duplicates(subset="id").reset_index(drop=True)
    print(f"  removed {before - len(df):,} duplicate ids")

    # --- Stratified sample across target subcategories --------------------
    # Take an equal-ish slice from each target subcategory so no single topic
    # dominates. Papers whose primary cat is cs.* but not in the target list
    # are still eligible via the "other" bucket.
    df["bucket"] = df["primary_category"].where(
        df["primary_category"].isin(config.TARGET_SUBCATEGORIES), other="cs.other"
    )

    per_bucket = config.SAMPLE_SIZE // (len(config.TARGET_SUBCATEGORIES) + 1)
    sampled = []
    for bucket, grp in df.groupby("bucket"):
        n = min(per_bucket, len(grp))
        sampled.append(grp.sample(n=n, random_state=config.RANDOM_SEED))
    sample_df = pd.concat(sampled).reset_index(drop=True)

    # Top up to SAMPLE_SIZE if rounding left us short
    if len(sample_df) < config.SAMPLE_SIZE:
        remaining = df.drop(sample_df.index, errors="ignore")
        extra = remaining.sample(
            n=min(config.SAMPLE_SIZE - len(sample_df), len(remaining)),
            random_state=config.RANDOM_SEED,
        )
        sample_df = pd.concat([sample_df, extra]).reset_index(drop=True)

    sample_df = sample_df.drop(columns=["bucket"])
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample_df.to_parquet(config.SAMPLE_PARQUET, index=False)

    print(f"\nWrote {len(sample_df):,} sampled papers to {config.SAMPLE_PARQUET}")
    print("Subcategory distribution in sample:")
    print(sample_df["primary_category"].value_counts().head(12).to_string())


if __name__ == "__main__":
    main()
