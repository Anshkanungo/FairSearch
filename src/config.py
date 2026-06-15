"""
config.py — Central configuration for the FairSearch-arXiv pipeline.

Keeping every tunable knob in one place makes experiments reproducible:
change a value here and the whole pipeline (indexing, retrieval, evaluation)
picks it up. No magic numbers scattered across scripts.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent      # repo root
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CHROMA_DIR = DATA_DIR / "chroma_db"                 # persistent vector store

# Raw Kaggle dump (download separately — see README).
# https://www.kaggle.com/datasets/Cornell-University/arxiv
RAW_ARXIV_JSON = DATA_DIR / "arxiv-metadata-oai-snapshot.json"

# Cleaned, sampled subset produced by data_ingest.py
SAMPLE_PARQUET = DATA_DIR / "cs_sample.parquet"

# ---------------------------------------------------------------------------
# Sampling parameters
# ---------------------------------------------------------------------------
SAMPLE_SIZE = 50_000          # number of abstracts to keep
YEAR_MIN = 2018               # inclusive lower bound on update_date year
YEAR_MAX = 2025               # inclusive upper bound
RANDOM_SEED = 42              # reproducible sampling

# Only keep papers whose primary category starts with one of these.
# (arXiv stores categories space-delimited; the FIRST token is primary.)
CS_PREFIX = "cs."

# Subcategories we stratify across so the sample is not topic-skewed.
TARGET_SUBCATEGORIES = [
    "cs.LG",  # machine learning
    "cs.CL",  # computation & language
    "cs.CV",  # computer vision
    "cs.AI",  # artificial intelligence
    "cs.IR",  # information retrieval
    "cs.CR",  # cryptography & security
    "cs.DB",  # databases
    "cs.SE",  # software engineering
]

# ---------------------------------------------------------------------------
# Embedding + vector store
# ---------------------------------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim, fast, CPU-friendly
EMBED_DIM = 384
EMBED_BATCH_SIZE = 256        # abstracts encoded per batch
COLLECTION_NAME = "arxiv_cs_abstracts"

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = 10                    # documents returned per query

# ---------------------------------------------------------------------------
# LLM synthesis (Gemini)
# ---------------------------------------------------------------------------
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_MAX_TOKENS = 1024
# API key is read from the environment variable GEMINI_API_KEY (see .env.example)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
# A retrieved paper counts as RELEVANT to a query if BOTH hold:
#   (1) it shares the query's primary cs.* subcategory, AND
#   (2) its abstract shares at least MIN_TERM_OVERLAP distinctive terms
#       with the query text.
# This is a reproducible proxy for human relevance judgments (see report).
MIN_TERM_OVERLAP = 2
NUM_EVAL_QUERIES = 100        # held-out queries for precision/recall
