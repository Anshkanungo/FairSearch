# FairSearch-arXiv

**Evaluating and Mitigating Institutional Bias in Academic RAG**

A study of *institutional homophily* in Retrieval-Augmented Generation over the
arXiv corpus: do dense retrievers and LLM synthesis systematically over-represent
papers from elite institutions, and can we mitigate it without sacrificing
retrieval quality?

Course project, Information Retrieval (SEC 01, Summer 2026).

**Team**
- **Ansh Kanungo** — RQ1 (retrieval parity) and the fair re-ranking algorithm
- **Andrew Hannah** — RQ2 (synthesis neutrality): citation-share and coverage analysis
- **Wan Chi Kao** — RQ3 (fairness–utility tradeoff), dashboard, and presentation

---

## Research questions

- **RQ1 — Retrieval Parity.** Does dense vector search over arXiv cs.\* over-rank
  papers from elite (top QS-ranked) institutions relative to comparable work from
  regional ones?
- **RQ2 — Synthesis Neutrality.** When the LLM synthesizes an answer, does it
  amplify institutional concentration relative to the retrieved set (citation-share),
  and does it preferentially cite elite-institution papers (coverage rate)?
- **RQ3 — Fairness–Utility Tradeoff.** How much retrieval quality (NDCG@10, MRR) do
  we lose as we increase the fairness intervention?

This repository covers the **Week 6 baseline**: a working Naive RAG pipeline and its
Precision/Recall evaluation. Fairness measurement and mitigation come in later phases.

---

## Pipeline

```
data_ingest.py  ->  build_index.py  ->  rag_pipeline.py  ->  evaluate_baseline.py
   (filter,          (embed +            (retrieve +          (Precision@K,
    sample,           store in            Gemini              Recall@K)
    clean)            ChromaDB)           synthesis)
```

| Stage | Script | What it does |
|-------|--------|--------------|
| 1 | `src/data_ingest.py` | Stream the raw arXiv JSON, filter to cs.\* (2018–2025), dedupe, clean, write a stratified 50K-paper sample. |
| 2 | `src/build_index.py` | Embed abstracts with `all-MiniLM-L6-v2` (384-dim) and store vectors + metadata in a persistent ChromaDB collection. |
| 3 | `src/rag_pipeline.py` | The Naive RAG baseline: embed query → cosine top-K retrieval → Gemini 1.5 Flash synthesis. |
| 4 | `src/evaluate_baseline.py` | Compute Precision@K and Recall@K against a reproducible relevance proxy. |
| — | `src/make_figures.py` | Render result figures for the report/slides. |
| — | `src/config.py` | All tunable parameters in one place. |

---

## Setup

```bash
# 1. Clone and enter
git clone https://github.com/anshkanungo/FairSearch.git
cd FairSearch

# 2. Install dependencies (a virtualenv is recommended)
pip install -r requirements.txt

# 3. Download the arXiv metadata dump from Kaggle and place the .json in data/
#    https://www.kaggle.com/datasets/Cornell-University/arxiv
#    Expected path: data/arxiv-metadata-oai-snapshot.json

# 4. (Optional, for synthesis) add your Gemini key
cp .env.example .env        # then edit .env and paste your key
```

> Retrieval and evaluation run **without** a Gemini key. The key is only needed
> for the LLM synthesis step in `rag_pipeline.py`.

---

## Running

```bash
python src/data_ingest.py        # ~2-4 min, writes data/cs_sample.parquet
python src/build_index.py        # ~5-15 min on CPU, builds the vector store
python src/evaluate_baseline.py  # prints Precision@10 / Recall@10, writes results/
python src/make_figures.py       # optional: figures for slides/report

# Try one query end-to-end (needs GEMINI_API_KEY for the synthesis part):
python src/rag_pipeline.py "recent advances in graph neural networks"
```

Outputs land in `results/`:
- `baseline_metrics.csv` — per-query precision/recall
- `baseline_summary.json` — aggregate means
- `*.png` — figures

---

## How relevance is defined (baseline evaluation)

arXiv has no human relevance judgments, so we use a transparent proxy. A retrieved
paper counts as **relevant** to a query if it (1) shares the query's primary cs.\*
subcategory **and** (2) shares at least two distinctive (non-stopword, length ≥ 4)
terms with the query. Queries are the titles of held-out papers, whose own
subcategory defines the relevant set.

This is a *proxy*, not ground truth — see the report's limitations section. It is
fully reproducible (no API, no manual labeling) which is the right tradeoff for a
baseline.

---

## Tech stack

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, CPU-friendly)
- **Vector store:** ChromaDB (local, persistent, cosine similarity)
- **LLM:** Google Gemini 1.5 Flash (synthesis)
- **Evaluation:** pandas + scikit-learn

---

## Repository layout

```
FairSearch/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── src/
│   ├── config.py            # all parameters
│   ├── data_ingest.py       # stage 1: sample + clean
│   ├── build_index.py       # stage 2: embed + index
│   ├── rag_pipeline.py      # stage 3: retrieve + synthesize
│   ├── evaluate_baseline.py # stage 4: precision/recall
│   └── make_figures.py      # figures for report/slides
├── data/                    # (gitignored) raw dump, sample, chroma db
├── results/                 # metrics + figures
└── notebooks/               # exploratory analysis
```

---

## Roadmap

- [x] **Phase I (Wk 3–5):** proposal, Naive RAG baseline, Precision/Recall
- [ ] **Phase II (Wk 6–8):** institution labeling, retrieval bias audit (RQ1, RQ2)
- [ ] **Phase III (Wk 9–12):** fair re-ranking, λ sweep, fairness–utility tradeoff (RQ3)
- [ ] **Phase IV (Wk 13–14):** Streamlit dashboard, final report
