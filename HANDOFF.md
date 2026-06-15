# FairSearch-arXiv — Week 6 Internal Handoff

**Purpose of this document:** This is an *internal* working doc, not the
deliverable. It captures everything we built and learned today, plus all the
raw material the actual ACM report and the 10-slide deck need, organized
against the Week 6 rubric. Andrew and Wan: you can read this top-to-bottom (or
paste it into an AI tool) to generate the real report and slides.

**Team & roles**
- **Ansh Kanungo** — RQ1 (retrieval parity) + the fair re-ranking algorithm; built the baseline pipeline.
- **Andrew Hannah** — RQ2 (synthesis neutrality): citation-share + coverage-rate analysis.
- **Wan Chi Kao** — RQ3 (fairness–utility tradeoff) + dashboard + presentation.

---

## ⚠️ READ FIRST — things that will save you hours

1. **Use Python 3.11, NOT 3.13.** ChromaDB silently crashes on Python 3.13
   (Windows access-violation during `collection.add`). The identical chromadb
   version (1.5.9) works perfectly on 3.11. This cost us most of a day. Setup:
   ```
   py -3.11 -m venv .venv311
   .\.venv311\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. **`pyarrow` is required** to read the parquet sample — it's in requirements now.
3. **The dataset is NOT in the repo** (5.3 GB). Download from Kaggle:
   https://www.kaggle.com/datasets/Cornell-University/arxiv → put the json in `data/`.
4. **Numbers in this doc are REAL** (we ran them) unless marked `[TODO]`.

---

## 1. What we did today (the narrative)

We built and ran the complete **Naive RAG baseline** end-to-end:

```
data_ingest.py → build_index.py → rag_pipeline.py → evaluate_baseline.py → make_figures.py
```

- Streamed the full **3,073,376-record** arXiv metadata dump, filtered to cs.\*
  papers from 2018–2025 (**557,514** qualifying), and sampled a balanced
  **50,000**-abstract subset stratified across 8 cs.\* subcategories.
- Embedded all abstracts with **all-MiniLM-L6-v2** (384-dim) and indexed them in
  **ChromaDB** (cosine similarity). Final index: **49,909** documents.
- Built the Naive RAG pipeline: query → top-10 cosine retrieval → Gemini 1.5
  Flash synthesis.
- Evaluated retrieval quality on **18 curated neutral queries** using a
  reproducible relevance proxy. **Result: Precision@10 = 0.71.**

We also did real **data exploration** that reshaped the project plan (see §4).

---

## 2. Problem Description  *(rubric: Problem Description — carry forward & refine)*

RAG-based academic search embeds a query, retrieves the most similar paper
abstracts by cosine similarity, and has an LLM synthesize an answer. This
concentrates two biases into one opaque pipeline: **representational bias** in
pretrained embeddings and **consensus bias** in LLM summarization.

We study **institutional homophily**: the tendency of dense retrievers to
over-rank papers from a small set of well-resourced (typically elite, Western)
institutions, even when comparable work exists from regional or emerging
research hubs. Mechanism is compounding: (1) embedding models are trained on
corpora over-representing elite-institution writing; (2) heavily-cited
institutions' papers cluster more tightly in embedding space, biasing
nearest-neighbor retrieval; (3) the LLM faithfully amplifies whatever
distribution it receives, presenting an institutional imbalance to the user as
the objective state of the field. The downstream effect is an algorithmic
Matthew effect in science — the already-visible become more visible.

---

## 3. Research Questions & Hypotheses  *(rubric: RQs and Hypotheses)*

- **RQ1 — Retrieval Parity (Ansh).** Does dense retrieval over arXiv cs.\*
  systematically rank elite-institution papers above comparable
  regional-institution papers?
  **H1:** Baseline retrieval shows a Selection Rate Ratio ≥ 2.0 favoring top-20
  QS-ranked institutions across neutral queries.

- **RQ2 — Synthesis Neutrality (Andrew).** Does the LLM amplify institutional
  concentration during synthesis? Two measurements: (a) **citation-share** —
  compare the institutional distribution of *cited* papers vs *retrieved*
  papers (if retrieved is 60/40 elite/regional but cited is 85/15, the LLM
  amplifies); (b) **coverage rate** — fraction of retrieved papers actually
  cited, broken down elite vs regional.
  **H2:** Generated answers over-represent elite-institution sources relative to
  the retrieved context.

- **RQ3 — Fairness–Utility Tradeoff (Wan).** When mitigation (FairMMR-Inst) is
  applied, what is the tradeoff between fairness (Selection Rate Ratio) and
  utility (NDCG@10, Precision@10)?
  **H3:** A dual-criterion re-ranker reduces institutional concentration with
  limited precision loss; we expect a smooth Pareto frontier as the fairness
  weight is swept.

*(Note: RQ1/RQ3 fairness numbers depend on institution labels, which require
enrichment — see §4 and §7. This update establishes the baseline; bias
measurement is next phase per the rubric.)*

---

## 4. Dataset  *(rubric: Dataset — describe sample, filtering, preprocessing, justify)*

**Source.** Cornell University arXiv Metadata snapshot (Kaggle), newline-delimited
JSON, **3,073,376 total records** (we verified this by counting; correct the
old "2.5M" estimate). One record per submission.

**Fields present (14):** `id`, `submitter`, `authors`, `authors_parsed`,
`title`, `comments`, `journal-ref`, `doi`, `report-no`, `categories`,
`license`, `abstract`, `versions`, `update_date`.

**Filtering & sampling.**
- Kept papers whose **primary category** starts with `cs.` and whose
  `update_date` year is **2018–2025** → 557,514 qualifying papers.
- Sampled **50,000**, stratified across 8 target subcategories so no single
  topic dominates fairness measurements.
- Dropped abstracts < 50 chars; deduplicated on arXiv `id`.

**Sample subcategory distribution (real):**

| Subcat | Count | | Subcat | Count |
|--------|-------|---|--------|-------|
| cs.CV  | 5,739 | | cs.IR  | 5,575 |
| cs.LG  | 5,739 | | cs.SE  | 5,575 |
| cs.CL  | 5,652 | | cs.DB  | 4,723 |
| cs.AI  | 5,598 | | (+ cs.RO, cs.IT, cs.HC, cs.DC overflow) |
| cs.CR  | 5,596 | | | |

**Preprocessing.** Whitespace/newline normalization on titles and abstracts;
year extracted from `update_date`; primary category parsed as first token of
`categories`.

**🔑 KEY FINDING — affiliations are essentially absent.** This reshapes Phase II.
We measured institution-data availability on 50k recent cs.\* papers:

| Signal | Coverage |
|--------|----------|
| `authors_parsed` affiliation slot | **0.3%** |
| `report-no` | 1.1% |
| `doi` (bridge to external sources) | 24.9% |
| `journal-ref` | 17.9% |

So institution labels **cannot** come from arXiv metadata. We also tested
**OpenAlex** (by DOI and by filter) on sample papers → returned **empty
institutions / 0 results**. Conclusion: institution enrichment is a real Phase
II engineering task (robust OpenAlex queries + Semantic Scholar fallback +
author-name matching), not a quick lookup. This is a genuine finding, not a
failure — it redefines the bias-measurement plan.

---

## 5. Experimentation & Results  *(rubric: Naive RAG + Precision/Recall, with figures)*

**System architecture.**
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, CPU, cosine, normalized).
- Vector store: **ChromaDB** (local persistent, cosine, HNSW).
- LLM: **Google Gemini 1.5 Flash** for synthesis.
- Pipeline: query → embed → top-10 cosine retrieval → Gemini synthesis with inline citations.

**Evaluation method.** 18 curated **neutral natural-phrase queries** (e.g.,
"Recent advances in graph neural networks"), each mapped to an expected cs.\*
subcategory. A retrieved paper is **relevant** if it (1) is in the expected
subcategory AND (2) shares ≥ 2 distinctive terms with the query. This is a
reproducible proxy (no human labels, no API).

**Headline results (REAL):**

| Metric | Value |
|--------|-------|
| Mean **Precision@10** | **0.7056** |
| Mean **Recall@10** | 0.0085 |
| Queries | 18 |
| Index size | 49,909 docs |

**Per-query precision (selected):** perfect (1.0) for graph neural networks,
LLMs for text generation, neural information retrieval; strong (0.8–0.9) for
object detection, semantic segmentation, QA, software testing; low (0.2–0.3)
for "RL for robotics control" and "distributed training" — see interpretation.

**Interpreting Precision@10 = 0.71:** strong for an untuned baseline — ~7 of 10
retrieved papers are on-topic. The retriever clearly works.

**Interpreting low Recall@10 (HONEST — include this):** Recall@10 is
structurally tiny because each broad query has hundreds of proxy-relevant
papers in the corpus while K=10 (so the max possible recall is ~0.02 regardless
of quality). We report it per the assignment but treat **Precision@10 as the
primary baseline-quality signal**. A more meaningful recall would require a
small human-labeled ground-truth set or known-item queries — noted for future
work.

**Interpreting the low-precision queries (HONEST):** "RL for robotics" (0.2) and
"distributed training" (0.2) score low not because retrieval failed but because
those papers cross-file in arXiv — most RL and distributed-training papers sit
in cs.LG rather than the cs.RO/cs.DC subcategories we assigned. This reflects
fuzzy subcategory boundaries, not retrieval failure.

**Figures (in `results/`, use in report + slides):**
- `precision_recall_distribution.png` — histogram of per-query precision, mean
  line at 0.706. **Lead with this one** — shows most queries 0.6–1.0 with a
  small hard tail.
- `precision_recall_by_subcategory.png` — precision/recall bars per subcategory;
  recall bars are near-invisible (visually makes the recall point).

---

## 6. Background  *(rubric: expand to 10 peer-reviewed papers, 3-part structure)*

**STATUS: 5 papers done (from the proposal), 5 MORE NEEDED. [TODO — Andrew/Ansh]**

Already reviewed (in the proposal, keep these):
1. **Carbonell & Goldstein (1998), SIGIR** — MMR diversity reranking.
2. **Singh & Joachims (2018), KDD** — Fairness of Exposure in Rankings.
3. **Karako & Manggala (2018), UMAP** — FMMR (fairness-aware MMR).
4. **Rekabsaz, Kopeinik & Schedl (2021), SIGIR** — societal bias in BERT rankers.
5. **Kim & Diaz (2024), ICTIR** — fair ranking in RAG.

**[TODO] Find 5 more peer-reviewed papers** (SIGIR, FAccT, ECIR, CIKM, WWW, or
ACL — no blogs/news). Each needs the 3-part structure: *What is it about? / What
did they do? / Limitations or conclusions.* Suggested search topics to find
relevant ones: fair ranking metrics (e.g., Zehlike FA\*IR CIKM 2017), exposure
fairness in recommendation, bias in dense retrieval, LLM citation bias, fairness
in retrieval-augmented generation, demographic parity in IR. **Cannot be
auto-generated — must be real papers you actually read and cite.**

---

## 7. Plan for Next Phase  *(rubric: Plan for Next)*

- **Institution enrichment (unblocks all bias work):** build a robust pipeline
  to resolve arXiv IDs → author affiliations via OpenAlex (proper endpoints) +
  Semantic Scholar fallback + CrossRef for the 25% with DOIs. Report coverage
  *by group* to detect non-random missingness (regional schools may be labeled
  less often — flagged by Andrew).
- **Demographic mapping:** label institutions Elite (top-20 QS) vs Regional;
  prepare robustness proxies (CS h5-index, Global North/South).
- **RQ1 bias audit:** compute Selection Rate Ratio on the 18+ neutral queries.
- **RQ2 (Andrew):** citation-share + coverage-rate via Gemini inline citations.
- **RQ3 (Wan):** implement FairMMR-Inst, sweep λ, plot fairness–utility Pareto.
- **Dashboard (Wan):** Streamlit side-by-side baseline vs fair, with scorecard.

---

## 8. GitHub repo status  *(rubric: organized, README, commented, preprocessing + RAG code)*

Repo: **github.com/anshkanungo/FairSearch**

```
FairSearch/
├── README.md
├── requirements.txt          # pinned, Python 3.11 warning
├── requirements-lock.txt     # exact working snapshot
├── .env.example              # GEMINI_API_KEY template
├── .gitignore                # excludes data/, .env, venv
├── src/
│   ├── config.py             # all parameters
│   ├── data_ingest.py        # stage 1: filter + sample + clean
│   ├── build_index.py        # stage 2: embed + index (ChromaDB)
│   ├── rag_pipeline.py       # stage 3: retrieve + Gemini synthesis
│   ├── eval_queries.py       # 18 curated neutral queries
│   ├── evaluate_baseline.py  # stage 4: Precision@10 / Recall@10
│   └── make_figures.py       # figures for report/slides
├── results/
│   ├── baseline_metrics.csv
│   ├── baseline_summary.json
│   ├── precision_recall_distribution.png
│   └── precision_recall_by_subcategory.png
└── data/                     # (gitignored) raw dump, sample, chroma db
```

How to run (after Python 3.11 setup + Kaggle download):
```
python src/data_ingest.py        # → data/cs_sample.parquet
python src/build_index.py        # → ChromaDB index (5–15 min)
python src/evaluate_baseline.py  # → prints Precision@10, writes results/
python src/make_figures.py       # → figures
```

---

## 9. Suggested 10-slide deck structure  *(rubric: 10 slides, 10-min talk)*

1. **Title & Team** — project title, three names + roles.
2. **Problem Recap & Motivation** — institutional homophily, why it matters for scientific equity (§2).
3. **Research Questions & Hypotheses** — RQ1/RQ2/RQ3 + H1/H2/H3 (§3).
4. **Dataset & Sampling** — 3M records → 50K cs.\* sample, balanced subcats, preprocessing (§4).
5. **System Architecture** — diagram: query → MiniLM embed → ChromaDB → Gemini (§5).
6. **Baseline Results** — Precision@10 = 0.71, the distribution figure, per-query table (§5).
7. **Early Fairness Observations** — the affiliation-coverage finding (0.3%), why bias measurement needs enrichment (§4). *This is our "early observation."*
8. **Challenges & Design Decisions** — Python 3.13→3.11 (ChromaDB), relevance-proxy choice, recall interpretation, OpenAlex dead-end (§1, §4, §5).
9. **Plan for Next Phase** — enrichment → bias audit → FairMMR-Inst → dashboard (§7).
10. **Questions & Discussion.**

---

## 10. Honest status checklist

| Item | Status |
|------|--------|
| Naive RAG pipeline | ✅ Built, runs |
| Precision@10 = 0.71 | ✅ Real |
| Recall@10 + honest caveat | ✅ Real |
| Figures | ✅ Generated |
| Dataset stats (3M, 50K, 0.3% affil) | ✅ Real |
| Code on GitHub + README | ✅ (push pending) |
| Background: 5 papers | ✅ From proposal |
| **Background: +5 new papers** | ❌ **TODO — must be real, can't auto-gen** |
| Institution-bias numbers | ⏭ Next phase (rubric expects this) |
| Actual ACM report (4–5 pg) | ⏭ To write from this doc |
| 10 slides | ⏭ To build from §9 |
