"""
build_index.py — Stage 2 of the pipeline.

Loads the sampled abstracts, embeds them with all-MiniLM-L6-v2, and stores
the vectors (plus metadata) in a persistent ChromaDB collection.

Run:
    python src/build_index.py

We register a CUSTOM sentence-transformers embedding function so ChromaDB
never falls back to its default ONNX-based embedder (which fails to load on
some Python 3.13 / Windows setups). Our embedding function reuses the same
SentenceTransformer model everywhere, keeping indexing and querying consistent.
"""

import sys

import pandas as pd
from tqdm import tqdm

import config


# ---------------------------------------------------------------------------
# Custom embedding function (shared by indexing and querying).
# Defined at module level so both build_index.py and rag_pipeline.py import it.
# ---------------------------------------------------------------------------
def get_embedding_function():
    """Return a ChromaDB-compatible embedding function backed by sentence-transformers."""
    from chromadb import Documents, EmbeddingFunction, Embeddings
    from sentence_transformers import SentenceTransformer

    class STEmbeddingFunction(EmbeddingFunction):
        """Wraps a SentenceTransformer so Chroma can call it directly."""

        def __init__(self, model_name: str):
            self._model_name = model_name
            self._model = SentenceTransformer(model_name)

        def __call__(self, input: Documents) -> Embeddings:
            # normalize so cosine distance behaves well
            vecs = self._model.encode(
                list(input),
                batch_size=config.EMBED_BATCH_SIZE,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return vecs.tolist()

        @staticmethod
        def name() -> str:
            return "st-minilm"

        def get_config(self):
            return {"model_name": self._model_name}

        @staticmethod
        def build_from_config(cfg):
            return STEmbeddingFunction(cfg["model_name"])

    return STEmbeddingFunction(config.EMBED_MODEL)


def main():
    if not config.SAMPLE_PARQUET.exists():
        sys.exit("Run data_ingest.py first — sample parquet not found.")

    import chromadb

    df = pd.read_parquet(config.SAMPLE_PARQUET)
    print(f"Loaded {len(df):,} papers for indexing.")

    print(f"Loading embedding model {config.EMBED_MODEL} ...")
    embed_fn = get_embedding_function()

    # Persistent client writes to disk so we only index once.
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

    # Fresh collection each run to avoid stale duplicates.
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass

    # New-style API: HNSW settings go in `configuration`, and we pass our
    # custom embedding function so Chroma never loads its ONNX default.
    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=embed_fn,
    )

    # We pass precomputed embeddings explicitly (faster + fully under our control).
    model = embed_fn._model
    n = len(df)
    for start in tqdm(range(0, n, config.EMBED_BATCH_SIZE), desc="indexing"):
        chunk = df.iloc[start:start + config.EMBED_BATCH_SIZE]
        abstracts = chunk["abstract"].tolist()
        embeddings = model.encode(
            abstracts,
            batch_size=config.EMBED_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        collection.add(
            ids=chunk["id"].tolist(),
            embeddings=embeddings,
            documents=abstracts,
            metadatas=[
                {
                    "title": str(row.title),
                    "primary_category": str(row.primary_category),
                    "year": int(row.year),
                    "authors": str(row.authors)[:300],
                    "comments": str(row.comments)[:300],
                    "submitter": str(row.submitter)[:120],
                }
                for row in chunk.itertuples()
            ],
        )

    print(f"\nIndexed {collection.count():,} documents into "
          f"'{config.COLLECTION_NAME}' at {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()