"""
rag_pipeline.py — Stage 3: the Naive RAG baseline.

Given a text query, this module:
    1. embeds the query with the same model used for indexing,
    2. retrieves the top-K most similar abstracts from ChromaDB (cosine),
    3. asks Gemini 1.5 Flash to synthesize an answer from that context.

This is the *baseline* (no fairness intervention). It is what we audit for
institutional homophily in later phases.

Usage (interactive):
    python src/rag_pipeline.py "recent advances in graph neural networks"

Or import NaiveRAG in a notebook / evaluation script.
"""

import os
import sys

import config
from build_index import get_embedding_function


class NaiveRAG:
    def __init__(self):
        import chromadb
        from sentence_transformers import SentenceTransformer

        # Reuse the same embedding model used at index time.
        self.model = SentenceTransformer(config.EMBED_MODEL)

        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        # Pass the custom embedding function so Chroma never loads its ONNX
        # default when resolving the collection.
        self.collection = client.get_collection(
            config.COLLECTION_NAME,
            embedding_function=get_embedding_function(),
        )

        self._gemini = None  # lazy-init; retrieval works without a key

    # --- Retrieval ---------------------------------------------------------
    def retrieve(self, query: str, k: int = None):
        """Return the top-k documents with their metadata and distances."""
        k = k or config.TOP_K
        q_emb = self.model.encode(
            [query], normalize_embeddings=True
        ).tolist()
        res = self.collection.query(
            query_embeddings=q_emb,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append({
                "abstract": doc,
                "metadata": meta,
                "distance": dist,             # cosine distance (lower = closer)
                "similarity": 1.0 - dist,     # convenience
            })
        return hits

    # --- Synthesis ---------------------------------------------------------
    def _init_gemini(self):
        import google.generativeai as genai

        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Export it or add it to a .env file "
                "(see .env.example). Retrieval works without it; synthesis does not."
            )
        genai.configure(api_key=key)
        self._gemini = genai.GenerativeModel(config.GEMINI_MODEL)

    def synthesize(self, query: str, hits: list) -> str:
        """Ask Gemini to answer the query using the retrieved abstracts."""
        if self._gemini is None:
            self._init_gemini()

        context = "\n\n".join(
            f"[{i+1}] {h['metadata'].get('title','')}\n{h['abstract']}"
            for i, h in enumerate(hits)
        )
        prompt = (
            "You are a scientific research assistant. Using ONLY the abstracts "
            "below, write a concise synthesis answering the question. Cite "
            "sources inline by their bracket number.\n\n"
            f"Question: {query}\n\n"
            f"Abstracts:\n{context}\n\nSynthesis:"
        )
        resp = self._gemini.generate_content(
            prompt,
            generation_config={"max_output_tokens": config.GEMINI_MAX_TOKENS},
        )
        return resp.text

    # --- Convenience: full pipeline ---------------------------------------
    def answer(self, query: str, k: int = None):
        hits = self.retrieve(query, k)
        synthesis = self.synthesize(query, hits)
        return synthesis, hits


def _cli():
    if len(sys.argv) < 2:
        sys.exit('Usage: python src/rag_pipeline.py "your query here"')
    query = " ".join(sys.argv[1:])
    rag = NaiveRAG()
    print(f"\nQuery: {query}\n")

    hits = rag.retrieve(query)
    print(f"Top {len(hits)} retrieved papers:")
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        print(f"  {i:2d}. [{m['primary_category']}] {m['title'][:70]} "
              f"(sim={h['similarity']:.3f})")

    if os.environ.get("GEMINI_API_KEY"):
        print("\n--- Gemini synthesis ---")
        print(rag.synthesize(query, hits))
    else:
        print("\n(Set GEMINI_API_KEY to see the synthesized answer.)")


if __name__ == "__main__":
    _cli()