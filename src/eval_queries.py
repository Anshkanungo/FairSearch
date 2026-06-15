"""
eval_queries.py — Curated neutral evaluation queries.

These natural-phrase queries mimic how researchers actually search (per the
project brief's example, "Recent advances in Graph Neural Networks"). Each is
mapped to the cs.* subcategory it most naturally belongs to; a retrieved paper
is considered relevant if it is in that subcategory AND shares distinctive
terms with the query.

This same set is reused in Phase II for the institutional-bias audit, so the
fairness measurements run over identical, neutral queries.
"""

# (query text, expected primary subcategory)
EVAL_QUERIES = [
    ("Recent advances in graph neural networks", "cs.LG"),
    ("Large language models for text generation", "cs.CL"),
    ("Vision language models for image understanding", "cs.CV"),
    ("Deep reinforcement learning for robotics control", "cs.RO"),
    ("Adversarial attacks on deep neural networks", "cs.CR"),
    ("Federated learning for privacy preservation", "cs.LG"),
    ("Transformer architectures for sequence modeling", "cs.CL"),
    ("Object detection in computer vision", "cs.CV"),
    ("Neural information retrieval and dense retrieval", "cs.IR"),
    ("Knowledge graph embeddings and reasoning", "cs.AI"),
    ("Self-supervised representation learning", "cs.LG"),
    ("Semantic segmentation of images", "cs.CV"),
    ("Question answering with large language models", "cs.CL"),
    ("Graph representation learning for recommendation", "cs.IR"),
    ("Differential privacy in machine learning", "cs.CR"),
    ("Automated software testing techniques", "cs.SE"),
    ("Distributed training of deep learning models", "cs.DC"),
    ("Generative adversarial networks for image synthesis", "cs.CV"),
]