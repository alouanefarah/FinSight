#!/usr/bin/env python3
"""
FinSight RAG Retriever (Enhanced)
=================================

Retrieves the most relevant banking document chunks
from the Chroma vector database for a given user query.

Features:
- Semantic + optional keyword hybrid search
- Cosine-score normalization and reranking
- Configurable top-k and score thresholds
- Lazy model initialization
- Optional JSON export for API use
- Pretty display for CLI testing

Author: FinSight AI Team
Date: November 2025
"""

import os
import json
import chromadb
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from pathlib import Path

# ==============================================
# CONFIGURATION
# ==============================================
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_CHROMA_PATH = "rag_index/bge"
DEFAULT_COLLECTION = "finsight_bge"


# ==============================================
# INITIALIZATION HELPERS
# ==============================================
@lru_cache(maxsize=1)
def load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Load embedding model once and cache it."""
    print(f"🚀 Loading model: {model_name}")
    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def load_collection(path: str = DEFAULT_CHROMA_PATH, name: str = DEFAULT_COLLECTION):
    """Initialize Chroma persistent collection."""
    try:
        client = chromadb.PersistentClient(path=path)
        collection = client.get_collection(name=name)
        print(f"💾 Loaded Chroma collection '{name}' from {path}")
        return collection
    except Exception as e:
        raise RuntimeError(f"❌ Failed to initialize Chroma collection: {e}")


# ==============================================
# CORE RETRIEVAL FUNCTION
# ==============================================
def retrieve_docs(
    query: str,
    top_k: int = 3,
    model_name: str = DEFAULT_MODEL,
    chroma_path: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION,
    score_threshold: float = 0.3,
    hybrid_weight: float = 0.2,
) -> List[Dict]:
    """
    Retrieve top-k most relevant chunks from Chroma for a given query.

    Args:
        query (str): User query text.
        top_k (int): Number of results to return.
        model_name (str): SentenceTransformer model name.
        chroma_path (str): Path to Chroma persistence folder.
        collection_name (str): Chroma collection name.
        score_threshold (float): Minimum normalized score to include.
        hybrid_weight (float): Weight (0–1) for keyword-based boosting.

    Returns:
        List[Dict]: Retrieved chunks with text, metadata, and scores.
    """
    model = load_model(model_name)
    collection = load_collection(chroma_path, collection_name)

    # --- Encode the query ---
    q_vec = model.encode(query, normalize_embeddings=True).tolist()

    # --- Query the vector DB ---
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k * 2,  # fetch extra for reranking
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = np.array(results.get("distances", [[]])[0], dtype=np.float32)

    # --- Convert distances → similarity (1 - normalized distance) ---
    scores = 1 - (distances / max(distances.max(), 1e-8))

    # --- Optional keyword hybrid boosting ---
    boosted_scores = []
    for doc, score in zip(docs, scores):
        keyword_boost = 0
        if hybrid_weight > 0:
            q_terms = set(query.lower().split())
            keyword_hits = sum(1 for t in q_terms if t in doc.lower())
            keyword_boost = hybrid_weight * (keyword_hits / max(len(q_terms), 1))
        boosted_scores.append(float(score + keyword_boost))

    # --- Normalize & rerank ---
    boosted_scores = np.array(boosted_scores)
    norm_scores = (boosted_scores - boosted_scores.min()) / (np.ptp(boosted_scores) + 1e-8)
    sorted_idx = np.argsort(-norm_scores)

    # --- Format results ---
    formatted = []
    for i in sorted_idx[:top_k]:
        if norm_scores[i] < score_threshold:
            continue
        formatted.append({
            "text": docs[i],
            "title": metas[i].get("title", "Unknown"),
            "doc_id": metas[i].get("doc_id", "N/A"),
            "score": round(float(norm_scores[i]), 3),
            "chunk_id": metas[i].get("chunk_id"),
            "chunk_type": metas[i].get("chunk_type"),
        })

    return formatted


# ==============================================
# OPTIONAL HELPERS
# ==============================================
def display_results(results: List[Dict]):
    """Pretty-print retrieved results."""
    if not results:
        print("[!] No relevant results found.")
        return

    print("\n📚 Top Retrieved Results:")
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r['title']} ({r['doc_id']}) – Score: {r['score']}")
        print(f"Chunk ID: {r['chunk_id']}  Type: {r['chunk_type']}")
        snippet = r['text'][:300].replace("\n", " ")
        print(f"Snippet: {snippet}...\n")


def export_results(results: List[Dict], output_path: str):
    """Save retrieved results to a JSON file for inspection."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 Results exported to {output_path}")


# ==============================================
# MAIN TEST
# ==============================================
if __name__ == "__main__":
    test_queries = [
        "What are the conditions to open a savings account?",
        "Explain the foreign transaction policy of the bank.",
        "How can students apply for personal loans?",
    ]

    for q in test_queries:
        print("\n" + "=" * 80)
        print(f"[Query] {q}")
        print("=" * 80)
        results = retrieve_docs(q, top_k=3)
        display_results(results)
