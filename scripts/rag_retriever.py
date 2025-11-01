#!/usr/bin/env python3
"""
FinSight RAG Retriever
======================

This module retrieves the most relevant banking document chunks
from the Chroma vector database for a given user query.

It is designed to integrate directly into an LLM pipeline
(e.g., LLaMA, Mistral, GPT-4) for Retrieval-Augmented Generation (RAG).

Author: FinSight AI Team
Version: 1.0
Date: 2025-11-01
"""

from sentence_transformers import SentenceTransformer
import chromadb
import os
from typing import List, Dict

# ================================================
# CONFIGURATION
# ================================================
CHROMA_PATH = "rag_index/bge"
COLLECTION_NAME = "finsight_bge"
MODEL_NAME = "BAAI/bge-base-en-v1.5"

# ================================================
# INITIALIZATION
# ================================================
try:
    print("🔹 Initializing retriever...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    model = SentenceTransformer(MODEL_NAME)
    print("✅ Retriever initialized successfully.\n")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    collection = None
    model = None


# ================================================
# RETRIEVE FUNCTION
# ================================================
def retrieve_docs(query: str, top_k: int = 3) -> List[Dict]:
    """
    Retrieve top-k most relevant document chunks from Chroma for a given query.

    Args:
        query (str): User question or input text.
        top_k (int): Number of chunks to retrieve (default = 3).

    Returns:
        List[Dict]: Each item contains text, title, doc_id, score.
    """
    if not collection or not model:
        raise RuntimeError("Retriever not initialized properly.")

    # --- Encode the query ---
    q_vec = model.encode(query, normalize_embeddings=True).tolist()

    # --- Query the Chroma collection ---
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    scores = results.get("distances", [[]])[0]

    # --- Format the results ---
    formatted = []
    for doc, meta, score in zip(docs, metas, scores):
        formatted.append({
            "text": doc,
            "title": meta.get("title", "Unknown"),
            "doc_id": meta.get("doc_id", "N/A"),
            "score": float(score)
        })

    return formatted


# ================================================
# DISPLAY FUNCTION (for manual testing)
# ================================================
def display_results(results: List[Dict]):
    """Pretty-print retrieved results for debugging."""
    if not results:
        print("❌ No results found.")
        return

    for i, r in enumerate(results, 1):
        print(f"\n📘 Result {i}:")
        print(f"   • Title: {r['title']} ({r['doc_id']})")
        print(f"   • Similarity score: {r['score']:.4f}")
        print(f"   • Snippet: {r['text'][:300]}...\n")


# ================================================
# MAIN TEST
# ================================================
if __name__ == "__main__":
    test_queries = [
        "What documents do I need to open a savings account?",
        "Explain the bank's policy on foreign transactions.",
        "How can a student apply for a loan?",
    ]

    for q in test_queries:
        print("\n" + "=" * 70)
        print(f"🔹 Query: {q}")
        print("=" * 70)

        results = retrieve_docs(q, top_k=3)
        display_results(results)
