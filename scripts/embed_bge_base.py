#!/usr/bin/env python3
"""
FinSight – Embedding Generation Pipeline
========================================

Encodes all text chunks into vector embeddings using BGE-base
for downstream storage in Chroma or AstraDB.

Features:
- Configurable input/output paths
- Table-aware text extraction
- Batch encoding with GPU/CPU fallback
- Proper saving (no post-fix step needed)
- Progress bars and metadata summary

Author: FinSight AI Team
Date: November 2025
"""

import os
import json
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ============================================
# CONFIGURATION
# ============================================
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_INPUT = "data/chunks"
DEFAULT_OUTPUT = "docs_embeddings_bge_base.parquet"
BATCH_SIZE = 16


# ============================================
# HELPERS
# ============================================
def extract_text(chunk: dict) -> str:
    """Extracts plain text representation for both paragraph and table chunks."""
    if chunk.get("chunk_type") == "table":
        headers = chunk.get("headers", [])
        rows = chunk.get("rows", [])
        row_text = [r["values"] for r in rows if "values" in r]
        return f"Table with columns: {', '.join(headers)}. " + " | ".join(row_text)
    return chunk.get("text", "")


def load_chunks(input_dir: str) -> pd.DataFrame:
    """Load all chunk JSON files into a single DataFrame."""
    chunks = []
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"❌ Input folder not found: {input_path.resolve()}")

    for file in tqdm(os.listdir(input_path), desc="📄 Loading chunk files"):
        if file.endswith(".json"):
            with open(input_path / file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for c in data:
                    c["content"] = extract_text(c)
                    chunks.append(c)

    if not chunks:
        raise RuntimeError("❌ No chunk data found. Did you run chunk_documents.py first?")
    df = pd.DataFrame(chunks)
    print(f"✅ Loaded {len(df)} chunks from {input_path}")
    return df


def compute_embeddings(df: pd.DataFrame, model_name: str, batch_size: int = 16) -> np.ndarray:
    """Compute embeddings for all chunks with progress tracking and fallback."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using model: {model_name} on {device}")

    model = SentenceTransformer(model_name, device=device)
    texts = df["content"].tolist()

    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("⚠️ GPU memory issue — switching to CPU mode.")
            model = SentenceTransformer(model_name, device="cpu")
            embeddings = model.encode(
                texts,
                batch_size=8,
                convert_to_numpy=True,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
        else:
            raise e

    return embeddings


def save_embeddings(df: pd.DataFrame, embeddings: np.ndarray, out_path: str):
    """Save embeddings into a clean Parquet file."""
    df = df.copy()
    df["embedding"] = [emb.tolist() for emb in embeddings]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"✅ Saved {len(df)} embeddings to {out_path}")


# ============================================
# MAIN PIPELINE
# ============================================
def main(input_dir: str, output_file: str, model_name: str = DEFAULT_MODEL, batch_size: int = BATCH_SIZE):
    """Full pipeline: load → encode → save."""
    df = load_chunks(input_dir)

    # Sanity check
    required_cols = ["doc_id", "chunk_id", "chunk_type", "content"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    embeddings = compute_embeddings(df, model_name, batch_size)
    save_embeddings(df, embeddings, output_file)

    # Summary
    avg_tokens = np.mean([len(c.split()) for c in df["content"]])
    print(f"\n📊 Summary: {len(df)} chunks | Avg tokens ≈ {avg_tokens:.1f}")
    print("✅ Embedding generation complete.\n")


# ============================================
# CLI
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinSight – Generate BGE embeddings from chunked documents.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input folder containing JSON chunks")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output Parquet file path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model name")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size for encoding")

    args = parser.parse_args()
    main(args.input, args.output, args.model, args.batch)
