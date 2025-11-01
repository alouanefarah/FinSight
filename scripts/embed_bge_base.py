# ============================================
# STEP 3 — Embedding Generation (BGE-BASE)
# ============================================

from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from tqdm import tqdm
import os, json, torch

import ast
# --- Paths ---
CHUNKS_DIR = "data/chunks/"
OUT_PATH = "docs_embeddings_bge_base.parquet"

# --- 1️⃣ Load and normalize chunks ---
def extract_text(chunk):
    """Handles both paragraph and table chunks."""
    if chunk.get("chunk_type") == "table":
        headers = chunk.get("headers", [])
        rows = chunk.get("rows", [])
        table_text = []
        for r in rows:
            table_text.append(r["values"])
        return f"Table with columns: {', '.join(headers)}. " + " | ".join(table_text)
    else:
        return chunk.get("text", "")

all_chunks = []
for file in os.listdir(CHUNKS_DIR):
    if file.endswith(".json"):
        with open(os.path.join(CHUNKS_DIR, file), "r", encoding="utf-8") as f:
            data = json.load(f)
            for c in data:
                c["content"] = extract_text(c)
                all_chunks.append(c)

df = pd.DataFrame(all_chunks)
print(f"✅ Loaded {len(df)} chunks.")

# --- 2️⃣ Load embedding model ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "BAAI/bge-base-en-v1.5"
model = SentenceTransformer(model_name, device=device)
print(f"🚀 Using {model_name} on {device}")

# --- 3️⃣ Compute embeddings ---
embeddings = model.encode(
    df["content"].tolist(),
    batch_size=16,
    convert_to_numpy=True,
    show_progress_bar=True,
    normalize_embeddings=True
)

# --- 4️⃣ Save output ---
df["embedding"] = [emb.tolist() for emb in embeddings]
df.to_parquet(OUT_PATH, index=False)
print(f"✅ Saved embeddings to {OUT_PATH}")



df = pd.read_parquet("docs_embeddings_bge_base.parquet")

# Convert stringified lists to actual lists of floats
def fix_embedding(e):
    if isinstance(e, str):
        try:
            return ast.literal_eval(e)
        except Exception:
            return []
    return e

df["embedding"] = df["embedding"].apply(fix_embedding)

print(type(df['embedding'][0]), len(df['embedding'][0]))  # should now show <class 'list'> 768

# Save the fixed version
df.to_parquet("docs_embeddings_bge_base_fixed.parquet", index=False)
print("✅ Fixed embeddings saved to docs_embeddings_bge_base_fixed.parquet")


