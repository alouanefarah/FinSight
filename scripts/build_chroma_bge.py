# ============================================
# STEP 4 — Build Vector DB (BGE Model)
# ============================================

import contextlib
import sys
import io
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

# --- 1️⃣ Load embeddings file ---
df = pd.read_parquet("docs_embeddings_bge_base_fixed.parquet")
print(f"✅ Loaded {len(df)} rows")

# --- Sanity check ---
print("Columns:", df.columns.tolist())
print(df.head(2))

# --- 2️⃣ Initialize Chroma ---
persist_dir = "rag_index/bge"
client = chromadb.PersistentClient(path=persist_dir)
collection = client.get_or_create_collection(
    name="finsight_bge",
    metadata={"description": "FinSight embeddings using BGE-base"}
)

# --- 3️⃣ Prepare data for insertion ---
texts = df["content"].tolist() if "content" in df.columns else df["text"].tolist()
metadatas = df[["doc_id", "title", "chunk_id", "chunk_type"]].to_dict("records")
embeddings = df["embedding"].tolist()
ids = df["chunk_id"].tolist()

print("🧠 Example embedding shape:", len(embeddings[0]))



# --- 4️⃣ Insert into Chroma (silently) ---
print("⏳ Inserting embeddings into Chroma...")

# Temporarily suppress stdout to hide internal prints
with contextlib.redirect_stdout(io.StringIO()):
    collection.add(
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
        ids=ids
    )

print(f"✅ Inserted {len(texts)} embeddings into {persist_dir}")

