FinSight – Chunking Rules 

Purpose:
This document defines the standardized rules and parameters used to split FinSight Bank’s text documents into retrievable sections (“chunks”) for the RAG pipeline using local Hugging Face embeddings (BAAI/bge-base-en-v1.5).

1. Objective

The goal of chunking is to divide each policy, charter, or brochure into coherent text blocks that preserve context while enabling efficient retrieval and semantic embedding.
Each chunk must represent a complete and meaningful subsection and remain small enough for accurate vector encoding by the embedding model.

2. Core Parameters
Parameter	Value	Description
Chunk Size (max)	500 tokens	Upper limit per chunk (≈350–400 words).
Chunk Size (min)	120 tokens	Ensures short paragraphs are merged into coherent blocks.
Average Chunk Target	400 tokens	Ideal window for BGE-base embeddings.
Language	English (UTF-8)	All texts from Step 1 are standardized English.
Tokenizer	GPT2TokenizerFast	Used to estimate token count and control chunk boundaries.
Embedding Model	BAAI/bge-base-en-v1.5	Local Hugging Face model (768-dim vectors).
Embedding Framework	sentence-transformers	For encoding chunks locally without API calls.
Overlap	None (0%)	Non-overlapping chunks for simplicity.
File Type (input)	.txt	Clean text format from Step 1.
File Type (output)	.json and .csv	Structured outputs with metadata.
3. Structural and Boundary Rules

Sentence Integrity:
Never break a sentence mid-way. Use NLTK’s sent_tokenize() for boundary detection.

Section Awareness:
Prefer chunk boundaries to align with numbered headings (e.g., “1.”, “I.”, “Section”).

Merge Short Segments (<150 tokens):
Combine them with the next section to preserve meaning.

Paragraph Delimiters:
Double newlines (\n\n) represent paragraph breaks.

Token Thresholds:

If current chunk > 450 tokens → close and start a new one at the next full sentence.

If < 120 tokens → merge forward.

Chunk Granularity:

One chunk ≈ one subsection or topic.

Avoid overly large (> 600 token) sections.

No Overlap:
Chunks are non-redundant to simplify indexing.

4. Metadata Structure (Per Chunk)
Field	Description
doc_id	Unique ID from inventory (001–021).
chunk_id	Sequential index (doc_id_1, doc_id_2, ...).
title	Full document title.
type	Policy / Guide / Brochure / Charter etc.
section	Section heading or first sentence.
date	Version date (January 2025).
n_tokens	Token count of the chunk.
chunk_text	Text content of the chunk.

All entries are consolidated in chunks_metadata.csv for embedding.

5. Output Format
Per-Document File (/data/chunks/<file>_chunks.json)
[
  {
    "chunk_id": "015_1",
    "doc_id": "015",
    "title": "Accounts and Cards Brochure",
    "chunk_text": "FinSight Bank offers a complete range of accounts...",
    "n_tokens": 432
  },
  ...
]

Consolidated Index (/data/chunks_metadata.csv)

Contains all chunk texts + metadata for every FinSight document (internal + public).

6. Example Chunking Logic

Input:

1. Introduction
FinSight Bank ensures transparent, fair, and secure banking...
2. Governance
The Board of Directors supervises compliance with BCT rules...


Output:

Chunk 1 → Tokens: 384 (covers Introduction)

Chunk 2 → Tokens: 412 (covers Governance)

7. Embedding Configuration (for Step 3)
Setting	Value
Embedding Model	BAAI/bge-base-en-v1.5
Vector Size	768
Framework	sentence-transformers
Similarity Metric	Cosine Similarity
Storage Engine	FAISS or Chroma (local vector DB)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
embeddings = model.encode(chunk_texts, batch_size=32, show_progress_bar=True)

8. Versioning and Audit
Field	Value
Author	FinSight AI Documentation Team
Version	1.1 (October 2025)
Pipeline	RAG – Document Structuring and Chunking
Embedding Model	BAAI/bge-base-en-v1.5
Last Updated	2025-10-22