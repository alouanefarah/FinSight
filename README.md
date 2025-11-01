# FinSight
python -m venv venv \\
venv\Scripts\activate
pip install sentence-transformers chromadb

 1. Chunk the documents first
python scripts/chunk_documents.py

 2. Create embeddings 
python scripts/embed_bge_base.py

 3. Build the Chroma database
python scripts/build_chroma_bge.py

 4. Finally, run the retriever to test
python scripts/rag_retriever.py
