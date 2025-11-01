# FinSight
python -m venv venv 

venv\Scripts\activate


pip install -r requirements.txt




 1. Chunk the documents first


python scripts/chunk_documents.py

 3. Create embeddings

    
python scripts/embed_bge_base.py

 4. Build the Chroma database
    

python scripts/build_chroma_bge.py

 5. Finally, run the retriever to test

    
python scripts/rag_retriever.py


