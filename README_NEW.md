# 🏦 FinSight - RAG-Powered Banking Document Intelligence

<div align="center">

**Production-Ready Retrieval-Augmented Generation System for Financial Institutions**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Chroma](https://img.shields.io/badge/Vector%20DB-Chroma-green.svg)](https://www.trychroma.com/)
[![BGE](https://img.shields.io/badge/Embeddings-BGE--base-orange.svg)](https://huggingface.co/BAAI/bge-base-en-v1.5)
[![Ollama](https://img.shields.io/badge/LLM-Phi--3--Mini-purple.svg)](https://ollama.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Quick Start](#-quick-start) • [Documentation](#-documentation) • [Features](#-key-features) • [Architecture](#-architecture) • [Usage](#-usage)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Pipeline Execution](#-pipeline-execution)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Technical Details](#-technical-details)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**FinSight** is a complete Retrieval-Augmented Generation (RAG) system that transforms banking documents into an intelligent Q&A system. It processes 390+ document chunks from policies, procedures, guides, and FAQs to provide accurate, context-aware responses to banking queries.

### What It Does

- 📚 **Processes 22 Banking Documents** (13 internal policies + 9 customer guides)
- 🔍 **Semantic Search** across 390+ intelligently chunked sections
- 🤖 **AI-Powered Answers** using local LLM (Phi-3 Mini via Ollama)
- 💾 **Persistent Storage** with Chroma vector database
- 🚀 **Fully Local** - No cloud dependencies, complete privacy

### Document Coverage

**Internal Policies (13):**
- Credit policies, risk management, compliance frameworks
- Governance charters, AML/CFT procedures
- Operational risk, information security, prudential reporting

**Customer Documents (9):**
- Account brochures, digital banking guides
- FAQs, product catalogs, KYC notices
- Payment guides, complaint procedures

---

## ✨ Key Features

### 🔍 Intelligent Document Processing
- **Smart Chunking**: Respects section boundaries, sentence integrity (max 350 tokens)
- **Table Intelligence**: Parses banking tables into searchable key-value pairs
- **Context Preservation**: Maintains document structure and relationships

### 🧠 Advanced Retrieval
- **Semantic Search**: BGE-base-en-v1.5 embeddings (768 dimensions)
- **Hybrid Scoring**: Combines semantic similarity + keyword matching
- **Score Normalization**: Consistent 0-1 relevance scores
- **Fast Queries**: Sub-second retrieval (20-80ms)

### 🤖 LLM Integration
- **Local Inference**: Phi-3 Mini via Ollama (no API costs)
- **Grounded Responses**: Citations from retrieved documents
- **Configurable**: Temperature, length, system prompts
- **Multiple Interfaces**: CLI, styled chat, API-ready

### 🛠️ Production-Ready
- **NumPy 2.0 Compatible**: Modern dependency support
- **GPU Acceleration**: Optional CUDA support
- **Error Handling**: Robust retry logic and fallbacks
- **Logging**: Progress bars and status updates
- **Extensible**: Easy to add documents or change models

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Raw Documents  │  22 .txt files (policies, guides, FAQs)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  STEP 1: Chunking (chunk_documents.py)      │
│  • Section detection & paragraph splitting  │
│  • Table parsing → key:value format         │
│  • Token counting (max 350 tokens/chunk)    │
└────────┬────────────────────────────────────┘
         │  Output: 390 chunks (JSON + CSV)
         ▼
┌─────────────────────────────────────────────┐
│  STEP 2: Embedding (embed_bge_base.py)      │
│  • Model: BAAI/bge-base-en-v1.5             │
│  • 768-dimensional vectors                  │
│  • L2 normalization for cosine similarity   │
└────────┬────────────────────────────────────┘
         │  Output: embeddings.parquet (768-dim)
         ▼
┌─────────────────────────────────────────────┐
│  STEP 3: Indexing (build_chroma_bge.py)     │
│  • Persistent Chroma DB storage             │
│  • Collection: finsight_bge                 │
│  • HNSW indexing for fast search            │
└────────┬────────────────────────────────────┘
         │  Output: rag_index/bge/ (vector DB)
         ▼
┌─────────────────────────────────────────────┐
│  STEP 4: Retrieval (rag_retriever.py)       │
│  • Query encoding → vector search           │
│  • Hybrid scoring (semantic + keyword)      │
│  • Top-K ranking with score threshold       │
└────────┬────────────────────────────────────┘
         │  Output: Ranked relevant chunks
         ▼
┌─────────────────────────────────────────────┐
│  STEP 5: Generation (llm_client.py)         │
│  • Build context from retrieved chunks      │
│  • Query Ollama (Phi-3 Mini)                │
│  • Generate grounded response               │
└─────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **Ollama** installed and running ([Download](https://ollama.ai/download))
- **8GB RAM** minimum (16GB recommended)

### Installation (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/alouanefarah/FinSight.git
cd FinSight

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download and start Ollama LLM
ollama pull phi3:mini
ollama serve
```

### Run the Pipeline

```bash
# Step 1: Chunk documents (2 seconds)
python scripts/chunk_documents.py --input data/docs_public --output data/chunks
python scripts/chunk_documents.py --input data/data_clean --output data/chunks

# Step 2: Generate embeddings (2-5 minutes on CPU, ~30s on GPU)
python scripts/embed_bge_base.py

# Step 3: Build vector database (5 seconds)
python scripts/build_chroma_bge.py

# Step 4: Test retrieval (loads in ~10 seconds)
python scripts/rag_retriever.py

# Step 5: Start interactive chat
python LLM/chat_app.py
```

### First Query

```bash
You: What documents do I need to open a savings account?

FinSight AI: Based on FinSight Bank's documentation, to open a 
savings account you need:
1. Valid national ID or passport
2. Proof of address (recent utility bill or lease agreement)
3. Initial deposit (minimum amount varies by account type)
...
```

---

## 📦 Installation

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.10+ |
| RAM | 8GB | 16GB |
| Storage | 5GB | 10GB |
| GPU | Optional | CUDA 11.8+ |

### Step-by-Step Setup

#### 1. Clone and Navigate

```bash
git clone https://github.com/alouanefarah/FinSight.git
cd FinSight
```

#### 2. Virtual Environment

**Windows PowerShell:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

**Standard Installation (CPU):**
```bash
pip install -r requirements.txt
```

**GPU-Accelerated Installation:**
```bash
# For CUDA 11.8+
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1+
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### 4. Install Ollama

**Windows:**
1. Download from https://ollama.ai/download
2. Run installer
3. Open PowerShell and run:
   ```powershell
   ollama pull phi3:mini
   ollama serve
   ```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull phi3:mini
ollama serve &
```

**Verify Ollama:**
```bash
curl http://localhost:11434/api/tags
```

#### 5. Verify Installation

```bash
python -c "import chromadb, sentence_transformers, torch; print('✅ All dependencies OK')"
```

---

## 🔄 Pipeline Execution

### Complete Pipeline (Detailed)

#### **Step 1: Document Chunking**

```bash
# Process customer-facing documents
python scripts/chunk_documents.py --input data/docs_public --output data/chunks

# Process internal policy documents  
python scripts/chunk_documents.py --input data/data_clean --output data/chunks
```

**What it does:**
- Splits documents by sections (numbered headings)
- Detects and parses tables (pipe-delimited)
- Creates chunks of ~350 tokens max
- Generates metadata CSV

**Output:**
- `data/chunks/*.json` (one per document)
- `data/chunks/chunks_metadata.csv` (consolidated)
- Expected: ~390 total chunks

**Options:**
```bash
# Process single file
python scripts/chunk_documents.py --input data/docs_public --output data/chunks --file FAQ.txt

# Custom metadata location
python scripts/chunk_documents.py --input data/docs_public --output data/chunks --meta data/
```

---

#### **Step 2: Embedding Generation**

```bash
python scripts/embed_bge_base.py
```

**What it does:**
- Loads all chunks from `data/chunks/`
- Encodes using BGE-base-en-v1.5 (768-dim)
- Normalizes embeddings (L2 norm)
- Saves to parquet format

**Performance:**
- CPU: 2-5 minutes for 390 chunks
- GPU: 30-60 seconds

**Output:**
- `docs_embeddings_bge_base.parquet` (main file)

**Configuration:**
Edit `scripts/embed_bge_base.py`:
```python
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"  # Model name
BATCH_SIZE = 16  # Increase for GPU (32/64)
```

---

#### **Step 3: Vector Database Build**

```bash
python scripts/build_chroma_bge.py
```

**What it does:**
- Loads embeddings from parquet
- Creates Chroma persistent client
- Inserts vectors with metadata
- Builds HNSW index

**Output:**
- `rag_index/bge/` (persistent storage)
- `rag_index/bge/chroma.sqlite3` (metadata DB)

**Time:** ~5 seconds

**Options:**
```bash
# Custom embedding file
python scripts/build_chroma_bge.py --emb my_embeddings.parquet

# Custom storage location
python scripts/build_chroma_bge.py --persist my_vector_db/

# Custom collection name
python scripts/build_chroma_bge.py --collection my_collection

# Custom batch size
python scripts/build_chroma_bge.py --batch 500
```

---

#### **Step 4: Test Retrieval**

```bash
python scripts/rag_retriever.py
```

**What it does:**
- Loads BGE model and Chroma collection
- Runs 3 sample queries
- Displays results with scores

**Sample Output:**
```
[Query] What are the conditions to open a savings account?
======================================================================
[1] Accounts And Cards Brochure En (Accounts_and_Cards_Brochure_EN) – Score: 0.876
Chunk ID: Accounts_and_Cards_Brochure_EN_3  Type: table
Snippet: Account Type:Savings Account, Target Clients:All customers, Main Benefits:Interest earning, Monthly Fee:Free...
```

**Configuration:**
Edit query parameters in `rag_retriever.py`:
```python
results = retrieve_docs(
    query="Your question",
    top_k=5,                    # Number of results
    score_threshold=0.3,        # Minimum score (0-1)
    hybrid_weight=0.2          # Keyword boost weight
)
```

---

#### **Step 5: Interactive Chat**

**Option A: Basic Client**
```bash
python LLM/llm_client.py
```

**Option B: Styled Interface (Recommended)**
```bash
python LLM/chat_app.py
```

**Features:**
- Retrieves relevant chunks automatically
- Displays "thinking" status
- Cites source documents
- Colorized output (chat_app.py)

**Commands:**
- Type any question
- Type `exit` or `quit` to end session
- Press `Ctrl+C` to interrupt

---

## 💻 Usage

### Sample Queries by Category

#### Account Services
```
What documents are required to open a current account?
What is the minimum balance for a youth account?
How do I upgrade from a savings to a current account?
What are the monthly fees for different account types?
```

#### Digital Banking
```
How do I reset my mobile banking password?
What are the steps to activate online banking?
How can I block my debit card through the app?
What transactions can I do via mobile banking?
```

#### Loans & Credit
```
How can students apply for personal loans?
What is the credit approval process for SMEs?
What documents are needed for a mortgage application?
What is the maximum loan amount for personal credit?
```

#### Policies & Compliance
```
What is the bank's KYC procedure?
Explain the AML/CFT policy for high-risk clients.
What are the compliance requirements for foreign transactions?
How does the bank handle customer complaints?
```

#### Fees & Charges
```
What are the fees for international wire transfers?
How much does a checkbook cost?
What are the ATM withdrawal charges?
Is there a fee for account closure?
```

### Programmatic Usage

```python
from LLM.llm_client import LLMClient

# Initialize client
client = LLMClient()

# Generate response
response = client.generate_rag_response(
    user_query="What documents do I need for a savings account?",
    top_k=5,                    # Retrieve top 5 chunks
    score_threshold=0.35,       # Min relevance score
    hybrid_weight=0.25          # Keyword boost factor
)

print(response)
```

### Using Retriever Directly

```python
from scripts.rag_retriever import retrieve_docs

# Retrieve relevant chunks
results = retrieve_docs(
    query="credit policy",
    top_k=3
)

# Access results
for i, result in enumerate(results, 1):
    print(f"[{i}] {result['title']}")
    print(f"Score: {result['score']}")
    print(f"Text: {result['text'][:200]}...")
```

---

## ⚙️ Configuration

### LLM Settings

Edit `LLM/config/llm_config.json`:

```json
{
  "model": "phi3:mini",
  "temperature": 0.7,
  "top_p": 0.9,
  "num_predict": 256,
  "system_prompt": "You are FinSight AI...",
  "max_retries": 3,
  "base_url": "http://localhost:11434/api/generate"
}
```

**Parameters:**
- `model`: Ollama model (`phi3:mini`, `llama3`, `mistral`)
- `temperature`: 0.0 (focused) to 1.0 (creative)
- `num_predict`: Max response tokens
- `system_prompt`: AI personality/instructions

**Alternative Models:**
```bash
# More capable but slower
ollama pull llama3:8b
ollama pull mistral:7b

# Update config.json with new model name
```

### Retrieval Settings

In `scripts/rag_retriever.py`:

```python
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_CHROMA_PATH = "rag_index/bge"
DEFAULT_COLLECTION = "finsight_bge"
```

### Chunking Settings

In `scripts/chunk_documents.py`:

```python
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 350  # Maximum tokens per chunk
```

### Embedding Settings

In `scripts/embed_bge_base.py`:

```python
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 16  # Increase for GPU
```

---

## 📁 Project Structure

```
FinSight/
│
├── scripts/                              # Pipeline scripts
│   ├── chunk_documents.py                # [STEP 1] Document chunking
│   ├── embed_bge_base.py                 # [STEP 2] Embedding generation
│   ├── build_chroma_bge.py               # [STEP 3] Vector DB creation
│   └── rag_retriever.py                  # [STEP 4] Retrieval engine
│
├── LLM/                                  # LLM interface
│   ├── llm_client.py                     # RAG client (basic)
│   ├── chat_app.py                       # Interactive chat (styled)
│   └── config/
│       └── llm_config.json               # LLM configuration
│
├── data/                                 # Document repository
│   ├── docs_public/                      # Customer documents (9 files)
│   │   ├── Accounts_and_Cards_Brochure_EN.txt
│   │   ├── FAQ.txt
│   │   ├── Digital_Banking_Guide_EN.txt
│   │   └── ...
│   ├── data_clean/                       # Internal policies (13 files)
│   │   ├── General_Credit_Policy_2025_EN.txt
│   │   ├── KYC_and_AML_CFT_Procedure_EN.txt
│   │   └── ...
│   ├── chunks/                           # Generated chunks
│   │   ├── <document>_chunks.json        # Per-document chunks
│   │   └── chunks_metadata.csv           # Consolidated metadata
│   └── chunking_rules.md                 # Chunking specification
│
├── rag_index/                            # Vector database
│   └── bge/
│       ├── chroma.sqlite3                # Metadata storage
│       └── <uuid>/                       # Vector storage
│
├── docs_embeddings_bge_base.parquet      # Generated embeddings
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
├── SETUP_GUIDE.md                        # Detailed setup instructions
├── .gitignore                            # Git exclusions
└── .git/                                 # Git repository
```

---

## 🔧 Technical Details

### Chunking Algorithm

**Strategy:**
1. Split by numbered sections (1., 2., 2.1, etc.)
2. Detect tables (pipe-delimited `|`)
3. Parse table rows into key:value format
4. Split paragraphs by double newlines
5. Respect sentence boundaries
6. Enforce token limits (max 350)

**Example:**
```
Input:
| Account Type | Fee |
| Savings      | 0   |

Output Chunk:
"Account Type:Savings, Fee:0"
```

### Embedding Model

**BAAI/bge-base-en-v1.5**
- Architecture: BERT encoder
- Parameters: 109M
- Dimensions: 768
- Max sequence: 512 tokens
- Training: Contrastive learning on 1.4B pairs
- Performance: MTEB score 63.6

### Vector Database

**Chroma Architecture:**
- Backend: SQLite + DuckDB
- Index: HNSW (Hierarchical Navigable Small World)
- Similarity: Cosine distance
- Latency: 20-80ms per query (local)

**Storage:**
```
rag_index/bge/
├── chroma.sqlite3           # Metadata + documents
└── <uuid>/                  # Embeddings
    ├── data_level0.bin
    └── index_metadata.pkl
```

### LLM Integration

**Phi-3 Mini Specs:**
- Parameters: 3.8B
- Context: 128K tokens
- Quantization: Q4_K_M
- Latency: 1-2s per response

**API Flow:**
1. Retrieve chunks from Chroma
2. Build context prompt
3. POST to Ollama API
4. Stream/return response

---

## 🐛 Troubleshooting

### Installation Issues

**1. ChromaDB Installation Fails**
```bash
# Try with specific version
pip install chromadb==0.4.24
```

**2. PyTorch CUDA Issues**
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Runtime Issues

**3. "Collection finsight_bge does not exist"**
```bash
# Rebuild database
python scripts/build_chroma_bge.py
```

**4. "Ollama not reachable"**
```bash
# Start Ollama service
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

**5. NumPy 2.0 Compatibility**
Already fixed in code. If you see `ptp` errors, update:
```python
# In rag_retriever.py, use:
np.ptp(array)  # Instead of array.ptp()
```

**6. Slow Embedding Generation**
```bash
# Check if GPU is being used
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**7. Out of Memory**
```python
# Reduce batch size in embed_bge_base.py
BATCH_SIZE = 8  # Instead of 16
```

### Cache Issues

**8. Code Changes Not Reflected**
```powershell
# Clear Python cache
Remove-Item -Recurse -Force scripts\__pycache__
Remove-Item -Recurse -Force LLM\__pycache__
Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force
```

---

## 📚 Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Complete setup instructions
- **[data/chunking_rules.md](data/chunking_rules.md)** - Chunking specification
- **[LLM/README.md](LLM/README.md)** - LLM client documentation

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Development Setup:**
```bash
pip install -r requirements.txt
pip install pytest black flake8  # Dev dependencies
```

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **BAAI** - BGE embedding models
- **Chroma** - Vector database
- **Ollama** - Local LLM runtime
- **Hugging Face** - Transformers ecosystem
- **Sentence-Transformers** - Embedding framework

---

## 📞 Contact & Support

- **GitHub Issues:** [Report a bug](https://github.com/alouanefarah/FinSight/issues)
- **Discussions:** [Ask questions](https://github.com/alouanefarah/FinSight/discussions)
- **Email:** support@finsight-ai.com

---

<div align="center">

**Built with ❤️ by the FinSight AI Team**

⭐ [Star this repo](https://github.com/alouanefarah/FinSight) • 🍴 [Fork](https://github.com/alouanefarah/FinSight/fork) • 📖 [Documentation](SETUP_GUIDE.md)

**Version 1.0** | November 2025

</div>
