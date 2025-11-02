#!/usr/bin/env python3
"""
FinSight LLM Client (RAG-Only)
==============================

Retrieval-Augmented Generation client that:
1️⃣ Retrieves top-k relevant chunks from ChromaDB
2️⃣ Builds a contextual prompt
3️⃣ Queries the local Ollama LLM (e.g., Phi-3 Mini)
4️⃣ Returns a grounded, concise banking answer

Author: FinSight AI Team
Date: November 2025
"""

import json
import requests
import time
from pathlib import Path
from typing import List, Dict
import sys

# Add project root for retriever import
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.rag_retriever import retrieve_docs


# ==========================================================
# LLM CLIENT (RAG-ONLY)
# ==========================================================
class LLMClient:
    """FinSight RAG client using Ollama for grounded answers."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            # Use path relative to this file's location
            config_path = Path(__file__).parent / "config" / "llm_config.json"
        self.config_path = Path(config_path)
        self._load_config()

    # ------------------------------------------------------
    def _load_config(self):
        """Load model configuration from JSON file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"❌ Config file not found: {self.config_path.resolve()}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.model = cfg.get("model", "phi3:mini")
        self.temperature = cfg.get("temperature", 0.7)
        self.top_p = cfg.get("top_p", 0.9)
        self.num_predict = cfg.get("num_predict", 256)
        self.system_prompt = cfg.get(
            "system_prompt",
            "You are FinSight AI, the official virtual assistant of FinSight Bank. "
            "Answer professionally and precisely using retrieved documents."
        )
        self.max_retries = cfg.get("max_retries", 3)
        self.base_url = cfg.get("base_url", "http://localhost:11434/api/generate")

    # ------------------------------------------------------
    def _build_prompt(self, user_query: str, context_chunks: List[Dict]) -> str:
        """Construct the full RAG prompt from retrieved chunks."""
        context_text = "\n---\n".join(
            f"[{r['title']} | Score: {r['score']}]\n{r['text']}" for r in context_chunks
        )

        return (
            f"{self.system_prompt}\n\n"
            f"Retrieved Knowledge:\n{context_text}\n\n"
            f"User Question: {user_query}\n"
            f"Assistant:"
        )

    # ------------------------------------------------------
    def _query_ollama(self, prompt: str) -> str:
        """Send the constructed prompt to Ollama API."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.num_predict,
            "stream": False
        }

        try:
            response = requests.post(self.base_url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()

        except requests.exceptions.ConnectionError:
            return "⚠️ Ollama not reachable. Please ensure it is running."
        except requests.exceptions.Timeout:
            return "⏳ Request to Ollama timed out."
        except Exception as e:
            return f"❌ Unexpected error: {e}"

    # ------------------------------------------------------
    def _retry_query(self, prompt: str) -> str:
        """Retry mechanism for failed requests."""
        for attempt in range(1, self.max_retries + 1):
            result = self._query_ollama(prompt)
            if not result.startswith(("⚠️", "⏳", "❌")):
                return result
            print(f"[Retry {attempt}/{self.max_retries}] Retrying...")
            time.sleep(1)
        return "❌ Model request failed after several retries."

    # ------------------------------------------------------
    def generate_rag_response(
        self,
        user_query: str,
        top_k: int = 5,
        score_threshold: float = 0.35,
        hybrid_weight: float = 0.25
    ) -> str:
        """
        Main RAG generation:
        - Retrieve top-k relevant chunks from Chroma
        - Build context-aware prompt
        - Query Ollama model for grounded response
        """
        print("🔍 Retrieving context from FinSight knowledge base...")
        try:
            results = retrieve_docs(
                query=user_query,
                top_k=top_k,
                score_threshold=score_threshold,
                hybrid_weight=hybrid_weight
            )
        except Exception as e:
            return f"❌ Retrieval error: {e}"

        if not results:
            return "No relevant information found in FinSight Bank’s documentation."

        print(f"🧠 Retrieved {len(results)} relevant chunks.")
        prompt = self._build_prompt(user_query, results)
        reply = self._retry_query(prompt)
        return reply


# ==========================================================
# STANDALONE TEST MODE
# ==========================================================
if __name__ == "__main__":
    client = LLMClient()
    print("🤖 FinSight RAG-only assistant ready. Type 'exit' to quit.\n")

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye 👋")
            break

        answer = client.generate_rag_response(user_input, top_k=3)
        print(f"\nFinSight: {answer}\n")
