#!/usr/bin/env python3
"""
FinSight LLM Client
===================

Extended to support Retrieval-Augmented Generation (RAG):
- Standard generate_response(): simple one-shot LLM query
- New generate_rag_response(): retrieves context from Chroma DB before answering

Author: FinSight AI Team
Date: November 2025
"""

import json
import requests
import time
import sys
from pathlib import Path

# ✅ NEW: import retriever from your RAG pipeline
# Add parent directory to path to import scripts
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.rag_retriever import retrieve_docs


class LLMClient:
    """Local LLM client for one-shot or retrieval-augmented generation via Ollama."""

    def __init__(self, config_path: str = "config/llm_config.json"):
        """Initialize the client, load configuration, and store model parameters."""
        self.config_path = Path(config_path)
        self._load_config()

    # ------------------------------------------------------------------
    def _load_config(self):
        """Load model parameters and API settings from JSON config."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path.resolve()}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Core parameters
        self.model = cfg.get("model", "phi3:mini")
        self.temperature = cfg.get("temperature", 0.7)
        self.top_p = cfg.get("top_p", 0.9)
        self.num_predict = cfg.get("num_predict", 256)
        self.system_prompt = cfg.get(
            "system_prompt",
            "You are FinSight AI, a polite and professional banking assistant."
        )
        self.max_retries = cfg.get("max_retries", 3)
        self.base_url = cfg.get("base_url", "http://localhost:11434/api/generate")

    # ------------------------------------------------------------------
    def _build_prompt(self, user_input: str) -> str:
        """Combine the system prompt and user input into a formatted message."""
        prompt = (
            f"{self.system_prompt}\n\n"
            f"User: {user_input}\n"
            f"Assistant:"
        )
        return prompt

    # ------------------------------------------------------------------
    def _query_ollama(self, prompt: str) -> str:
        """Send the prompt to the local Ollama API and return the generated text."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.num_predict,
            "stream": False  # Disable streaming for simplicity
        }

        try:
            response = requests.post(self.base_url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()

            # Ollama returns the generated text in the "response" key
            text = data.get("response", "").strip()
            return self._clean_response(text)

        except requests.exceptions.ConnectionError:
            return "⚠️  Could not connect to Ollama. Make sure it's running (ollama serve)."

        except requests.exceptions.Timeout:
            return "⏳  Request to Ollama timed out. Try again."

        except Exception as e:
            return f"❌  Unexpected error: {e}"

    # ------------------------------------------------------------------
    def _retry_query(self, prompt: str) -> str:
        """Retry sending the query up to max_retries times if it fails."""
        for attempt in range(1, self.max_retries + 1):
            result = self._query_ollama(prompt)
            if not result.startswith(("⚠️", "⏳", "❌")):
                return result
            print(f"[Retry {attempt}/{self.max_retries}] Retrying after failure...")
            time.sleep(1)
        return "❌  Model could not generate a response after several retries."

    # ------------------------------------------------------------------
    def _clean_response(self, text: str) -> str:
        """Remove unnecessary prefixes, spacing, or artifacts from model output."""
        text = text.replace("Assistant:", "").replace("assistant:", "").strip()
        return text

    # ------------------------------------------------------------------
    def generate_response(self, user_query: str) -> str:
        """Basic method: build prompt, query model, and return final answer."""
        prompt = self._build_prompt(user_query)
        reply = self._retry_query(prompt)
        return reply

    # ------------------------------------------------------------------
    # ✅ NEW: Retrieval-Augmented method
    # ------------------------------------------------------------------
    def generate_rag_response(self, user_query: str, top_k: int = 3) -> str:
        """
        Retrieve context from the FinSight knowledge base (Chroma DB)
        and generate a grounded answer via the local LLM.

        Args:
            user_query (str): user's question
            top_k (int): number of retrieved chunks (default=3)
        """
        try:
            results = retrieve_docs(user_query, top_k=top_k)
        except Exception as e:
            return f"❌ Retrieval error: {e}"

        if not results:
            return "No relevant information found in FinSight Bank’s knowledge base."

        # Combine retrieved chunks into context text
        context = "\n\n".join([
            f"[Source: {r['title']}]\n{r['text']}" for r in results
        ])

        # Build final prompt with context
        prompt = (
            f"{self.system_prompt}\n\n"
            f"Context:\n{context}\n\n"
            f"User: {user_query}\n"
            f"Assistant:"
        )

        return self._retry_query(prompt)


# ----------------------------------------------------------------------
# Optional quick test (standalone mode)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    client = LLMClient()
    print("FinSight AI (RAG-enabled) is ready. Type 'exit' to quit.\n")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye 👋")
            break
        print("\n🔍 Retrieving relevant context...\n")
        response = client.generate_rag_response(user_input)
        print(f"FinSight: {response}\n")
