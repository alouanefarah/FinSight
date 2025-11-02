#!/usr/bin/env python3
"""
FinSight – Document Chunking Pipeline
=====================================

Converts raw .txt documents into structured JSON chunks
for downstream embedding and retrieval.

Features:
- Section & paragraph segmentation
- Markdown/pipe table detection
- Dynamic token-aware chunk sizing
- Robust metadata (doc_id, title, section, type, token_count, timestamps)
- CLI for folder- or file-level processing

Author: FinSight AI Team
Date: November 2025
"""

import os
import re
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer

# ======================================
# CONFIGURATION
# ======================================
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 350  # target maximum tokens per paragraph chunk
tokenizer = AutoTokenizer.from_pretrained(MODEL)
SECTION_RE = re.compile(r'^\s*(\d+(?:\.\d+)*)\s+(.*)$')


# ======================================
# HELPERS
# ======================================
def normalize_text(text: str) -> str:
    """Clean raw text from extra newlines, tabs, and spaces."""
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\t+', ' ', text)
    return text.strip()


def split_sections(text: str):
    """Yield (section_title, content) pairs based on numbered headings."""
    lines = text.split('\n')
    current_title, buffer = None, []
    for line in lines:
        m = SECTION_RE.match(line)
        if m and ('.' not in m.group(1) or len(m.group(1).split('.')) <= 2):
            if current_title is not None:
                yield current_title, "\n".join(buffer).strip()
            sec_num, sec_name = m.group(1), m.group(2).strip()
            current_title = f"{sec_num}. {sec_name}"
            buffer = []
        else:
            buffer.append(line)
    if current_title is not None:
        yield current_title, "\n".join(buffer).strip()
    else:
        yield None, text


def split_blocks_by_double_newline(text: str) -> List[str]:
    """Split text into logical blocks separated by blank lines."""
    return [b.strip() for b in re.split(r'\n{2,}', text) if b.strip()]


def detect_table(block: str) -> bool:
    """Detects markdown or pipe-based tables."""
    return '|' in block and re.search(r'\w+\s*\|\s*\w+', block)


def parse_table_block(block: str, section_title: str, parent_id: str):
    """Parse a '|' table into structured key:value rows."""
    lines = [l.strip() for l in block.split('\n') if l.strip()]
    header_idx = next((i for i, l in enumerate(lines) if '|' in l), None)
    if header_idx is None or header_idx == len(lines) - 1:
        return None, []

    headers = [h.strip() for h in lines[header_idx].split('|') if h.strip()]
    rows = []
    for i, row in enumerate(lines[header_idx + 1:], start=1):
        if '|' not in row:
            continue
        values = [v.strip() for v in row.split('|')]
        kv_pairs = [f"{headers[j]}:{values[j]}" for j in range(min(len(headers), len(values)))]
        joined = ", ".join(kv_pairs)
        rows.append({
            "chunk_id": f"{parent_id}_{i}",
            "values": joined
        })
    return headers, rows


def token_count(text: str) -> int:
    """Return approximate token count for given text."""
    return len(tokenizer.encode(text))


def chunk_long_paragraph(text: str, section_title: str) -> List[str]:
    """Split very long paragraphs into smaller chunks respecting token limits."""
    words = text.split()
    chunks, current = [], []
    for w in words:
        current.append(w)
        if token_count(" ".join(current)) > MAX_TOKENS:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return [f"{section_title}\n{c.strip()}" for c in chunks if c.strip()]


# ======================================
# MAIN PIPELINE
# ======================================
def process_documents(input_dir: str, output_dir: str, meta_dir: str = None, single_file: str = None):
    """Process one or multiple documents into structured JSON chunks."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    meta_path = Path(meta_dir or output_dir) / "chunks_metadata.csv"
    output_path.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    meta_rows = []

    txt_files = [single_file] if single_file else [
        f for f in os.listdir(input_path) if f.lower().endswith(".txt")
    ]

    for fname in txt_files:
        file_path = input_path / fname
        if not file_path.exists():
            print(f"⚠️ Skipping missing file: {fname}")
            continue

        doc_id = file_path.stem
        doc_title = re.sub(r'[_\-]', ' ', doc_id).strip().title()
        doc_date = datetime.now().strftime("%Y-%m-%d")
        source_folder = input_path.name

        raw_text = file_path.read_text(encoding="utf-8")
        text = normalize_text(raw_text)

        chunks, idx = [], 1
        for section_title, section_text in split_sections(text):
            for block in split_blocks_by_double_newline(section_text):
                if detect_table(block):
                    parent_id = f"{doc_id}_{idx}"
                    headers, rows = parse_table_block(block, section_title, parent_id)
                    if headers and rows:
                        token_sum = sum(token_count(r["values"]) for r in rows)
                        chunks.append({
                            "doc_id": doc_id,
                            "chunk_id": parent_id,
                            "chunk_type": "table",
                            "section": section_title.split('.')[0] if section_title else None,
                            "title": doc_title,
                            "section_title": section_title,
                            "headers": headers,
                            "rows": rows,
                            "token_count": token_sum,
                            "created_at": datetime.now().isoformat()
                        })
                        meta_rows.append([
                            doc_id, doc_title, section_title, 1, "table", doc_date, token_sum, source_folder
                        ])
                        idx += 1
                else:
                    for ch in chunk_long_paragraph(block, section_title):
                        tcount = token_count(ch)
                        chunks.append({
                            "doc_id": doc_id,
                            "chunk_id": f"{doc_id}_{idx}",
                            "chunk_type": "paragraph",
                            "section": section_title.split('.')[0] if section_title else None,
                            "title": doc_title,
                            "text": ch,
                            "token_count": tcount,
                            "created_at": datetime.now().isoformat()
                        })
                        meta_rows.append([
                            doc_id, doc_title, section_title, 1, "paragraph", doc_date, tcount, source_folder
                        ])
                        idx += 1

        # === Save per-document JSON ===
        out_file = output_path / f"{doc_id}_chunks.json"
        out_file.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ {fname}: {len(chunks)} chunks created")

    # === Append or create metadata CSV ===
    header = ["doc_id", "title", "section", "page_number", "type", "date", "token_count", "source_folder"]
    file_exists = meta_path.exists()
    with meta_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerows(meta_rows)

    print(f"📄 Metadata updated at: {meta_path}")
    print("✅ All documents processed successfully.")


# ======================================
# CLI
# ======================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinSight – Chunk banking policy documents.")
    parser.add_argument("--input", required=True, help="Input folder with .txt files")
    parser.add_argument("--output", required=True, help="Output folder for JSON chunks")
    parser.add_argument("--meta", default=None, help="Optional folder for chunks_metadata.csv")
    parser.add_argument("--file", default=None, help="Process only one file (inside input folder)")
    args = parser.parse_args()

    process_documents(args.input, args.output, args.meta, single_file=args.file)
