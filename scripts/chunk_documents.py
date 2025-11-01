import os, re, json, csv, argparse
from datetime import datetime
from transformers import AutoTokenizer

# ===============================
# CONFIGURATION
# ===============================
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
SECTION_RE = re.compile(r'^\s*(\d+(?:\.\d+)*)\s+(.*)$')

# ===============================
# HELPERS
# ===============================
def normalize_text(text):
    """Clean raw text from extra newlines and spaces."""
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def split_sections(text):
    """Split doc by true sections (1., 2., 3.1), ignore inner steps."""
    lines = text.split('\n')
    current_title, buffer = None, []
    for line in lines:
        m = SECTION_RE.match(line)
        if m and ('.' not in m.group(1) or len(m.group(1).split('.')) <= 2):
            if current_title is not None:
                yield current_title, "\n".join(buffer)
            sec_num, sec_name = m.group(1), m.group(2).strip()
            current_title = f"{sec_num}. {sec_name}"
            buffer = []
        else:
            buffer.append(line)
    if current_title is not None:
        yield current_title, "\n".join(buffer)
    else:
        yield None, text


def split_blocks_by_double_newline(text):
    """Split text into blocks separated by blank lines."""
    return [b.strip() for b in re.split(r'\n{2,}', text) if b.strip()]


def detect_table(block):
    """Detect tables that contain at least one '|'."""
    return '|' in block


def parse_table_block(block, section_title, parent_id):
    """Parse a '|' table into structured key:value rows."""
    lines = [l.strip() for l in block.split('\n') if l.strip()]

    # find header line (first containing '|')
    header_idx = next((i for i, l in enumerate(lines) if '|' in l), None)
    if header_idx is None or header_idx == len(lines) - 1:
        return None, []

    header_line = lines[header_idx]
    headers = [h.strip() for h in header_line.split('|') if h.strip()]
    rows = []

    # process all subsequent lines that contain '|'
    for i, row in enumerate(lines[header_idx + 1:], start=1):
        if '|' not in row:
            continue
        values = [v.strip() for v in row.split('|')]
        # Build "Header:Value" pairs
        kv_pairs = [f"{headers[j]}:{values[j]}" for j in range(min(len(headers), len(values)))]
        joined = ", ".join(kv_pairs)
        rows.append({
            "chunk_id": f"{parent_id}_{i}",
            "values": joined
        })

    return headers, rows


def parse_paragraph_block(block, section_title):
    """Return one paragraph chunk with section title."""
    if not block.strip():
        return []
    return [f"{section_title}\n{block.strip()}"]


def token_count(text):
    return len(tokenizer.encode(text))


# ===============================
# MAIN PIPELINE
# ===============================
def process_documents(input_dir, output_dir, meta_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(meta_dir or output_dir, "chunks_metadata.csv")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)

    meta_rows = []

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".txt"):
            continue

        doc_id = os.path.splitext(fname)[0]
        doc_title = re.sub(r'[_\-]', ' ', doc_id).strip().title()
        doc_date = datetime.now().strftime("%Y-%m-%d")
        source_folder = os.path.basename(os.path.normpath(input_dir))

        with open(os.path.join(input_dir, fname), "r", encoding="utf-8") as f:
            raw = f.read()
        text = normalize_text(raw)

        chunks, idx = [], 1

        for section_title, section_text in split_sections(text):
            for block in split_blocks_by_double_newline(section_text):
                if detect_table(block):
                    parent_id = f"{doc_id}_{idx}"
                    headers, rows = parse_table_block(block, section_title, parent_id)

                    if headers and rows:
                        parent_chunk = {
                            "doc_id": doc_id,
                            "chunk_id": parent_id,
                            "chunk_type": "table",
                            "section": section_title.split('.')[0] if section_title else None,
                            "title": doc_title,
                            "section_title": section_title,
                            "headers": headers,
                            "rows": rows,
                            "token_count": sum(len(tokenizer.encode(r["values"])) for r in rows),
                            "created_at": datetime.now().isoformat()
                        }
                        chunks.append(parent_chunk)
                        meta_rows.append([
                            doc_id, doc_title, section_title, 1, "table",
                            doc_date, parent_chunk["token_count"], source_folder
                        ])
                        idx += 1
                else:
                    for ch in parse_paragraph_block(block, section_title):
                        para_chunk = {
                            "doc_id": doc_id,
                            "chunk_id": f"{doc_id}_{idx}",
                            "chunk_type": "paragraph",
                            "section": section_title.split('.')[0] if section_title else None,
                            "title": doc_title,
                            "text": ch,
                            "token_count": token_count(ch),
                            "created_at": datetime.now().isoformat()
                        }
                        chunks.append(para_chunk)
                        meta_rows.append([
                            doc_id, doc_title, section_title, 1, "paragraph",
                            doc_date, token_count(ch), source_folder
                        ])
                        idx += 1

        # === Save JSON per document ===
        out_path = os.path.join(output_dir, f"{doc_id}_chunks.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"✅ {fname}: {len(chunks)} chunks created")

    # === Append or create metadata CSV ===
    header = ["doc_id", "title", "section", "page_number", "type", "date", "token_count", "source_folder"]
    file_exists = os.path.exists(meta_path)
    with open(meta_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerows(meta_rows)

    print(f"📄 Metadata updated at: {meta_path}")
    print("✅ All documents processed successfully.")


# ===============================
# CLI
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FinSight – Chunking single or multiple documents"
    )
    parser.add_argument("--input", required=True, help="Input folder with .txt files")
    parser.add_argument("--output", required=True, help="Output folder for chunks JSON")
    parser.add_argument("--meta", default=None, help="Optional folder for global chunks_metadata.csv")
    parser.add_argument("--file", default=None, help="Process only this file (inside input folder)")

    args = parser.parse_args()

    if args.file:
        # --- Process only one file ---
        input_file = os.path.join(args.input, args.file)
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"❌ File not found: {input_file}")

        print(f"🔹 Processing only: {args.file}")
        process_documents(os.path.dirname(input_file), args.output, meta_dir=args.meta)
    else:
        # --- Process all files in the folder ---
        process_documents(args.input, args.output, meta_dir=args.meta)
