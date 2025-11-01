# ============================================================
# FinSight - Group Banking FAQ CSV into Contextual Text (.txt)
# ============================================================
import pandas as pd
from pathlib import Path

# --- CONFIG ---
input_file = "C:/Users/user/Documents/Banking RAG/data/Dataset_Banking_chatbot.csv"
output_file = "faq_grouped.txt"

# --- Load CSV ---
df = pd.read_csv(input_file, encoding="latin-1")



# normalize column names
df.columns = [c.strip().lower() for c in df.columns]
print("🧾 Columns found:", df.columns.tolist())

# --- Check expected columns ---
expected_cols = {"context", "question", "answer"}
if not expected_cols.issubset(set(df.columns)):
    raise ValueError(f"❌ CSV must contain columns: {expected_cols}")

# --- Group by context ---
grouped_text = []
grouped = df.groupby("context", dropna=True)

for context, rows in grouped:
    grouped_text.append("=" * 30)
    grouped_text.append(f"CONTEXT: {context.strip()}")
    grouped_text.append("=" * 30)
    grouped_text.append("")

    for i, row in enumerate(rows.itertuples(), start=1):
        q = str(row.question).strip()
        a = str(row.answer).strip()
        grouped_text.append(f"Q{i}: {q}")
        grouped_text.append(f"A{i}: {a}")
        grouped_text.append("")  # blank line

    grouped_text.append("-" * 30)
    grouped_text.append("")

# --- Save output ---
Path(output_file).write_text("\n".join(grouped_text), encoding="utf-8")
print(f"✅ Grouped FAQ text saved to: {output_file}")
