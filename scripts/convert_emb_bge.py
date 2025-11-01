import pandas as pd
import ast

df = pd.read_parquet("docs_embeddings_bge_base.parquet")

# Convert stringified lists to actual lists of floats
def fix_embedding(e):
    if isinstance(e, str):
        try:
            return ast.literal_eval(e)
        except Exception:
            return []
    return e

df["embedding"] = df["embedding"].apply(fix_embedding)

print(type(df['embedding'][0]), len(df['embedding'][0]))  # should now show <class 'list'> 768

# Save the fixed version
df.to_parquet("docs_embeddings_bge_base_fixed.parquet", index=False)
print("✅ Fixed embeddings saved to docs_embeddings_bge_base_fixed.parquet")


df = pd.read_parquet("docs_embeddings_bge_base_fixed.parquet")
print(type(df['embedding'][0]), len(df['embedding'][0]))
