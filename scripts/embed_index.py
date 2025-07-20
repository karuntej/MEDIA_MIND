#!/usr/bin/env python3
"""
Usage: python scripts/embed_index.py
Creates data/faiss.index and data/meta.db (with ents JSON).
"""
import json
import sqlite3
import numpy as np
import faiss
import pathlib
from sentence_transformers import SentenceTransformer

DATA_DIR = pathlib.Path("data")
chunks   = json.load(open(DATA_DIR/"processed"/"all_chunks.json"))
model    = SentenceTransformer("all-MiniLM-L12-v2")

# ── Build FAISS index ────────────────────────────────────────────────────────
texts = [c["text"][:512] for c in chunks]
embs  = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
ids   = np.arange(len(texts)).astype("int64")

index = faiss.IndexFlatIP(embs.shape[1])
index = faiss.IndexIDMap2(index)
index.add_with_ids(embs, ids)
faiss.write_index(index, str(DATA_DIR/"faiss.index"))
print("✔️  FAISS index written")

# ── Build / Recreate metadata DB (with ents) ────────────────────────────────
con = sqlite3.connect(DATA_DIR/"meta.db")
cur = con.cursor()

# Drop old table if it exists, to avoid schema mismatches
cur.execute("DROP TABLE IF EXISTS meta")

# Create fresh table with 7 columns
cur.execute("""
CREATE TABLE meta (
  id         INTEGER PRIMARY KEY,
  chunk_id   TEXT,
  source     TEXT,
  doc_path   TEXT,
  loc        TEXT,
  text       TEXT,
  ents       TEXT
)
""")

# Prepare and insert all rows
to_insert = []
for i, c in enumerate(chunks):
    to_insert.append((
        i,
        c["chunk_id"],
        c["source"],
        c["doc_path"],
        json.dumps(c["loc"]),
        c["text"],
        json.dumps(c.get("ents", []))
    ))

cur.executemany("INSERT INTO meta VALUES (?,?,?,?,?,?,?)", to_insert)
con.commit()
con.close()

print("✔️  SQLite meta.db written (with ents)")
