#!/usr/bin/env python3
"""
Usage: python scripts/enrich_chunks.py

Adds spaCy NER entities to each chunk in data/processed/all_chunks.json.
Overwrites that file with an "ents" field on every chunk.
"""
import json
from pathlib import Path
import spacy
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).resolve().parent.parent
IN_FILE    = BASE / "data" / "processed" / "all_chunks.json"
OUT_FILE   = IN_FILE  # overwrite in-place; or set to "all_chunks_enriched.json" if you prefer

# ── Load spaCy once ───────────────────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm")  # make sure you've run: python -m spacy download en_core_web_sm

# ── Read existing chunks ──────────────────────────────────────────────────────
chunks = json.loads(IN_FILE.read_text(encoding="utf-8"))

# ── Add ents to each chunk ────────────────────────────────────────────────────
for c in tqdm(chunks, desc="spaCy NER"):
    text = c.get("text", "")[:5000]
    doc  = nlp(text)
    c["ents"] = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

# ── Write back ────────────────────────────────────────────────────────────────
OUT_FILE.write_text(
    json.dumps(chunks, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print(f"✔ ner-enriched chunks → {OUT_FILE}")
