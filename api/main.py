from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss, sqlite3, json, numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from api.ollama_client import ollama_generate
import uvicorn
from typing import List, Optional

DATA = Path(__file__).resolve().parents[1] / "data"
INDEX_PATH = DATA / "faiss.index"
DB_PATH    = DATA / "meta.db"

# load retrieval index + DB + embedder
index = faiss.read_index(str(INDEX_PATH))
con   = sqlite3.connect(str(DB_PATH), check_same_thread=False)
model = SentenceTransformer("all-MiniLM-L12-v2")

app = FastAPI(title="MediaMind PDF API", version="0.2")

class Query(BaseModel):
    question: str
    top_k:    int = 5
    entities: Optional[List[str]] = None

@app.post("/chat")
def chat(q: Query):
    # 1️⃣ retrieve evidence chunks
    passages = dense_search(q.question, q.top_k)

    # 2️⃣ optional entity‐type filtering
    if q.entities:
        filtered = []
        for p in passages:
            labels = {e["label"] for e in p.get("ents", [])}
            if labels.intersection(q.entities):
                filtered.append(p)
        passages = filtered

    # 3️⃣ short‐circuit: no chunks → no answer
    if not passages:
        return {
            "question": q.question,
            "answer":   "❓ No matching content found in your PDF corpus.",
            "passages": []
        }

    # 4️⃣ synthesize from the retrieved passages only
    answer = synthesize_answer(q.question, passages)

    return {
        "question": q.question,
        "answer":   answer,
        "passages": passages
    }

def dense_search(q: str, k: int):
    vec = model.encode([q], normalize_embeddings=True).astype("float32")
    D,I = index.search(vec, k)
    out = []
    for rank, idx in enumerate(I[0]):
        row = con.execute(
            "SELECT chunk_id, source, doc_path, loc, text, ents FROM meta WHERE id=?",
            (int(idx),)
        ).fetchone()
        chunk_id, source, path, loc, text, ents = row
        out.append({
            "rank":     rank,
            "score":    float(D[0][rank]),
            "doc_path": path,
            "loc":      json.loads(loc),
            "text":     text,
            "ents":     json.loads(ents)
        })
    return out

def synthesize_answer(question: str, passages: list[dict]) -> str:
    ctx = "\n".join(f"[{i}] {p['text']}" for i, p in enumerate(passages))
    prompt = (
        "Answer the user question using only the numbered context. "
        "Cite passages like [0], [1].\n\n### Context\n"
        f"{ctx}\n\n### Question\n{question}\n### Answer:\n"
    )
    return ollama_generate(prompt, model="llama3.2:latest")

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
