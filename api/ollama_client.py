import requests, json

OLLAMA_BASE = "http://localhost:11434"        # keep as-is

DEFAULT_MODEL = "llama3.2:latest"        # ← update to match /api/tags

def ollama_generate(prompt: str,
                    model: str = DEFAULT_MODEL,
                    max_tokens: int = 256) -> str:
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    url = f"{OLLAMA_BASE}/api/generate"
    resp = requests.post(url, json=data, timeout=300)
    resp.raise_for_status()
    return resp.json()["response"].strip()

