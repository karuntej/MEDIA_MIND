# Media Mind

# Install requirements.txt dependencies using
pip install -r requirements.txt

or 

pip install fastapi pydantic faiss-cpu numpy sentence-transformers requests PyMuPDF tqdm spacy streamlit uvicorn urllib3

To enable image OCR when extracting PDFs install the optional dependencies:

```bash
pip install pytesseract pillow
```
You also need the tesseract binary on your system (e.g. `apt install tesseract-ocr`).

## Run the application

Use the convenience script below to launch Ollama, the FastAPI server and the
Streamlit UI in one go:

```bash
python scripts/run_app.py
```

Any existing instances of these processes will be terminated automatically so
you don't need to clear the ports manually.


conda deactivate 
source .venv/bin/activate
uvicorn api.main:app --reload
lsof -i :8000 
kill -9 8000
streamlit run ui/app.py 
OLLAMA_API_URL=http://localhost:11434
API_PORT=8000

python scripts/extract_pdf.py
python scripts/enrich_chunks.py
python scripts/embed_index.py

python scripts/run_app.py 