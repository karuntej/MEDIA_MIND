#!/usr/bin/env python3
"""
Parallel robust PDF extractor:
  • Repairs via pikepdf (optional)
  • fitz → fallback to pdfplumber (optional)
  • OCR (pytesseract, optional)
  • Tables (camelot, optional)
  • Embedded images
  • Sentence‐level sliding‐window chunking
Outputs → data/processed/all_chunks.json,
          skipped_pdfs.json,
          skipped_pages.json,
          images/…
"""
import uuid
import json
import io
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image

# ── Optional repair ──────────────────────────────────────────────────────────
try:
    import pikepdf
    HAS_REPAIR = True
except ImportError:
    HAS_REPAIR = False

# ── Core PDF reader ──────────────────────────────────────────────────────────
import fitz  # PyMuPDF

# ── Optional pdfplumber fallback ─────────────────────────────────────────────
try:
    import pdfplumber
    HAS_PLUMBER = True
except ImportError:
    HAS_PLUMBER = False

# ── Optional OCR ──────────────────────────────────────────────────────────────
try:
    import pytesseract
    HAS_OCR = shutil.which("tesseract") is not None
except ImportError:
    HAS_OCR = False

# ── Optional table extraction ─────────────────────────────────────────────────
try:
    import camelot
    HAS_TABLES = True
except ImportError:
    HAS_TABLES = False

# ── Sentence‐splitting setup (spaCy) ──────────────────────────────────────────
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])
nlp.add_pipe("sentencizer")
WINDOW, OVERLAP = 5, 2
STEP = WINDOW - OVERLAP

# ── Paths ────────────────────────────────────────────────────────────────────
BASE    = Path(__file__).resolve().parents[1]
PDF_DIR = BASE / "data" / "raw" / "pdf"
OUT_DIR = BASE / "data" / "processed"
IMG_DIR = OUT_DIR / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)


def process_pdf(pdf_path_str: str):
    """
    Process one PDF: repair, open, per-page extract text/OCR/tables/images,
    sentence‐chunk, return (chunks, skipped_pages).
    Raises on PDF‐open failure.
    """
    pdf_path = Path(pdf_path_str)
    chunks = []
    skipped_pages = []

    # 1) Optional repair
    doc = None
    if HAS_REPAIR:
        try:
            fixed = pdf_path.with_suffix(".fixed.pdf")
            with pikepdf.Pdf.open(pdf_path) as pdf:
                pdf.save(fixed)
            doc = fitz.open(str(fixed))
        except Exception:
            doc = None

    # 2) Fallback open
    if doc is None:
        doc = fitz.open(str(pdf_path))  # may raise

    # 3) Iterate pages
    for page_no in range(doc.page_count):
        text = ""
        tables = []

        # Try native fitz
        try:
            page = doc.load_page(page_no)
            text = page.get_text()

            # OCR overlay
            if HAS_OCR:
                for img in page.get_images(full=True):
                    xref = img[0]
                    raw = page.get_image(xref)["image"] if hasattr(page, "get_image") \
                          else doc.extract_image(xref)["image"]
                    if raw:
                        im = Image.open(io.BytesIO(raw))
                        text += "\n" + pytesseract.image_to_string(im)
                        # save image
                        page_img_dir = IMG_DIR / pdf_path.stem / f"page_{page_no+1}"
                        page_img_dir.mkdir(parents=True, exist_ok=True)
                        im.save(page_img_dir / f"img_{xref}.png")

            # Camelot tables
            if HAS_TABLES:
                try:
                    tbls = camelot.read_pdf(
                        str(pdf_path), pages=str(page_no+1), flavor="stream"
                    )
                    for t in tbls:
                        tables.append(t.df.to_csv(index=False))
                except Exception:
                    pass

        except Exception as e:
            # Fallback via pdfplumber
            if HAS_PLUMBER:
                try:
                    with pdfplumber.open(str(pdf_path)) as pl:
                        pg = pl.pages[page_no]
                        text = pg.extract_text() or ""
                        # tables via pdfplumber
                        if HAS_TABLES:
                            for table in pg.extract_tables():
                                rows = [",".join(cell or "" for cell in r) for r in table]
                                tables.append("\n".join(rows))
                        # images via pdfplumber + OCR
                        if HAS_OCR:
                            for img in pg.images:
                                crop = pg.crop((img["x0"], img["top"], img["x1"], img["bottom"]))
                                b = crop.to_image(resolution=150).original
                                im = Image.open(io.BytesIO(b))
                                page_img_dir = IMG_DIR / pdf_path.stem / f"page_{page_no+1}"
                                page_img_dir.mkdir(parents=True, exist_ok=True)
                                im.save(page_img_dir / f"plumb_{img['object_id']}.png")
                                text += "\n" + pytesseract.image_to_string(im)
                except Exception as e2:
                    skipped_pages.append((pdf_path.name, page_no+1, f"{e} / {e2}"))
                    continue
            else:
                skipped_pages.append((pdf_path.name, page_no+1, str(e)))
                continue

        # 4) Sentence sliding‐window
        doc_nlp = nlp(text)
        sents = [s.text.strip() for s in doc_nlp.sents if s.text.strip()]
        base = {
            "chunk_id": str(uuid.uuid4()),
            "source": "pdf",
            "doc_path": str(pdf_path),
            "loc": {"page": page_no+1},
            "tables": tables
        }

        if sents:
            for i in range(0, len(sents), STEP):
                win = sents[i : i + WINDOW]
                chunks.append({
                    **base,
                    "elem_type": "text",
                    "loc": {
                        **base["loc"],
                        "start_sent": i+1,
                        "end_sent": i+len(win)
                    },
                    "text": " ".join(win)
                })
        else:
            chunks.append({**base, "elem_type": "text", "text": text})

        # 5) Emit table chunks
        for ti, tbl in enumerate(tables):
            chunks.append({
                **base,
                "elem_type": "table",
                "table_no": ti,
                "text": tbl
            })

    return chunks, skipped_pages


def main():
    # 0) Discover
    print(f"Scanning PDFs in {PDF_DIR}…")
    pdfs = list(PDF_DIR.rglob("*.pdf"))
    print(f"Found {len(pdfs)} PDF files\n")

    all_chunks = []
    all_skipped_pages = []
    skipped_pdfs = []

    # 1) Parallel execution
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_pdf, str(p)): p for p in pdfs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing PDFs"):
            pdf = futures[fut]
            try:
                chunks, skipped_pages = fut.result()
                all_chunks.extend(chunks)
                all_skipped_pages.extend(skipped_pages)
            except Exception as e:
                skipped_pdfs.append((pdf.name, str(e)))

    # 2) Write outputs
    (OUT_DIR / "all_chunks.json").write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "skipped_pdfs.json").write_text(
        json.dumps(skipped_pdfs, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "skipped_pages.json").write_text(
        json.dumps(all_skipped_pages, indent=2), encoding="utf-8"
    )

    # 3) Summary & tips
    print(f"\n✅ Extracted {len(all_chunks)} chunks")
    print(f"🚫 Skipped {len(skipped_pdfs)} PDFs & {len(all_skipped_pages)} pages")
    if not HAS_REPAIR:
        print("ℹ️  Tip: pip install pikepdf to auto-repair corrupt PDFs")
    if not HAS_PLUMBER:
        print("ℹ️  Tip: pip install pdfplumber to enable fallback extraction")
    if not HAS_OCR:
        print("ℹ️  Tip: pip install pytesseract Pillow & system tesseract for OCR")
    if not HAS_TABLES:
        print("ℹ️  Tip: pip install camelot-py[cv] Ghostscript for table extraction")


if __name__ == "__main__":
    main()
