# ui/app.py
import sys, os, textwrap, re, html, requests
from pathlib import Path
import streamlit as st

# ── Make ../src importable ───────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
SRC  = HERE.parent / "src"
sys.path.insert(0, str(SRC))

from utils_pdf import render_page_png


###############################################################################
# CONFIG
###############################################################################
API_URL     = "http://localhost:8000/chat"
TOP_K_MAX   = 10
PAGE_ZOOM   = 1.4

ENTITY_LABELS = [
    "PERSON","NORP","FAC","ORG","GPE","LOC","PRODUCT",
    "EVENT","WORK_OF_ART","LAW","LANGUAGE","DATE","TIME",
    "PERCENT","MONEY","QUANTITY","ORDINAL","CARDINAL"
]


###############################################################################
# SIDEBAR
###############################################################################
st.set_page_config(page_title="MediaMind PDF Chat", layout="wide")
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top-k passages", 1, TOP_K_MAX, 5)
entity_filter = st.sidebar.multiselect(
    "Filter by entity type",
    options=ENTITY_LABELS,
    help="Show only passages containing at least one of these entity labels"
)
st.sidebar.markdown("---")
st.sidebar.write("🔌 **API**:", API_URL)


###############################################################################
# MAIN
###############################################################################
st.title("📚 MediaMind · PDF Chat")
query = st.text_input("Ask a question about your PDFs and press ↵", key="query")


def highlight_terms(query: str, text: str, max_chars: int = 400) -> str:
    snippet = textwrap.shorten(text.replace("\n"," "), max_chars, placeholder=" …")
    tokens  = [re.escape(w) for w in query.split() if len(w) >= 4]
    if not tokens:
        return html.escape(snippet)
    pattern = re.compile("(" + "|".join(tokens) + ")", re.I)
    return pattern.sub(r"<mark>\1</mark>", html.escape(snippet))


if query:
    with st.spinner("Searching…"):
        try:
            res = requests.post(
                API_URL,
                json={
                    "question": query,
                    "top_k":    top_k,
                    "entities": entity_filter or None
                },
                timeout=60
            )
            res.raise_for_status()
            payload = res.json()
        except requests.exceptions.ReadTimeout:
            st.error("⚠️ The request timed out (60s). Try fewer passages or a faster backend.")
            st.stop()
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    # ── Show LLM answer ──────────────────────────────────────────────────────────
    if payload.get("answer"):
        st.subheader("💡 LLM-Synthesised Answer")
        st.success(payload["answer"])

    st.markdown("---")

    # ── Show supporting passages ────────────────────────────────────────────────
    for idx, p in enumerate(payload.get("passages", [])):
        pdf_name = Path(p["doc_path"]).name
        page_no  = p["loc"].get("page", "?")
        score    = p["score"]

        col1, col2 = st.columns([1, 3])
        with col1:
            pdf_path = Path(p["doc_path"]).expanduser().resolve()
            if pdf_path.exists():
                st.image(render_page_png(str(pdf_path), page_no), use_column_width=True)
            else:
                st.caption(":grey[Preview unavailable]")

        with col2:
            st.markdown(f"**{idx+1}. {pdf_name} · page {page_no} · score {score:.3f}**")
            st.write(
                highlight_terms(query, p["text"]),
                unsafe_allow_html=True
            )

            # ── Display extracted entities ────────────────────────────────────────
            ents = p.get("ents", [])
            if ents:
                ents_str = ", ".join(f"{e['text']}[{e['label']}]" for e in ents)
                st.caption(f"Entities: {ents_str}")

        st.divider()
else:
    st.info("Enter a question above to get started!")
