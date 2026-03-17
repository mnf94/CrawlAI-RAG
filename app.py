import streamlit as st
import time
from dotenv import load_dotenv

# Import fungsi backend internal Anda
from scraper.crawler import crawl_website
from rag.chunker import chunk_text
from rag.vectorstore import create_vectorstore
from rag.qa import get_qa_chain

# 1. Load environment
load_dotenv()

st.set_page_config(
    page_title="Crawl AI RAG",
    layout="centered"
)

# Custom Styling
st.markdown(
    """
    <style>
    div.stButton > button {
        font-size: 0.85rem;
        padding: 0.5rem 0.5rem;
        text-align: left;
        white-space: normal;
        line-height: 1.2;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize Session State
if "last_website" not in st.session_state:
    st.session_state.last_website = None
if "answer" not in st.session_state:
    st.session_state.answer = None

st.title("CrawlAI RAG")
st.caption("Crawl websites, index content, and ask questions directly")

# =====================================================
# 1. INDEX WEBSITE (Logic Terintegrasi)
# =====================================================
st.subheader("1. Index a Website")

with st.form("ingest_form"):
    website_url = st.text_input(
        "Enter website URL",
        placeholder="https://example.com"
    )
    submit_ingest = st.form_submit_button("Index Website")

if submit_ingest:
    if not website_url:
        st.warning("Please enter a website URL")
    else:
        progress = st.progress(0.0)
        status_text = st.empty()
        
        try:
            # Simulasi progress bar seperti UI lama Anda
            status_text.text("Launching headless browser...")
            progress.progress(15)
            time.sleep(0.5)
            
            # --- PROSES ASLI ---
            status_text.text("Crawling website pages...")
            pages = crawl_website(website_url)
            progress.progress(45)
            
            status_text.text("Chunking content...")
            chunks = chunk_text(pages)
            progress.progress(70)
            
            status_text.text("Creating vectorstore (Embedding)...")
            create_vectorstore(chunks, website_url)
            
            # Simpan status ke session
            st.session_state.last_website = website_url
            
            progress.progress(100)
            status_text.text("Indexing complete!")
            st.success(f"Website indexed successfully: {len(pages)} pages.")
            
        except Exception as e:
            st.error(f"Indexing failed: {str(e)}")

st.divider()

# =====================================================
# FUNGSI ASK (Direct Call)
# =====================================================
def ask_internal(question: str):
    if not st.session_state.last_website:
        st.error("Please index a website first!")
        return

    with st.spinner("Generating answer..."):
        try:
            # Panggil fungsi QA Anda langsung
            qa_chain = get_qa_chain(st.session_state.last_website)
            result = qa_chain.invoke(question)
            
            # Parsing hasil
            if isinstance(result, dict) and "result" in result:
                st.session_state.answer = result["result"]
            else:
                st.session_state.answer = str(result)
        except Exception as e:
            st.session_state.answer = f"Error: {str(e)}"

# =====================================================
# 2. QUICK QUESTIONS
# =====================================================
st.subheader("2. Quick Questions")

common_questions = [
    "What is this website about and what is its primary purpose?",
    "Who is the primary owner or creator of this website?",
]

cols = st.columns(2)
for idx, q in enumerate(common_questions):
    with cols[idx % 2]:
        if st.button(q, use_container_width=True, key=f"q_{idx}"):
            ask_internal(q)

st.divider()

# =====================================================
# 3. ASK YOUR OWN QUESTION
# =====================================================
st.subheader("3. Ask Your Own Question")

with st.form("ask_form"):
    user_question = st.text_input("Ask something", placeholder="Type here...")
    submit_ask = st.form_submit_button("Ask")

if submit_ask:
    if user_question:
        ask_internal(user_question)
    else:
        st.warning("Please enter a question")

# =====================================================
# 4. ANSWER DISPLAY
# =====================================================
if st.session_state.answer:
    st.markdown("### Answer")
    st.info(st.session_state.answer)
