import os
import streamlit as st
import time
from dotenv import load_dotenv

# 1. Load environment variables (Terutama GROQ_API_KEY)
load_dotenv()

# =====================================================
# SETUP PLAYWRIGHT UNTUK CLOUD
# =====================================================
@st.cache_resource
def install_playwright():
    """Menginstal browser Playwright otomatis di server saat aplikasi pertama jalan."""
    # PERHATIAN: install-deps dihapus karena akan diurus oleh file packages.txt
    os.system("playwright install chromium")

# Jalankan instalasi Playwright sebelum memanggil modul internal
install_playwright()

# --- IMPORT INTERNAL MODULES ---
from scraper.crawler import crawl_website
from rag.chunker import chunk_text
from rag.vectorstore import create_vectorstore
from rag.qa import get_qa_chain

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="CrawlAI RAG",
    page_icon="🌐",
    layout="centered"
)

# Custom CSS untuk Button Styling agar sesuai UI asli
st.markdown(
    """
    <style>
    div.stButton > button {
        font-size: 0.85rem;
        padding: 0.5rem 0.5rem;
        text-align: left;
        white-space: normal;
        line-height: 1.2;
        width: 100%;
    }
    .main-title { font-size: 2.5rem; font-weight: bold; text-align: center; margin-bottom: 0; }
    img { display: block; margin-left: auto; margin-right: auto; border-radius: 50%; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- INITIALIZE SESSION STATE ---
if "indexed_websites" not in st.session_state:
    st.session_state.indexed_websites = []
if "last_url" not in st.session_state:
    st.session_state.last_url = None
if "answer" not in st.session_state:
    st.session_state.answer = None

# --- HEADER & BRANDING ---
st.markdown('<p align="center"><img src="https://github.com/user-attachments/assets/4a7ead5e-cf4a-427d-b225-5b853966e0da" width="200"></p>', unsafe_allow_html=True)
st.markdown('<div class="main-title">CrawlAI RAG</div>', unsafe_allow_html=True)
st.caption("<center>Crawl websites, index content, and ask questions with LLaMA 3.3</center>", unsafe_allow_html=True)

# =====================================================
# 1. INDEX WEBSITE
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
        progress = st.progress(0)
        status = st.empty()

        try:
            # Step-by-step progress simulation with real logic
            status.text("🚀 Launching crawler...")
            progress.progress(10)
            
            pages = crawl_website(website_url)
            status.text(f"📄 Found {len(pages)} pages. Extracting text...")
            progress.progress(40)
            
            chunks = chunk_text(pages)
            status.text(f"✂️ Created {len(chunks)} text chunks...")
            progress.progress(70)
            
            create_vectorstore(chunks, website_url)
            status.text("📦 Storing embeddings in ChromaDB...")
            
            # Save to state
            st.session_state.last_url = website_url
            if website_url not in st.session_state.indexed_websites:
                st.session_state.indexed_websites.append(website_url)
                
            progress.progress(100)
            status.text("✅ Indexing complete!")
            st.success(f"Successfully indexed {website_url}")
            
        except Exception as e:
            st.error(f"Indexing failed: {str(e)}")

st.divider()

# =====================================================
# FUNGSI UNTUK BERTANYA (Direct Call & LCEL Update)
# =====================================================
def perform_ask(question: str):
    if not st.session_state.last_url:
        st.error("Please index a website first.")
        return

    with st.spinner("🤖 Thinking..."):
        try:
            # Panggil Chain RAG dari qa.py
            rag_chain = get_qa_chain(st.session_state.last_url)
            
            # Memakai LCEL murni dan StrOutputParser
            response = rag_chain.invoke(question)
            
            st.session_state.answer = response
        except Exception as e:
            st.session_state.answer = f"Error: {str(e)}"

# =====================================================
# 2. QUICK QUESTIONS
# =====================================================
if st.session_state.last_url:
    st.subheader("2. Quick Questions")
    st.info(f"Context: **{st.session_state.last_url}**")

    common_questions = [
        "What is this website about and what is its primary purpose?",
        "Who is the primary owner or creator of this website?",
    ]

    cols = st.columns(2)
    for idx, q in enumerate(common_questions):
        with cols[idx % 2]:
            if st.button(q, key=f"q_{idx}"):
                perform_ask(q)

    st.divider()

    # =====================================================
    # 3. ASK YOUR OWN QUESTION
    # =====================================================
    st.subheader("3. Ask Your Own Question")

    with st.form("ask_form"):
        user_question = st.text_input(
            "Ask something about the website",
            placeholder="Type your question here"
        )
        submit_ask = st.form_submit_button("Ask")

    if submit_ask:
        if user_question:
            perform_ask(user_question)
        else:
            st.warning("Please enter a question")

    # =====================================================
    # 4. ANSWER DISPLAY
    # =====================================================
    if st.session_state.answer:
        st.markdown("### 💡 Answer")
        st.info(st.session_state.answer)

else:
    st.info("👈 Start by indexing a website to unlock questions.")

st.divider()
st.caption("CrawlAI RAG | Built by Ankit Kumar Nayak")
