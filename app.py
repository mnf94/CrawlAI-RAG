import streamlit as st
import time
from dotenv import load_dotenv

# Import internal modules (Sesuaikan dengan struktur folder Anda)
from scraper.crawler import crawl_website
from rag.chunker import chunk_text
from rag.vectorstore import create_vectorstore
from rag.qa import get_qa_chain

# 1. Load Environment (API Keys untuk Groq/OpenAI)
load_dotenv()

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="CrawlAI RAG",
    page_icon="🌐",
    layout="centered"
)

# Custom CSS untuk Button & UI
st.markdown("""
    <style>
    .main-title { font-size: 2.5rem; font-weight: bold; text-align: center; margin-bottom: 0; }
    .stButton > button { width: 100%; border-radius: 5px; height: 3em; }
    img { display: block; margin-left: auto; margin-right: auto; border-radius: 50%; }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER & BRANDING (Sesuai Panduan) ---
st.markdown('<p align="center"><img src="https://github.com/user-attachments/assets/4a7ead5e-cf4a-427d-b225-5b853966e0da" width="200"></p>', unsafe_allow_html=True)
st.markdown('<div class="main-title">CrawlAI RAG</div>', unsafe_allow_html=True)
st.caption("<center>Crawl websites, index content, and ask questions using RAG</center>", unsafe_allow_html=True)

# Initialize Session State
if "indexed_websites" not in st.session_state:
    st.session_state.indexed_websites = []
if "answer" not in st.session_state:
    st.session_state.answer = None

# --- SECTION 1: INDEXING ---
st.header("1. Index a Website")
with st.container(border=True):
    website_url = st.text_input("Enter website URL", placeholder="https://example.com")
    submit_ingest = st.button("Index Website", type="primary")

    if submit_ingest:
        if not website_url:
            st.warning("Please enter a website URL")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Crawling
                status_text.text("🔍 Crawling internal pages...")
                pages = crawl_website(website_url)
                progress_bar.progress(30)
                
                # Step 2: Chunking
                status_text.text("✂️ Splitting text into chunks...")
                chunks = chunk_text(pages)
                progress_bar.progress(60)
                
                # Step 3: Vector Storage (ChromaDB)
                status_text.text("📦 Generating embeddings & storing in ChromaDB...")
                create_vectorstore(chunks, website_url)
                
                # Update State
                if website_url not in st.session_state.indexed_websites:
                    st.session_state.indexed_websites.append(website_url)
                
                progress_bar.progress(100)
                status_text.text("✅ Indexing complete!")
                st.success(f"Indexed {len(pages)} pages from {website_url}")
            except Exception as e:
                st.error(f"Error during indexing: {e}")

# --- SECTION 2: QUICK QUESTIONS ---
if st.session_state.indexed_websites:
    st.divider()
    st.header("2. Quick Questions")
    
    # Menampilkan website mana saja yang sudah masuk ke Vector DB
    st.info(f"Currently searching in: {', '.join(st.session_state.indexed_websites)}")
    
    common_questions = [
        "What is this website about and what is its primary purpose?",
        "List all services or products mentioned in this website.",
    ]
    
    cols = st.columns(2)
    for idx, q in enumerate(common_questions):
        with cols[idx % 2]:
            if st.button(q, key=f"btn_{idx}"):
                with st.spinner("Analyzing content..."):
                    # Gunakan URL terakhir atau logika multi-index di get_qa_chain
                    qa_chain = get_qa_chain(st.session_state.indexed_websites[-1])
                    res = qa_chain.invoke(q)
                    st.session_state.answer = res.get("result") if isinstance(res, dict) else str(res)

    # --- SECTION 3: CUSTOM QUESTION ---
    st.divider()
    st.header("3. Ask Your Own Question")
    with st.form("ask_form", clear_on_submit=False):
        user_q = st.text_input("Ask something specific:", placeholder="e.g. Who is the author?")
        submit_q = st.form_submit_button("Ask AI")
        
        if submit_q and user_q:
            with st.spinner("Groq LLaMA 3.3 is thinking..."):
                qa_chain = get_qa_chain(st.session_state.indexed_websites[-1])
                res = qa_chain.invoke(user_q)
                st.session_state.answer = res.get("result") if isinstance(res, dict) else str(res)

# --- SECTION 4: DISPLAY ANSWER ---
if st.session_state.answer:
    st.subheader("💡 Answer")
    st.markdown(f"> {st.session_state.answer}")
    
# --- FOOTER ---
st.divider()
st.caption("Built with Groq, LangChain, and ChromaDB | Author: Ankit Kumar Nayak")
