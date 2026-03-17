import streamlit as st
from dotenv import load_dotenv
from scraper.crawler import crawl_website
from rag.chunker import chunk_text
from rag.vectorstore import create_vectorstore
from rag.qa import get_qa_chain

# 1. Load environment variables (OpenAI API Key, dsb)
load_dotenv()

# Konfigurasi Halaman
st.set_page_config(page_title="RAG Website Q&A", layout="wide")
st.title("🌐 Website RAG Assistant")

# Initialize Session State untuk menyimpan status website
if "last_website" not in st.session_state:
    st.session_state.last_website = None

# --- SIDEBAR: Ingest Website ---
with st.sidebar:
    st.header("Konfigurasi Data")
    url_input = st.text_input("Masukkan URL Website:", placeholder="https://example.com")
    
    if st.button("Index Website"):
        if url_input:
            with st.spinner("Sedang memproses website..."):
                try:
                    # Proses yang sebelumnya ada di @app.post("/ingest")
                    pages = crawl_website(url_input)
                    chunks = chunk_text(pages)
                    create_vectorstore(chunks, url_input)
                    
                    # Simpan ke session state
                    st.session_state.last_website = url_input
                    
                    st.success(f"Berhasil! {len(pages)} halaman diindeks dalam {len(chunks)} chunks.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")
        else:
            st.warning("Mohon masukkan URL terlebih dahulu.")

# --- MAIN AREA: Q&A Interface ---
st.divider()

if st.session_state.last_website:
    st.info(f"Sedang mengobrol dengan: **{st.session_state.last_website}**")
    
    question = st.text_input("Tanyakan sesuatu tentang website ini:")
    
    if st.button("Tanya"):
        if question:
            with st.spinner("Mencari jawaban..."):
                # Proses yang sebelumnya ada di @app.post("/ask")
                qa_chain = get_qa_chain(st.session_state.last_website)
                result = qa_chain.invoke(question)
                
                # Parsing jawaban
                answer = ""
                if isinstance(result, dict) and "result" in result:
                    answer = result["result"]
                else:
                    answer = str(result)
                
                st.markdown("### Jawaban:")
                st.write(answer)
        else:
            st.warning("Silakan masukkan pertanyaan.")
else:
    st.write("Silakan masukkan URL di sidebar untuk memulai.")
