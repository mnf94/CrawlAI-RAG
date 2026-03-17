import os
import shutil
from urllib.parse import urlparse
from langchain_chroma import Chroma  # Menggunakan package terbaru
from langchain_huggingface import HuggingFaceEmbeddings # Standar terbaru

def get_domain_path(website_url: str, base_dir="vector_db"):
    """Fungsi pembantu untuk konsistensi folder penyimpanan"""
    domain = urlparse(website_url).netloc.replace(".", "_")
    return os.path.join(base_dir, domain)

def create_vectorstore(chunks, website_url: str, base_dir="vector_db"):
    persist_dir = get_domain_path(website_url, base_dir)
    
    # Hapus data lama agar tidak duplikat saat indexing ulang URL yang sama
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Inisialisasi dan simpan data
    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    # .persist() sudah tidak diperlukan di versi terbaru, 
    # Chroma otomatis menyimpan saat objek dibuat.
    return persist_dir

def get_vectorstore(website_url: str, base_dir="vector_db"):
    """Fungsi baru untuk dipanggil oleh qa.py"""
    persist_dir = get_domain_path(website_url, base_dir)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Memuat kembali database dari folder
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    return vectordb
