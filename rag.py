import streamlit as st
import PyPDF2
import io
import os
import time
from pinecone import Pinecone, ServerlessSpec
import openai
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv() 

############################
# 1. AYARLAR
############################

# OpenAI API ayarları
openai.api_key = os.environ.get("OPENAI_API_KEY")     # OPEN AI API_KEY

# Pinecone ayarları (Burayı değiştirmiyoruz)
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY") # app.pinecone.io API_KEY
PINECONE_ENV = os.environ.get("PINECONE_ENV")         # us-east-1
INDEX_NAME = os.environ.get("INDEX_NAME")             # EKER

# Pinecone Client - 2.0
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)
EMBED_DIM = 1536  # text-embedding-3-small

# Index var mı? Yoksa Oluştur
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="euclidean",
        spec=spec
    )

############################
# 2. YARDIMCI FONKSİYONLAR
############################

def pdf_to_text(pdf_file: io.BytesIO) -> str:
    """
    Yüklenmiş PDF dosyasından metin içeriğini döndürür.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text() + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Metni belirli uzunlukta parçalara ayırır (chunk).
    overlap parametresi, parçalar arasında üst üste binme miktarını kontrol eder.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    OpenAI Embedding API'yi kullanarak metinlerin vektör gösterimlerini döndürür.
    """
    response = openai.Embedding.create(
        input=texts,
        model="text-embedding-3-small"
    )
    
    # response["data"], her bir metin için embedding döndürür
    embeddings = [res["embedding"] for res in response["data"]]
    return embeddings

def upsert_chunks_to_pinecone(chunks: list[str], metadata: dict):
    """
    Chunkları Pinecone'a vektör olarak yükler.
    """
    embeddings = get_embeddings(chunks)
    

    # Pinecone'a upsert için hazırlık
    to_upsert = []
    for i, embed in enumerate(embeddings):
        chunk_id = f"{metadata.get('filename', 'doc')}_{i}"
        to_upsert.append((chunk_id, embed, {"chunk_text": chunks[i], **metadata}))

    # Pinecone 2.0 client'ında index'e bu şekilde erişip upsert yapıyoruz
    pc.Index(INDEX_NAME).upsert(vectors=to_upsert)

############################
# 3. STREAMLIT ARAYÜZÜ
############################

def main():
    # Sayfa yapılandırması
    st.set_page_config(
        page_title="EKER - RAG Sistemi",
        page_icon="📚",
        layout="wide"
    )

    # Custom CSS ekleyelim
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #FF4B4B;
            color: white;
            border-radius: 5px;
        }
        .css-1v0mbdj.etr89bj1 {
            margin-top: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar oluşturalım
    with st.sidebar:
        st.image("eker_logo.png", width=200)
        st.markdown("---")
        st.markdown("### Sistem Bilgileri")
        st.info("""
        - Model: text-embedding-3-small
        - Vectorel Veritabanı: Pinecone
        - Chunk Boyutu: 500 karakter
        """)

    # Ana içerik
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("📚 EKER RAG Sistemi")
        st.markdown("""
        Bu sistem, PDF dosyalarınızı:
        1. Parçalara ayırır
        2. OpenAI Embedding modeliyle vektöre çevirir
        3. Pinecone veritabanına kaydeder
        """)

    # PDF Yükleme Bölümü
    st.markdown("### 📂 PDF Yükleme")
    pdf_file = st.file_uploader(
        "PDF dosyanızı sürükleyip bırakın veya 'Browse files' butonuna tıklayın",
        type=["pdf"],
        help="Maksimum dosya boyutu: 200MB"
    )

    if pdf_file is not None:
        # İşlem Adımları
        st.markdown("### 🔄 İşlem Adımları")
        col1, col2 = st.columns(2)
        
        with col1:
            # PDF Bilgileri
            text = pdf_to_text(pdf_file)
            st.metric(
                label="PDF Metin Uzunluğu",
                value=f"{len(text)} karakter"
            )

        with col2:
            # Chunk Bilgileri
            chunks = chunk_text(text)
            st.metric(
                label="Oluşturulan Parça Sayısı",
                value=f"{len(chunks)} chunk"
            )

        # İşlem Butonu
        if st.button("📤 PDF'yi Pinecone'a Kaydet", help="Tıklayarak işlemi başlatın"):
            with st.status("İşlem devam ediyor...") as status:
                # İlerleme çubuğu
                progress_bar = st.progress(0)
                
                st.write("✨ PDF metin dönüşümü tamamlandı")
                progress_bar.progress(25)
                time.sleep(0.5)
                
                st.write("📝 Metin parçalara ayrılıyor...")
                progress_bar.progress(50)
                time.sleep(0.5)
                
                st.write("🔄 Embedding'ler oluşturuluyor...")
                progress_bar.progress(75)
                
                metadata = {
                    "filename": pdf_file.name,
                    "upload_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "chunk_count": len(chunks)
                }
                
                upsert_chunks_to_pinecone(chunks, metadata)
                progress_bar.progress(100)
                status.update(label="İşlem tamamlandı!", state="complete")
                
            st.success("✅ PDF başarıyla Pinecone'a yüklendi!")
            
            # İşlem Özeti
            st.markdown("### 📊 İşlem Özeti")
            st.json({
                "Dosya Adı": pdf_file.name,
                "Metin Uzunluğu": len(text),
                "Parça Sayısı": len(chunks),
                "Yükleme Tarihi": metadata["upload_date"]
            })

if __name__ == "__main__":
    main()
