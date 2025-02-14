# rag_utils.py
import os
import openai
from pinecone import Pinecone

# .env dosyasını yükle (eğer .env kullanıyorsanız)
from dotenv import load_dotenv
load_dotenv()

# OpenAI ve Pinecone API anahtarlarını ve ayarlarını .env veya ortam değişkenlerinden alın
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")

openai.api_key = OPENAI_API_KEY

# OpenAI Client'ı başlat (YENİ SATIR)
client = openai.OpenAI()

# Pinecone client'ı başlat
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

EMBED_MODEL = "text-embedding-3-large" # Embedding modeli

def get_embedding(text, model=EMBED_MODEL):
    """
    OpenAI Embedding API kullanarak metin için embedding oluşturur.
    """
    text = text.replace("\n", " ")
    response = client.embeddings.create( # GÜNCELLENMİŞ SATIR: client.embeddings.create
        input=[text],
        model=model
    )
    return response.data[0].embedding # response.data[0].embedding ile embedding'e erişiyoruz.

def get_relevant_context(query, top_k=3):
    """
    Pinecone'dan verilen soruya en alakalı bağlamı getirir.
    """
    query_embedding = get_embedding(query)

    # Pinecone'da benzer embedding'leri ara
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    contexts = []
    for match in results.matches:
        contexts.append(match.metadata.get('chunk_text', '')) # chunk_text metadata'sını al
    return "\n\n".join(contexts) # Bağlamları birleştir ve döndür

