import os
import uuid

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from flask import Flask, request, render_template, redirect, url_for
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

############################
# .env ve Flask
############################
load_dotenv()
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

############################
# API Keys and Model
############################
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("INDEX_NAME", "gpt4o")

# Model'i .env içinden al, yoksa gpt-4o kullan
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")

############################
# OpenAI Client (Yeni Sürüm)
############################
openai_client = OpenAI(api_key=OPENAI_API_KEY)

############################
# Pinecone 2.0
############################
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)
EMBED_DIM = 768  # S-BERT

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="euclidean",
        spec=spec
    )

pinecone_index = pc.Index(INDEX_NAME)

############################
# S-BERT Model (Burada Farklı Embedding Modelleri kullanabiliriz. Araştırmak lazım.) 
# Bu Model Local'de çalışıyor. Cloud Modeller'de denenebilir.
############################
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
embedder = SentenceTransformer(model_name)

############################
# PDF to text
############################
def pdf_to_text(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

############################
# Text Split
# En doğru Chunksize nedir overlap nedir araştırmak lazım. 
############################
def split_text_into_chunks(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.create_documents([text])
    return chunks

############################
# Upsert to Pinecone
############################
def upsert_documents(chunks):
    vectors = []
    for chunk in chunks:
        content = chunk.page_content
        emb = embedder.encode(content).tolist()
        vector_id = str(uuid.uuid4())
        metadata = {"text": content[:200]}
        vectors.append((vector_id, emb, metadata))
    pinecone_index.upsert(vectors=vectors)

############################
# Retrieve Context from Pinecone
############################
def get_relevant_context(query, top_k=3):
    query_emb = embedder.encode(query).tolist()
    response = pinecone_index.query(
        vector=query_emb,
        top_k=top_k,
        include_metadata=True
    )
    if "matches" in response:
        return "\n\n".join([m["metadata"]["text"] for m in response["matches"]])
    return ""

############################
# Chat Completion (Yeni)
############################
def get_chat_completion(prompt, max_tokens=200):
    # Burada MODEL_NAME .env'den geliyor
    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    # Dikkat: response -> ChatCompletion nesnesi, subscriptable değil
    return response.choices[0].message.content

############################
# Flask Routes
############################

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "pdf_file" in request.files:
            pdf_file = request.files["pdf_file"]
            if pdf_file.filename != "":
                pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_file.filename)
                pdf_file.save(pdf_path)
                text = pdf_to_text(pdf_path)
                chunks = split_text_into_chunks(text)
                upsert_documents(chunks)
                return redirect(url_for("chat"))
    return render_template("index.html")

@app.route("/chat", methods=["GET", "POST"])
def chat():
    answer = ""
    query = ""
    if request.method == "POST":
        query = request.form.get("query", "")
        if query:
            context = get_relevant_context(query, top_k=3)
            prompt = f"Soru: {query}\n\nBağlam:\n{context}\n\nCevap:"
            answer = get_chat_completion(prompt, max_tokens=200)
    return render_template("chat.html", answer=answer, query=query)

if __name__ == "__main__":
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])
    app.run(debug=True)
