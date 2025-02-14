# proxy.py
import os
import json

from flask import Flask, request, Response
from flask_cors import CORS
from dotenv import load_dotenv

# OpenAI yeni kütüphane
import openai

# RAG fonksiyonları import
from rag_utils import get_relevant_context

load_dotenv()

app = Flask(__name__)
CORS(app)  # Tüm origin'lere izin ver

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
openai.api_key = os.environ.get("OPENAI_API_KEY")     # OPEN AI API_KEY

# Varsayılan system_msg tanımı (fonksiyon DIŞINDA)
system_msg = (
    "Sen yardımsever bir chatbotsın. Kullanıcının sorularına Türkçe cevap ver."
    "Eğer kullanıcının sorusu bağlamla alakalı değilse veya çok genel bir soru ise, sadece genel bir karşılama mesajı ver."
    "Eğer soru bağlam ile ilgiliyse, bağlamı kullanarak detaylı ve bilgilendirici bir cevap ver."
)

# OpenAI Client'ı başlat
client = openai.OpenAI()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    if not data:
        return {"error": "No JSON payload provided"}, 400

    user_prompt = data.get("prompt", "")
    system_msg = data.get("system", system_msg) # İSTEKTEN GELEN system MESAJINI AL, YOKSA VARSAYILANI KULLAN
    requested_model = data.get("model", OPENAI_MODEL)  # Eğer JSON'da yoksa .env deki model

    # 1) Bağlam getir
    context = get_relevant_context(user_prompt, top_k=3)
    # Varsayılan prompt oluştur
    # (Kullanıcıya eğer bulduğumuz context varsa ekle)
    if context.strip():
        # Bağlam bulunmuşsa, user prompt'un sonuna ekleyebiliriz
        # veya system mesajına ekleyebiliriz.
        # Örnek: user sorusuna ekliyoruz:
        user_prompt = (
            f"Soru: {user_prompt}\n\n"
            f"Bağlam:\n{context}\n\n"
            "Cevap:"
        )

    # 2) Streaming yanıt oluşturacak generator fonksiyonu
    def generate():
        try:
            # ChatCompletion streaming
            response = client.chat.completions.create( # GÜNCELLENMİŞ SATIR: client.chat.completions.create
                model=requested_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                stream=True
            )
            # response bir iterator döndürecek
            for chunk in response:
                # chunk structure: ChatCompletionChunk
                if hasattr(chunk, "choices"):
                    for c in chunk.choices:
                        if hasattr(c, "delta") and c.delta:
                            text = c.delta.content
                            if text:
                                # Chunk'ı JSON olarak yolluyoruz
                                # front-end satır satır parse edecek
                                chunk_data = {"response": text}
                                yield f"data: {json.dumps(chunk_data)}\n\n"

        except Exception as e:
            # Hata durumunda da bir satır gönderelim
            error_msg = {"response": f"Üzgünüm, bir hata oluştu: {str(e)}"}
            yield f"data: {json.dumps(error_msg)}\n\n"

    # 3) Chunked response döndür
    return Response(generate(), mimetype="text/event-stream")

if __name__ == "__main__":
    # Örnek: Bu proxy 0.0.0.0:8000 portunda çalışsın
    app.run(host="0.0.0.0", port=8000, debug=True)