# GPT4 + Pinecone + Flask (RAG Yapısı)

Bu proje, **Flask** tabanlı bir web uygulaması kullanarak,
1. **PDF** dosyalarını **chunks** (parçalar) halinde **Pinecone** vektör veritabanına yüklüyor,
2. Kullanıcı bir soru (chat) sorduğunda, en alakalı chunk’ları (bağlam) **Pinecone** üzerinden çekiyor,
3. Bu bağlamı **OpenAI** (ChatCompletion) API’sine göndererek kullanıcıya **bağlamlı bir cevap** veriyor.

Özellikle **Türkçe** içeriği daha iyi işlemek için **Sentence-Transformers** (`paraphrase-multilingual-mpnet-base-v2`) modeliyle embedding’ler üretiyoruz.  
OpenAI’nin **yeni Python Client** arayüzü (`from openai import OpenAI`) baz alınarak oluşturuldu.

## Özellikler

- **Flask** arayüzü üzerinden PDF sürükle-bırak veya dosya seçimi.
- **S-BERT modeli** ile metin embedding (Türkçe dahil).
- **Pinecone 2.0** (serverless) üzerinde vektör veritabanı.
- **OpenAI ChatCompletion** (1.0.0+ kütüphanesi) ile metne dayalı sohbet (RAG).
- **Bağlam olmadığı** durumlarda veya **bağlamın yetersiz kaldığı** durumlarda, model (ChatGPT) kendi genel bilgisinden faydalanır.

## Kurulum

1. **Depoyu klonlayın** (veya indirin):
   ```bash
   git clone https://github.com/kullanici-adi/gpt4-n-vectordb-flask.git
   cd gpt4-n-vectordb-flask
