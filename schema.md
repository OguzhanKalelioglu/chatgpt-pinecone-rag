   [PDF Dokümanları]
                   │
                   ▼
         [Metin Çıkarımı (PyPDFLoader)]
                   │
                   ▼
    [Metin Bölme (TextSplitter - örn. RecursiveCharacterTextSplitter)]
                   │
                   ▼
     [Embedding Uygulaması (Embedding Model)]
                   │
                   ▼
[Vektör Veritabanı (Chroma, Pinecone, vb.)]
                   │
                   │  ← (Sorgu: Kullanıcı Sorgusunun Embedding'i)
                   ▼
  [Benzerlik Araması (Retrieval - k en benzer parça)]
                   │
                   ▼
  [Bağlam Olarak Seçilen Parçaların Birleştirilmesi]
                   │
                   ▼
 [Prompt Oluşturma: "Soru: ...  Bağlam: ... Cevap:"]
                   │
                   ▼
         [GPT-4o API ile Yanıt Üretimi]
                   │
                   ▼
              [Son Yanıt]