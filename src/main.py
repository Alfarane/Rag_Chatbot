from pdf_loader import load_pdf_text
from chunker import chunk_text
from gemini_client import embed_text   # <- NOTE: changed here
from build_index import build_faiss_index
from rag_answer import rag_answer

PDF_PATH = "data/jio_info.pdf"


# 1) Load PDF
print("Loading PDF...")
text = load_pdf_text(PDF_PATH)

# 2) Chunk text
print("Chunking...")
chunks = chunk_text(text)

# 3) Embed chunks one by one
print("Embedding chunks with Gemini...")
embeddings = []
for c in chunks:
    emb = embed_text(c)     # <- FIXED: use embed_text
    embeddings.append(emb)

# 4) Build FAISS index
print("Building index...")
index, _ = build_faiss_index(embeddings)

# 5) Metadata
metadata = [{"text": c, "source": PDF_PATH} for c in chunks]

print("\nRAG Chatbot Ready! Type 'exit' to quit.\n")

while True:
    q = input("You: ")
    if q.lower() == "exit":
        break

    answer = rag_answer(q, index, metadata)
    print("Bot:", answer)
    print()
