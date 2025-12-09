import numpy as np
import faiss
from gemini_client import embed_text

def retrieve(query, index, metadata, top_k=4):
    q_emb = embed_text(query)
    q = np.array([q_emb], dtype="float32")
    faiss.normalize_L2(q)

    D, I = index.search(q, top_k)

    return [metadata[i]["text"] for i in I[0]]
