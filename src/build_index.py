import faiss
import numpy as np

def build_faiss_index(embeddings):
    arr = np.array(embeddings, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = np.ascontiguousarray(arr, dtype="float32")
    faiss.normalize_L2(arr)

    print('Embedding array shape:', arr.shape)
    index = faiss.IndexFlatIP(arr.shape[1])   # cosine similarity
    index.add(arr)

    return index, arr
