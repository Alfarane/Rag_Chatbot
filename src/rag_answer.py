from retrieval import retrieve
from gemini_client import generate_answer

def rag_answer(question, index, metadata):
    retrieved_chunks = retrieve(question, index, metadata)

    context = "\n\n---\n\n".join(retrieved_chunks)

    messages = [
        {"role": "system", "content": "Use the provided CONTEXT to answer the QUESTION."},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer clearly and accurately."}
    ]

    return generate_answer(messages)
