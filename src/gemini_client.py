
import os
from google import genai

def embed_text(text, model_name="gemini-embedding-001"):
    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    resp = client.models.embed_content(
        model=model_name,
        contents=text
    )
    # resp.embeddings is a list of ContentEmbedding, use the first one's values
    if not resp.embeddings or len(resp.embeddings) == 0:
        raise RuntimeError("No embeddings returned from Gemini API. Response: {}".format(resp))
    return resp.embeddings[0].values

def generate_answer(messages, model_name="gemini-2.5-flash"):
    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    prompt = "\n".join([msg["content"] for msg in messages])
    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    return response.text
