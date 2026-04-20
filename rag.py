# ----------- disable warnings that clutter the user interface ---------
import os
import warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# ----------------------------------------------------------------------

import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)
logging.getLogger("torchvision").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)

MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
TOP_K = 5
TEMPERATURE = 0.1

_encoder = None


def get_encoder():
    """Lazy-load the embedding model once."""
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(MODEL_NAME)
    return _encoder


def retrieve(query, index, chunks, k=TOP_K):
    encoder = get_encoder()
    query_vec = encoder.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "page": chunks[idx]["page"],
            "distance": round(float(dist), 4)
        })
    return results


def generate(query, results):
    context = ""
    for i, r in enumerate(results):
        context += f"[{i+1}] (Source: {r['source']}, Page {r['page']})\n{r['text']}\n\n"

    prompt = f"""You are a technical assistant answering questions about engineering documents.
    Answer the question using only the provided context. Be specific and thorough - 
    synthesize across all relevant sources rather than treating each in isolation.
    For every claim you make, cite the source using its bracketed number e.g. [1].
    If the context does not contain enough information to answer, say so explicitly.
    
    Context:{context}
    
    Question: {query}
    
    Answer:
    """

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": TEMPERATURE}
    )
    return response["message"]["content"]


def rag_query(query, index, chunks, top_K=TOP_K):
    results = retrieve(query, index, chunks, top_K)
    answer = generate(query, results)
    return answer, results