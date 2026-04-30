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
from rank_bm25 import BM25Okapi

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

def retrieve_dense(query, index, chunks, k):
    encoder = get_encoder()
    query_vec = encoder.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_vec, k)
    results = {}
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        results[int(idx)] = {
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "page": chunks[idx]["page"],
            "distance": round(float(dist), 4),
            "dense_rank": rank
        }
    return results

def retrieve_bm25(query, chunks, bm25, k):    
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(bm25_scores)[::-1][:k]
    results = {}
    for rank, idx in enumerate(top_indices):
        results[int(idx)] = {
            "text": chunks[idx]["text"],
            "source": chunks[idx]["source"],
            "page": chunks[idx]["page"],
            "sparse_rank": rank
        }
    return results

def retrieve(query, index, chunks, bm25=None, k=TOP_K):
    dense_results = retrieve_dense(query, index, chunks, k)
    sparse_results = retrieve_bm25(query, chunks, bm25, k) if bm25 is not None else {}

    all_indices = set(dense_results.keys()) | set(sparse_results.keys())
    K_RRF = 60
    rrf_scores = {}
    for idx in all_indices:
        score = 0
        if idx in dense_results:
            score += 1 / (K_RRF + dense_results[idx]["dense_rank"])
        if idx in sparse_results:
            score += 1 / (K_RRF + sparse_results[idx]["sparse_rank"])
        rrf_scores[idx] = score

    top_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]

    results = []
    for idx in top_indices:
        source = dense_results.get(idx) or sparse_results.get(idx)
        results.append({
            "text": source["text"],
            "source": source["source"],
            "page": source["page"],
            "distance": round(1 - rrf_scores[idx], 4)
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

def rag_query(query, index, chunks, bm25=None, top_K=TOP_K):
    results = retrieve(query, index, chunks, bm25, top_K)
    answer = generate(query, results)
    return answer, results