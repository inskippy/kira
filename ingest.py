import pdfplumber
import numpy as np
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

INDEX_DIR = "indexes"
# keep this so kira.py import doesn't break — point it at "nasa"
INDEX_PATH = str(Path(INDEX_DIR) / "nasa" / "rag_index.faiss")
# INDEX_PATH = "rag_index.faiss"
# CHUNKS_PATH = "rag_chunks.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

def index_paths(name="default"):
    base = Path(INDEX_DIR) / name
    base.mkdir(parents=True, exist_ok=True)
    return str(base / "rag_index.faiss"), str(base / "rag_chunks.pkl")

# load and chunk the knowledgebase
def load_and_chunk(pdf_dir, chunk_size=500, overlap=50):
    chunks = []  # list of dicts: {text, source, page}
    
    for pdf_path in Path(pdf_dir).glob("*.pdf"):
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue
                
                # slide window across page text
                words = text.split()
                for i in range(0, len(words), chunk_size - overlap):
                    chunk_words = words[i : i + chunk_size]
                    if len(chunk_words) < 20:  # skip tiny tail chunks
                        continue
                    chunks.append({
                        "text": " ".join(chunk_words),
                        "source": pdf_path.name,
                        "page": page_num
                    })
    
    return chunks

def show_document_word_distribution(pdf_dir):
    # distribution of words per page
    word_counts = []
    empty_pages = 0
    page_count = 0
    for pdf_path in Path(pdf_dir).glob("*.pdf"):
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_count += 1
                text = page.extract_text()
                if text:
                    word_counts.append(len(text.split()))
                else:
                    empty_pages += 1

    print(f"Total page count: {page_count}")
    print(f"Pages with no extractable text: {empty_pages}")
    print(f"Median words/page: {np.median(word_counts):.0f}")
    print(f"Mean words/page: {np.mean(word_counts):.0f}")
    print(f"Pages over 500 words: {sum(1 for w in word_counts if w > 500)}")
    print(f"Pages under 100 words: {sum(1 for w in word_counts if w < 100)}")

def build_index(pdf_dir, name):
    idx_path, chunks_path = index_paths(name)
    print("Loading and chunking PDFs...")
    chunks = load_and_chunk(pdf_dir)
    print(f"  {len(chunks)} chunks from {len(list(Path(pdf_dir).glob('*.pdf')))} PDFs")

    print("Embedding chunks...")
    # embed the chunks and build the index
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        [c["text"] for c in chunks],
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype(np.float32) # FAISS expects float32

    print("Building FAISS index...")
    # IndexFlatL2 = exact nearest-neighbor search by L2 distance
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # faiss.write_index(index, INDEX_PATH)
    # with open(CHUNKS_PATH, "wb") as f:
    faiss.write_index(index, idx_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    # print(f"Saved: {INDEX_PATH}, {CHUNKS_PATH}")
    print(f"Saved: {idx_path}, {chunks_path}")
    return index, chunks

def add_documents(pdf_dir, name):
    """Add new PDFs to an existing index without rebuilding."""
    idx_path, chunks_path = index_paths(name)
    if not Path(idx_path).exists():
        raise FileNotFoundError("No existing index found. Run build_index() first.")
# def add_documents(pdf_dir):
    # if not Path(INDEX_PATH).exists():

    print("Loading existing index...")
    # index = faiss.read_index(INDEX_PATH)
    # with open(CHUNKS_PATH, "rb") as f:
    index = faiss.read_index(idx_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    existing_count = index.ntotal

    print("Chunking new documents...")
    new_chunks = load_and_chunk(pdf_dir)
    if not new_chunks:
        print("No new chunks found.")
        return index, chunks

    print("Embedding new chunks...")
    model = SentenceTransformer(MODEL_NAME)
    new_embeddings = model.encode(
        [c["text"] for c in new_chunks],
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype(np.float32)

    index.add(new_embeddings)
    chunks.extend(new_chunks)

    # faiss.write_index(index, INDEX_PATH)
    # with open(CHUNKS_PATH, "wb") as f:
    faiss.write_index(index, idx_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Added {index.ntotal - existing_count} vectors. Total: {index.ntotal}")
    return index, chunks

# def load_index():
#     """Load existing index and chunks from disk."""
#     if not Path(INDEX_PATH).exists():
#         raise FileNotFoundError("No index found. Run ingest.py first.")
#     index = faiss.read_index(INDEX_PATH)
#     with open(CHUNKS_PATH, "rb") as f:

def load_index(name):
    idx_path, chunks_path = index_paths(name)
    if not Path(idx_path).exists():
        raise FileNotFoundError(f"No index found for '{name}'. Run build_index() first.")
    index = faiss.read_index(idx_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


if __name__ == "__main__":
    # build_index("NASA_Docs")
    import sys
    pdf_dir = sys.argv[1] if len(sys.argv) > 1 else "NASA_Docs"
    name = sys.argv[2] if len(sys.argv) > 2 else "nasa"
    build_index(pdf_dir, name)