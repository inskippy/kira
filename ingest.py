import pdfplumber
import numpy as np
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

INDEX_DIR = "indexes"
# keep this so kira.py import doesn't break — point it at "nasa"
INDEX_PATH = str(Path(INDEX_DIR) / "nasa" / "rag_index.faiss")
# INDEX_PATH = "rag_index.faiss"
# CHUNKS_PATH = "rag_chunks.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

def index_paths(name="default"):
    base = Path(INDEX_DIR) / name
    base.mkdir(parents=True, exist_ok=True)
    return (
            str(base / "rag_index.faiss"),
            str(base / "rag_chunks.pkl"),
            str(base / "bm25.pkl")
        )


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
    idx_path, chunks_path, bm25_path = index_paths(name)
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
    
    print("Building BM25 index...")
    tokenized_chunks = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"Saved: {bm25_path}")
    
    return index, chunks, bm25

def add_documents(pdf_dir, name):
    """Add new PDFs to an existing index without rebuilding."""
    idx_path, chunks_path, bm25_path = index_paths(name)
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
        bm25 = None
        if Path(bm25_path).exists():
            with open(bm25_path, "rb") as f:
                bm25 = pickle.load(f)
        return index, chunks, bm25

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

    print("Rebuilding BM25 index...")
    tokenized_chunks = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"Saved: {bm25_path}")

    return index, chunks, bm25

# def load_index():
#     """Load existing index and chunks from disk."""
#     if not Path(INDEX_PATH).exists():
#         raise FileNotFoundError("No index found. Run ingest.py first.")
#     index = faiss.read_index(INDEX_PATH)
#     with open(CHUNKS_PATH, "rb") as f:

def load_index(name):
    idx_path, chunks_path, bm25_path = index_paths(name)
    if not Path(idx_path).exists():
        raise FileNotFoundError(f"No index found for '{name}'. Run build_index() first.")
    index = faiss.read_index(idx_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    bm25 = None
    if Path(bm25_path).exists():
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
    return index, chunks, bm25

# --------------------- csv parsing & chunking, standalone for now ------------------


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


def load_and_chunk_csv(csv_dir, text_fields=None, id_field="ID"):
    """
    Ingest a CSV where each row becomes one chunk.
    text_fields: optional list of column names to embed.
                 If None, uses all columns except id_field.
    id_field: column to use as the source citation (ticket ID).
    """
    import pandas as pd
    chunks = []
    for csv_path in Path(csv_dir).glob("*.csv"):
        df = pd.read_csv(csv_path)
        
        print(f"  Columns available: {list(df.columns)}")
        print(f"  Rows: {len(df)}")

        for idx, row in df.iterrows():
            if text_fields:
                text = " | ".join(
                    f"{f}: {row[f]}" for f in text_fields
                    if f in row and pd.notna(row[f]) and str(row[f]).strip()
                )
            else:
                text = " | ".join(
                    f"{col}: {val}" for col, val in row.items()
                    if pd.notna(val) and str(val).strip() and col != id_field
                )

            if len(text.split()) < 5:
                continue

            chunk_id = str(row[id_field]) if id_field in row and pd.notna(row[id_field]) else f"row_{idx+1}"

            chunks.append({
                "text": text,
                "source": chunk_id,
                "page": idx + 1
            })

        print(f"  {len(chunks)} chunks from {Path(csv_path).name}")
    return chunks


def build_index_csv(csv_path, name, text_fields=None, id_field="Issue key"):
    idx_path, chunks_path, bm25_path = index_paths(name)
    print("Loading and chunking CSV...")
    chunks = load_and_chunk_csv(csv_path, text_fields, id_field)

    print("Embedding chunks...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        [c["text"] for c in chunks],
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype(np.float32)

    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, idx_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved: {idx_path}, {chunks_path}")

    print("Building BM25 index...")
    tokenized_chunks = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"Saved: {bm25_path}")

    return index, chunks, bm25

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("path", help="Path to source material")
    parser.add_argument("name", help="Name for identifying built index")

    parser.add_argument("--csv", help="Comma-separated column names to ingest")
    parser.add_argument("--id_col", help="ID column for CSV")

    args = parser.parse_args()

    text_fields = None
    if args.csv is not None:
        text_fields = args.csv.split(",") if args.csv.strip() else None

    id_col = args.id_col

    is_csv = args.csv is not None

    if is_csv:
        build_index_csv(args.path, args.name, text_fields=text_fields, id_field=id_col)
    else:
        build_index(args.path, args.name)

