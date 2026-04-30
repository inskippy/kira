from rag import rag_query
from ingest import load_index, build_index, INDEX_PATH
from pathlib import Path
import sys

PDF_DIRS = {
    "nasa": "NASA_Docs",
    "research": "Research_Docs",
}

# PDF_DIR = "NASA_Docs"


def print_response(query, answer, results):
    print(f"\nQuery: {query}")
    print(f"\nAnswer:\n{answer}")
    print("\n--- Sources ---")
    for i, r in enumerate(results):
        print(f"  [{i+1}] {r['source']} p.{r['page']} (distance: {r['distance']})")
    print()


def main():
    # load or build index
    # if Path(INDEX_PATH).exists():
    #     print("Loading existing index...")
    #     index, chunks = load_index()
    # else:
    #     print("No index found, building from scratch...")
    #     index, chunks = build_index(PDF_DIR)
    name = sys.argv[1] if len(sys.argv) > 1 else "nasa"
    try:
        index, chunks, bm25 = load_index(name)
    except FileNotFoundError:
        index, chunks, bm25 = build_index(PDF_DIRS.get(name, name), name)
    print(f"Ready. {index.ntotal} vectors loaded.\n")

    print("Launching KIRA: Knowledge Intelligence and Retrieval Assistant")
    print("Enter queries or type 'quit' to exit")
    while True:
        query = input("Query: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break
        answer, results = rag_query(query, index, chunks)
        print_response(query, answer, results)


if __name__ == "__main__":
    main()