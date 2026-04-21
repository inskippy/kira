# KIRA — Knowledge Intelligence and Retrieval Assistant

KIRA is a lightweight, local-first Retrieval-Augmented Generation (RAG) 
system for semantic search and question answering over knowledgebases.
It is designed to run entirely locally with no external API calls; 
ideal for IP-sensitive and regulatory-constrained environments.

---

## Features

- Natural language querying over local PDF document collections
- Cited responses with source document and page number traceability
- Negative rejection — explicitly declines to answer when relevant 
  content is not in the index
- Fully local inference via Ollama (no data leaves your machine)
- Streamlit web UI and CLI interfaces
- Incremental document ingestion without full index rebuilds

---

## Architecture

KIRA is organized into four layers:

| Layer | Component | Technology |
|---|---|---|
| Ingestion | PDF parsing | pdfplumber |
| Ingestion | Chunking | Fixed-size, 500 words, 50-word overlap |
| Ingestion | Embedding | all-MiniLM-L6-v2 (sentence-transformers) |
| Retrieval | Vector store | FAISS (IndexFlatL2) |
| Retrieval | Search | Top-K nearest neighbor (L2 distance) |
| Inference | LLM | llama3.2 via Ollama |
| Interface | UI | Streamlit / CLI |

### Key design decisions

**No external framework dependency**
Implemented in Python without LangChain or similar abstractions; easily
debuggable and modifiable, prioritizing transparency.

**On-premises by design**
All inference runs locally via Ollama. No document content, queries, 
or responses are transmitted to external services; perfect for
private or proprietary data.

**K=5 retrieval (prototype)**
KIRA uses K=5 nearest-neighbor retrieval, constrained by the 
context window of small LLMs (llama3.2, 3B parameters). 
Enterprise deployment should use K=10, consistent with Lewis et 
al. (2021) finding valid passages in the top 10 results 90% of
the time. 

**Fixed-size chunking**
500-word chunks with 50-word overlap are used for prototype validation 
against NASA technical reports. This chunk size targets dense, 
page-level information in engineering documents. Future scope should
use semantic/variable chunking tuned per document type.

**Citation-first prompting**
The prompt template enforces source citation for all claims, negative
rejection if information is not in the knowledge base, or explicit
disclaimers when generated content is not grounded in retrieved sources.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) with llama3.2 pulled locally

```bash
ollama pull llama3.2
```

---

## Installation

```bash
git clone https://github.com/inskippy/kira.git
cd kira
pip install -r requirements.txt
```

---

## Usage

### 1. Organize your documents

KIRA supports multiple named indexes. Place PDFs in a subdirectory 
within a `knowledge` directory and choose a name for the index:
```
knowledge/
    NASA_Docs/       → index name: "nasa"
    Research_Docs/   → index name: "research"
    my_docs/         → index name: "anything"
```

### 2. Build an index

```bash
python ingest.py <pdf_dir> <index_name>
```

Example:
```bash
python ingest.py knowledge/NASA_Docs nasa
python ingest.py knowledge/Research_Docs research
```

This creates `indexes/<index_name>/rag_index.faiss` and 
`indexes/<index_name>/rag_chunks.pkl`. The `indexes/` directory 
is gitignored and never committed - same for `knowledge/`.

### 3. Add documents to an existing index

To add new PDFs without rebuilding from scratch:

```python
from ingest import add_documents
add_documents("path/to/new/docs", "index_name")
```

### 4. Query via CLI

```bash
python kira.py <index_name>
```

Example:
```bash
python kira.py nasa
```

### 5. Query via web UI

```bash
streamlit run app.py
```

Select one or more indexes from the sidebar to search across 
them simultaneously. Results are merged and ranked by relevance.

---

## Evaluation

KIRA was validated against five NASA Technical Reports 
(Systems Engineering Handbook, Expanded Guidance Vols. 1–2, 
Requirement Assurance, Interface Management) using 14 test 
queries across three categories:

| Metric | Result |
|---|---|
| Accuracy (top-1) | 69% |
| Accuracy (top-5) | 100% |
| Citation traceability | 85%* |
| Negative rejection | 100% |
| Avg. latency | 23s |

*Traceability failures resolved on query repetition; 
prompt engineering identified as corrective action.

---

## Roadmap

Feel free to contribute and open PR's!
- [x] Multi-index support (named collections with merged retrieval)
- [ ] Semantic/variable chunking
- [ ] Hybrid retrieval (dense + BM25)
- [ ] Conversation memory (multi-turn queries)
- [ ] Cross-encoder reranking
- [ ] Docx/xlsx ingestion
- [ ] OCR support for scanned documents

---

## Citation

If you use KIRA in your research, please cite:

```bibtex
@misc{inskip2026kira,
  author = {Inskip, Adam T.},
  title = {KIRA: Knowledge Intelligence and Retrieval Assistant},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/inskippy/kira}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## References

Lewis, P. et al. (2021). Retrieval-Augmented Generation for 
Knowledge-Intensive NLP Tasks. arXiv:2005.11401.

## User Interface
![KIRA UI](docs\kira_default_ui.png)
![KIRA UI WITH QUERY](docs\kira_ui_with_response.png)

## Enterprise Deployment Architecture Diagram
![KIRA DEFAULT UI](docs\enterprise_deployment_architecture.png)