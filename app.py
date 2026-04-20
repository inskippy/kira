import streamlit as st
# from ingest import load_index, build_index, INDEX_PATH
from ingest import load_index, build_index, INDEX_DIR
from rag import rag_query, get_encoder
from pathlib import Path
import time

TOP_K = 5

# PDF_DIR = "NASA_Docs"
PDF_DIRS = {
    "nasa": "NASA_Docs",
    "research": "Research_Docs",
}

def available_indexes():
    p = Path(INDEX_DIR)
    if not p.exists():
        return []
    return [d.name for d in sorted(p.iterdir()) if d.is_dir()]

indexes = available_indexes()

with st.sidebar:
    st.header("Knowledgebase")
    selected = st.multiselect(
        "Select one or more indexes:",
        options=indexes,
        default=[indexes[0]] if indexes else []
    )

st.set_page_config(page_title="KIRA", page_icon="🔍", layout="wide")
st.title("KIRA")
# st.caption("Knowledge Intelligence and Retrieval Assistant")

# load index once per session using streamlit cache
@st.cache_resource
def load_kira(name):
    try:
        return load_index(name)
    except FileNotFoundError:
        return build_index(PDF_DIRS.get(name, name), name)

loaded = {name: load_kira(name) for name in selected}
total_vectors = sum(idx.ntotal for idx, _ in loaded.values()) if loaded else 0


# warmup code - without this, first query has high latency
if "warmed_up" not in st.session_state:
    st.session_state.warmed_up = False

if not st.session_state.warmed_up and loaded:
    get_encoder().encode(["warmup"], convert_to_numpy=True)
    first_name = next(iter(loaded))
    warmup_index, warmup_chunks = loaded[first_name]
    rag_query("warmup", warmup_index, warmup_chunks)
    st.session_state.warmed_up = True


# def load_kira():
#     if Path(INDEX_PATH).exists():
#         return load_index()
#     return build_index(PDF_DIR)
# index, chunks = load_kira()



# st.caption(f"Ready: {index.ntotal} vectors loaded across your knowledgebase.")

# if "index" not in st.session_state:
#     with st.spinner("Loading KIRA... please wait..."):
#         index, chunks = load_index() if Path(INDEX_PATH).exists() else build_index(PDF_DIR)
#         st.session_state.index = index
#         st.session_state.chunks = chunks
# index = st.session_state.index
# chunks = st.session_state.chunks
# st.caption(f"Ready: {index.ntotal} vectors loaded across your knowledgebase.")

# initialize history on first load
if "history" not in st.session_state:
    st.session_state.history = []


st.caption("Knowledge Intelligence and Retrieval Assistant")
st.divider()

col1, col2 = st.columns([3,1])
with col1:
    query = st.text_input("Ask a question about your engineering documents:", 
                          placeholder="e.g. What is the purpose of Phase E in NASA Systems Engineering?")
with col2:
    # st.metric("Documents indexed", index.ntotal)
    st.metric("Vectors loaded", total_vectors)


# query input
# query = st.text_input("Ask a question about your engineering documents:")

# if query:
if query and loaded:
    start_time = time.time()
    with st.spinner("Retrieving and generating answer..."):
        # answer, results = rag_query(query, index, chunks)
        from rag import generate
        all_results = []
        for name, (index, chunks) in loaded.items():
            _, results = rag_query(query, index, chunks, TOP_K)
            for r in results:
                r["index"] = name
            all_results.extend(results)
        all_results.sort(key=lambda r: r["distance"])
        answer = generate(query, all_results[:TOP_K])
    
    # prepend so newest appears at top
    st.session_state.history.insert(0, {
        "query": query,
        "answer": answer,
        # "results": results
        "results": all_results[:TOP_K],
        "indexes": selected
    })
    print("-------------------------------------------------")
    print(query)
    print(f"Execution Time: {(time.time() - start_time):.4f}")
    print("-------------------------------------------------")

# render all history
for entry in st.session_state.history:
    st.markdown(f"### Q: {entry['query']}")
    st.caption(f"Searched: {', '.join(entry['indexes'])}")
    st.write(entry["answer"])
    st.markdown("#### Sources")
    for i, r in enumerate(entry["results"]):
        # with st.expander(f"[{i+1}] {r['source']} — p.{r['page']} (distance: {r['distance']})"):
        with st.expander(f"[{i+1}] [{r['index']}] {r['source']} — p.{r['page']} (distance: {r['distance']})"):
            st.write(r["text"])
    st.divider()

# # query input
# query = st.text_input("Ask a question about information in your knowledgebase:")

# if query:
#     with st.spinner("Retrieving and generating answer..."):
#         answer, results = rag_query(query, index, chunks)

#     st.markdown("### Answer")
#     st.write(answer)

#     st.markdown("### Sources")
#     for i, r in enumerate(results):
#         with st.expander(f"[{i+1}] {r['source']} — p.{r['page']} (distance: {r['distance']})"):
#             st.write(r["text"])