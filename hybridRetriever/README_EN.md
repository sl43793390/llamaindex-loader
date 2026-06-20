# hybridRetriever

> Hybrid retrieval module for RAG — fuse dense vector + BM25 + keyword search to improve recall quality.

---

## 1. Introduction

`hybridRetriever/` adds **multi-recall + result fusion** on top of the Milvus vector index in `vectorStore/`. It covers:

- **Keyword search** — when queries contain "proper nouns / codes / acronyms" that pure vector retrieval often misses
- **BM25 search** — classic IR algorithm, strong on short / rare terms
- **Dense vector search** — semantic matching (synonyms, paraphrases)
- **Hybrid search** — multiple retrievers in parallel, fused via RRF / weighted / concat
- **Milvus native hybrid** — dense + sparse vector fields in the same collection, mixed in one search

---

## 2. Directory Layout

```
hybridRetriever/
├── __init__.py                # Public API exports
├── utils.py                   # tokenize / inverted index / rrf_fuse / weighted_fuse / score_normalize
├── bm25_retriever.py          # BM25 (rank_bm25 — Okapi / Plus / L variants)
├── keyword_retriever.py       # TFIDFKeywordRetriever + LlamaIndexKeywordRetriever
├── vector_retriever.py        # VectorRetriever + score_normalize
├── hybrid_retriever.py        # HybridRetriever + build_hybrid_retriever factory
├── milvus_hybrid.py           # Milvus native hybrid (dual-vector field + hybrid_ranker)
├── demo.py                    # End-to-end demos for 4 scenarios
├── README.md                  # Chinese docs (this file is the English one)
└── README_EN.md               # English docs (this file)
```

---

## 3. Core API

### 3.1 Base retrievers

| Class | Purpose | Backend |
|---|---|---|
| `BM25Retriever` | BM25 search | `rank_bm25.BM25Okapi / Plus / L` |
| `TFIDFKeywordRetriever` | TF-IDF keyword | Pure-Python inverted index |
| `LlamaIndexKeywordRetriever` | LlamaIndex KeywordTable | `KeywordTableSimpleRetriever / RAKERetriever` |
| `VectorRetriever` | Dense vector + score normalization | `VectorStoreIndex` |
| `MilvusHybridRetriever` | Milvus native dense+sparse | `MilvusVectorStore(enable_sparse=True)` |

### 3.2 Hybrid retrieval

| Class / Function | Purpose |
|---|---|
| `HybridRetriever(retrievers, method, top_k)` | Multi-recall + fusion |
| `build_hybrid_retriever(nodes, index, ...)` | One-liner factory for common cases |
| `method="rrf"` | Paper-style RRF, parameter-free |
| `method="weighted"` | Weighted fusion (requires score normalization) |
| `method="concat"` | Concat + dedup by node_id |

### 3.3 Utilities

| Function | Purpose |
|---|---|
| `tokenize(text, remove_stopwords=True)` | Chinese / English mixed tokenizer |
| `rrf_fuse(ranked_lists, k=60, top_k=10)` | RRF fusion |
| `weighted_fuse(ranked_lists, weights, ...)` | Weighted fusion |
| `score_normalize(hits, invert=False)` | Min-max normalize to [0, 1] |
| `dedup_by_id(hits)` | Dedupe by `node_id` |

---

## 4. Quick Start

### 4.1 Install

```bash
# LlamaIndex deps are already in requirements.txt
pip install rank-bm25
```

### 4.2 Minimal example — zero-dependency hybrid

```python
from llama_index.core.schema import TextNode
from hybridRetriever import (
    BM25Retriever, TFIDFKeywordRetriever, HybridRetriever,
)

nodes = [TextNode(text=t) for t in [
    "Milvus is an open source vector database, supporting billion-scale search.",
    "BM25 is a classic IR algorithm based on term frequency and length normalization.",
    "Hybrid Search usually combines dense and sparse retrieval to improve recall.",
]]

bm25 = BM25Retriever.from_nodes(nodes, variant="okapi", k1=1.5, b=0.75)
kw   = TFIDFKeywordRetriever.from_nodes(nodes)

hybrid = HybridRetriever(retrievers=[bm25, kw], method="rrf", top_k=5)

hits = hybrid.retrieve("Milvus hybrid retrieval", top_k=5)
for h in hits:
    print(h.score, h.node.get_content()[:60])
```

### 4.3 Plug into an existing Milvus index (recommended for production)

```python
from vectorStore.milvus_store import (
    build_milvus_store, create_index, configure_settings,
)
from hybridRetriever import build_hybrid_retriever

configure_settings()
store = build_milvus_store(overwrite=True)
index = create_index(nodes, milvus_store=store)

# vector + BM25 + keyword, weighted fusion
hybrid = build_hybrid_retriever(
    nodes      = nodes,
    index      = index,
    retrievers = ["vector", "bm25", "keyword"],
    weights    = [0.6, 0.3, 0.1],
    method     = "weighted",
    top_k      = 5,
)
hits = hybrid.retrieve("What is hybrid search?")
```

### 4.4 Milvus native hybrid (for 100M+ vectors)

```python
from hybridRetriever.milvus_hybrid import (
    build_milvus_hybrid_store, MilvusHybridRetriever,
)
from llama_index.core import VectorStoreIndex, StorageContext

# In real projects use BGE-M3 / SPLADE-V2 etc.
def bge_m3_sparse(texts):
    return [{t: 1.0 for t in tx.split()[:20]} for tx in texts]

store = build_milvus_hybrid_store(
    collection_name           = "rag_hybrid",
    dim                       = 1024,
    sparse_embedding_function = bge_m3_sparse,
    hybrid_ranker             = "rrf",
    hybrid_ranker_params      = {"k": 60},
)
index = VectorStoreIndex(
    nodes,
    storage_context=StorageContext.from_defaults(vector_store=store),
)
retriever = MilvusHybridRetriever(index, similarity_top_k=5)
hits = retriever.retrieve("Milvus hybrid search")
```

### 4.5 Plug into a Chat Engine for RAG

```python
from vectorStore.chat import build_chat_engine

engine = build_chat_engine(index=index, retriever=hybrid, debug=True)
print(engine.chat("What is Milvus?").response)
```

---

## 5. Run the demos

```bash
# Zero-dependency (BM25 + TF-IDF; no API key needed)
python -m hybridRetriever.demo --mode zero

# Plug into Milvus index
python -m hybridRetriever.demo --mode milvus

# Milvus native hybrid
python -m hybridRetriever.demo --mode milvus_hybrid

# Hybrid + Chat Engine
python -m hybridRetriever.demo --mode chat

# Run all
python -m hybridRetriever.demo --mode all
```

See [demo.py](./demo.py) for full source.

---

## 6. Selection Guide

| Scenario | Recommended approach |
|---|---|
| < 10k nodes, no Embedding budget | `BM25` + `TFIDF`, `HybridRetriever(method="rrf")` |
| Existing Milvus index, want better recall | `VectorRetriever` + `BM25Retriever`, `method="weighted"` |
| Queries heavy on "proper nouns / IDs" | Always add BM25 or keyword (vector will miss) |
| > 100k nodes, billions of vectors | Milvus native hybrid (`enable_sparse=True`) |
| Mostly Chinese | BM25 + Chinese stopwords / jieba (built-in uses naive char split) |
| Need RAKE keyword extraction | `LlamaIndexKeywordRetriever(mode="rake")` |

---

## 7. Tuning Tips

- **`weights[bm25_idx]`**: more jargon / proper nouns → raise to 0.4 ~ 0.5
- **`weights[vector_idx]`**: more paraphrases → raise to 0.7 ~ 0.8
- **RRF `k`**: keep the default 60 (paper value)
- **Per-retriever `top_k`**: at least `2 × final top_k` to avoid over-filtering after fusion

---

## 8. Known Limitations

- Built-in `tokenize()` uses naive single-char CJK splitting; swap in `jieba` for production
- `TFIDFKeywordRetriever` does not support phrase queries (only single tokens)
- `MilvusHybridRetriever` requires Milvus 2.4+ sparse field support
- Incremental `add_nodes` re-builds the index for each retriever (fine for < 50k nodes)
