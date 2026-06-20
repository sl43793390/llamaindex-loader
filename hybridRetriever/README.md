# hybridRetriever

> RAG 混合检索模块 —— 把"稠密向量 + BM25 + 关键词"等多种召回源融合,提升 RAG 召回质量。

---

## 1. 简介

`hybridRetriever/` 在 `vectorStore` 已有的 Milvus 向量索引之上,提供**多路召回 + 融合排序**的能力。
覆盖以下场景:

- **关键词检索** —— 用户提问里有"专属名词 / 编号 / 缩写"时,纯向量检索会漏(同义词)
- **BM25 检索** —— 经典信息检索算法,对短词、稀有词命中率高
- **稠密向量检索** —— 语义匹配(同义、改写)
- **混合检索** —— 上述多路并行召回,RRF / 加权融合
- **Milvus 原生混合** —— 同一 collection 存稠密 + 稀疏向量,Milvus 内部一次 search 出混合结果

---

## 2. 目录结构

```
hybridRetriever/
├── __init__.py                # 公共 API 导出
├── utils.py                   # 分词 / 倒排索引 / 融合函数(score_normalize / rrf_fuse / weighted_fuse)
├── bm25_retriever.py          # BM25 检索(基于 rank_bm25,Okapi/Plus/L 三种变体)
├── keyword_retriever.py       # 关键词检索(TFIDFKeywordRetriever + LlamaIndexKeywordRetriever)
├── vector_retriever.py        # 稠密向量检索(VectorRetriever + score_normalize 归一化)
├── hybrid_retriever.py        # 混合检索(HybridRetriever + build_hybrid_retriever 工厂)
├── milvus_hybrid.py           # Milvus 原生混合(双向量字段 + hybrid_ranker)
├── demo.py                    # 4 个场景的端到端 demo
├── README.md                  # 中文文档(本文件)
└── README_EN.md               # 英文文档
```

---

## 3. 核心 API

### 3.1 基础检索器

| 类 | 作用 | 后端 |
|---|---|---|
| `BM25Retriever` | BM25 检索(经典信息检索算法) | `rank_bm25.BM25Okapi / Plus / L` |
| `TFIDFKeywordRetriever` | TF-IDF 关键词检索 | 纯 Python 倒排索引 |
| `LlamaIndexKeywordRetriever` | LlamaIndex 内置 KeywordTable | `KeywordTableSimpleRetriever / RAKERetriever` |
| `VectorRetriever` | 稠密向量检索 + 分数归一化 | `VectorStoreIndex` |
| `MilvusHybridRetriever` | Milvus 原生稠密+稀疏 | `MilvusVectorStore(enable_sparse=True)` |

### 3.2 混合检索

| 类 / 函数 | 作用 |
|---|---|
| `HybridRetriever(retrievers, method, top_k)` | 多路召回 + 融合排序 |
| `build_hybrid_retriever(nodes, index, ...)` | 一行构造常用混合检索 |
| `HybridRetriever(method="rrf")` | 论文经典 RRF,无参数 |
| `HybridRetriever(method="weighted")` | 加权融合,需先 `score_normalize` |
| `HybridRetriever(method="concat")` | 多路拼接 + 去重 |

### 3.3 工具函数

| 函数 | 作用 |
|---|---|
| `tokenize(text, remove_stopwords=True)` | 中英文混合分词 |
| `rrf_fuse(ranked_lists, k=60, top_k=10)` | RRF 融合 |
| `weighted_fuse(ranked_lists, weights, ...)` | 加权融合 |
| `score_normalize(hits, invert=False)` | 分数归一化到 [0, 1] |
| `dedup_by_id(hits)` | 按 `node_id` 去重 |

---

## 4. 快速开始

### 4.1 安装依赖

```bash
# 基础 LlamaIndex 已有;只需补一个 rank-bm25
pip install rank-bm25
```

### 4.2 最简示例 —— 零依赖混合检索

```python
from llama_index.core.schema import TextNode
from hybridRetriever import (
    BM25Retriever, TFIDFKeywordRetriever, HybridRetriever,
)

# 准备节点(实际场景里来自 spliter/ 或 dataLoader/)
nodes = [TextNode(text=t) for t in [
    "Milvus 是一个开源的向量数据库,支持亿级向量的毫秒级检索。",
    "BM25 是经典的信息检索算法,基于词频和文档长度归一化。",
    "Hybrid Search 通常结合稠密向量检索和稀疏检索。",
]]

# 1) 构造两个稀疏检索器
bm25 = BM25Retriever.from_nodes(nodes, variant="okapi", k1=1.5, b=0.75)
kw   = TFIDFKeywordRetriever.from_nodes(nodes)

# 2) 融合(RRF,无参数)
hybrid = HybridRetriever(retrievers=[bm25, kw], method="rrf", top_k=5)

# 3) 检索
hits = hybrid.retrieve("Milvus 混合检索", top_k=5)
for h in hits:
    print(h.score, h.node.get_content()[:60])
```

### 4.3 接入现有 Milvus 索引(推荐生产用法)

```python
from vectorStore.milvus_store import (
    build_milvus_store, create_index, configure_settings,
)
from hybridRetriever import build_hybrid_retriever

configure_settings()
store  = build_milvus_store(overwrite=True)
index  = create_index(nodes, milvus_store=store)

# 一行构造:向量 + BM25 + 关键词,加权融合
hybrid = build_hybrid_retriever(
    nodes  = nodes,
    index  = index,
    retrievers = ["vector", "bm25", "keyword"],
    weights    = [0.6, 0.3, 0.1],   # 语义为主,稀疏补强
    method     = "weighted",
    top_k      = 5,
)
hits = hybrid.retrieve("什么是混合检索?")
```

### 4.4 Milvus 原生混合(亿级数据)

```python
from hybridRetriever.milvus_hybrid import (
    build_milvus_hybrid_store, MilvusHybridRetriever,
)
from llama_index.core import VectorStoreIndex, StorageContext

# 真实项目里用 BGE-M3 / SPLADE-V2 之类的稀疏函数
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
hits = retriever.retrieve("Milvus 混合检索")
```

### 4.5 接 Chat Engine 做 RAG 对话

```python
from vectorStore.chat import build_chat_engine

# 用 HybridRetriever 替换默认的向量 retriever
engine = build_chat_engine(index=index, retriever=hybrid, debug=True)
print(engine.chat("Milvus 是什么?").response)
```

---

## 5. Demo 跑法

```bash
# 零依赖 demo(BM25 + TF-IDF,不需要 API Key)
python -m hybridRetriever.demo --mode zero

# 接入 Milvus 索引(需要 OPENAI_API_KEY + 跑过 vectorStore 索引流程)
python -m hybridRetriever.demo --mode milvus

# Milvus 原生混合
python -m hybridRetriever.demo --mode milvus_hybrid

# 混合检索 + Chat Engine
python -m hybridRetriever.demo --mode chat

# 全跑
python -m hybridRetriever.demo --mode all
```

详见 [demo.py](./demo.py)。

---

## 6. 选择指南

| 场景 | 推荐方案 |
|---|---|
| < 1w 节点,不想花 Embedding 钱 | `BM25` + `TFIDF`,`HybridRetriever(method="rrf")` |
| 已有 Milvus 索引,想提升召回 | `VectorRetriever` + `BM25Retriever`,`weighted` 融合 |
| 提问含大量"专属名词 / 编号" | 必加 BM25 或关键词(纯向量会漏) |
| > 10w 节点,亿级数据 | Milvus 原生混合(`enable_sparse=True`) |
| 中文为主 | BM25 配合中文停用词 / jieba(目前用内置简单分词) |
| 需要 RAKE 关键词提取 | `LlamaIndexKeywordRetriever(mode="rake")` |

---

## 7. 调参建议

- **BM25 权重(`weights[bm25_idx]`)**:数据量大、专业术语多 → 调到 0.4 ~ 0.5
- **向量权重(`weights[vector_idx]`)**:同义改写多 → 调到 0.7 ~ 0.8
- **RRF k**:保持默认 60(论文值)
- **top_k 召回宽度**:每个 retriever 内部 top_k 至少 = `2 × 最终 top_k`,防止融合后过少

---

## 8. 已知限制

- 内置 `tokenize()` 是无字典的简单 CJK 单字切分,工业场景建议替换为 `jieba`
- `TFIDFKeywordRetriever` 没有"短语查询"能力(只匹配单字 / 单词)
- `MilvusHybridRetriever` 依赖 Milvus 2.4+ 的稀疏字段支持
- 增量添加节点时各 retriever 都会重建索引(适合 < 5w 节点)
