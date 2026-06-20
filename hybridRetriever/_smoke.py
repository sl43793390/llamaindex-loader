"""
hybridRetriever 冒烟测试。

只跑不依赖 Embedding / LLM / Milvus 服务的部分:
    - 纯 Python 的 BM25、TF-IDF
    - HybridRetriever 走 RRF / Weighted / Concat 三种融合

要看完整流程(含向量、Milvus 混合检索)请配置好 API Key / Milvus 后
把对应代码块的 ``if True`` 打开。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llama_index.core import Document
from llama_index.core.schema import TextNode

from hybridRetriever import (
    BM25Retriever,
    HybridRetriever,
    LlamaIndexKeywordRetriever,
    TFIDFKeywordRetriever,
    VectorRetriever,
    build_hybrid_retriever,
    rrf_fuse,
    tokenize,
    weighted_fuse,
)


# ============== 测试语料 ==============
DOCS = [
    "Milvus 是一个开源的向量数据库,支持亿级向量的毫秒级检索。",
    "BM25 是经典的信息检索算法,基于词频和文档长度归一化,常被用作稀疏检索的 baseline。",
    "LlamaIndex 是一个用于构建 RAG 应用的框架,支持多种数据加载、索引、检索方式。",
    "FAISS 是 Facebook 开发的稠密向量 ANN 检索库,适合单机大规模场景。",
    "Elasticsearch 是基于倒排索引的全文搜索引擎,内置 BM25,适合做 keyword search。",
    "Hybrid Search 通常结合稠密向量检索和稀疏检索,通过 RRF 或加权融合提升召回质量。",
    "Sentence-Transformers 提供多种预训练 embedding 模型,支持中英文双语。",
    "Milvus 2.x 支持多向量字段,可以同时存稠密向量和稀疏向量做混合检索。",
    "RRF(Reciprocal Rank Fusion) 是一种无参数的检索结果融合算法。",
    "Agentic Chunking 使用 LLM 提取命题,然后用 Embedding 聚类成主题块。",
]
NODES = [TextNode(text=t, metadata={"i": i}) for i, t in enumerate(DOCS)]
QUERIES = [
    "Milvus 支持稀疏向量吗?",
    "什么是 BM25?",
    "RAG 怎么混合检索?",
    "LlamaIndex 有什么功能?",
]


def _print_hits(title: str, hits, top: int = 5) -> None:
    print(f"--- {title} ---")
    for h in hits[:top]:
        s = f"{h.score:.4f}" if h.score is not None else "n/a"
        print(f"  [{s}] {h.node.get_content()[:70]}")
    print()


def main() -> None:
    # 1. tokenize
    print("=== [1] tokenize ===")
    print("  Milvus 向量数据库 =>", tokenize("Milvus 向量数据库, 100 万向量")[:15])
    print()

    # 2. BM25
    print("=== [2] BM25Retriever ===")
    bm25 = BM25Retriever.from_nodes(NODES, k1=1.5, b=0.75)
    print("  ", bm25)
    for q in QUERIES[:2]:
        _print_hits(f'BM25 q="{q}"', bm25.retrieve(q, top_k=3))
    print()

    # 3. TF-IDF
    print("=== [3] TFIDFKeywordRetriever ===")
    kw = TFIDFKeywordRetriever.from_nodes(NODES)
    print("  ", kw)
    for q in QUERIES[:2]:
        _print_hits(f'TF-IDF q="{q}"', kw.retrieve(q, top_k=3))
    print()

    # 4. Hybrid (RRF) — BM25 + TFIDF
    print("=== [4] HybridRetriever(method=rrf) ===")
    hybrid_rrf = HybridRetriever(
        retrievers=[bm25, kw],
        method="rrf",
        top_k=5,
    )
    print(hybrid_rrf.describe())
    _print_hits('RRF q="Milvus 混合检索"', hybrid_rrf.retrieve("Milvus 混合检索", top_k=5))
    print()

    # 5. Hybrid (Weighted) — BM25 * 0.7 + TFIDF * 0.3
    print("=== [5] HybridRetriever(method=weighted) ===")
    hybrid_w = HybridRetriever(
        retrievers=[bm25, kw],
        weights=[0.7, 0.3],
        method="weighted",
        top_k=5,
    )
    print(hybrid_w.describe())
    _print_hits('Weighted q="RAG 混合检索"', hybrid_w.retrieve("RAG 混合检索", top_k=5))
    print()

    # 6. Hybrid (Concat)
    print("=== [6] HybridRetriever(method=concat) ===")
    hybrid_c = HybridRetriever(
        retrievers=[bm25, kw],
        method="concat",
        top_k=5,
    )
    _print_hits('Concat q="什么是 BM25?"', hybrid_c.retrieve("什么是 BM25?", top_k=5))
    print()

    # 7. 工具函数:rrf_fuse / weighted_fuse
    print("=== [7] rrf_fuse / weighted_fuse ===")
    a = bm25.retrieve("Milvus", top_k=3)
    b = kw.retrieve("Milvus", top_k=3)
    _print_hits("rrf_fuse(a, b)", rrf_fuse([a, b], top_k=5))
    _print_hits("weighted_fuse(a, b, 0.6/0.4)", weighted_fuse([a, b], [0.6, 0.4], top_k=5))
    print()

    # 8. build_hybrid_retriever 工厂
    print("=== [8] build_hybrid_retriever ===")
    factory = build_hybrid_retriever(
        nodes=NODES,
        retrievers=["bm25", "keyword"],
        weights=[0.6, 0.4],
        method="rrf",
        top_k=5,
    )
    print(factory.describe())
    _print_hits("factory q='FAISS'", factory.retrieve("FAISS"))
    print()

    # 9. LlamaIndex KeywordTable + Vector —— 需要 LLM / Embedding,默认跳过
    if os.environ.get("ADV_SMOKE_FULL") == "1":
        print("=== [9] LlamaIndexKeywordRetriever (full) ===")
        try:
            rk = LlamaIndexKeywordRetriever.from_nodes(NODES, mode="rake")
            _print_hits('rake q="Milvus"', rk.retrieve("Milvus", top_k=3))
        except Exception as e:
            print(f"  [skipped] {type(e).__name__}: {str(e)[:80]}")

        # Vector —— 需要 index + embed_model
        from llama_index.core import VectorStoreIndex, Settings
        from vectorStore.milvus_store import get_embed_model
        Settings.embed_model = get_embed_model()
        index = VectorStoreIndex(nodes=NODES)
        vr = VectorRetriever(index, normalize_score=True)
        _print_hits('vector q="Milvus"', vr.retrieve("Milvus", top_k=3))
    else:
        print("=== [9] LlamaIndexKeywordRetriever / VectorRetriever skipped (set ADV_SMOKE_FULL=1) ===")

    print("=== smoke OK ===")


if __name__ == "__main__":
    main()
