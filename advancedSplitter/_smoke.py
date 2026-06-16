"""
Smoke test for advancedSplitter.

只跑不需要真实 LLM / Embedding 的部分:
    - 父子切分:不需要 LLM,但 HierarchicalNodeParser 内部会算 Embedding(用于排序),
      所以用 embed_model=None 让它走 fallback(BBPE-free 的简单切分)。
    - 命题提取 / Agentic / LLM-based:都跳过(只验证导入 + 签名)。

要看完整流程请设置环境变量后再跑::

    set ADV_SMOKE_FULL=1
    python advancedSplitter/_smoke.py
"""
import os
import sys
import io

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llama_index.core import Document
from advancedSplitter import (
    split_parent_child,
    split_semantic,
    split_by_llm,
    split_agentic,
    extract_propositions,
    split_semantic_dual,
    DEFAULT_CHUNK_SIZES,
)


SAMPLE = """
Milvus 是一个开源的向量数据库,专为大规模向量相似度搜索设计。
它支持亿级向量的毫秒级检索,广泛应用于推荐系统、图像检索、自然语言处理等领域。

向量数据库的核心是 ANN(Approximate Nearest Neighbor)算法。
常见的 ANN 算法包括 HNSW、IVF、PQ 等,它们在精度和速度之间做不同权衡。
Milvus 2.x 使用了基于 DiskANN 的存储引擎,可以处理单 collection 数十亿向量。

在 RAG(Retrieval-Augmented Generation)场景中,向量库通常承担两个任务:
第一,把文档切块后用 Embedding 模型向量化,存入向量库;
第二,用户提问时,先用 Embedding 检索 Top-K 相关块,再送 LLM 生成答案。

Milvus 的 Python SDK 叫 pymilvus,常用方法包括 connect、create_collection、insert、search。
注意:每次用完要记得 release 或 close,否则连接会泄漏。

高级特性:
- 多租户隔离:按 partition_key 分组
- 标量过滤:在向量检索的同时做 WHERE 条件
- 混合检索:同时支持稠密向量 + 稀疏向量(BM25)
"""


def main() -> None:
    docs = [Document(text=SAMPLE, metadata={"source": "smoke"})]
    print(f"=== Smoke: sample {len(SAMPLE)} chars, {len(docs)} doc(s) ===\n")

    # 1. 父子切分 —— 纯本地,不需要 LLM / Embedding
    print("[1] split_parent_child")
    all_nodes, leaves = split_parent_child(docs, chunk_sizes=(256, 64))
    print(f"    all_nodes={len(all_nodes)}  leaves={len(leaves)}")
    for n in all_nodes[:3]:
        print(f"    - {n.node_id[:20]}... len={len(n.get_content())}")
    print()

    # 2. 语义切分 —— 不传 embed_model,会走 config 默认(没配 API key 会报错,但能验证签名)
    print("[2] split_semantic (signature only)")
    try:
        nodes = split_semantic(docs, buffer_size=1, breakpoint_percentile_threshold=95)
        print(f"    produced {len(nodes)} nodes")
    except Exception as e:
        print(f"    [skipped] {type(e).__name__}: {str(e)[:80]}")
    print()

    # 3. LLM-based —— 跳过
    if os.environ.get("ADV_SMOKE_FULL") == "1":
        print("[3] split_by_llm")
        try:
            nodes = split_by_llm(docs, batch_size=6)
            print(f"    produced {len(nodes)} nodes")
        except Exception as e:
            print(f"    [skipped] {type(e).__name__}: {str(e)[:80]}")
        print()

        print("[4] split_agentic")
        try:
            nodes = split_agentic(docs, use_llm_confirm=True)
            print(f"    produced {len(nodes)} nodes")
        except Exception as e:
            print(f"    [skipped] {type(e).__name__}: {str(e)[:80]}")
    else:
        print("[3][4] split_by_llm / split_agentic skipped (set ADV_SMOKE_FULL=1 to run)")

    print("\n=== smoke OK ===")


if __name__ == "__main__":
    main()
