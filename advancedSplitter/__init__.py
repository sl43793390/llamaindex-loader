"""
advancedSplitter
~~~~~~~~~~~~~~~~
RAG 高级切分方法集合。

    - parent_child        : 父子切分(HierarchicalNodeParser + AutoMergingRetriever)
    - semantic            : 语义切分(SemanticSplitterNodeParser)
    - llm_chunking        : LLM-based 切分(自实现 LLMSemanticChunker)
    - propositions        : 命题提取(Agentic Chunking 的原料步骤)
    - agentic_chunking    : Agentic Chunking(命题 → 主题聚类 → 合并)
    - utils               : 共享的 Embedding / LLM 客户端 & 文本工具

入口示例::

    from advancedSplitter import (
        split_parent_child,
        split_semantic,
        split_by_llm,
        split_agentic,
    )

    docs = [Document(text=...)]
    nodes = split_semantic(docs, breakpoint_percentile_threshold=80)
"""
from .parent_child import (
    split_parent_child,
    DEFAULT_CHUNK_SIZES,
    build_storage_context_from_hierarchy,
    auto_merging_retrieve,
)
from .semantic import split_semantic, split_semantic_dual
from .llm_chunking import split_by_llm
from .propositions import extract_propositions, propositions_to_nodes
from .agentic_chunking import split_agentic

__all__ = [
    # parent_child
    "split_parent_child",
    "DEFAULT_CHUNK_SIZES",
    "build_storage_context_from_hierarchy",
    "auto_merging_retrieve",
    # semantic
    "split_semantic",
    "split_semantic_dual",
    # llm_chunking
    "split_by_llm",
    # propositions
    "extract_propositions",
    "propositions_to_nodes",
    # agentic
    "split_agentic",
]
