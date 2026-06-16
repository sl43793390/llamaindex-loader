"""
advancedSplitter.semantic
~~~~~~~~~~~~~~~~~~~~~~~~~~
语义切分(Semantic Chunking)。

原理:
    1. 把文本按句子切分;
    2. 逐句算 Embedding;
    3. 相邻句子若 Embedding 余弦距离突然变大(> 阈值),
       就在那里断一刀 —— 即"语义换题"的位置。

何时该用:
    - 文档主题常变(产品手册、FAQ、对话日志)
    - 硬切(定长句子数)会在主题切换处切坏,需要自适应

依赖:llama_index.core.node_parser.SemanticSplitterNodeParser
"""
from __future__ import annotations

from typing import List, Optional

from llama_index.core import Document
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode

from .utils import get_embed_model, make_node_id


def split_semantic(
    documents: List[Document],
    embed_model: Optional[BaseEmbedding] = None,
    buffer_size: int = 1,
    breakpoint_percentile_threshold: int = 95,
    embed_model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> List[BaseNode]:
    """
    基于 Embedding 相似度的语义切分。

    Args:
        documents: 输入 Document 列表。
        embed_model: 已构造好的 Embedding 模型。None 时按 config 自动构造。
        buffer_size: 局部平滑窗口大小(把当前句与前后 N 句的 Embedding 平均,
                     减少偶发抖动)。1 表示仅当前句。
        breakpoint_percentile_threshold: 切分阈值百分位。
            95 = 仅在 top-5% 距离最大的地方切(更保守)
            80 = 在 top-20% 处切(更激进,块更多更短)
        embed_model_name / api_key / api_base: 自定义 Embedding。

    Returns:
        BaseNode 列表。
    """
    if embed_model is None:
        embed_model = get_embed_model(
            model=embed_model_name,
            api_key=api_key,
            api_base=api_base,
        )

    parser = SemanticSplitterNodeParser(
        embed_model=embed_model,
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
    )
    nodes = parser.get_nodes_from_documents(documents)

    for i, n in enumerate(nodes):
        if not n.node_id:
            n.node_id = make_node_id("sem", i, n.get_content())

    return nodes


def split_semantic_dual(
    documents: List[Document],
    embed_model: Optional[BaseEmbedding] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    **kwargs,
) -> List[BaseNode]:
    """
    语义切分 + 句子窗口双策略(SemanticDoubleMergingSplitter)。

    先按语义切;切完如果某段太长,再按句子窗口二次切;反过来如果太短,就合并。
    适合长 + 主题不均的文档。
    """
    from llama_index.core.node_parser import (
        SemanticDoubleMergingSplitterNodeParser,
        SentenceSplitter,
    )

    if embed_model is None:
        embed_model = get_embed_model()

    sentence_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    parser = SemanticDoubleMergingSplitterNodeParser(
        embed_model=embed_model,
        sentence_splitter=sentence_parser,
        **kwargs,
    )
    nodes = parser.get_nodes_from_documents(documents)

    for i, n in enumerate(nodes):
        if not n.node_id:
            n.node_id = make_node_id("sem2", i, n.get_content())

    return nodes
