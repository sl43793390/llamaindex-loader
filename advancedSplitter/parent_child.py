"""
advancedSplitter.parent_child
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
父子切分(Hierarchical / Auto-Merging Retrieval)。

原理:
    1. 用 HierarchicalNodeParser 把每个 Document 切成多层节点:
        - L0:最大粒度(整段 / 整节)
        - L1:中等粒度
        - L2:最小粒度(叶子,用于检索)
    2. 检索时先用叶子(L2)匹配,若多个相邻叶子同属一个 L1 父节点,
       且都相关,则把整个 L1 节点返回 → 上下文更完整。

何时该用:
    - 文档有明显段落 / 章节结构(报告、论文、长文)
    - 单块不够上下文,整段又太大

依赖:llama_index.core.node_parser.HierarchicalNodeParser
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import BaseNode, TextNode

from .utils import get_embed_model, make_node_id


# 默认三层粒度:大块 2048、中块 512、小块 128
# 注意:每个 chunk_size 必须是后一个的整数倍,否则 HierarchicalNodeParser 会报 warning
DEFAULT_CHUNK_SIZES: Tuple[int, ...] = (2048, 512, 128)


def split_parent_child(
    documents: List[Document],
    chunk_sizes: Optional[Sequence[int]] = None,
    chunk_overlap: int = 20,
) -> Tuple[List[BaseNode], List[BaseNode]]:
    """
    父子切分。

    Args:
        documents: 输入 Document 列表。
        chunk_sizes: 从大到小的粒度序列。默认 ``(2048, 512, 128)``。
        chunk_overlap: 相邻块重叠字符数。

    Returns:
        ``(all_nodes, leaf_nodes)``:
            - ``all_nodes``    : 全部层级的节点(用于构建存储,父节点也入库)
            - ``leaf_nodes``   : 最细粒度节点(用于初始检索)

    Example:
        >>> all_nodes, leaves = split_parent_child(docs)
        >>> # 配合 AutoMergingRetriever 检索时,父节点会自动合并进上下文
    """
    sizes = tuple(sorted(chunk_sizes or DEFAULT_CHUNK_SIZES, reverse=True))

    parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=list(sizes),
        chunk_overlap=chunk_overlap,
    )

    all_nodes = parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(all_nodes)

    # 给 id 加 prefix 便于排查
    for i, n in enumerate(all_nodes):
        if not n.node_id:
            n.node_id = make_node_id("hier", i, n.get_content())

    return all_nodes, leaf_nodes


def build_storage_context_from_hierarchy(
    all_nodes: List[BaseNode],
    embed_model=None,
):
    """
    把层级节点包装成 ``StorageContext``,供 VectorStoreIndex 使用。

    Returns:
        ``(storage_context, leaf_index_node_ids)``
    """
    from llama_index.core.storage import StorageContext
    from llama_index.core.indices.utils import NodeIdStorageContextPair  # noqa: F401

    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_nodes(all_nodes)

    leaf_ids = {n.node_id for n in get_leaf_nodes(all_nodes)}
    return storage_context, leaf_ids


def auto_merging_retrieve(
    query_engine_or_retriever,
    query: str,
):
    """
    便捷封装:用 AutoMergingRetriever 包一层 index.as_retriever。
    库已经提供了现成的 AutoMergingRetriever,这里只是包装一下示例。
    """
    from llama_index.core.retrievers import AutoMergingRetriever

    base = query_engine_or_retriever
    if hasattr(base, "as_retriever"):
        base = base.as_retriever(similarity_top_k=6)
    return AutoMergingRetriever(
        base_retriever=base,
        storage_context=base._storage_context,  # noqa: SLF001
        simple_ratio_thresh=0.4,  # 兄弟节点重叠比例阈值
    )
