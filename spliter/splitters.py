"""
spliter.splitters
~~~~~~~~~~~~~~~~~~~
封装 LlamaIndex 提供的各类文本切分器,统一返回 ``List[BaseNode]``。

支持的切分方式
--------------
- split_by_sentence      : 句子切分(默认,推荐)
- split_by_token         : Token 切分
- split_simple           : 简单定长切分
- split_sentence_window  : 句子窗口切分(检索时还原上下文)
- split_markdown         : 按 Markdown 标题层级切分
- split_html             : 按 HTML 标签结构切分
- split_json             : 按 JSON 嵌套结构切分
- split_code             : 按代码语义(类/函数等)切分
- split_semantic         : 基于 Embedding 相似度的语义切分

可使用 ``auto_split`` 按 ``doc_type`` 自动选择切分器。
"""
from typing import List, Optional

from llama_index.core import Document
from llama_index.core.node_parser import (
    SentenceSplitter,
    TokenTextSplitter,
    SentenceWindowNodeParser,
    SimpleNodeParser,
    MarkdownNodeParser,
    HTMLNodeParser,
    JSONNodeParser,
    CodeSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import BaseNode


# ============================================================
# 1. 句子切分(默认,推荐)
# ============================================================
def split_by_sentence(
    documents: List[Document],
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
) -> List[BaseNode]:
    """
    按句子切分,保留段落结构。

    Args:
        documents: 待切分的 Document 列表。
        chunk_size: 单个块的目标字符数。
        chunk_overlap: 相邻块之间的重叠字符数。

    Returns:
        BaseNode 列表。
    """
    parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return parser.get_nodes_from_documents(documents)


# ============================================================
# 2. Token 切分
# ============================================================
def split_by_token(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    separator: str = " ",
) -> List[BaseNode]:
    """
    按 token 切分,适合对接 LLM 的 token 上限约束。

    Args:
        documents: 待切分的 Document 列表。
        chunk_size: 单块 token 数。
        chunk_overlap: 相邻块重叠 token 数。
        separator: 备选分割字符。

    Returns:
        BaseNode 列表。
    """
    parser = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=separator,
    )
    return parser.get_nodes_from_documents(documents)


# ============================================================
# 3. 简单定长切分
# ============================================================
def split_simple(
    documents: List[Document],
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
    separator: str = " ",
) -> List[BaseNode]:
    """
    按 separator 简单切分(轻量、速度快)。

    Args:
        documents: 待切分的 Document 列表。
        chunk_size: 单块字符数。
        chunk_overlap: 相邻块重叠字符数。
        separator: 分割字符。

    Returns:
        BaseNode 列表。
    """
    parser = SimpleNodeParser.from_defaults(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=separator,
    )
    return parser.get_nodes_from_documents(documents)


# ============================================================
# 4. 句子窗口切分
# ============================================================
def split_sentence_window(
    documents: List[Document],
    window_size: int = 3,
    window_metadata_key: str = "window",
    original_text_metadata_key: str = "original_sentence",
) -> List[BaseNode]:
    """
    句子窗口切分:以单句为节点,周围 N 句作为上下文窗口,
    检索时使用 MetadataReplacementPostProcessor 把节点替换为窗口。

    Args:
        documents: 待切分的 Document 列表。
        window_size: 上下文窗口大小(前后各取多少句)。
        window_metadata_key: 窗口内容写入的元数据 key。
        original_text_metadata_key: 原句内容写入的元数据 key。

    Returns:
        BaseNode 列表。
    """
    parser = SentenceWindowNodeParser.from_defaults(
        window_size=window_size,
        window_metadata_key=window_metadata_key,
        original_text_metadata_key=original_text_metadata_key,
    )
    return parser.get_nodes_from_documents(documents)


# ============================================================
# 5. Markdown 结构化切分
# ============================================================
def split_markdown(documents: List[Document]) -> List[BaseNode]:
    """
    基于 Markdown 标题层级切分,每个章节对应一个节点。

    Args:
        documents: 待切分的 Document 列表。

    Returns:
        BaseNode 列表。
    """
    parser = MarkdownNodeParser()
    return parser.get_nodes_from_documents(documents)


# ============================================================
# 6. HTML 标签结构化切分
# ============================================================
def split_html(documents: List[Document]) -> List[BaseNode]:
    """
    按 HTML 标签结构切分。

    Args:
        documents: 待切分的 Document 列表。

    Returns:
        BaseNode 列表。
    """
    parser = HTMLNodeParser()
    return parser.get_nodes_from_documents(documents)


# ============================================================
# 7. JSON 层级切分
# ============================================================
def split_json(documents: List[Document]) -> List[BaseNode]:
    """
    按 JSON 嵌套结构切分。

    Args:
        documents: 待切分的 Document 列表。

    Returns:
        BaseNode 列表。
    """
    parser = JSONNodeParser()
    return parser.get_nodes_from_documents(documents)


# ============================================================
# 8. 代码切分
# ============================================================
def split_code(
    documents: List[Document],
    language: str = "python",
    chunk_lines: int = 40,
    chunk_line_overlap: int = 15,
    max_chars: int = 1500,
) -> List[BaseNode]:
    """
    按代码语义(类/函数等)切分代码文件,基于 tree-sitter。

    Args:
        documents: 待切分的 Document 列表。
        language: 语言类型,例如 ``"python"`` / ``"javascript"``。
        chunk_lines: 单块最大行数。
        chunk_line_overlap: 相邻块重叠行数。
        max_chars: 单块最大字符数。

    Returns:
        BaseNode 列表。
    """
    parser = CodeSplitter(
        language=language,
        chunk_lines=chunk_lines,
        chunk_line_overlap=chunk_line_overlap,
        max_chars=max_chars,
    )
    return parser.get_nodes_from_documents(documents)


# ============================================================
# 9. 语义切分(需 Embedding 模型)
# ============================================================
def split_semantic(
    documents: List[Document],
    embed_model: BaseEmbedding,
    buffer_size: int = 1,
    breakpoint_percentile_threshold: int = 95,
) -> List[BaseNode]:
    """
    基于语义相似度切分:在语义变化最大的位置断开。

    Args:
        documents: 待切分的 Document 列表。
        embed_model: Embedding 模型实例(用于计算句向量)。
        buffer_size: 参与比较的句子窗口大小。
        breakpoint_percentile_threshold: 断点百分位阈值,
            越大 -> 切分粒度越细。

    Returns:
        BaseNode 列表。
    """
    parser = SemanticSplitterNodeParser.from_defaults(
        embed_model=embed_model,
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
    )
    return parser.get_nodes_from_documents(documents)


# ============================================================
# 10. 通用入口(按 doc_type 自动选择切分器)
# ============================================================
def auto_split(
    documents: List[Document],
    doc_type: str = "text",
) -> List[BaseNode]:
    """
    根据 doc_type 自动选择切分器。

    Args:
        documents: 待切分的 Document 列表。
        doc_type: 文档类型,可选:
            ``"text"`` / ``"sentence"`` / ``"token"`` /
            ``"simple"`` / ``"window"`` /
            ``"markdown"`` / ``"html"`` / ``"json"`` / ``"code"``。

    Returns:
        BaseNode 列表。

    Raises:
        ValueError: doc_type 不在受支持列表中时抛出。
    """
    mapping = {
        "text": split_by_sentence,
        "sentence": split_by_sentence,
        "token": split_by_token,
        "simple": split_simple,
        "window": split_sentence_window,
        "markdown": split_markdown,
        "html": split_html,
        "json": split_json,
        "code": split_code,
    }
    if doc_type not in mapping:
        raise ValueError(f"不支持的 doc_type: {doc_type}")
    return mapping[doc_type](documents)
