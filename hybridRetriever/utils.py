"""
hybridRetriever.utils
~~~~~~~~~~~~~~~~~~~~~~~
混合检索共用的工具:
    - 文本分词(中英文混合,简单实现)
    - 倒排索引 / TF-IDF
    - 检索结果融合:RRF、加权和、Max
    - 节点工具函数

设计原则:
    - 全部纯 Python + llama_index 现成类型,不引第三方
    - 失败降级:分词失败 → 退化为字符切分
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle, TextNode


# ============================================================
# 分词
# ============================================================
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
# 中文常见停用词(精简版,够做 BM25)
_CN_STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "那", "些", "里", "为", "与", "及", "或", "但", "而", "以", "把",
    "被", "让", "使", "由", "从", "向", "到", "对", "跟", "比", "等", "等等",
}
_EN_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "by", "for", "with", "to", "of", "and", "or", "but",
    "this", "that", "these", "those", "it", "its", "as", "if", "or",
}


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    中英文混合分词。

    规则:
        - 连续 CJK 字符 → 按单字切(无字典的简单做法)
        - 连续 ASCII 字母/数字 → 当作一个 token(英文词 / 数字)
        - 长度 < 2 的 token 去掉
        - 停用词可选去掉

    注意:
        这是简单实现,够用于 BM25 演示 / 中等规模库(< 10w 文档)。
        工业场景建议用 jieba + 自定义词典。
    """
    if not text:
        return []
    tokens: List[str] = []
    # 1) ASCII 单词
    for w in _WORD_RE.findall(text):
        w_low = w.lower()
        if len(w_low) < 2:
            continue
        if remove_stopwords and w_low in _EN_STOPWORDS:
            continue
        tokens.append(w_low)
    # 2) CJK 单字
    for ch in _CJK_RE.findall(text):
        if remove_stopwords and ch in _CN_STOPWORDS:
            continue
        tokens.append(ch)
    return tokens


# ============================================================
# 倒排索引 / TF-IDF
# ============================================================
def build_inverted_index(
    docs: Sequence[Sequence[str]],
) -> Tuple[Dict[str, List[int]], List[Counter]]:
    """
    构建倒排索引。

    Args:
        docs: token 列表的列表,docs[i] = 第 i 个文档的 tokens。

    Returns:
        ``(inv, tf_list)``:
            - ``inv[term] = [doc_idx, ...]``(倒排表,doc 出现 term 的下标)
            - ``tf_list[i] = Counter``(第 i 个文档的词频)
    """
    inv: Dict[str, List[int]] = defaultdict(list)
    tf_list: List[Counter] = []
    for i, tokens in enumerate(docs):
        c = Counter(tokens)
        tf_list.append(c)
        for term in c:
            inv[term].append(i)
    return dict(inv), tf_list


def tfidf_score(
    query_tokens: Sequence[str],
    inv: Dict[str, List[int]],
    tf_list: Sequence[Counter],
    n_docs: int,
) -> Counter:
    """
    用 TF-IDF 算每个文档对 query 的相关度。
    返回 ``Counter{doc_idx: score}``。
    """
    scores: Counter = Counter()
    if not query_tokens or n_docs == 0:
        return scores
    df = {term: len(inv.get(term, [])) for term in set(query_tokens)}
    for term in query_tokens:
        if term not in inv:
            continue
        # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
        idf = math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1)
        for doc_idx in inv[term]:
            tf = tf_list[doc_idx].get(term, 0)
            # 归一化 tf
            doc_len = sum(tf_list[doc_idx].values()) or 1
            scores[doc_idx] += idf * (tf * (1 + math.log(tf)) / doc_len)
    return scores


# ============================================================
# 检索结果融合
# ============================================================
def rrf_fuse(
    ranked_lists: Sequence[Sequence[NodeWithScore]],
    k: int = 60,
    top_k: int = 10,
) -> List[NodeWithScore]:
    """
    Reciprocal Rank Fusion(论文 Cormack et al. 2009)。

    score(node) = Σ 1 / (k + rank_i)

    Args:
        ranked_lists: 多个 retriever 返回的按 score 降序的 NodeWithScore 列表。
        k: 平滑常数(论文推荐 60)。
        top_k: 返回前 N 个。

    Returns:
        融合 + 按 RRF 分数降序的新 NodeWithScore 列表。
    """
    score_sum: Dict[str, float] = defaultdict(float)
    node_map: Dict[str, NodeWithScore] = {}
    for ranked in ranked_lists:
        for rank, nws in enumerate(ranked, 1):
            nid = nws.node.node_id
            score_sum[nid] += 1.0 / (k + rank)
            # 用第一次出现的 node 实体(避免重复)
            if nid not in node_map:
                node_map[nid] = nws
    # 按 RRF 分数排序
    sorted_ids = sorted(score_sum, key=lambda x: -score_sum[x])[:top_k]
    return [
        NodeWithScore(node=node_map[nid].node, score=score_sum[nid], )
        for nid in sorted_ids
    ]


def weighted_fuse(
    ranked_lists: Sequence[Sequence[NodeWithScore]],
    weights: Sequence[float],
    top_k: int = 10,
    normalize: bool = True,
) -> List[NodeWithScore]:
    """
    加权融合。

    Args:
        ranked_lists: 同 rrf_fuse。
        weights: 每个 retriever 的权重(和 ranked_lists 等长)。
        top_k: 返回前 N。
        normalize: True 时把每个 retriever 的分数归一化到 [0, 1](否则按原 score 加权)。
    """
    assert len(ranked_lists) == len(weights)
    if not ranked_lists:
        return []
    score_sum: Dict[str, float] = defaultdict(float)
    node_map: Dict[str, NodeWithScore] = {}
    for ranked, w in zip(ranked_lists, weights):
        if not ranked:
            continue
        scores = [n.score or 0.0 for n in ranked]
        if normalize and max(scores) > min(scores):
            lo, hi = min(scores), max(scores)
            span = hi - lo or 1.0
            for nws, s in zip(ranked, scores):
                s_norm = (s - lo) / span
                score_sum[nws.node.node_id] += w * s_norm
                if nws.node.node_id not in node_map:
                    node_map[nws.node.node_id] = nws
        else:
            for nws in ranked:
                score_sum[nws.node.node_id] += w * (nws.score or 0.0)
                if nws.node.node_id not in node_map:
                    node_map[nws.node.node_id] = nws
    sorted_ids = sorted(score_sum, key=lambda x: -score_sum[x])[:top_k]
    return [
        NodeWithScore(node=node_map[nid].node, score=score_sum[nid])
        for nid in sorted_ids
    ]


# ============================================================
# 节点工具
# ============================================================
def nodes_to_texts(nodes: Iterable[BaseNode]) -> List[str]:
    """把节点取文本。"""
    return [n.get_content() for n in nodes]


def attach_scores(
    nodes: Sequence[BaseNode],
    scores: Sequence[float],
) -> List[NodeWithScore]:
    """把 BaseNode 列表配上 score 打包成 NodeWithScore。"""
    return [NodeWithScore(node=n, score=float(s)) for n, s in zip(nodes, scores)]


def top_k_by_score(
    nodes_with_score: Sequence[NodeWithScore],
    top_k: int,
) -> List[NodeWithScore]:
    """按 score 降序截前 top_k。"""
    return sorted(nodes_with_score, key=lambda x: -(x.score or 0.0))[:top_k]


def dedup_by_id(
    nodes_with_score: Sequence[NodeWithScore],
) -> List[NodeWithScore]:
    """按 node_id 去重(保留第一次出现 + 最高分)。"""
    seen: Dict[str, NodeWithScore] = {}
    for nws in nodes_with_score:
        nid = nws.node.node_id
        if nid not in seen or (nws.score or 0) > (seen[nid].score or 0):
            seen[nid] = nws
    return list(seen.values())
