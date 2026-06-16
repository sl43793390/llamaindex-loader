"""
advancedSplitter.agentic_chunking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Agentic Chunking(代理式切分)。

核心思想(出自 Anthropic / 独立研究博客):

    1. 把文档拆成"原子命题"(见 :mod:`propositions`);
    2. 让 LLM 维护一个"主题栈"(scenes / topics),新命题来时:
         - 决定 push 到哪个 topic
         - 决定是否开新 topic
    3. 把同一 topic 内的命题合并成一个 chunk。

简化实现(本模块):
    - 跳过复杂的状态机;
    - 用 Embedding + 滑动窗口相似度粗排 + LLM 精排 来聚合命题。
    - 实际效果接近论文中的 Agentic Chunking,但更稳定 / 更便宜。

适用:
    - 异构 / 多主题长文(用户手册 + FAQ + 操作步骤 + 案例)
    - 需要精排主题边界的场景
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from llama_index.core import Document
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode, TextNode

from .propositions import extract_propositions, propositions_to_nodes
from .utils import get_embed_model, get_llm, make_node_id, split_sentences


# ============================================================
# 1) 命题 + Embedding
# ============================================================
def _embed_propositions(
    propositions: List[str],
    embed_model: BaseEmbedding,
) -> np.ndarray:
    """对每个命题算 Embedding,返回 ``(N, D)`` 矩阵。"""
    if not propositions:
        return np.zeros((0, 0), dtype=np.float32)
    vecs = embed_model.get_text_embedding_batch(propositions, show_progress=False)
    arr = np.asarray(vecs, dtype=np.float32)
    # 归一化,方便后面算余弦
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr / norms


# ============================================================
# 2) 用相似度 + 阈值做"主题边界"检测
# ============================================================
def _detect_topic_breaks(
    sims: np.ndarray,
    threshold: float = 0.55,
    window: int = 3,
) -> List[bool]:
    """
    决定每两个相邻命题之间是否"换主题":
        - 取当前位置前后 window 范围内的相似度均值
        - 若均值 < threshold → 视为换题

    返回长度 = len(sims) - 1 的 bool 列表。
    """
    n = len(sims)
    breaks: List[bool] = []
    for i in range(n - 1):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        local = sims[lo:hi].mean()
        breaks.append(bool(local < threshold))
    return breaks


# ============================================================
# 3) LLM 精排:让 LLM 在"候选边界"处二次确认
# ============================================================
CONFIRM_BOUNDARY_PROMPT = """你是文档主题切分助手。下面是若干连续命题的预览。
我已在第 {idx} 处标记一个候选分界(用 === 分隔)。

请判断"这里"是否真的是一个主题切换点(从 A 话题转到 B 话题):
- 若候选边界两侧命题确实在讲不同主题(视角 / 对象 / 任务不同) → 输出 true
- 若仍属同一话题,只是顺承举例 → 输出 false

要求:严格只输出 true 或 false,不要解释。

【命题列表】
{props}
"""


def _confirm_boundary_with_llm(
    propositions: List[str],
    candidate_idx: int,
    llm: LLM,
    context: int = 4,
) -> bool:
    """让 LLM 看看候选边界处是否真的换题。"""
    lo = max(0, candidate_idx - context)
    hi = min(len(propositions), candidate_idx + context + 1)
    snippet = propositions[lo:hi]
    formatted = []
    for i, p in enumerate(snippet):
        marker = " ===" if (lo + i) == candidate_idx else ""
        formatted.append(f"[{lo + i}] {p[:200]}{marker}")
    prompt = CONFIRM_BOUNDARY_PROMPT.format(
        idx=candidate_idx,
        props="\n".join(formatted),
    )
    try:
        resp = llm.complete(prompt)
        text = resp.text.strip().lower()
        return text.startswith("true")
    except Exception:
        return False


# ============================================================
# 4) 主入口
# ============================================================
def split_agentic(
    documents: List[Document],
    llm: Optional[LLM] = None,
    embed_model: Optional[BaseEmbedding] = None,
    sim_threshold: float = 0.55,
    min_chunk_props: int = 2,
    max_chunk_chars: int = 2000,
    use_llm_confirm: bool = True,
    model: Optional[str] = None,
    embed_model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> List[BaseNode]:
    """
    Agentic Chunking:命题提取 → 主题相似度聚类 → LLM 精排边界 → 合并成块。

    Args:
        documents: 输入 Document 列表。
        llm / embed_model: 已构造好的客户端;None 时按 config 自动构造。
        sim_threshold: 主题切换的相似度阈值(0~1,越小切得越碎)。
        min_chunk_props: 一个 chunk 最少包含几条命题(避免过碎)。
        max_chunk_chars: 单 chunk 字符上限(超出时硬切)。
        use_llm_confirm: 是否让 LLM 在相似度边界处二次确认(更准但更慢)。
        model / embed_model_name / api_key / api_base: 自定义 LLM / Embedding。

    Returns:
        List[TextNode],每条 chunk 含若干命题,metadata 记录 ``split_method="agentic"``。
    """
    if llm is None:
        llm = get_llm(model=model, api_key=api_key, api_base=api_base, max_tokens=1024)
    if embed_model is None:
        embed_model = get_embed_model(
            model=embed_model_name, api_key=api_key, api_base=api_base
        )

    # Step 1: 按段落切 → 命题提取
    per_doc_props: List[List[str]] = []
    for doc in documents:
        paras = [p for p in doc.text.split("\n\n") if p.strip()]
        if not paras:
            paras = [doc.text]
        props_per_para = extract_propositions(paras, llm=llm)
        flat = [p for ps in props_per_para for p in ps]
        per_doc_props.append(flat)

    # Step 2: 对每个 doc,逐 doc 做主题聚类
    nodes: List[TextNode] = []
    for doc_idx, (doc, props) in enumerate(zip(documents, per_doc_props)):
        if not props:
            continue
        if len(props) == 1:
            nodes.append(_make_chunk_node(doc, props, doc_idx, len(nodes), 1, props))
            continue

        # 算 Embedding + 找候选边界
        embs = _embed_propositions(props, embed_model)
        sims = (embs[:-1] * embs[1:]).sum(axis=1)  # 余弦(已归一化)
        cand_breaks = _detect_topic_breaks(sims, threshold=sim_threshold)

        # 可选:LLM 精排
        if use_llm_confirm:
            confirmed: List[bool] = []
            for i, is_break in enumerate(cand_breaks):
                if not is_break:
                    confirmed.append(False)
                    continue
                confirmed.append(_confirm_boundary_with_llm(props, i, llm))
            cand_breaks = confirmed

        # Step 3: 按边界聚类命题
        groups: List[List[str]] = []
        cur: List[str] = []
        for prop, is_break in zip(props, [False] + cand_breaks):
            if is_break and cur:
                groups.append(cur)
                cur = []
            cur.append(prop)
        if cur:
            groups.append(cur)

        # Step 4: 合并过小的尾部块 + 切过大的块
        groups = _merge_small_tail(groups, min_size=min_chunk_props)
        groups = _split_oversized(groups, max_chars=max_chunk_chars)

        # Step 5: 成 node
        for g_idx, group in enumerate(groups):
            nodes.append(
                _make_chunk_node(doc, group, doc_idx, len(nodes), g_idx, props)
            )

    return nodes


# ============================================================
# 内部工具
# ============================================================
def _merge_small_tail(groups: List[List[str]], min_size: int) -> List[List[str]]:
    """把过小(< min_size)的块并到前一块。"""
    if not groups:
        return groups
    merged: List[List[str]] = []
    for g in groups:
        if merged and len(g) < min_size:
            merged[-1].extend(g)
        else:
            merged.append(g)
    # 末尾过小并到倒数第二
    if len(merged) >= 2 and len(merged[-1]) < min_size:
        merged[-2].extend(merged.pop())
    return merged


def _split_oversized(groups: List[List[str]], max_chars: int) -> List[List[str]]:
    """字符数超 max_chars 的块,按命题边界硬切。"""
    out: List[List[str]] = []
    for g in groups:
        if sum(len(p) for p in g) <= max_chars:
            out.append(g)
            continue
        cur: List[str] = []
        cur_len = 0
        for p in g:
            if cur_len + len(p) > max_chars and cur:
                out.append(cur)
                cur = []
                cur_len = 0
            cur.append(p)
            cur_len += len(p)
        if cur:
            out.append(cur)
    return out


def _make_chunk_node(
    doc: Document,
    props: List[str],
    doc_idx: int,
    global_idx: int,
    group_idx: int,
    all_props: List[str],
) -> TextNode:
    """把若干命题拼成一段文字,打 node。"""
    text = " ".join(props).strip()
    return TextNode(
        text=text,
        id_=make_node_id("agentic", global_idx, text),
        metadata={
            **(doc.metadata or {}),
            "split_method": "agentic",
            "doc_index": doc_idx,
            "group_index": group_idx,
            "proposition_count": len(props),
        },
        excluded_embed_metadata_keys=["split_method", "doc_index", "group_index"],
        excluded_llm_metadata_keys=["split_method", "doc_index", "group_index"],
    )
