"""
advancedSplitter.llm_chunking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LLM-based 切分(LLMSemanticChunker)。

LlamaIndex 0.14 没有内置 LLMSemanticChunker(早期版本有,
0.10+ 后被移除并被改造到 langchain 生态)。
这里用 LlamaIndex 自带的 LLM + JSON 模式,自己实现:

    1. 文本预处理(按段落 / 句子切)
    2. 每 8~10 句一组,送 LLM 判断"是否该在此处断章"
    3. LLM 返回 JSON 数组 [true/false, true/false, ...]
    4. true 处为分块边界

特点:
    - 比纯 Embedding 语义切分更准(能理解主题转折、对话角色变化)
    - 比 Agentic Chunking(命题提取)更便宜(不需要逐句提取)
    - 慢:每批要 1 次 LLM 调用,长文档可能要几十次

适用:
    - 高质量长文(法律、医疗、技术规范)需要按"意群"切
    - 可接受 LLM 调用成本
"""
from __future__ import annotations

import json
import re
from typing import List, Optional, Sequence, Tuple

from llama_index.core import Document
from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode, TextNode

from .utils import get_llm, make_node_id, split_sentences


# ============================================================
# Prompt 模板
# ============================================================
PROMPT_BOUNDARY = """你是一个文档切分助手。下面是一段连续文本,已被切分为若干"句"。

任务:判断相邻两个句子之间是否应该"换章"。
如果语义主题 / 视角 / 角色发生明显变化(比如从介绍跳到具体实现、从原因跳到结果),返回 true;
否则(还在讲同一件事、同一段论述)返回 false。

要求:
- 输出严格 JSON 数组,长度等于句数减 1
- 数组里的每个元素是 true 或 false
- 不要输出任何解释、代码块标记

【句子的标号】
{sentences}

【输出示例】
[false, true, false, true]
"""


def _format_prompt(sentences: Sequence[str]) -> str:
    """把句子列表拼成 prompt 的"标号"段,避免 LLM 误读。"""
    lines = []
    for i, s in enumerate(sentences, 1):
        # 截断过长单句(防御性)
        s_disp = s if len(s) <= 300 else s[:300] + "..."
        lines.append(f"[{i}] {s_disp}")
    return "\n".join(lines)


def _parse_json_array(raw: str, expected_len: int) -> List[bool]:
    """
    解析 LLM 返回的 JSON 数组。容忍以下格式:
        - 纯 JSON:   [true, false, true]
        - 包了 ```json ... ``` 围栏
        - 后面带了多余文字
    长度不够时右侧补 False;多了截断。
    """
    if not raw:
        return [False] * expected_len

    raw = raw.strip()
    # 去围栏
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    # 找第一个 [ 和最后一个 ]
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return [False] * expected_len
    snippet = raw[start: end + 1]
    try:
        data = json.loads(snippet)
    except Exception:
        return [False] * expected_len
    if not isinstance(data, list):
        return [False] * expected_len
    out: List[bool] = []
    for x in data[:expected_len]:
        if isinstance(x, bool):
            out.append(x)
        elif isinstance(x, str) and x.strip().lower() in ("true", "false"):
            out.append(x.strip().lower() == "true")
        else:
            out.append(False)
    while len(out) < expected_len:
        out.append(False)
    return out


# ============================================================
# 主入口
# ============================================================
def split_by_llm(
    documents: List[Document],
    llm: Optional[LLM] = None,
    batch_size: int = 8,
    min_chunk_chars: int = 200,
    max_chunk_chars: int = 2000,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
) -> List[BaseNode]:
    """
    用 LLM 判断"句间是否换章"来切分。

    Args:
        documents: 输入 Document 列表。
        llm: 已构造好的 LLM。None 时按 config 自动构造。
        batch_size: 一次送给 LLM 判定的句子数。越大越省调用次数,
                    但单次 prompt 越长越容易出错。8~12 通常较好。
        min_chunk_chars: 切出来的块如果小于这个字符数,会与下一块合并。
        max_chunk_chars: 切出来的块如果大于这个字符数,会强制在最近的句号处
                         切一刀(避免 LLM 全程判 false 导致超长块)。
        model / api_key / api_base / temperature: 自定义 LLM。

    Returns:
        TextNode 列表,带 ``metadata["split_method"] = "llm"``。
    """
    if llm is None:
        llm = get_llm(
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=1024,
        )

    nodes: List[TextNode] = []

    for doc_idx, doc in enumerate(documents):
        text = doc.text
        sentences = split_sentences(text)
        if not sentences:
            continue
        if len(sentences) == 1:
            nodes.append(_make_node(doc, [sentences[0]], doc_idx, 0))
            continue

        # 1) 滑动窗口 + LLM 判定句间边界
        boundaries = _detect_boundaries(sentences, llm, batch_size)

        # 2) 按边界切块
        raw_chunks: List[List[str]] = []
        cur: List[str] = []
        for sent, is_break in zip[tuple[str, bool]](sentences, [False] + boundaries):
            if is_break and cur:
                raw_chunks.append(cur)
                cur = []
            cur.append(sent)
        if cur:
            raw_chunks.append(cur)

        # 3) 后处理:太小合并 / 太大硬切
        cleaned = _post_process_chunks(
            raw_chunks, min_chunk_chars=min_chunk_chars, max_chunk_chars=max_chunk_chars
        )

        # 4) 打包成 nodes
        for ck_idx, ck in enumerate(cleaned):
            nodes.append(_make_node(doc, ck, doc_idx, ck_idx))

    return nodes


def _detect_boundaries(
    sentences: List[str], llm: LLM, batch_size: int
) -> List[bool]:
    """
    返回长度为 len(sentences) - 1 的 bool 列表,
    boundaries[i] 表示 sentences[i] 与 sentences[i+1] 之间是否换章。
    """
    boundaries: List[bool] = []
    n = len(sentences)
    i = 0
    while i < n - 1:
        batch = sentences[i: i + batch_size + 1]  # 取 batch_size+1 句,产出 batch_size 个边界
        prompt = PROMPT_BOUNDARY.format(sentences=_format_prompt(batch))
        try:
            resp = llm.complete(prompt)
            raw = resp.text
        except Exception:
            raw = ""
        flags = _parse_json_array(raw, expected_len=len(batch) - 1)
        boundaries.extend(flags)
        i += batch_size
    # 截断到 n-1
    return boundaries[: n - 1]


def _post_process_chunks(
    chunks: List[List[str]],
    min_chunk_chars: int,
    max_chunk_chars: int,
) -> List[List[str]]:
    """
    1. 太小(< min_chunk_chars) → 与下一块合并
    2. 还不够 → 与上一块合并
    3. 太大(> max_chunk_chars) → 按句号在中间硬切
    """
    if not chunks:
        return []

    # Step 1: 合并过小的块
    merged: List[List[str]] = []
    for ck in chunks:
        ck_text = "".join(ck)
        if merged and len(ck_text) < min_chunk_chars:
            merged[-1].extend(ck)
        else:
            merged.append(ck)

    # Step 2: 末尾块还小 → 合并到上一个
    if len(merged) >= 2:
        last_text = "".join(merged[-1])
        if len(last_text) < min_chunk_chars:
            merged[-2].extend(merged.pop())

    # Step 3: 切过大的块(在最近的 "。" / "." 处切)
    out: List[List[str]] = []
    for ck in merged:
        if len("".join(ck)) <= max_chunk_chars:
            out.append(ck)
            continue
        # 拆
        cur_text = ""
        cur_sents: List[str] = []
        for s in ck:
            if len(cur_text) + len(s) > max_chunk_chars and cur_sents:
                out.append(cur_sents)
                cur_sents = []
                cur_text = ""
            cur_sents.append(s)
            cur_text += s
        if cur_sents:
            out.append(cur_sents)
    return out


def _make_node(
    doc: Document,
    sentences: List[str],
    doc_idx: int,
    chunk_idx: int,
) -> TextNode:
    text = "".join(sentences).strip()
    return TextNode(
        text=text,
        id_=make_node_id("llm", doc_idx * 10000 + chunk_idx, text),
        metadata={
            **(doc.metadata or {}),
            "split_method": "llm",
            "doc_index": doc_idx,
            "chunk_index": chunk_idx,
        },
        excluded_embed_metadata_keys=["split_method", "doc_index", "chunk_index"],
        excluded_llm_metadata_keys=["split_method", "doc_index", "chunk_index"],
    )
