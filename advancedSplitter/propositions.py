"""
advancedSplitter.propositions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
命题提取(Proposition Extraction)。

Agentic Chunking 的核心前置步骤(出自
"Agentic Chunking: Semi-Automated Text Chunking" 一文):

    1. 让 LLM 把段落拆成"原子命题"(atomic propositions):
       每条命题是 1 句可独立成立的陈述,包含主谓宾和必要修饰。
    2. 每条命题单独算 Embedding。
    3. (在 agentic_chunking.py 中)再让 LLM 把"主题相近"的命题聚成块。

这样做的优点:
    - 检索时不会把"苹果和橙子的比较"和"苹果的营养价值"混在一起;
    - 命题粒度比句子细、比段落粗,适合精排。

注意:
    - 这是 Agentic Chunking 的"原料准备"步骤;
      想直接用聚类得到的块,见 :mod:`agentic_chunking`。
    - 单独使用也可,命题列表本身就是高质量的索引单元。
"""
from __future__ import annotations

import json
import re
from typing import List, Optional, Sequence

from llama_index.core import Document
from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode, TextNode

from .utils import get_llm, make_node_id


PROPOSITION_PROMPT = """你是文档预处理助手。请把下面这段文字拆成"原子命题"(atomic propositions)。

要求:
- 每条命题是一句完整、可独立成立的陈述(主谓宾清晰);
- 拆分粒度:把含多个事实的句子拆成多条;
- 不要修改原意,不要添加解释;
- 保持原文语言(中文保持中文,英文保持英文);
- 如果某句是标题、列表标记、空白,跳过即可。

输出严格 JSON 数组,每项是字符串:
["命题 1", "命题 2", "命题 3", ...]

【待处理段落】
{paragraph}
"""


def extract_propositions(
    paragraphs: Sequence[str],
    llm: Optional[LLM] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.0,
) -> List[List[str]]:
    """
    对每个段落调用 LLM,提取命题列表。

    Args:
        paragraphs: 段落字符串序列(一般来自 :func:`utils.split_sentences` 后再按段聚合)。
        llm: 已构造好的 LLM;None 时按 config 自动构造。
        model / api_key / api_base / temperature: 自定义 LLM。

    Returns:
        与 paragraphs 等长的 List[List[str]]:每个元素是该段落的命题列表。
        LLM 失败 / 解析失败时,退化为把整段当成 1 个命题。
    """
    if llm is None:
        llm = get_llm(
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=2048,
        )

    out: List[List[str]] = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            out.append([])
            continue
        if len(p) < 30:  # 太短,直接当命题
            out.append([p])
            continue
        prompt = PROPOSITION_PROMPT.format(paragraph=p)
        try:
            resp = llm.complete(prompt)
            props = _parse_propositions_json(resp.text)
        except Exception:
            props = []
        if not props:
            props = [p]  # fallback:整段作为 1 个命题
        out.append(props)
    return out


def _parse_propositions_json(raw: str) -> List[str]:
    """解析 LLM 返回的 JSON 数组字符串,容忍围栏。"""
    if not raw:
        return []
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        data = json.loads(raw[start: end + 1])
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [str(x).strip() for x in data if str(x).strip()]


def propositions_to_nodes(
    propositions: List[List[str]],
    documents: List[Document],
    embed_model=None,
) -> List[BaseNode]:
    """
    把命题展开成 TextNode(每条命题一个 node)。

    Args:
        propositions: ``[doc_idx][prop_idx] -> str``,
                      通常由 :func:`extract_propositions` 配合多文档生成。
        documents:   原 Document 列表,用来取 metadata。
        embed_model: 若给,会立刻给每个 node 算 Embedding 并挂到 ``node.embedding`` 上。

    Returns:
        List[TextNode]
    """
    nodes: List[TextNode] = []
    counter = 0
    for doc_idx, props in enumerate(propositions):
        base_meta = documents[doc_idx].metadata if doc_idx < len(documents) else {}
        for p_idx, prop in enumerate(props):
            node = TextNode(
                text=prop,
                id_=make_node_id("prop", counter, prop),
                metadata={
                    **base_meta,
                    "split_method": "proposition",
                    "doc_index": doc_idx,
                    "proposition_index": p_idx,
                },
            )
            nodes.append(node)
            counter += 1

    if embed_model is not None and nodes:
        # 批量算 embedding(LlamaIndex 会在 .get_text_embedding_batch 内做并发)
        texts = [n.get_content() for n in nodes]
        try:
            embeddings = embed_model.get_text_embedding_batch(texts, show_progress=False)
            for n, e in zip(nodes, embeddings):
                n.embedding = e
        except Exception:
            pass
    return nodes
