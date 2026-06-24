"""
ragasEvaluator.testset_generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
将切分后的 BaseNode 列表转换为 RAGAS 评估所需的 (question, ground_truth) 问答对。

核心流程
--------
1. 遍历每个 BaseNode，提取文本内容
2. 对过长文本做截断（避免 prompt 超出模型上下文窗口）
3. 调用 LLM，让模型基于文本生成若干问答对
4. 解析 LLM 返回的 JSON（含多种容错处理）

设计要点
--------
- **不依赖 ragas.testset**（ragas 自带的测试集生成器需要额外安装更多模型），
  而是直接复用项目已有的 ``vectorStore.milvus_store.get_llm`` 获取 OpenAI 兼容 LLM。
- **严格 JSON 输出 + 多层容错**：LLM 经常在 JSON 前后加 `````json```` 包裹或多余文字，
  ``_parse_qa_json`` 会自动处理这些情况。
- **单条失败不阻塞**：某个 chunk 生成失败时打印警告并跳过，不影响后续 chunk。

输出格式::

    [
        {"question": "...", "ground_truth": "...", "source_chunk": "..."},
        ...
    ]
"""
from __future__ import annotations

import json
import logging
import re
from typing import List, Optional, Sequence

from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 默认 Prompt：让 LLM 根据一段参考资料生成 N 个问答对
# ---------------------------------------------------------------------------
GENERATE_QA_PROMPT = """你是一个 RAG 评测集构造专家。请根据下面这段参考资料，生成 {n_questions} 个高质量的问答对。

要求:
1. 问题必须仅依据参考资料就能回答（不能引用资料外知识）
2. 问题要多样化：事实型、解释型、对比型各占一定比例
3. ground_truth 必须从参考资料中提取或总结，不能编造
4. 输出严格的 JSON 数组，不要任何额外文字、注释或 Markdown 包裹

参考资料:
\"\"\"
{chunk}
\"\"\"

输出格式（JSON 数组，每项包含 question 和 ground_truth）:
[
  {{\"question\": \"...\", \"ground_truth\": \"...\"}}
]
"""


# ---------------------------------------------------------------------------
# JSON 解析容错
# ---------------------------------------------------------------------------
def _parse_qa_json(raw: str) -> List[dict]:
    """
    从 LLM 原始输出中提取问答对列表。

    兼容以下 LLM 输出格式:
        - 纯 JSON 数组
        - `````json ... `````  Markdown 包裹
        - JSON 前后有废话文字
        - 单引号替代双引号（少见但会出现）

    Args:
        raw: LLM 返回的原始文本。

    Returns:
        解析成功的问答对列表，格式为 ``[{"question": ..., "ground_truth": ...}]``。
        空问题/空答案/缺字段的条目会被自动过滤。
    """
    if not raw:
        return []

    s = raw.strip()

    # 1) 去掉 Markdown 代码块包裹
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    # 2) 截取第一个 [ ... ] 段（丢弃前后多余文字）
    m = re.search(r"\[.*\]", s, flags=re.DOTALL)
    if m:
        s = m.group(0)

    # 3) 单引号 → 双引号（部分 LLM 会输出单引号 JSON）
    s = s.replace("'", '"')

    # 4) 尝试解析
    try:
        data = json.loads(s)
    except Exception:
        logger.debug("JSON 解析失败，原始输出: %s", raw[:200])
        return []

    if not isinstance(data, list):
        return []

    # 5) 逐条校验：必须有 question 和 ground_truth，且均为非空字符串
    out: List[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q = (item.get("question") or "").strip()
        g = (item.get("ground_truth") or "").strip()
        if not q or not g:
            continue
        out.append({"question": q, "ground_truth": g})

    return out


class TestsetGenerator:
    """
    用 LLM 将 BaseNode 列表转换为 (question, ground_truth) 问答对列表。

    Args:
        llm: OpenAI 兼容的 LLM 实例；为 None 时自动调用
            ``vectorStore.milvus_store.get_llm()`` 获取。
        n_questions_per_chunk: 每个 chunk 生成几个问答对（推荐 1~3）。
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        n_questions_per_chunk: int = 1,
    ) -> None:
        if llm is None:
            from vectorStore.milvus_store import get_llm
            llm = get_llm()
        self.llm = llm
        self.n_questions_per_chunk = max(1, int(n_questions_per_chunk))

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def generate_from_nodes(self, nodes: Sequence) -> List[dict]:
        """
        批量从 BaseNode 列表生成问答对。

        Args:
            nodes: ``BaseNode`` 列表（``TextNode`` / ``Document`` 均可），
                通常由 ``spliter.split_by_sentence`` 等切分函数产出。

        Returns:
            问答对列表，每项格式::

                {"question": "...", "ground_truth": "...", "source_chunk": "..."}
        """
        results: List[dict] = []
        total = len(nodes)
        for idx, node in enumerate(nodes, 1):
            # 提取文本
            text = node.get_content() if hasattr(node, "get_content") else str(node)
            if not text or not text.strip():
                logger.debug("node %d/%d 为空，跳过", idx, total)
                continue

            # 过长文本截断，避免 prompt 超出模型上下文窗口
            text = self._truncate(text, max_chars=2400)

            try:
                pairs = self._gen_for_one_chunk(text)
            except Exception as e:
                logger.warning("node %d/%d 生成失败: %s: %s", idx, total, type(e).__name__, e)
                continue

            # 为每条问答对附加来源 chunk，方便后续追溯
            for p in pairs:
                p["source_chunk"] = text
                results.append(p)

            logger.info("node %d/%d: 生成 %d 个问答对", idx, total, len(pairs))

        return results

    def generate_from_texts(self, texts: Sequence[str]) -> List[dict]:
        """
        直接传入字符串列表生成问答对（绕过 BaseNode）。

        适用于已有纯文本、不需要走 LlamaIndex Document 加载流程的场景。

        Args:
            texts: 文本字符串列表。

        Returns:
            同 :meth:`generate_from_nodes`。
        """
        from llama_index.core.schema import TextNode
        nodes = [TextNode(text=t) for t in texts]
        return self.generate_from_nodes(nodes)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _gen_for_one_chunk(self, chunk: str) -> List[dict]:
        """对单个 chunk 调用一次 LLM，返回解析后的问答对列表。"""
        prompt = GENERATE_QA_PROMPT.format(
            n_questions=self.n_questions_per_chunk,
            chunk=chunk,
        )
        resp = self.llm.complete(prompt)
        return _parse_qa_json(resp.text)

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        """
        截断过长文本。

        超过 max_chars 时在截断位置加 ``...`` 提示 LLM 内容被截断。
        """
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."
