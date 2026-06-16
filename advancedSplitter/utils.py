"""
advancedSplitter.utils
~~~~~~~~~~~~~~~~~~~~~~~
高级切分器共用的辅助:
    - Embedding / LLM 客户端懒构造
    - 文本预处理(规范化空白、句子切分、token 估算)
    - 缓存目录

设计原则:
    - 客户端默认走项目根目录的 ``config.py``,但允许在函数调用时显式传入。
    - 任何对 LLM / Embedding 的访问失败都不致命 —— 退回到轻量算法。
"""
from __future__ import annotations

import os
import re
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence

# 复用项目根目录的 config(支持 OpenAI 兼容服务)
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import EMBED, LLM  # noqa: E402


# ============================================================
# 客户端构造(懒加载)
# ============================================================
@lru_cache(maxsize=1)
def get_embed_model(model: Optional[str] = None,
                    api_key: Optional[str] = None,
                    api_base: Optional[str] = None):
    """构造 Embedding 模型(OpenAI 兼容)。"""
    from llama_index.core.embeddings import BaseEmbedding  # noqa: F401
    from llama_index.embeddings.openai import OpenAIEmbedding

    return OpenAIEmbedding(
        model=model or EMBED.model,
        api_key=api_key or EMBED.api_key,
        api_base=api_base or EMBED.api_base,
    )


@lru_cache(maxsize=1)
def get_llm(model: Optional[str] = None,
            api_key: Optional[str] = None,
            api_base: Optional[str] = None,
            temperature: float = 0.2,
            max_tokens: int = 4096):
    """构造 LLM(OpenAI 兼容)。"""
    from llama_index.llms.openai import OpenAI

    return OpenAI(
        model=model or LLM.model,
        api_key=api_key or LLM.api_key,
        api_base=api_base or LLM.api_base,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=LLM.timeout,
    )


# ============================================================
# 文本预处理
# ============================================================
_WS_RE = re.compile(r"[ \t\u00a0\u3000]+")
_NL_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """
    规范化空白:
        - 合并连续空格为单空格
        - 3+ 连续换行压成 2 个(保留段落间隔)
    """
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    return text.strip()


# 中英文混合的句子切分(。！？.!?) + 引号包裹的不切
_SENT_END_RE = re.compile(r"(?<=[。！？!?\.])")


def split_sentences(text: str) -> List[str]:
    """
    中英文句子切分。

    简单规则:
        - 在 。 ！？ ! ? . 之后切
        - 但引号("" '')内不算
        - 段落(\n\n)也算分隔

    不依赖 NLTK,够用于切分器预处理。
    """
    if not text.strip():
        return []
    # 先按段落切
    parts: List[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        # 简单 split,然后用 _SENT_END_RE 二次切
        chunks = _SENT_END_RE.split(para)
        for c in chunks:
            c = c.strip()
            if c:
                parts.append(c)
    return parts


# token 估算:英文按词 / 4 ≈ 1 token,中文按字符 / 1.5 ≈ 1 token
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数,用于切分大小控制。"""
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    return int(cjk / 1.5 + other / 4) + 1


def text_hash(text: str) -> str:
    """对文本生成短 hash,用作 node id 后缀。"""
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:10]


# ============================================================
# 节点 id 辅助
# ============================================================
def make_node_id(prefix: str, idx: int, text: str) -> str:
    """生成稳定且可读的 node id: ``<prefix>_<idx>_<hash>``。"""
    return f"{prefix}_{idx:04d}_{text_hash(text)}"
