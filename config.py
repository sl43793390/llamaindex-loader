"""
config
~~~~~~~
集中管理项目配置(Embedding、LLM、Milvus、Chat)。

覆盖优先级:
    1. 函数调用时显式传入的参数
    2. 环境变量
    3. 代码中的默认值
"""
import os
from dataclasses import dataclass, field
from typing import Optional


def _env(key: str, default: str) -> str:
    """
    读取环境变量,缺失时返回默认值。

    Args:
        key: 环境变量名。
        default: 默认值。

    Returns:
        环境变量值或默认值。
    """
    return os.environ.get(key, default)


@dataclass
class EmbeddingConfig:
    """Embedding 模型配置(OpenAI 兼容)。"""

    #: Embedding 模型名,如 ``text-embedding-3-small`` 或各厂商自带。
    model: str = field(default_factory=lambda: _env("EMBED_MODEL", "text-embedding-3-small"))
    #: API Key。
    api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", "sk-xxxxxx"))
    #: API Base URL(支持任意 OpenAI 兼容服务,如 dmxapi / DeepSeek / 通义等)。
    api_base: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", "https://www.dmxapi.cn/v1"))
    #: Embedding 维度(Milvus 建表需要,需与所选模型匹配)。
    dim: int = field(default_factory=lambda: int(_env("EMBED_DIM", "1536")))


@dataclass
class LLMConfig:
    """LLM 配置(OpenAI 兼容,可以是 DeepSeek / 通义千问 / Ollama / vLLM 等)。"""

    #: 模型名。
    model: str = field(default_factory=lambda: _env("LLM_MODEL", "gpt-4.1-nano"))
    #: API Key。
    api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", "sk-xxxxxx"))
    #: API Base URL。
    api_base: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", "https://www.dmxapi.cn/v1"))
    #: 采样温度,越低越确定。
    temperature: float = 0.2
    #: 单次生成最大 token 数。
    max_tokens: int = 8192
    #: 请求超时秒数。
    timeout: float = 60.0


@dataclass
class MilvusConfig:
    """Milvus 向量库配置。"""

    #: Milvus 连接字符串。
    #: 嵌入式模式:本地文件路径,如 ``./milvus.db``。
    #: 集群/单机模式:服务器地址,如 ``http://localhost:19530``。
    uri: str = field(default_factory=lambda: _env("MILVUS_URI", "./milvus_llamaindex.db"))
    #: Milvus 集群访问 token(嵌入式不需要)。
    token: Optional[str] = field(default_factory=lambda: _env("MILVUS_TOKEN", None))
    #: 集合(表)名。
    collection_name: str = field(default_factory=lambda: _env("MILVUS_COLLECTION", "llamaindex_rag"))
    #: 是否每次启动覆盖已有 collection。
    overwrite: bool = field(default_factory=lambda: _env("MILVUS_OVERWRITE", "true").lower() == "true")


@dataclass
class ChatConfig:
    """对话引擎配置。"""

    #: 是否对多轮问题进行改写(CondenseQuestionChatEngine)。
    #: True  = 改写(适合口语化、上下文依赖问题)
    #: False = 不改写(CustomRAGChatEngine,保留原问题直接检索)
    enable_question_rewriting: bool = field(
        default_factory=lambda: _env("ENABLE_QUESTION_REWRITING", "true").lower() == "true"
    )
    #: 检索返回的最相关节点数。
    similarity_top_k: int = field(default_factory=lambda: int(_env("SIMILARITY_TOP_K", "5")))
    #: 对话历史 token 上限(超过会被裁剪)。
    memory_token_limit: int = field(default_factory=lambda: int(_env("MEMORY_TOKEN_LIMIT", "3000")))
    #: 是否开启 RAG 全链路调试日志(打印检索节点、提示词、模型响应)。
    debug: bool = field(default_factory=lambda: _env("RAG_DEBUG", "false").lower() == "true")
    #: 调试打印时,单段文本的截断长度(字符数)。
    debug_text_limit: int = field(default_factory=lambda: int(_env("RAG_DEBUG_TEXT_LIMIT", "800")))


# ============================================================
# 默认配置实例(可直接 import 使用)
# ============================================================
EMBED = EmbeddingConfig()
LLM = LLMConfig()
MILVUS = MilvusConfig()
CHAT = ChatConfig()
