"""
vectorStore.chat
~~~~~~~~~~~~~~~~~
基于 Milvus 索引 + OpenAI 兼容 LLM 的检索对话(RAG Chat)。

实现思路
--------
- ``enable_question_rewriting=False`` (默认)
    使用自实现的 :class:`CustomRAGChatEngine`,完全控制消息拼装,
    确保"参考资料 + 用户问题"一定在 user 消息里送给 LLM。

- ``enable_question_rewriting=True``
    使用 LlamaIndex 内置的 ``CondenseQuestionChatEngine``(多轮问题改写),
    底层 query_engine 显式传入 ``text_qa_template``,user 消息里也一定带 context_str。

debug 开关(详见 ``config.CHAT.debug`` / :class:`RAGDebugHandler`)会打印:
    - 系统提示词 / QA 模板 / Context 模板
    - Milvus 检索出的节点
    - 送入 LLM 的 messages
    - LLM 返回
"""
from typing import List, Optional

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.llms import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.prompts import (
    ChatMessage,
    ChatPromptTemplate,
    MessageRole,
)

from config import CHAT
from vectorStore.callbacks import RAGDebugHandler
from vectorStore.custom_engine import CustomRAGChatEngine


# ============================================================
# 提示词模板
# ============================================================
DEFAULT_SYSTEM_PROMPT = (
    "你是一个专业的知识库助手,请根据下方提供的参考资料用中文回答用户问题。"
    "若参考资料不足,请明确告知用户并给出建议。"
)

# Query Engine(改写模式)用 —— 占位符 {context_str} / {query_str}
QA_USER_TEMPLATE_STR = (
    "以下是参考资料:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "请根据上述参考资料用中文回答用户的问题。"
    "如果参考资料不足,请明确告知用户并给出建议。\n"
    "问题: {query_str}\n"
    "回答:"
)

# 自实现 Engine(非改写模式)用 —— 占位符 {context_str} / {message}
CONTEXT_USER_TEMPLATE_STR = (
    "以下是参考资料:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "请根据上述参考资料用中文回答用户的问题。"
    "如果参考资料不足,请明确告知用户并给出建议。\n"
    "问题: {message}\n"
    "回答:"
)


# ============================================================
# 内部工具
# ============================================================
def _print_block(title: str, content: str) -> None:
    """
    用统一格式打印一块信息(便于 grep)。

    与 :mod:`vectorStore.callbacks` 中的同名函数功能一致,这里
    复制一份是为了避免 :mod:`chat` 反向依赖 :mod:`callbacks` 的私有符号。

    Args:
        title: 块标题。
        content: 块内容。
    """
    print("\n" + "=" * 70)
    print(f"【{title}】")
    print("=" * 70)
    print(content)


def _build_callback_manager(debug: bool) -> CallbackManager:
    """
    构造 CallbackManager,根据 debug 决定是否挂载 :class:`RAGDebugHandler`。

    Args:
        debug: 是否开启调试日志。

    Returns:
        配置好的 CallbackManager。
    """
    cm = CallbackManager([])
    if debug:
        cm.add_handler(RAGDebugHandler(text_limit=CHAT.debug_text_limit))
    return cm


def _print_system_prompt_info(system_prompt: str) -> None:
    """
    在 debug 开启时打印所有提示词模板,方便确认 prompt 是否生效。

    Args:
        system_prompt: 系统提示词。
    """
    _print_block("系统提示词(System Prompt)", system_prompt)
    _print_block("QA User 模板(改写模式)", QA_USER_TEMPLATE_STR)
    _print_block("Context User 模板(自实现模式)", CONTEXT_USER_TEMPLATE_STR)


# ============================================================
# 对话引擎构造
# ============================================================
def build_chat_engine(
    index: VectorStoreIndex,
    llm: Optional[LLM] = None,
    similarity_top_k: int = 5,
    system_prompt: Optional[str] = None,
    memory_token_limit: int = 3000,
    enable_question_rewriting: Optional[bool] = None,
    debug: Optional[bool] = None,
):
    """
    构造带记忆的对话引擎。

    Args:
        index: 已构建好的 ``VectorStoreIndex``。
        llm: LLM 实例;为 None 时从 ``Settings.llm`` 读取。
        similarity_top_k: 检索返回的最相关节点数。
        system_prompt: 系统提示词;为 None 时使用 ``DEFAULT_SYSTEM_PROMPT``。
        memory_token_limit: 对话历史的 token 上限。
        enable_question_rewriting:
            - True :使用 ``CondenseQuestionChatEngine``(多轮问题改写)
            - False:使用 :class:`CustomRAGChatEngine`(自实现,完全控制消息拼装)
            - None :从 ``config.CHAT.enable_question_rewriting`` 读取
        debug: 是否开启 RAG 全链路日志;None 时从 ``config.CHAT.debug`` 读取。

    Returns:
        一个 ``BaseChatEngine`` 子类实例。
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    if enable_question_rewriting is None:
        enable_question_rewriting = CHAT.enable_question_rewriting
    if debug is None:
        debug = CHAT.debug

    if debug:
        _print_system_prompt_info(system_prompt)

    # 把 debug 回调挂到全局 Settings,Engine 内部所有步骤都会触发
    cm = _build_callback_manager(debug)
    Settings.callback_manager = cm

    # retriever / memory 共用
    retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    memory = ChatMemoryBuffer.from_defaults(token_limit=memory_token_limit)

    if enable_question_rewriting:
        # 模式 A:多轮问题先改写为独立问题,再检索
        chat_qa_template = ChatPromptTemplate(
            message_templates=[
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=MessageRole.USER, content=QA_USER_TEMPLATE_STR),
            ]
        )
        query_engine = index.as_query_engine(
            similarity_top_k=similarity_top_k,
            text_qa_template=chat_qa_template,
        )
        return CondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine,
            llm=llm,
            memory=memory,
            verbose=debug,
        )

    # 模式 B(默认):自实现 Chat Engine —— 完全控制消息拼装
    return CustomRAGChatEngine(
        retriever=retriever,
        llm=llm or Settings.llm,
        memory=memory,
        system_prompt=system_prompt,
        context_template=CONTEXT_USER_TEMPLATE_STR,
    )


def build_simple_query_engine(
    index: VectorStoreIndex,
    similarity_top_k: int = 5,
    system_prompt: Optional[str] = None,
    debug: Optional[bool] = None,
):
    """
    构造一次性查询引擎(无对话记忆,适合脚本化单轮问答)。

    Args:
        index: 已构建好的 ``VectorStoreIndex``。
        similarity_top_k: 检索返回的最相关节点数。
        system_prompt: 系统提示词;为 None 时使用 ``DEFAULT_SYSTEM_PROMPT``。
        debug: 是否开启 RAG 全链路日志;None 时从 ``config.CHAT.debug`` 读取。

    Returns:
        ``RetrieverQueryEngine`` 实例。
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    if debug is None:
        debug = CHAT.debug

    if debug:
        _print_system_prompt_info(system_prompt)
        cm = _build_callback_manager(debug)
        Settings.callback_manager = cm

    chat_qa_template = ChatPromptTemplate(
        message_templates=[
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=QA_USER_TEMPLATE_STR),
        ]
    )
    return index.as_query_engine(
        similarity_top_k=similarity_top_k,
        text_qa_template=chat_qa_template,
    )


# ============================================================
# 命令行交互
# ============================================================
def chat_loop(chat_engine) -> None:
    """
    命令行交互式对话循环。

    接受用户输入直到 ``exit`` / ``quit`` / EOF / Ctrl-C。

    Args:
        chat_engine: ``build_chat_engine`` 返回的引擎。
    """
    print("\n=== RAG 对话已就绪,输入 'exit' / 'quit' 退出 ===")
    while True:
        try:
            user_input = input("\n请输入: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        _print_block("用户输入", user_input)
        response = chat_engine.chat(user_input)
        print(f"\n助手: {response}")
