"""
vectorStore.custom_engine
~~~~~~~~~~~~~~~~~~~~~~~~~~
完全自控的 RAG Chat Engine —— 显式控制"参考资料 + 用户问题"在 user 消息里
送给 LLM,避免 LlamaIndex 内置 ContextChatEngine 在不同版本下行为不一致的问题。

消息拼装规则
------------
system: {system_prompt}\\n\\n{chat_history}
user  : {context_template.format(context_str=<joined nodes>, message=<user query>)}
"""
from typing import List

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    MessageRole,
)
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    BaseChatEngine,
    StreamingAgentChatResponse,
)
from llama_index.core.llms import LLM
from llama_index.core.memory import BaseMemory
from llama_index.core.schema import NodeWithScore


def _format_chat_history(chat_history: List[ChatMessage]) -> str:
    """
    把对话历史格式化成纯文本,追加到 system 消息尾部。

    Args:
        chat_history: ChatMessage 列表。

    Returns:
        追加到 system 末尾的格式化文本;若历史为空,返回空串。
    """
    if not chat_history:
        return ""
    lines = ["\n\n【对话历史】"]
    for msg in chat_history:
        role = "用户" if msg.role == MessageRole.USER else "助手"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


class CustomRAGChatEngine(BaseChatEngine):
    """
    完全自控的 RAG 对话引擎。

    与 LlamaIndex 内置 ``ContextChatEngine`` 的关键差异:
    - 显式将 ``context_str`` 渲染到 user 消息里,避免版本升级后模板被忽略。
    - system 消息只放 system_prompt + chat_history,职责清晰。
    - 调用 LLM 时直接传 ``List[ChatMessage]``,可被 RAGDebugHandler 完整捕获。
    """

    def __init__(
        self,
        retriever,
        llm: LLM,
        memory: BaseMemory,
        system_prompt: str,
        context_template: str,
    ) -> None:
        """
        Args:
            retriever: 任意实现了 ``retrieve(str) -> List[NodeWithScore]`` 的检索器。
            llm: LLM 实例。
            memory: 对话记忆(``ChatMemoryBuffer`` 等),可为 None。
            system_prompt: 系统提示词。
            context_template: user 消息模板,占位符 ``{context_str}`` ``{message}``。
        """
        self._retriever = retriever
        self._llm = llm
        self._memory = memory
        self._system_prompt = system_prompt
        self._context_template = context_template

    # ---------- BaseChatEngine 抽象方法 ----------
    @property
    def chat_history(self) -> List[ChatMessage]:
        """返回当前对话历史(来自 memory)。"""
        return self._memory.get() if self._memory else []

    def reset(self) -> None:
        """清空对话历史。"""
        if self._memory:
            self._memory.reset()

    # ---------- 内部工具方法 ----------
    def _retrieve_nodes(self, message: str) -> List[NodeWithScore]:
        """
        显式调用 retriever。
        走显式调用而不是 context 内部隐式调用,确保 RAGDebugHandler 能正确触发 RETRIEVE 事件。

        Args:
            message: 用户原始问题。

        Returns:
            命中节点(含相似度分数)。
        """
        return self._retriever.retrieve(message)

    def _build_messages(
        self,
        message: str,
        context_str: str,
        chat_history: List[ChatMessage],
    ) -> List[ChatMessage]:
        """
        显式拼装 system + user 两条消息。

        Args:
            message: 用户问题。
            context_str: 检索出的文本(已 join)。
            chat_history: 已有对话历史。

        Returns:
            准备送入 LLM 的消息列表。
        """
        # system: 提示词 + 格式化的历史
        system_content = self._system_prompt
        hist_str = _format_chat_history(chat_history)
        if hist_str:
            system_content = system_content + hist_str

        # user: 模板显式 format,占位符一定被替换
        user_content = self._context_template.format(
            context_str=context_str,
            message=message,
        )

        return [
            ChatMessage(role=MessageRole.SYSTEM, content=system_content),
            ChatMessage(role=MessageRole.USER, content=user_content),
        ]

    # ---------- 核心:同步聊天 ----------
    def chat(self, message: str) -> AgentChatResponse:
        """
        同步对话入口。

        流程:
            1) 调 retriever 拿节点
            2) 拼接 context_str
            3) 拼装 messages
            4) 调 LLM
            5) 写回 memory

        Args:
            message: 用户输入。

        Returns:
            AgentChatResponse,含 ``response`` 与 ``source_nodes``。
        """
        chat_history = self.chat_history

        # 1) 检索
        nodes = self._retrieve_nodes(message)

        # 2) 拼接 context
        context_str = (
            "\n\n".join([n.get_content() for n in nodes])
            if nodes
            else "(无参考资料)"
        )

        # 3) 拼装消息
        messages = self._build_messages(message, context_str, chat_history)

        # 4) 调用 LLM
        chat_response: ChatResponse = self._llm.chat(messages)
        assistant_text = chat_response.message.content or ""

        # 5) 更新记忆
        if self._memory is not None:
            self._memory.put(ChatMessage(role=MessageRole.USER, content=message))
            self._memory.put(
                ChatMessage(role=MessageRole.ASSISTANT, content=assistant_text)
            )

        return AgentChatResponse(
            response=assistant_text,
            source_nodes=list(nodes),
        )

    # ---------- BaseChatEngine 抽象方法(异步/流式) ----------
    async def achat(self, message: str) -> AgentChatResponse:
        """
        异步对话入口。

        实现简化:直接复用 ``chat()``。如果上游 LLM 自身提供 async 接口,
        可改为 ``await self._llm.achat(messages)``。

        Args:
            message: 用户输入。

        Returns:
            AgentChatResponse。
        """
        return self.chat(message)

    def stream_chat(self, message: str) -> StreamingAgentChatResponse:
        """
        流式对话入口。

        实现简化:流式 = 非流式(一次性返回完整结果后再包装)。
        若需真正的 token 级流式,可改写为迭代 ``self._llm.stream_chat(messages)``。

        Args:
            message: 用户输入。

        Returns:
            StreamingAgentChatResponse。
        """
        result = self.chat(message)
        return StreamingAgentChatResponse(
            response=result.response,
            sources=result.sources,
        )

    async def astream_chat(self, message: str) -> StreamingAgentChatResponse:
        """
        异步流式对话入口。

        Args:
            message: 用户输入。

        Returns:
            StreamingAgentChatResponse。
        """
        result = self.chat(message)
        return StreamingAgentChatResponse(
            response=result.response,
            sources=result.sources,
        )
