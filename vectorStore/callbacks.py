"""
vectorStore.callbacks
~~~~~~~~~~~~~~~~~~~~~~~
自定义 LlamaIndex 回调处理器,用于打印 RAG 全链路关键信息:

  1. 用户原始问题 / 改写后的问题
  2. 从 Milvus 检索出的节点(文本 + 相似度分数)
  3. 拼装后送入 LLM 的提示词(prompt / messages)
  4. LLM 的最终回答
  5. LLM 调用参数(token 上限等)

挂载方式::

    from llama_index.core import Settings
    from vectorStore.callbacks import RAGDebugHandler
    Settings.callback_manager.add_handler(RAGDebugHandler())
"""
from typing import Any, Optional

from llama_index.core.callbacks.base import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload


def _truncate(text: str, limit: int = 800) -> str:
    """
    过长文本截断,避免刷屏。

    Args:
        text: 原始字符串。
        limit: 截断阈值(字符数)。

    Returns:
        截断后的字符串(若超过 limit,尾部追加省略说明)。
    """
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [截断,共 {len(text)} 字符]"


def _print_block(title: str, content: str) -> None:
    """
    用统一格式打印一块调试信息(便于 grep)。

    Args:
        title: 块标题。
        content: 块内容。
    """
    print("\n" + "=" * 70)
    print(f"【{title}】")
    print("=" * 70)
    print(content)


class RAGDebugHandler(BaseCallbackHandler):
    """
    RAG 调试回调:在 RETRIEVE / LLM 事件结束时打印详细信息。

    Args:
        show_prompt: 是否打印送入 LLM 的 messages / prompt。
        show_response: 是否打印 LLM 的最终回答。
        show_retrieval: 是否打印 Milvus 检索节点。
        text_limit: 单段调试文本的最大字符数,超过会被截断。
    """

    def __init__(
        self,
        show_prompt: bool = True,
        show_response: bool = True,
        show_retrieval: bool = True,
        text_limit: int = 800,
    ) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.show_prompt = show_prompt
        self.show_response = show_response
        self.show_retrieval = show_retrieval
        self.text_limit = text_limit
        # 用来给 LLM / RETRIEVE 事件编号
        self._llm_event_count = 0
        self._retrieve_event_count = 0

    # ---------- 抽象方法:trace 生命周期 ----------
    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """
        开始一次 trace(每次 chat()/query() 调用对应一次 trace)。

        Args:
            trace_id: 内部生成的 trace 标识。
        """
        self._llm_event_count = 0
        self._retrieve_event_count = 0
        _print_block("Trace 开始", f"trace_id={trace_id}")

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[dict] = None,
    ) -> None:
        """
        结束一次 trace。

        Args:
            trace_id: 内部生成的 trace 标识。
            trace_map: LlamaIndex 内部维护的 trace 状态表(本回调不使用)。
        """
        _print_block(
            "Trace 结束",
            f"trace_id={trace_id} | LLM 调用 {self._llm_event_count} 次",
        )

    # ---------- 事件开始(本回调不在开始时打印) ----------
    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[dict] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """
        事件开始回调,直接返回 event_id 供内部关联。

        Args:
            event_type: 事件类型。
            payload: 事件负载(本回调不使用)。
            event_id: 事件 ID。
            parent_id: 父事件 ID。
            **kwargs: 透传参数。

        Returns:
            原样返回 event_id。
        """
        return event_id

    # ---------- 事件结束(实际打印逻辑) ----------
    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[dict] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> None:
        """
        事件结束回调,按事件类型分发到具体打印逻辑。

        Args:
            event_type: 事件类型。
            payload: 事件负载。
            event_id: 事件 ID(本回调不使用)。
            parent_id: 父事件 ID(本回调不使用)。
            **kwargs: 透传参数。
        """
        if payload is None:
            return

        if event_type == CBEventType.RETRIEVE and self.show_retrieval:
            self._retrieve_event_count += 1
            self._print_retrieval(payload)
        elif event_type == CBEventType.LLM:
            self._llm_event_count += 1
            self._print_llm_event(payload)

    # ---------- 检索节点 ----------
    def _print_retrieval(self, payload: dict) -> None:
        """
        打印 RETRIEVE 事件的 payload:查询串 + 命中节点列表。

        Args:
            payload: 事件负载,期望包含 QUERY_STR / NODES。
        """
        query_str = payload.get(EventPayload.QUERY_STR, "")
        nodes = payload.get(EventPayload.NODES, []) or []

        _print_block(
            f"检索 #{self._retrieve_event_count} - 查询: {query_str}",
            f"命中节点数: {len(nodes)}",
        )

        for i, node in enumerate(nodes, 1):
            # 相似度
            score = ""
            if hasattr(node, "score") and node.score is not None:
                score = f"  score={node.score:.4f}"
            # 来源
            source = ""
            if hasattr(node, "metadata") and node.metadata:
                src = (
                    node.metadata.get("file_path")
                    or node.metadata.get("source")
                    or node.metadata.get("filename")
                    or ""
                )
                if src:
                    source = f"  source={src}"
            print(f"\n--- Node {i}{score}{source} ---")
            print(_truncate(node.get_content(), self.text_limit))

    # ---------- LLM 调用 ----------
    def _print_llm_event(self, payload: dict) -> None:
        """
        打印 LLM 事件的 payload:提示词 / 响应 / 调用参数。

        Args:
            payload: 事件负载,期望包含 MESSAGES / PROMPT / RESPONSE / ADDITIONAL_KWARGS。
        """
        # LlamaIndex 在 payload 里提供 messages / prompt / response / additional_kwargs
        messages = payload.get(EventPayload.MESSAGES, None)
        prompt = payload.get(EventPayload.PROMPT, None)
        response = payload.get(EventPayload.RESPONSE, None)
        additional = payload.get(EventPayload.ADDITIONAL_KWARGS, None) or {}

        # ---- 提示词 ----
        if self.show_prompt:
            if messages:
                _print_block(
                    f"LLM 调用 #{self._llm_event_count} - 发送的 messages",
                    "",
                )
                for i, msg in enumerate(messages):
                    role = getattr(msg, "role", "unknown")
                    content = getattr(msg, "content", str(msg))
                    if isinstance(content, list):
                        # 部分模型把 content 拆成多段(OpenAI 多模态)
                        content = "\n".join(
                            [
                                c.get("text", str(c)) if isinstance(c, dict) else str(c)
                                for c in content
                            ]
                        )
                    print(f"\n[Message {i}] role={role}")
                    print(_truncate(content, self.text_limit))
            elif prompt:
                _print_block(
                    f"LLM 调用 #{self._llm_event_count} - 发送的 prompt",
                    _truncate(prompt, self.text_limit),
                )

        # ---- 响应 ----
        if self.show_response and response is not None:
            text = getattr(response, "text", None) or str(response)
            _print_block(
                f"LLM 调用 #{self._llm_event_count} - 模型返回",
                _truncate(text, self.text_limit),
            )

        # ---- Token / 参数 ----
        if additional:
            _print_block(
                f"LLM 调用 #{self._llm_event_count} - 调用参数",
                str(additional),
            )
