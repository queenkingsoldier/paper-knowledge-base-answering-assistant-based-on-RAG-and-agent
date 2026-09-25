"""Agent 决策层。

对外只暴露一个组装入口 `build_agent()`：把检索器、LLM、RAG 引擎注入到
各工具模块，再构造手写 ReAct Agent。

为什么用「注入」而不是在工具模块里 import 期构造：Streamlit 每次交互都会
重跑整个脚本，若工具在 import 期就把 Chroma 连接、BM25 索引建好，每次重跑
都会重建一遍，几秒的启动开销会变成常态。注入让重资源走 `lru_cache`
只构造一次。
"""

from __future__ import annotations

from typing import Any

from core.logging_utils import get_logger

logger = get_logger(__name__)


def _inject_dependencies(engine: Any, retriever: Any, llm: Any) -> list[str]:
    """把引擎 / 向量库 / LLM 注入到各工具模块。

    逐个模块独立 try/except：某个工具模块缺失或注入失败，不应该让整个
    Agent 组装失败——与 `agent.tools.base.build_registry` 的容错策略一致。
    返回成功注入的模块名列表，便于冒烟测试断言。
    """
    injected: list[str] = []

    def _try(module_name: str, attr: str, value: Any) -> None:
        try:
            module = __import__(f"agent.tools.{module_name}", fromlist=[attr])
        except ImportError:
            logger.debug("工具模块 %s 不存在，跳过依赖注入", module_name)
            return
        setter = getattr(module, attr, None)
        if setter is None:
            logger.debug("工具模块 %s 没有 %s，跳过", module_name, attr)
            return
        try:
            setter(value)
            injected.append(module_name)
        except Exception as exc:  # noqa: BLE001 - 单个注入失败不该拖垮组装
            logger.warning("注入 %s.%s 失败：%s", module_name, attr, exc)

    _try("kb_search", "set_engine", engine)
    _try("paper_compare", "set_engine", engine)
    _try("paper_meta", "set_vectorstore", retriever.vectorstore)
    _try("summarize", "set_vectorstore", retriever.vectorstore)
    _try("summarize", "set_llm", llm)

    return injected


def build_agent(retriever: Any = None, llm: Any = None, memory: Any = None,
                recorder: Any = None) -> Any:
    """组装一个可用的 ReAct Agent。

    参数
    ----
    retriever : HybridRetriever | None
        不传则用 `retrieval.hybrid.build_retriever()`（自带 lru_cache）。
    llm : LLMProvider | None
        不传则用 `core.providers.get_llm()`。
    memory : ConversationMemory | None
        传入后 Agent 会读历史、写回本轮。
    recorder : TraceRecorder | None
        挂到返回对象的 `.recorder` 属性上。**记录动作用调用方完成**
        （见 `streamlit_app.answer_once`）——因为 session_id 是调用方的语境，
        Agent 自己并不知道当前是哪个会话。

    返回
    ----
    ReActAgent（其 `.tools` 是工具注册表，`.recorder` 可能为 None）
    """
    from agent.react import ReActAgent
    from agent.tools import build_registry
    from core.providers import get_llm
    from generation.rag import RAGEngine
    from retrieval.hybrid import build_retriever

    retriever = retriever if retriever is not None else build_retriever()
    llm = llm if llm is not None else get_llm()

    engine = RAGEngine(retriever=retriever, llm=llm)
    injected = _inject_dependencies(engine, retriever, llm)
    logger.debug("已注入依赖的工具模块：%s", injected)

    tools = build_registry()
    agent = ReActAgent(llm=llm, tools=tools, memory=memory)
    agent.retriever = retriever
    agent.engine = engine
    agent.recorder = recorder
    return agent


__all__ = ["build_agent"]
