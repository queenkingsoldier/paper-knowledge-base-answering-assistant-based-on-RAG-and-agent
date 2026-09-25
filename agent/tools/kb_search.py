"""知识库检索工具（`kb_search`）：把 `RAGEngine.retrieve_only` 的结果变成 Agent 的 Observation。

三条设计约束：

1. **依赖注入，不在 import 期构造**。`RAGEngine` 背后是向量库 + BM25 + 重排三件套，
   构造代价大；Streamlit 每次交互都会重跑整个脚本，在 import 期构造等于每次重跑
   都重建一遍。所以这里只持有模块级 `_ENGINE`，由 `build_agent()` 调 `set_engine()`
   注入一次（`build_registry()` 只负责 import 模块触发注册，不做任何构造）。

2. **永不抛异常**。工具返回值是交还给 Agent 的 Observation：异常穿出去会打断 ReAct
   循环。未注入引擎、检索失败、参数非法，一律返回可读的中文说明——Agent 看到
   「错误：…」才能按 System Prompt 的规则换工具、换参数或直接作答。

3. **输出格式与 `agent/react.py` 的引用正则逐字对齐**：`[n] (文件名 第p页) 片段正文`。
   `react.py` 的 `_CITATION_RE` 是**逐行**匹配的，因此片段正文里的换行必须压成空格：
   PDF 解析出的片段常带换行与页脚，不压平就会让引用只截到第一行、甚至整条丢失。

`top_k` / `mode` 都带默认值并做上下界保护：LLM 在 `Action Input` 里省略参数是常态，
参数写错（`top_k: 999`、`mode: "faiss"`）也不该让一次检索变成一次报错。
"""

from __future__ import annotations

import re
from typing import Any

from agent.tools.base import register_tool
from core.logging_utils import get_logger

logger = get_logger(__name__)

#: 可用的检索档位，与 `retrieval.hybrid.MODES` 一致（这里不 import 它，
#: 避免工具层反向依赖检索层的实现细节；两边取值不一致时下面会兜到默认档）。
MODES: tuple[str, ...] = ("vector", "hybrid", "hybrid_rerank")

#: 默认档位：混合检索 + 交叉编码器精排，与 `generation.rag.DEFAULT_MODE` 相同。
DEFAULT_MODE = "hybrid_rerank"

DEFAULT_TOP_K = 5
MIN_TOP_K = 1
MAX_TOP_K = 20

#: 未注入引擎时的提示。含「未初始化」字样——Agent 的 System Prompt 会告诉它
#: 观察里出现「错误：」时要换工具或直接说明情况。
NOT_INITIALIZED = "错误：检索引擎未初始化，无法检索知识库。请改用其他工具或直接说明。"
EMPTY_NOTICE = "知识库中未找到与该问题相关的内容。"

#: 片段正文压平成单行后，单条允许的最大长度。超出部分截断并标注，
#: 避免一份 PDF 里的超长表格把整轮上下文挤爆。
MAX_SNIPPET_CHARS = 800

_WS_RE = re.compile(r"\s+")

#: 模块级引擎引用：`build_agent()` 通过 `set_engine()` 写入，测试可直接读写。
_ENGINE: Any | None = None


def set_engine(engine: Any | None) -> None:
    """注入检索引擎（幂等；传 `None` 表示清空，便于测试与热重载）。"""
    global _ENGINE
    _ENGINE = engine


def get_engine() -> Any | None:
    """返回当前注入的引擎。

    公开这个读取口是为了让同包的其他工具（`paper_compare` 等）在自身
    未被注入时回退到同一份引擎实例，而不是各自再造一个。
    """
    return _ENGINE


# ---------------------------------------------------------------------------
# 参数归一化
# ---------------------------------------------------------------------------


def _clamp_top_k(value: Any) -> int:
    """把 LLM 给的 `top_k` 夹到 `[MIN_TOP_K, MAX_TOP_K]`；非数字则用默认值。

    不做静默的「零结果」：`top_k=0` 会让 Agent 以为知识库是空的，
    夹到 1 至少还能看到一条证据。
    """
    try:
        number = int(value)
    except (TypeError, ValueError):
        return DEFAULT_TOP_K
    return max(MIN_TOP_K, min(MAX_TOP_K, number))


def _normalize_mode(value: Any) -> str:
    """档位为空或拼错时回落到默认档，而不是把 `ValueError` 抛给 Agent。"""
    mode = str(value or "").strip()
    return mode if mode in MODES else DEFAULT_MODE


def _one_line(text: Any) -> str:
    """把片段正文压成单行（见模块 docstring 第 3 条）。"""
    return _WS_RE.sub(" ", str(text or "")).strip()


def _snippet(text: Any) -> str:
    flat = _one_line(text)
    if len(flat) <= MAX_SNIPPET_CHARS:
        return flat
    return flat[:MAX_SNIPPET_CHARS] + "……（片段过长已截断）"


def _format_hits(chunks: list[Any], limit: int) -> list[str]:
    """渲染 `[n] (文件名 第p页) 片段正文`，编号从 1 连续排。"""
    lines: list[str] = []
    for scored in chunks[:limit]:
        chunk = getattr(scored, "chunk", scored)
        filename = str(getattr(chunk, "filename", "") or "未知文档")
        try:
            page_no = int(getattr(chunk, "page_no", 0) or 0)
        except (TypeError, ValueError):
            page_no = 0
        lines.append(f"[{len(lines) + 1}] ({filename} 第{page_no}页) {_snippet(getattr(chunk, 'text', ''))}")
    return lines


def _degraded_note(result: Any) -> str:
    """把检索链路的降级原因附在末尾，让 Agent 知道「结果可能不完整」。"""
    reason = getattr(result, "degraded", None)
    if not reason:
        return ""
    return f"\n（提示：本次检索走了降级链路：{reason}，结果可能不完整。）"


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def kb_search(query: str = "", top_k: int = DEFAULT_TOP_K, mode: str = DEFAULT_MODE) -> str:
    """在论文知识库中检索与 `query` 相关的片段。

    Args:
        query: 检索问题或关键词。
        top_k: 返回片段数，夹到 1..20；非数字时用默认值。
        mode: 检索档位，取值见 `MODES`；非法取值回落到默认档。

    Returns:
        每行一个片段：`[1] (vit.pdf 第3页) 片段正文`；
        无命中返回空库提示，未注入引擎或检索失败返回可读的错误说明。**不抛异常。**
    """
    question = str(query or "").strip()
    if not question:
        return "错误：检索问题为空，请给出要检索的问题或关键词。"

    engine = _ENGINE
    if engine is None:
        return NOT_INITIALIZED

    limit = _clamp_top_k(top_k)
    used_mode = _normalize_mode(mode)

    try:
        # 位置参数调用（而非 mode=...）：引擎与测试替身只要实现
        # `retrieve_only(question, mode)` 就够了，不强制它们接受关键字。
        result = engine.retrieve_only(question, used_mode)
    except Exception as exc:  # noqa: BLE001 - Observation 必须能交还给 Agent
        logger.warning("kb_search 检索失败：%s", exc)
        return f"错误：知识库检索失败（{type(exc).__name__}: {exc}）。请换一种问法或改用其他工具。"

    chunks = list(getattr(result, "chunks", None) or [])
    note = _degraded_note(result)
    if not chunks:
        return EMPTY_NOTICE + note

    lines = _format_hits(chunks, limit)
    if not lines:  # pragma: no cover - chunks 非空时必有一行
        return EMPTY_NOTICE + note
    return "\n".join(lines) + note


@register_tool(
    name="kb_search",
    description="检索论文知识库，返回带文件名与页码的原文片段；回答论文事实问题前应先调用它",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "要检索的问题或关键词"},
            "top_k": {"type": "integer", "description": "返回片段数，默认 5，范围 1-20"},
            "mode": {
                "type": "string",
                "description": "检索档位：vector | hybrid | hybrid_rerank，默认 hybrid_rerank",
            },
        },
        "required": ["query"],
    },
)
def kb_search_tool(query: str = "", top_k: int = DEFAULT_TOP_K, mode: str = DEFAULT_MODE) -> str:
    """工具入口：注册表里叫 `kb_search`。

    参数全部带默认值：LLM 省略 `query` 时也要拿到一句可读的提示，
    而不是 `TypeError: kb_search() missing 1 required positional argument`。
    """
    return kb_search(query, top_k=top_k, mode=mode)
