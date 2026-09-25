"""论文对比工具（`paper_compare`）：对两篇论文各检索一次，交出可直接对比的片段。

**为什么必须按 `doc_id` 过滤**：`engine.retrieve_only()` 是全库检索，返回的是
top-k 里「整体最相关」的片段。做对比时如果不过滤，很可能两篇里有一篇一个片段
都没进 top-k，模型就会拿 A 的片段去编 B 的结论——这是本工具最危险的失败模式。
所以每篇都单独检索一次，并只保留 `chunk.doc_id == 目标` 的片段；
某篇颗粒无收时**必须明说**，让模型知道证据缺口在哪。

检索档位固定走 `hybrid_rerank`（生产默认档）：对比场景要求召回质量稳定，
不该因为 LLM 随手写了个模式就换一条链路。

与 `kb_search` / `paper_meta` 一致：**不抛异常**，一切失败都返回可读字符串。
`set_engine()` 未被调用时回退到 `kb_search` 注入的引擎（同一进程只该有一份引擎）。
"""

from __future__ import annotations

import re
from typing import Any

from agent.tools.base import register_tool
from core.logging_utils import get_logger

logger = get_logger(__name__)

#: 检索档位：与 `generation.rag.DEFAULT_MODE` 一致。
DEFAULT_MODE = "hybrid_rerank"

#: 每篇最多展示的片段数。对比的 Observation 要短——两篇各 3 段已经足够定性，
#: 再多只会挤占上下文，还会让模型倾向于罗列而不是对比。
PER_DOC_HITS = 3

#: 未注入引擎时的提示（含「未初始化」，与 kb_search 保持一致的口径）。
NOT_INITIALIZED = "错误：检索引擎未初始化，无法对比论文。请改用其他工具或直接说明。"

#: 检索用的查询模板：把维度词与论文里的常见小节词拼在一起。
#: 单给 `aspect`（如 "method"）太短，向量与 BM25 都容易召回到摘要里的泛泛之谈。
QUERY_TEMPLATE = "{aspect} 方法 数据集 结果"

MAX_SNIPPET_CHARS = 800
_WS_RE = re.compile(r"\s+")

#: 模块级引擎引用，由 `build_agent()` 通过 `set_engine()` 注入。
_ENGINE: Any | None = None


def set_engine(engine: Any | None) -> None:
    """注入检索/生成引擎（幂等；传 `None` 表示清空，便于测试与热重载）。"""
    global _ENGINE
    _ENGINE = engine


def _resolve_engine() -> Any | None:
    """取当前可用的引擎：优先本模块注入的，其次回退到 `kb_search` 的。"""
    if _ENGINE is not None:
        return _ENGINE
    from agent.tools import kb_search as kb_module

    return kb_module.get_engine()


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _normalize_aspect(aspect: Any) -> str:
    text = str(aspect or "").strip()
    return text or "方法"


def _one_line(text: Any) -> str:
    """压成单行：`agent/react.py` 的引用正则逐行匹配，换行会截断引用。"""
    flat = _WS_RE.sub(" ", str(text or "")).strip()
    if len(flat) <= MAX_SNIPPET_CHARS:
        return flat
    return flat[:MAX_SNIPPET_CHARS] + "……（片段过长已截断）"


def _vectorstore_of(engine: Any) -> Any | None:
    return getattr(getattr(engine, "retriever", None), "vectorstore", None)


def _known_documents(engine: Any) -> tuple[dict[str, str], bool]:
    """读 `{doc_id: filename}`。

    Returns:
        `(映射, 清单是否可用)`。清单不可用（向量库缺失或读失败）时第二项为
        `False`，调用方应跳过「文档是否存在」的前置校验，改由检索结果判断——
        宁可少一次友好提示，也不要因为读不到清单就断言文档不存在。
    """
    vectorstore = _vectorstore_of(engine)
    if vectorstore is None:
        return {}, False
    try:
        rows = list(vectorstore.list_documents() or [])
    except Exception as exc:  # noqa: BLE001 - 只是校验用的清单，读失败不致命
        logger.warning("读取文档清单失败，跳过 doc_id 前置校验：%s", exc)
        return {}, False
    return {str(r.get("doc_id", "")): str(r.get("filename", "")) for r in rows}, True


def _retrieve_hits(engine: Any, doc_id: str, query: str) -> tuple[list[Any], str | None]:
    """对全库检索一次，只留下属于 `doc_id` 的片段。

    Returns:
        `(片段列表, 错误说明)`；检索本身失败时错误说明非空、片段列表为空。
    """
    try:
        # 位置参数调用：只要求实现 `retrieve_only(question, mode)`。
        result = engine.retrieve_only(query, DEFAULT_MODE)
    except Exception as exc:  # noqa: BLE001 - Observation 必须能交还给 Agent
        logger.warning("paper_compare 检索失败：%s", exc)
        return [], f"错误：检索失败（{type(exc).__name__}: {exc}）"
    chunks = list(getattr(result, "chunks", None) or [])
    hits = [
        scored
        for scored in chunks
        if str(getattr(getattr(scored, "chunk", None), "doc_id", "")) == doc_id
    ]
    return hits[:PER_DOC_HITS], None


def _render_hits(hits: list[Any], start_index: int) -> list[str]:
    """渲染 `[n] (文件名 第p页) 片段正文`，编号接着上一篇继续排。"""
    lines: list[str] = []
    for scored in hits:
        chunk = getattr(scored, "chunk", scored)
        filename = str(getattr(chunk, "filename", "") or "未知文档")
        try:
            page_no = int(getattr(chunk, "page_no", 0) or 0)
        except (TypeError, ValueError):
            page_no = 0
        lines.append(
            f"[{start_index + len(lines)}] ({filename} 第{page_no}页) "
            f"{_one_line(getattr(chunk, 'text', ''))}"
        )
    return lines


def _resolve_name(known: dict[str, str], doc_id: str, hits: list[Any]) -> str:
    """展示用文件名：优先文档清单，其次片段自带，最后退回 doc_id。"""
    if known.get(doc_id):
        return known[doc_id]
    for scored in hits:
        name = str(getattr(getattr(scored, "chunk", None), "filename", "") or "")
        if name:
            return name
    return doc_id


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def paper_compare(doc_id_a: str = "", doc_id_b: str = "", aspect: str = "method") -> str:
    """对比两篇论文在某一维度上的内容。

    Args:
        doc_id_a: 第一篇论文的 doc_id（先用 `paper_meta()` 查）。
        doc_id_b: 第二篇论文的 doc_id。
        aspect: 对比维度，如 `method` / `数据集` / `结论` / `实验结果`。

    Returns:
        两篇各自的片段块（`[n] (文件名 第p页) 正文`），某篇无命中时明确说明缺口；
        任一 doc_id 不存在（或两个相同、引擎未注入）→ 含「未找到」或「错误」的可读提示。
        **不抛异常。**
    """
    first = str(doc_id_a or "").strip()
    second = str(doc_id_b or "").strip()
    if not first or not second:
        return "错误：需要两个 doc_id 才能对比，请先用 paper_meta() 查看已入库文档。"
    if first == second:
        return f"错误：两篇论文的 doc_id 相同（{first}），请给出两个不同的 doc_id。"

    engine = _resolve_engine()
    if engine is None:
        return NOT_INITIALIZED

    dimension = _normalize_aspect(aspect)
    known, has_list = _known_documents(engine)
    if has_list:
        missing = [d for d in (first, second) if d not in known]
        if missing:
            existing = "、".join(
                f"{name or '未知'}({doc})" for doc, name in sorted(known.items())
            )
            return (
                f"未找到文档：{'、'.join(missing)}。知识库现有文档：{existing or '无'}。"
                "请先用 paper_meta() 确认 doc_id。"
            )

    query = QUERY_TEMPLATE.format(aspect=dimension)
    hits_a, error_a = _retrieve_hits(engine, first, query)
    hits_b, error_b = _retrieve_hits(engine, second, query)
    if error_a or error_b:
        return error_a or error_b or "错误：检索失败。"

    lines = [f"【对比维度】{dimension}"]

    lines.append(f"【论文 A】{_resolve_name(known, first, hits_a)}（doc_id={first}）")
    if hits_a:
        lines.extend(_render_hits(hits_a, 1))
    else:
        lines.append(f"未找到与该维度相关的片段（doc_id={first} 的这篇论文在本次检索中没有命中）。")

    lines.append(f"【论文 B】{_resolve_name(known, second, hits_b)}（doc_id={second}）")
    if hits_b:
        lines.extend(_render_hits(hits_b, len(hits_a) + 1))
    else:
        lines.append(f"未找到与该维度相关的片段（doc_id={second} 的这篇论文在本次检索中没有命中）。")

    if hits_a and hits_b:
        lines.append(f"【提示】请基于以上两组片段对比两篇论文的「{dimension}」，并指出差异与共同点。")
    else:
        lines.append("【提示】有一篇没有命中片段，请如实说明证据不足，不要凭空推断该篇的内容。")
    return "\n".join(lines)


@register_tool(
    name="paper_compare",
    description="对比知识库中两篇论文在某个维度（方法/数据集/结论等）上的异同，返回各自的相关原文片段",
    parameters={
        "type": "object",
        "properties": {
            "doc_id_a": {"type": "string", "description": "第一篇论文的 doc_id"},
            "doc_id_b": {"type": "string", "description": "第二篇论文的 doc_id"},
            "aspect": {
                "type": "string",
                "description": "对比维度，如 method | 数据集 | 结论，默认 method",
            },
        },
        "required": ["doc_id_a", "doc_id_b"],
    },
)
def paper_compare_tool(doc_id_a: str = "", doc_id_b: str = "", aspect: str = "method") -> str:
    """工具入口：注册表里叫 `paper_compare`。

    参数带默认值：LLM 少写一个 doc_id 时要拿到「需要两个 doc_id」的可读提示，
    而不是一个 TypeError。
    """
    return paper_compare(doc_id_a=doc_id_a, doc_id_b=doc_id_b, aspect=aspect)
