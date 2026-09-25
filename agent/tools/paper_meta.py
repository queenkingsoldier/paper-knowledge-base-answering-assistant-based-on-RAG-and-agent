"""论文元信息工具（`paper_meta`）：回答「知识库里有哪些论文 / 这篇论文的 doc_id 是什么」。

为什么值得单独一个工具：`kb_search` / `paper_compare` / `summarize_paper` 都要 `doc_id`，
而用户（和 LLM）通常只知道文件名。没有这个工具，Agent 只能靠猜 doc_id，一猜错就是
「文档不存在」。它把「先看清有什么、再动手」变成一步低成本的动作。

依赖：只读向量库的**公开**接口 `list_documents()` / `iter_chunks(doc_id)`，
不碰任何私有字段——上游换存储实现时这里不用改。

`set_vectorstore()` 未被调用时（例如 `build_agent()` 只注入了引擎），会回退到
`kb_search` 里注入的那份引擎的 `retriever.vectorstore`：同一个进程里只该有一份
向量库句柄，重复构造既慢又可能踩到 Chroma 的并发写限制。

与 `kb_search` 一样：**不抛异常**，一切失败都变成可读字符串交给 Agent。
"""

from __future__ import annotations

from typing import Any

from agent.tools.base import register_tool
from core.logging_utils import get_logger

logger = get_logger(__name__)

NOT_INITIALIZED = "错误：向量库未初始化，无法读取论文元信息。请改用其他工具或直接说明。"
EMPTY_KB = "知识库中还没有任何已入库文档。请先在「文档管理」上传 PDF。"

#: 模块级向量库引用，由 `build_agent()` 通过 `set_vectorstore()` 注入。
_VECTORSTORE: Any | None = None


def set_vectorstore(vectorstore: Any | None) -> None:
    """注入向量库（幂等；传 `None` 表示清空，便于测试与热重载）。"""
    global _VECTORSTORE
    _VECTORSTORE = vectorstore


def _resolve_vectorstore() -> Any | None:
    """取当前可用的向量库：优先本模块注入的，其次回退到 `kb_search` 的引擎。"""
    if _VECTORSTORE is not None:
        return _VECTORSTORE
    # 函数内 import：避免 `agent.tools` 包内模块之间的导入顺序耦合。
    from agent.tools import kb_search as kb_module

    engine = kb_module.get_engine()
    return getattr(getattr(engine, "retriever", None), "vectorstore", None)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _list_documents(vectorstore: Any) -> list[dict]:
    """读文档清单；存储故障时返回空列表（由调用方给出可读提示）。"""
    try:
        return list(vectorstore.list_documents() or [])
    except Exception as exc:  # noqa: BLE001 - 读元信息失败不该打断 Agent
        logger.warning("读取文档清单失败：%s", exc)
        return []


def _page_range(vectorstore: Any, doc_id: str) -> tuple[int, int] | None:
    """该文档的页码范围 `(最小页, 最大页)`；取不到时返回 `None`。"""
    try:
        rows = list(vectorstore.iter_chunks(doc_id=doc_id) or [])
    except Exception as exc:  # noqa: BLE001 - 页码只是补充信息，拿不到不影响主输出
        logger.warning("读取文档 %s 的片段失败：%s", doc_id, exc)
        return None
    pages = [int(r.get("page_no") or 0) for r in rows]
    valid = [p for p in pages if p > 0]
    return (min(valid), max(valid)) if valid else None


def _format_range(pages: tuple[int, int]) -> str:
    first, last = pages
    return f"第 {first} 页" if first == last else f"第 {first}-{last} 页"


def _describe(doc: dict, vectorstore: Any) -> str:
    """单篇文档的详细信息（含页码范围）。"""
    doc_id = str(doc.get("doc_id", ""))
    lines = [
        "文档信息：",
        f"- 文件名：{doc.get('filename') or '未知'}",
        f"- doc_id：{doc_id}",
        f"- 片段数：{int(doc.get('chunk_count') or 0)}",
    ]
    pages = _page_range(vectorstore, doc_id)
    lines.append(f"- 页码范围：{_format_range(pages) if pages else '未知'}")
    ingested = str(doc.get("ingested_at") or "")
    if ingested:
        lines.append(f"- 入库时间：{ingested}")
    return "\n".join(lines)


def _list_all(docs: list[dict]) -> str:
    lines = [f"知识库中共有 {len(docs)} 篇已入库文档："]
    for i, doc in enumerate(docs, start=1):
        lines.append(
            f"{i}. {doc.get('filename') or '未知'}（doc_id={doc.get('doc_id', '')}，"
            f"{int(doc.get('chunk_count') or 0)} 个片段）"
        )
    lines.append("如需某篇的详细信息（页码范围、入库时间），用 paper_meta(doc_id=...) 再查一次。")
    return "\n".join(lines)


def _match_documents(docs: list[dict], doc_id: str, filename: str) -> tuple[list[dict], str]:
    """按 doc_id / filename 匹配文档。

    匹配分三档，先精确后模糊，避免「文件名少打一个词」直接判成不存在：
    精确 doc_id → 精确文件名（忽略大小写）→ 文件名子串（忽略大小写）。

    Returns:
        `(命中的文档列表, 说明用的键名)`；无命中时列表为空。
    """
    want_id = str(doc_id or "").strip()
    want_name = str(filename or "").strip()

    if want_id:
        exact = [d for d in docs if str(d.get("doc_id", "")) == want_id]
        if exact:
            return exact, f"doc_id={want_id}"
        # 用户（或 LLM）常把文件名塞进 doc_id 参数，顺手兼容一下。
        by_name = [
            d for d in docs if str(d.get("filename", "")).lower() == want_id.lower()
        ]
        if by_name:
            return by_name, f"文件名={want_id}"

    if want_name:
        exact = [
            d for d in docs if str(d.get("filename", "")).lower() == want_name.lower()
        ]
        if exact:
            return exact, f"文件名={want_name}"
        partial = [
            d for d in docs if want_name.lower() in str(d.get("filename", "")).lower()
        ]
        if partial:
            return partial, f"文件名包含「{want_name}」"

    key = f"doc_id={want_id}" if want_id else f"文件名={want_name}"
    return [], key


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def paper_meta(doc_id: str = "", filename: str = "") -> str:
    """查询知识库的文档清单或某一篇论文的元信息。

    Args:
        doc_id: 文档 ID（`paper_meta()` 列出的那个 `d...` 串）。
        filename: 文件名（可只给其中一段，按子串匹配）。与 `doc_id` 至少给一个。

    Returns:
        `doc_id` / `filename` 都为空 → 列出全部已入库文档；
        命中一篇 → 输出文件名、doc_id、片段数、页码范围、入库时间；
        命中多篇 → 列出候选，提示用 doc_id 精确指定；
        匹配不到 → 含「未找到」的可读提示。**不抛异常。**
    """
    vectorstore = _resolve_vectorstore()
    if vectorstore is None:
        return NOT_INITIALIZED

    docs = _list_documents(vectorstore)
    want_id = str(doc_id or "").strip()
    want_name = str(filename or "").strip()

    if not want_id and not want_name:
        return _list_all(docs) if docs else EMPTY_KB
    if not docs:
        return EMPTY_KB

    hits, key = _match_documents(docs, want_id, want_name)
    if not hits:
        available = "、".join(
            f"{d.get('filename') or '未知'}({d.get('doc_id', '')})" for d in docs
        )
        return f"未找到匹配的文档（{key}）。知识库现有文档：{available or '无'}。"
    if len(hits) == 1:
        return _describe(hits[0], vectorstore)

    lines = [f"匹配到 {len(hits)} 篇文档（{key}），请用 doc_id 精确指定："]
    for i, doc in enumerate(hits, start=1):
        lines.append(
            f"{i}. {doc.get('filename') or '未知'}（doc_id={doc.get('doc_id', '')}，"
            f"{int(doc.get('chunk_count') or 0)} 个片段）"
        )
    return "\n".join(lines)


@register_tool(
    name="paper_meta",
    description="查询知识库文档清单，或某篇论文的文件名/doc_id/片段数/页码范围；不知道 doc_id 时先用它",
    parameters={
        "type": "object",
        "properties": {
            "doc_id": {"type": "string", "description": "文档 ID；留空则列出全部文档"},
            "filename": {"type": "string", "description": "文件名（支持部分匹配），与 doc_id 二选一"},
        },
        "required": [],
    },
)
def paper_meta_tool(doc_id: str = "", filename: str = "") -> str:
    """工具入口：注册表里叫 `paper_meta`。两个参数都可省略（省略即列出全部文档）。"""
    return paper_meta(doc_id=doc_id, filename=filename)
