"""单篇论文摘要工具（`summarize_paper`）：取该文档**开头**若干片段，产出四段式摘要。

两个刻意的取舍：

1. **只喂前 N 个片段，并且如实标注**。完整长文会超出上下文预算，这里按 `seq` 升序
   取文档开头的前 N 个片段（`seq` 是切分顺序，等于「从头读」）。这种摘要天然不完整，
   所以输出里**必须**带上「以下摘要由模型基于前 N 个片段生成，可能不完整」——
   用户若把它当成全文摘要，就会拿一份基于前几页的结论去做决策。宁可自我贬损，
   也不要制造这种误解。

2. **片段文本直接来自向量库的 `iter_chunks()`，不走向量检索**。摘要是「通读开头」，
   不是「按问题找相关段」，用检索反而会按相似度把片段打乱顺序、丢掉行文脉络。

依赖注入：向量库与 LLM 都靠 `set_*` 注入（`build_agent()` 调一次），不在 import 期
构造——Streamlit 每次交互都重跑脚本，import 期构造等于每次重跑都重建一次 provider。
`set_llm()` 未被调用时会回退到 `core.providers.get_llm()` 工厂；工厂本身失败
（如未配置 key）也会被兜成可读字符串。**本工具不抛异常。**
"""

from __future__ import annotations

from typing import Any

from agent.tools.base import register_tool
from core.logging_utils import get_logger

logger = get_logger(__name__)

#: 送入模型的片段数上限。四段式摘要只需要「开头讲清了什么」，6 段足够覆盖
#: 摘要/引言/方法开头，再多只是把摘要变成一次昂贵的全文朗读。
MAX_CHUNKS = 6

#: 未注入依赖时的提示（含「未初始化」，与 kb_search 口径一致）。
VECTORSTORE_NOT_INITIALIZED = "错误：向量库未初始化，无法生成论文摘要。请改用其他工具或直接说明。"

#: 输出里必须出现的免责标注（模板）。`{count}` 是实际使用的片段数。
USAGE_NOTICE = "（以下摘要由模型基于前 {count} 个片段生成，可能不完整，请以原文为准。）"

_SYSTEM_PROMPT = (
    "你是科研论文阅读助手。只能依据用户给出的片段作答，不得引入片段之外的事实；"
    "若片段不足以支撑某一段，就写「片段中未提及」，不要推测。"
    "输出必须严格分为四段，每段以「背景：」「方法：」「结果：」「结论：」开头，"
    "不要添加其他标题、前言或结语。"
)

#: 模块级依赖，由 `build_agent()` 通过 `set_*` 注入。
_VECTORSTORE: Any | None = None
_LLM: Any | None = None


def set_vectorstore(vectorstore: Any | None) -> None:
    """注入向量库（幂等；传 `None` 表示清空，便于测试与热重载）。"""
    global _VECTORSTORE
    _VECTORSTORE = vectorstore


def set_llm(llm: Any | None) -> None:
    """注入 LLM provider（幂等；传 `None` 表示回退到 `get_llm()`）。"""
    global _LLM
    _LLM = llm


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _resolve_vectorstore() -> Any | None:
    """取当前可用的向量库：优先本模块注入的，其次回退到 `kb_search` 的引擎。"""
    if _VECTORSTORE is not None:
        return _VECTORSTORE
    from agent.tools import kb_search as kb_module

    engine = kb_module.get_engine()
    return getattr(getattr(engine, "retriever", None), "vectorstore", None)


def _resolve_llm() -> tuple[Any | None, str]:
    """取 LLM：优先注入的，其次工厂；工厂失败时返回可读错误说明。"""
    if _LLM is not None:
        return _LLM, ""
    try:
        from core.providers import get_llm

        return get_llm(), ""
    except Exception as exc:  # noqa: BLE001 - 缺 key / 配置错都必须是可读字符串
        logger.warning("获取 LLM 失败：%s", exc)
        return None, f"错误：LLM 未初始化（{type(exc).__name__}: {exc}）。请检查模型配置后重试。"


def _list_documents(vectorstore: Any) -> list[dict]:
    try:
        return list(vectorstore.list_documents() or [])
    except Exception as exc:  # noqa: BLE001 - 读清单失败不该打断 Agent
        logger.warning("读取文档清单失败：%s", exc)
        return []


def _first_chunks(vectorstore: Any, doc_id: str) -> list[dict]:
    """按 `seq` 升序取该文档的前 `MAX_CHUNKS` 个片段。

    `seq` 相同（跨页切分边界）时用 `page_no` 兜底排序，保证顺序稳定——
    每次重跑得到不同顺序的 prompt，会让摘要看起来「随机变化」。
    """
    try:
        rows = list(vectorstore.iter_chunks(doc_id=doc_id) or [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取文档 %s 的片段失败：%s", doc_id, exc)
        return []

    def sort_key(row: dict) -> tuple[int, int, str]:
        try:
            return (int(row.get("seq") or 0), int(row.get("page_no") or 0), str(row.get("chunk_id", "")))
        except (TypeError, ValueError):  # pragma: no cover - 脏数据兜底
            return (0, 0, str(row.get("chunk_id", "")))

    return sorted(rows, key=sort_key)[:MAX_CHUNKS]


def _build_messages(filename: str, chunks: list[dict]) -> list[dict]:
    """拼 Prompt：片段带页码，便于模型在四段里引用具体页。"""
    blocks = [
        f"[片段 {i}｜第 {int(row.get('page_no') or 0)} 页] {str(row.get('text') or '').strip()}"
        for i, row in enumerate(chunks, start=1)
    ]
    user = (
        f"论文文件名：{filename}\n"
        f"以下是该论文开头的 {len(chunks)} 个片段（按原文顺序）：\n\n"
        + "\n\n".join(blocks)
        + "\n\n请据此写出「背景-方法-结果-结论」四段式摘要。"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def summarize_paper(doc_id: str = "") -> str:
    """对某一篇论文生成「背景-方法-结果-结论」四段式摘要。

    Args:
        doc_id: 目标论文的 doc_id（先用 `paper_meta()` 查）。

    Returns:
        带免责标注与四段式正文的字符串；文档不存在 → 含「未找到」的可读提示；
        依赖未初始化或模型调用失败 → 含「错误」的可读提示。**不抛异常。**
    """
    target = str(doc_id or "").strip()
    if not target:
        return "错误：未指定 doc_id，无法确定要摘要哪篇论文。请先用 paper_meta() 查看已入库文档。"

    vectorstore = _resolve_vectorstore()
    if vectorstore is None:
        return VECTORSTORE_NOT_INITIALIZED

    docs = _list_documents(vectorstore)
    match = next((d for d in docs if str(d.get("doc_id", "")) == target), None)
    if match is None and docs:
        available = "、".join(
            f"{d.get('filename') or '未知'}({d.get('doc_id', '')})" for d in docs
        )
        return f"未找到文档：{target}。知识库现有文档：{available}。请先用 paper_meta() 确认 doc_id。"

    filename = str((match or {}).get("filename") or target)
    chunks = _first_chunks(vectorstore, target)
    if not chunks:
        return f"未找到文档 {target} 的可用片段（可能尚未切分入库或已损坏），无法生成摘要。"

    llm, error = _resolve_llm()
    if llm is None:
        return error

    try:
        # 不传额外生成参数：provider 的 temperature/max_tokens 一律由 config 决定，
        # 这里也不强制 provider 接受关键字参数（工具只依赖 `complete(messages)`）。
        answer = str(llm.complete(_build_messages(filename, chunks)) or "").strip()
    except Exception as exc:  # noqa: BLE001 - Observation 必须能交还给 Agent
        logger.warning("summarize_paper 调用 LLM 失败：%s", exc)
        return f"错误：生成摘要失败（{type(exc).__name__}: {exc}）。请稍后重试或改用 kb_search。"

    if not answer:
        return "错误：模型没有返回摘要内容（返回空文本）。请稍后重试。"

    header = f"【论文摘要】{filename}（doc_id={target}）"
    notice = USAGE_NOTICE.format(count=len(chunks))
    return f"{header}\n{notice}\n{answer}"


@register_tool(
    name="summarize_paper",
    description="对知识库中某一篇论文生成「背景-方法-结果-结论」四段式摘要（基于该文档开头若干片段）",
    parameters={
        "type": "object",
        "properties": {
            "doc_id": {"type": "string", "description": "要摘要的论文 doc_id，可用 paper_meta() 查询"},
        },
        "required": ["doc_id"],
    },
    # LLM 生成比纯检索慢得多，30s 对长摘要有被误杀的风险。
    timeout_s=60.0,
)
def summarize_paper_tool(doc_id: str = "") -> str:
    """工具入口：注册表里叫 `summarize_paper`。`doc_id` 省略时给出可读提示。"""
    return summarize_paper(doc_id=doc_id)
