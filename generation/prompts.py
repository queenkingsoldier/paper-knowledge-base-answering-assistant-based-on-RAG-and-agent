"""Prompt 模板与引用溯源。

本模块负责 RAG 生成链路里「把检索结果安全地塞进上下文」这一件事，做三件事：

1. **组装 Prompt**（`build_rag_messages`）：按相关性降序拼接参考资料，
   编号在拼接时即固定并内联来源标签，生成后无需二次匹配；
2. **Token 预算控制**：逐条累加，超预算即停止，绝不截断单条片段；
3. **引用校验**（`extract_citation_indices` / `validate_citations`）：
   把模型输出里的 `[n]` 抽出来，越界编号原样保留并单独标记，
   交给前端渲染成「未验证引用」提示。

依赖约束：只依赖 `core.schemas`，不依赖任何 Provider 与上层模块。
"""

from __future__ import annotations

import re

from core.logging_utils import get_logger
from core.schemas import Citation, ScoredChunk, estimate_tokens

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------

#: RAG 主 System Prompt（Spec §5.2.1 四段式）。
#: 两条硬约束：「只能依据参考资料」「不足以回答时必须明确说明」。
#: 参考资料与用户问题本身由 `build_rag_messages` 放进 user 消息，
#: 这里固定的是不随问题变化的角色、约束与输出格式。
RAG_SYSTEM_PROMPT = """[系统角色]
你是严谨的科研助理。你只能依据"参考资料"回答，不得编造。
若参考资料不足以回答，必须明确说明"知识库中未找到相关内容"。

[参考资料]
参考资料随用户问题一起给出，每条形如 "[1] (来源: 文件名 第N页)"，其后紧跟片段正文。
编号在上下文拼接时即已固定，你只能引用这些编号，不得自行编号、改动编号或虚构来源。

[用户问题]
用户问题位于参考资料之后的 "[用户问题]" 段落中。

[输出要求]
1. 每个事实性陈述后必须紧跟引用标记，格式 [1] 或 [1][2]
2. 引用编号必须对应上方参考资料的编号，不得虚构
3. 若多个来源冲突，需指出冲突并分别标注
4. 用中文回答，专业术语保留英文原文
"""

#: 检索零命中时使用的 Prompt。刻意不直接拒答：
#: 此时链路的既定行为是「退到纯 LLM 生成 + 明确告知用户」，所以要求模型
#: 必须先声明这是自身知识、并保留「无法确定就直说」的出口。
NO_CONTEXT_PROMPT = (
    "知识库中未找到相关内容的参考资料。\n"
    "请遵守以下约束：\n"
    "1. 不要假装看过资料，也不得虚构文献、页码或数据；\n"
    "2. 若你依据自身知识作答，必须在回答开头明确说明"
    "「知识库中未找到相关内容，以下回答来自模型自身知识，请谨慎参考」；\n"
    "3. 若你无法确定，请直接说明无法回答。"
)

#: 引用标记正则。只认 1~3 位纯数字，`[abc]` / `[]` / 裸 `[` 自然被排除；
#: 3 位上界同时挡住 `[2024]` 这类年份误判。
_CITATION_RE = re.compile(r"\[(\d{1,3})\]")

#: Citation.snippet 的长度上界：前端只用来做悬浮预览，不需要全文。
SNIPPET_CHARS = 120


# ---------------------------------------------------------------------------
# 上下文拼接
# ---------------------------------------------------------------------------


def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """压缩空白并截断，供前端做引用预览。"""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def _format_block(index: int, scored: ScoredChunk) -> str:
    """把一条片段渲染成带内联来源标签的上下文块。

    来源标签必须内联：模型生成时直接看到 `[1] (来源: x.pdf 第3页)`，
    引用编号与真实来源的映射就此固定，省掉一次「生成后再匹配」的猜测。
    """
    chunk = scored.chunk
    return f"[{index}] (来源: {chunk.filename} 第{chunk.page_no}页)\n{chunk.text}"


def _to_citation(index: int, scored: ScoredChunk) -> Citation:
    chunk = scored.chunk
    return Citation(
        index=index,
        filename=chunk.filename,
        page_no=chunk.page_no,
        chunk_id=chunk.chunk_id,
        snippet=_snippet(chunk.text),
    )


def build_rag_messages(
    question: str, chunks: list[ScoredChunk], max_context_tokens: int
) -> tuple[list[dict], list[Citation]]:
    """组装 RAG 对话消息，并返回与编号严格对应的引用表。

    返回 `([{"role": "system", ...}, {"role": "user", ...}], citations)`。

    关键约定：

    - **按 `rank` 升序排列**（即相关性降序）。LLM 对上下文的开头与结尾注意力更强
      （"lost in the middle"），所以最高分片段必须放最前。
    - **编号从 1 开始，且 `citations[i].index == i + 1`** —— 与返回列表下标一一对应，
      前端拿 `citations[n-1]` 就能还原答案里的 `[n]`。
    - **逐条累加 `estimate_tokens(整块)`，超预算立即停止，不截断单条**。
      保留一个语义完整的片段，比塞进半句话更有价值；半句话还可能把模型带偏。
    - `chunks` 为空（或预算小到一条都放不下）时返回 `NO_CONTEXT_PROMPT` 且 `citations == []`。

    Args:
        question: 用户问题。
        chunks: 检索结果，`rank` 从 1 开始。
        max_context_tokens: 参考资料部分的 token 预算（不含 System Prompt）。

    Returns:
        (messages, citations)
    """
    ordered = sorted(chunks or [], key=lambda sc: sc.rank)

    citations: list[Citation] = []
    blocks: list[str] = []
    used_tokens = 0
    dropped = 0

    for scored in ordered:
        index = len(citations) + 1  # 编号只随「已收入」的片段递增，因此与下标对齐
        block = _format_block(index, scored)
        cost = estimate_tokens(block)
        if used_tokens + cost > max_context_tokens:
            # 超预算即停止：不截断、也不尝试后面的片段（后面只会更低分）。
            dropped = len(ordered) - len(blocks)
            break
        used_tokens += cost
        blocks.append(block)
        citations.append(_to_citation(index, scored))

    if blocks:
        user_content = (
            "[参考资料]\n" + "\n\n".join(blocks) + f"\n\n[用户问题]\n{question}"
        )
    else:
        # 两条路径合流：真的没检索到，或预算小到一条都塞不下。
        # 对模型而言两者的处境相同——手上没有任何可用资料，只能如实声明。
        if ordered:
            logger.warning(
                "上下文预算 %d token 放不下任何片段（共 %d 条），已退化为无资料模式",
                max_context_tokens,
                dropped or len(ordered),
            )
        user_content = f"{NO_CONTEXT_PROMPT}\n\n[用户问题]\n{question}"

    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages, citations


# ---------------------------------------------------------------------------
# 引用溯源
# ---------------------------------------------------------------------------


def extract_citation_indices(answer: str) -> list[int]:
    """抽出答案中的引用编号，去重并升序返回。

    只识别 `[1]`~`[999]`；`[abc]`、`[]`、裸 `[` 一律忽略——
    宁可漏掉一个畸形标记，也不要把正文里的方括号当引用。
    """
    if not answer:
        return []
    return sorted({int(m) for m in _CITATION_RE.findall(answer)})


def validate_citations(answer: str, valid: set[int]) -> tuple[str, list[int]]:
    """校验答案里的引用编号是否落在真实引用表内。

    越界编号**原样保留**（引用编号是模型输出的一部分，删掉会让句子变得难以理解），
    只在后面追加 `(未验证)` 标记，并把编号收集进 `bad` 交给前端渲染「未验证引用」提示。

    Args:
        answer: 模型输出的完整答案。
        valid: 合法编号集合，通常是 `{c.index for c in citations}`。

    Returns:
        (cleaned, bad_indices)，`bad_indices` 升序去重；全部合法时 `cleaned == answer`。
    """
    if not answer:
        return answer, []

    allowed = {int(v) for v in (valid or ())}
    bad: list[int] = []

    def _mark(match: re.Match) -> str:
        number = int(match.group(1))
        if number in allowed:
            return match.group(0)
        if number not in bad:
            bad.append(number)
        return f"{match.group(0)}(未验证)"

    cleaned = _CITATION_RE.sub(_mark, answer)
    if bad:
        logger.info("答案中存在未验证引用: %s", sorted(bad))
    return cleaned, sorted(bad)
