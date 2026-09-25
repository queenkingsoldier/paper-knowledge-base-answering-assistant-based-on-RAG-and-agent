"""统一数据结构定义。

本模块是整个项目的「接口契约」：所有模块之间只通过这些结构通信。
任何一层的内部实现都可以替换，只要它仍然输入/输出这些结构。

依赖方向约束：本模块只能被依赖，不得依赖任何其他项目模块。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# CJK 及日韩文字范围。这些字符按「一字一 token」估算。
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

# 英文 token 与单词的经验比例：1 token ≈ 0.75 个英文单词。
_EN_WORDS_PER_TOKEN = 0.75


def estimate_tokens(text: str) -> int:
    """粗略估算文本的 token 数。

    CJK 字符按一字一 token；其余文本按「英文单词数 / 0.75」向上取整。
    这是估算而非精确计数——用途是控制上下文预算与滑窗裁剪，
    宁可略微高估（提前截断）也不要低估（超模型上限）。
    """
    if not text:
        return 0
    cjk_count = len(_CJK_RE.findall(text))
    non_cjk = _CJK_RE.sub(" ", text)
    word_count = len(_WORD_RE.findall(non_cjk))
    return cjk_count + math.ceil(word_count / _EN_WORDS_PER_TOKEN)


@dataclass
class Page:
    """文档中的一页。page_no 从 1 开始，是引用溯源的关键字段。"""

    text: str
    page_no: int


@dataclass
class Document:
    """一份解析完成的原始文档。"""

    doc_id: str  # sha256(文件内容)[:16]，内容寻址，天然幂等
    filename: str
    filetype: str  # pdf | docx | txt | md
    pages: list[Page]
    meta: dict  # 标题/作者/年份/DOI，尽力抽取，抽不到留空
    ingested_at: str


@dataclass
class Chunk:
    """切分后的最小检索单元，携带完整来源坐标。"""

    chunk_id: str  # f"{doc_id}::{page_no}::{seq}"
    doc_id: str
    filename: str
    page_no: int
    seq: int
    text: str
    token_count: int


@dataclass
class ScoredChunk:
    """带分数的检索结果。

    source 取值：vector | bm25 | rrf | rerank
    rank 从 1 开始，由产生它的那一层重新编号。
    """

    chunk: Chunk
    score: float
    source: str
    rank: int


@dataclass
class Citation:
    """答案中 [n] 标记对应的真实来源。"""

    index: int
    filename: str
    page_no: int
    chunk_id: str
    snippet: str


@dataclass
class Message:
    """一轮对话消息。"""

    role: str  # user | assistant | system
    content: str
    citations: list[Citation] = field(default_factory=list)


@dataclass
class AgentStep:
    """ReAct 循环中的一步：Thought → Action → Observation。"""

    index: int
    thought: str
    action: str
    action_input: dict
    observation: str
    latency_ms: int
    tokens: int
    error: str | None = None


@dataclass
class AgentResult:
    """一次完整问答的产出，前端与评测都消费这个结构。"""

    answer: str
    citations: list[Citation]
    steps: list[AgentStep]
    tokens: dict  # {"prompt": int, "completion": int}
    latency_ms: dict  # {"retrieve": int, "generate": int, "total": int}
    route: str  # direct_tool | agent | rag | cache
    degraded: str | None = None


@dataclass
class RetrievalResult:
    """一次检索的产出。degraded 非空表示走了降级链路。"""

    chunks: list[ScoredChunk]
    mode: str  # vector | hybrid | hybrid_rerank
    latency_ms: int
    degraded: str | None = None
