"""文本分块策略：`Document` → `list[Chunk]`。

三种策略共用同一契约 `BaseChunker.split(doc)`，可在配置里一键切换
（`settings.chunk_strategy`），用于「分块策略对比实验」：

- ``fixed``     ：基线。按估算 token 数滑动窗口硬切，不理会语义边界。
- ``recursive`` ：**默认策略**。按分隔符优先级递归降级，尽量在自然边界断开。
- ``semantic``  ：按相邻句向量余弦相似度在语义断点处切开；没有 `embed_fn`
                  时自动降级为 ``recursive``（**不报错**——能跑起来比报错有用）。

统一约定：

- `chunk_id = f"{doc_id}::{page_no}::{seq}"`，`seq` 在**整篇文档内**单调递增，
  所以同一文档内 ID 必然唯一（下游用它做引用溯源与去重）。
- 纯空白的片段一律丢弃，绝不产出空 chunk（Review Focus #5）。
- 长度判断全部走 `estimate_tokens`，与模型上下文预算用同一把尺子。

术语：本模块里的 `chunk_size` / `overlap` 都是**估算 token 数**，不是字符数；
中英混排下两者差别很大，靠 `_prefix_len_within_budget` 换算成字符位置。
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence

from config import settings
from core.logging_utils import get_logger
from core.schemas import Chunk, Document, Page, estimate_tokens

logger = get_logger(__name__)

#: 递归分块的分隔符优先级：从「最强的语义边界」到「最弱的兜底」。
#: 中文全角标点（。！？；）必须排在英文句点之前——中文文本里 `.` 常出现在
#: 数字与缩写中，用它切句会切出大量碎片。最后的空串是硬切兜底。
SEPARATORS: tuple[str, ...] = (
    "\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", " ", "",
)

#: 把 token 预算换算成字符搜索窗口时的上界（每 token 最多按这么多字符预留）。
#: 只影响二分搜索的范围，不影响结果正确性。
_MAX_CHARS_PER_TOKEN = 8

#: 语义分块的默认断点阈值。`Settings` 里没有对应字段（config.py 已冻结），
#: 因此这里给出常量默认值，调用方可通过构造参数覆盖。
_DEFAULT_SEMANTIC_THRESHOLD = 0.5

#: 页眉页脚判定：出现在超过这个比例的页面上，且长度短于 _BOILERPLATE_MAX_LEN。
_BOILERPLATE_RATIO = 0.6
_BOILERPLATE_MAX_LEN = 100

#: 页脚候选的形态：整行就是一个页码（可带短横线/括号/点等装饰），如
#: "12"、"- 12 -"、"[3]"、"(7)"。详见 `strip_boilerplate` 的说明。
_PAGE_NUMBER_RE = re.compile(r"^[\s\-–—_·.()\[\]|]*\d{1,5}[\s\-–—_·.()\[\]|]*$")

#: 参考文献章节的标题特征（小写化后按前缀匹配）。
_REFERENCE_HEADS: tuple[str, ...] = (
    "references", "reference", "bibliography", "参考文献", "引用文献",
)

_SENTENCE_RE = re.compile(r"[^。！？；!?;\n]+[。！？；!?;]?[ \t]*")


# ------------------------------------------------------------------ 长度换算


def _prefix_len_within_budget(text: str, start: int, budget: int) -> int:
    """`text[start:]` 中最长的前缀长度（字符数），其 token 估算 ≤ budget。

    `estimate_tokens` 对前缀单调不减，所以能二分。搜索窗口先按
    「每 token 最多 `_MAX_CHARS_PER_TOKEN` 个字符」粗扩，再在窗口内二分：
    否则每切一块都要对整篇剩余文本切片，长文档下是 O(n²)。
    """
    if budget <= 0 or start >= len(text):
        return 0
    remaining = len(text) - start
    window = min(remaining, max(budget * _MAX_CHARS_PER_TOKEN + 1, 1))
    while window < remaining and estimate_tokens(text[start:start + window]) <= budget:
        window = min(remaining, window * 2)
    if estimate_tokens(text[start:start + window]) <= budget:
        return remaining
    # 不变量：f(0)=0 ≤ budget < f(window)，二分找最大的「不超预算」长度。
    lo, hi = 0, window
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if estimate_tokens(text[start:start + mid]) <= budget:
            lo = mid
        else:
            hi = mid
    return lo


def _suffix_len_within_budget(text: str, end: int, budget: int) -> int:
    """`text[:end]` 中最长的后缀长度（字符数），其 token 估算 ≤ budget。

    用于「重叠窗口」：把上一块的尾巴原样接到下一块开头，保住跨块上下文。
    """
    if budget <= 0 or end <= 0:
        return 0
    if estimate_tokens(text[:end]) <= budget:
        return end
    lo, hi = 0, end
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if estimate_tokens(text[end - mid:end]) <= budget:
            lo = mid
        else:
            hi = mid
    return lo


def _hard_split(text: str, chunk_size: int) -> list[str]:
    """无标点可用时的最终兜底：按 token 预算硬切，保证每片非空。

    单字符就超预算的极端情况（chunk_size 配得极小）下强制取 1 个字符——
    宁可略微超预算，也不能产出空片或原地打转。
    """
    pieces: list[str] = []
    start = 0
    total = len(text)
    while start < total:
        length = max(1, _prefix_len_within_budget(text, start, chunk_size))
        pieces.append(text[start:start + length])
        start += length
    return pieces


def _split_keep_separator(text: str, sep: str) -> list[str]:
    """按 `sep` 切分，并把分隔符留在前一片段的末尾——不丢任何一个字符。"""
    parts = text.split(sep)
    out = [part + sep for part in parts[:-1]]
    if parts[-1]:
        out.append(parts[-1])
    return [part for part in out if part]


def _split_sentences(text: str) -> list[str]:
    """按中英文句末标点切句，保留标点，并保留行尾换行。

    保留换行是为了让「段落」在合并回 chunk 时不至于被压成一行——
    段落结构对阅读与 BM25 都有价值。
    """
    out: list[str] = []
    for line in text.split("\n"):
        found = [part for part in _SENTENCE_RE.findall(line) if part.strip()]
        if not found:
            continue
        out.extend(found)
        out[-1] = out[-1].rstrip(" \t") + "\n"
    return out


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度。维度不一致或零向量返回 0.0（视为「不相似」而不是崩掉）。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# -------------------------------------------------------------------- 编号


class _SeqCounter:
    """整篇文档内单调递增的分块序号。

    刻意不挂在 chunker 实例上：chunker 保持无状态、可复用、可并发，
    序号只在一次 `split()` 调用内有效。
    """

    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = 0

    def next(self) -> int:
        self.value += 1
        return self.value


def _emit(doc: Document, page_no: int, texts: Iterable[str], counter: _SeqCounter) -> list[Chunk]:
    """把若干文本片段变成 `Chunk`。

    所有策略共用这一个出口，因此：`chunk_id` 格式、`token_count` 口径、
    「过滤空白片段」的规则天然一致，不会各写各的写歪。
    """
    chunks: list[Chunk] = []
    for raw in texts:
        text = (raw or "").strip()
        if not text:  # 绝不产出空 chunk
            continue
        seq = counter.next()
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{page_no}::{seq}",
                doc_id=doc.doc_id,
                filename=doc.filename,
                page_no=page_no,
                seq=seq,
                text=text,
                token_count=estimate_tokens(text),
            )
        )
    return chunks


# ------------------------------------------------------------------ 策略实现


class BaseChunker(ABC):
    """分块策略统一契约。"""

    @abstractmethod
    def split(self, doc: Document) -> list[Chunk]:
        """把文档切成带完整来源坐标的 chunk 列表。"""
        raise NotImplementedError


class FixedChunker(BaseChunker):
    """基线策略：按估算 token 数滑动窗口硬切。

    不理会句子与段落边界，因此最容易切坏语义——它的价值在于
    「对照实验的基线」：如果更聪明的策略没比它好，那复杂就是白加的。
    """

    def __init__(self, chunk_size: int | None = None, overlap: int | None = None) -> None:
        self.chunk_size = max(1, int(chunk_size if chunk_size is not None else settings.chunk_size))
        raw_overlap = max(0, int(overlap if overlap is not None else settings.chunk_overlap))
        step = self.chunk_size - raw_overlap
        if step <= 0:
            # overlap ≥ chunk_size 会让步长变成 0 或负数——直接死循环。
            # 此时放弃重叠，强制步长为 chunk_size：少一点上下文，好过挂死。
            logger.warning(
                "FixedChunker: overlap=%d 不小于 chunk_size=%d，已忽略重叠以避免死循环",
                raw_overlap, self.chunk_size,
            )
            raw_overlap, step = 0, self.chunk_size
        self.overlap = raw_overlap
        self._step_tokens = max(1, step)

    def split(self, doc: Document) -> list[Chunk]:
        counter = _SeqCounter()
        chunks: list[Chunk] = []
        for page in doc.pages:
            chunks.extend(_emit(doc, page.page_no, self._windows(page.text), counter))
        return chunks

    def _windows(self, text: str) -> list[str]:
        """滑动窗口：窗口长 `chunk_size` 个 token，每次前进 `step` 个 token。

        窗口与步长都用 token 预算换算，所以中英混排下也不会「英文块特别大」。
        """
        if not text or not text.strip():
            return []
        windows: list[str] = []
        start = 0
        total = len(text)
        while start < total:
            length = max(1, _prefix_len_within_budget(text, start, self.chunk_size))
            windows.append(text[start:start + length])
            advance = max(1, _prefix_len_within_budget(text, start, self._step_tokens))
            start += advance
        return windows


class RecursiveChunker(BaseChunker):
    """默认策略：按分隔符优先级递归降级，尽量在自然边界断开。

    先试 `\\n\\n`（段落），不够再试 `\\n`（行）、`。！？；`（中文句末）、
    `. ! ?`（英文句末）、空格，最后才硬切。每一步只处理「上一级切完仍然
    超长」的片段——所以正常段落永远不会被硬切开。
    """

    def __init__(self, chunk_size: int | None = None, overlap: int | None = None) -> None:
        self.chunk_size = max(1, int(chunk_size if chunk_size is not None else settings.chunk_size))
        # 重叠不能把「本块 + 下一片」撑爆 chunk_size，否则合并阶段会反复抖动。
        self.overlap = min(
            max(0, int(overlap if overlap is not None else settings.chunk_overlap)),
            max(0, self.chunk_size - 1),
        )

    def split(self, doc: Document) -> list[Chunk]:
        counter = _SeqCounter()
        chunks: list[Chunk] = []
        for page in doc.pages:
            chunks.extend(_emit(doc, page.page_no, self.split_text(page.text), counter))
        return chunks

    def split_text(self, text: str) -> list[str]:
        """把一段文本切成若干片段（只出文本，不出 `Chunk`）。

        公开它是为了让其他策略（如 semantic）复用它做二次切分，
        而不是各自重写一遍「按边界断开」的逻辑。
        """
        if not text or not text.strip():
            return []
        return self._merge(self._split_text(text, 0))

    # -- 递归降级 ---------------------------------------------------------

    def _split_text(self, text: str, sep_index: int) -> list[str]:
        if not text or estimate_tokens(text) <= self.chunk_size:
            return [text] if text else []
        if sep_index >= len(SEPARATORS):
            return _hard_split(text, self.chunk_size)
        sep = SEPARATORS[sep_index]

        parts = _split_keep_separator(text, sep) if sep else _hard_split(text, self.chunk_size)
        if len(parts) <= 1:
            # 这一级分隔符切不动（文本里根本没有它），降级到下一级。
            return self._split_text(text, sep_index + 1)

        out: list[str] = []
        for part in parts:
            if estimate_tokens(part) <= self.chunk_size:
                out.append(part)
            else:
                out.extend(self._split_text(part, sep_index + 1))
        return out

    # -- 贪心合并 + 重叠 ---------------------------------------------------

    def _merge(self, pieces: list[str]) -> list[str]:
        """把细碎片段贪心合并到接近 `chunk_size`，并在块之间补重叠。"""
        merged: list[str] = []
        buffer = ""
        buf_tokens = 0
        for piece in pieces:
            if not piece:
                continue
            piece_tokens = estimate_tokens(piece)
            if buffer and buf_tokens + piece_tokens > self.chunk_size:
                merged.append(buffer)
                buffer = self._overlap_tail(buffer)
                buf_tokens = estimate_tokens(buffer)
            buffer += piece
            buf_tokens += piece_tokens
        if buffer.strip():
            merged.append(buffer)
        return merged

    def _overlap_tail(self, text: str) -> str:
        """取上一块末尾约 `overlap` 个 token 的原文，作为下一块的开头。"""
        if self.overlap <= 0 or not text:
            return ""
        length = _suffix_len_within_budget(text, len(text), self.overlap)
        return text[len(text) - length:] if length > 0 else ""


class SemanticChunker(BaseChunker):
    """语义分块：在相邻句向量相似度骤降处断开。

    `embed_fn is None` 时**直接委托 `RecursiveChunker`**：没有向量化能力
    只是拿不到语义边界，不该变成异常。同理，`embed_fn` 抛错或返回数量
    对不上时也降级——入库流程不该被一次 embedding 抖动整篇打断。
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        threshold: float | None = None,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        self.chunk_size = max(1, int(chunk_size if chunk_size is not None else settings.chunk_size))
        self.threshold = float(threshold) if threshold is not None else _DEFAULT_SEMANTIC_THRESHOLD
        self.embed_fn = embed_fn
        # 降级链路与二次切分都以 recursive(overlap=0) 为基准：语义块之间
        # 靠向量断点衔接，再叠一层文本重叠只会重复计入上下文预算。
        self._recursive = RecursiveChunker(chunk_size=self.chunk_size, overlap=0)

    def split(self, doc: Document) -> list[Chunk]:
        if self.embed_fn is None:
            return self._recursive.split(doc)
        counter = _SeqCounter()
        chunks: list[Chunk] = []
        for page in doc.pages:
            chunks.extend(_emit(doc, page.page_no, self._split_page(page.text), counter))
        return chunks

    def _split_page(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            return self._recursive.split_text(text)

        vectors = self._embed(sentences)
        if vectors is None:
            return self._recursive.split_text(text)

        pieces: list[str] = []
        for group in self._group(sentences, vectors):
            merged = "".join(group)
            if estimate_tokens(merged) <= self.chunk_size:
                pieces.append(merged)
            else:
                # 语义完整但太长：交给 recursive 二次切分，别硬切。
                pieces.extend(self._recursive.split_text(merged))
        return pieces

    def _embed(self, sentences: list[str]) -> list[list[float]] | None:
        """批量向量化；任何失败都返回 None 让调用方降级，而不是向上抛。"""
        try:
            vectors = self.embed_fn(list(sentences))  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001 - 降级优先，见类 docstring
            logger.warning("语义分块 embed_fn 调用失败，本页降级为递归分块：%s", exc)
            return None
        if not vectors or len(vectors) != len(sentences):
            logger.warning(
                "语义分块 embed_fn 返回 %d 个向量，与 %d 句不符，本页降级为递归分块",
                len(vectors or []), len(sentences),
            )
            return None
        return vectors

    def _group(self, sentences: list[str], vectors: list[list[float]]) -> list[list[str]]:
        """相邻句相似度低于阈值处断开，其余归为一组。"""
        groups: list[list[str]] = [[sentences[0]]]
        for index in range(1, len(sentences)):
            if _cosine(vectors[index - 1], vectors[index]) < self.threshold:
                groups.append([])
            groups[-1].append(sentences[index])
        return groups


def get_chunker(strategy: str | None = None, **kw) -> BaseChunker:
    """按策略名构造 chunker。未知策略直接抛 `ValueError`（配置错误要早暴露）。"""
    name = (strategy or settings.chunk_strategy or "").strip().lower()
    if name == "fixed":
        return FixedChunker(**kw)
    if name == "recursive":
        return RecursiveChunker(**kw)
    if name == "semantic":
        return SemanticChunker(**kw)
    raise ValueError(f"未知的分块策略：{strategy!r}；可选 fixed | recursive | semantic")


# -------------------------------------------------------------- 学术文本特化


def strip_boilerplate(pages: list[Page]) -> list[Page]:
    """删除在多数页面上反复出现的短行（页眉 / 页脚 / 水印）。

    判定分两种角色：

    - **页眉**：某行作为某页的**首行**出现，且在 > 60% 的页面上如此，长度 < 100。
    - **页脚**：某行作为某页的**末行**出现，在 > 60% 的页面上如此，长度 < 100，
      **并且整行就是一个页码**（`_PAGE_NUMBER_RE`）。

    末行为什么多一道「必须像页码」的门槛：行尾是正文的常见位置，
    正文行完全可能因为章节结构相似而在多数页面上重复出现。按单纯的行频删
    会**误删正文**——这是不可逆的信息损失；而漏删一个非数字页脚只是多几行
    噪声。两类错误的代价不对称，所以规则偏向「宁可少删」。

    单页文档直接原样返回：一页之内没有「跨页重复」可言，
    照搬比例判定会把标题和末行误删。
    """
    if len(pages) < 2:
        return list(pages)

    total = len(pages)
    counts: dict[str, int] = {}
    for page in pages:
        lines = [line.strip() for line in page.text.splitlines() if line.strip()]
        if not lines:
            continue
        # 用 set 去重：首行与末行可能是同一行（单行页），只应计一次。
        candidates = {lines[0]}
        if _PAGE_NUMBER_RE.match(lines[-1]):
            candidates.add(lines[-1])
        for candidate in candidates:
            if len(candidate) < _BOILERPLATE_MAX_LEN:
                counts[candidate] = counts.get(candidate, 0) + 1

    boilerplate = {line for line, n in counts.items() if n / total > _BOILERPLATE_RATIO}
    if not boilerplate:
        return list(pages)

    cleaned: list[Page] = []
    for page in pages:
        kept = [line for line in page.text.splitlines() if line.strip() not in boilerplate]
        cleaned.append(Page(text="\n".join(kept), page_no=page.page_no))
    return cleaned


def is_reference_section(text: str) -> bool:
    """判断文本是否位于参考文献章节。

    参考文献列表对问答价值低，却极容易命中关键词（作者名、方法名都在里面），
    因此下游会据此降权。容忍 `6. References`、`## Bibliography` 这类编号前缀。
    """
    if not text:
        return False
    stripped = text.strip().lstrip("#*0123456789.、）) \t")
    return stripped.lower().startswith(_REFERENCE_HEADS)


__all__ = [
    "SEPARATORS",
    "BaseChunker",
    "FixedChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "get_chunker",
    "is_reference_section",
    "strip_boilerplate",
]
