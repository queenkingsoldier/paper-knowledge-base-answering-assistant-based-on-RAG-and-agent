"""BM25 关键词索引。

为什么需要它：向量检索擅长语义相近，但对**专有名词、缩写、公式符号**
这类低频精确词很弱（`BatchNorm`、`mAP`、`ResNet-50`）。BM25 补上这一路，
两路结果再由 RRF 融合。

分词：`jieba` 不在依赖清单里，改用零依赖的「CJK bigram + 英文小写词 + 数字」。
中文 bigram（相邻二字）对学术文本的召回可接受，且不会引入词典与加载开销。

依赖方向：retrieval → core，本模块不发任何网络请求。
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from core.logging_utils import get_logger
from core.schemas import Chunk, ScoredChunk

logger = get_logger(__name__)

INDEX_FILENAME = "bm25.pkl"

_EN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]*")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

_MIN_EN_LEN = 2
# 命中兜底分：BM25 分数退化为 0 时用它保证「有命中」依然得正分。
_MATCH_FLOOR = 1e-6


def tokenize(text: str) -> list[str]:
    """轻量分词：英文小写单词（长度 ≥2）+ CJK bigram + 数字。

    设计取舍：
    - CJK 用 bigram 而非单字：单字召回噪声太大（"的"、"在" 满篇都是），
      bigram 已经能表达「注意力」「机制」这类学术词组；
    - 单字 CJK 片段（如句尾孤字）直接保留，避免整段被丢成空列表；
    - 全角标点、空串、纯数字都不允许抛异常——它们在生产里必然出现。
    """
    if not text:
        return []
    lowered = text.lower()
    tokens = [w for w in _EN_RE.findall(lowered) if len(w) >= _MIN_EN_LEN]
    for seg in _CJK_RE.findall(text):
        if len(seg) == 1:
            tokens.append(seg)
        else:
            tokens.extend(seg[i : i + 2] for i in range(len(seg) - 1))
    # 数字单独保留：'3.14' 这类纯数字查询不能返回空分词结果。
    tokens.extend(_NUMBER_RE.findall(lowered))
    return tokens


class BM25Index:
    """BM25Okapi 索引，支持增量 add 与 pickle 持久化。"""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = Path(index_dir)
        self.chunks: dict[str, Chunk] = {}
        self.order: list[str] = []  # 下标即 BM25 语料下标
        self.bm25: BM25Okapi | None = None
        self._tokens: dict[str, list[str]] = {}
        self._token_sets: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # 构建与增量
    # ------------------------------------------------------------------
    def _register(self, chunks: list[Chunk]) -> int:
        """登记新 chunk，返回新增条数（同 chunk_id 已存在则跳过，保证幂等）。"""
        added = 0
        for chunk in chunks:
            if chunk.chunk_id in self.chunks:
                continue
            tokens = tokenize(chunk.text)
            self.chunks[chunk.chunk_id] = chunk
            self.order.append(chunk.chunk_id)
            self._tokens[chunk.chunk_id] = tokens
            self._token_sets[chunk.chunk_id] = set(tokens)
            added += 1
        return added

    def _rebuild(self) -> None:
        if not self.order:
            # 空语料会让 BM25Okapi 在 avgdl/average_idf 上除零，必须短路。
            self.bm25 = None
            return
        self.bm25 = BM25Okapi([self._tokens[cid] for cid in self.order])

    def build(self, chunks: list[Chunk]) -> None:
        """全量重建索引（覆盖已有内容）。"""
        self.chunks = {}
        self.order = []
        self._tokens = {}
        self._token_sets = {}
        self._register(chunks)
        self._rebuild()
        logger.info("BM25 索引全量重建：%d 个 chunk", len(self.order))

    def add(self, chunks: list[Chunk]) -> None:
        """增量添加。BM25Okapi 不支持增量，故整体重建；
        语料只存 token 列表，万级 chunk 的重建仍在秒级。"""
        added = self._register(chunks)
        if added:
            self._rebuild()
            logger.info("BM25 索引增量添加 %d 个 chunk，当前总计 %d", added, len(self.order))

    def remove_document(self, doc_id: str) -> int:
        removed = [cid for cid in self.order if self.chunks[cid].doc_id == doc_id]
        if not removed:
            return 0
        drop = set(removed)
        for cid in removed:
            self.chunks.pop(cid, None)
            self._tokens.pop(cid, None)
            self._token_sets.pop(cid, None)
        self.order = [cid for cid in self.order if cid not in drop]
        self._rebuild()
        return len(removed)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def query(self, query: str, top_k: int) -> list[ScoredChunk]:
        if top_k <= 0 or self.bm25 is None or not self.order:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        q_set = set(q_tokens)
        raw = self.bm25.get_scores(q_tokens)

        scored: list[tuple[float, int, str]] = []
        for i, cid in enumerate(self.order):
            matched = len(q_set & self._token_sets[cid])
            score = float(raw[i])
            if score <= 0:
                # BM25Okapi 的 idf 在「查询词恰好出现在一半文档」或语料极小
                # （如 1~2 篇）时会退化成 0 甚至负数——此时一个真实的关键词
                # 命中也会拿到 0 分。用命中的查询词元数兜一个极小正分，
                # 使「有命中」与「得分 > 0」严格等价，避免把真命中过滤掉。
                score = _MATCH_FLOOR * matched
            if score <= 0:
                # 完全没有查询词元命中：BM25 恒为 0。必须丢弃，
                # 否则会返回大量与问题无关的 chunk。
                continue
            scored.append((score, i, cid))
        if not scored:
            return []

        scored.sort(key=lambda t: (-t[0], t[1]))  # 分数降序；同分按语料序，保证确定性
        return [
            ScoredChunk(chunk=self.chunks[cid], score=score, source="bm25", rank=rank)
            for rank, (score, _i, cid) in enumerate(scored[:top_k], start=1)
        ]

    # ------------------------------------------------------------------
    # 持久化与状态
    # ------------------------------------------------------------------
    @property
    def index_path(self) -> Path:
        return self.index_dir / INDEX_FILENAME

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("wb") as f:
            # 只存 chunk 与顺序：token 可由文本重算，避免索引文件随分词规则漂移。
            pickle.dump({"chunks": self.chunks, "order": self.order}, f)
        logger.info("BM25 索引已保存: %s", self.index_path)

    def load(self) -> bool:
        if not self.index_path.exists():
            return False
        try:
            with self.index_path.open("rb") as f:
                payload = pickle.load(f)
        except Exception as exc:  # noqa: BLE001 - 索引损坏应降级为「未加载」而非崩溃
            logger.warning("BM25 索引读取失败，将忽略: %s", exc)
            return False
        self.chunks = dict(payload.get("chunks") or {})
        self.order = [cid for cid in (payload.get("order") or []) if cid in self.chunks]
        self._tokens = {cid: tokenize(self.chunks[cid].text) for cid in self.order}
        self._token_sets = {cid: set(toks) for cid, toks in self._tokens.items()}
        self._rebuild()
        logger.info("BM25 索引已加载：%d 个 chunk", len(self.order))
        return True

    def count(self) -> int:
        return len(self.order)
