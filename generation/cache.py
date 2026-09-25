"""语义缓存：让「换个说法问同一个问题」也能命中。

设计取舍（与 Spec §5.2.3 一致）：

1. **阈值刻意偏高**（默认 `settings.cache_similarity_threshold = 0.95`）。
   问答场景里「差一点」的问题答案可能完全不同——"BERT 用了什么数据集" 与
   "BERT 用了什么优化器" 向量相似度可以很高，答案却毫无关系。
   **召回不足比答错更可接受**，所以宁可漏命中也不返回错误答案。
2. **内存 list + pickle 持久化**：条目数在万级以下时线性扫描耗时可忽略，
   没必要为此引入向量库依赖。
3. **`hit_count` 记录真实高频问题**，供后续分析用户需求分布。
4. **缓存故障绝不冒泡**：`get`/`put` 内部吞掉向量化异常并退化为
   「未命中」/「跳过写入」——缓存是加速层，不该让一次问答失败。

依赖约束：只依赖 `config`、`core.providers`、`core.schemas`。
"""

from __future__ import annotations

import json
import pickle
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from config import settings
from core.logging_utils import get_logger
from core.providers import EmbeddingProvider
from core.schemas import AgentResult, AgentStep, Citation

logger = get_logger(__name__)

#: 持久化文件名，落在 `settings.cache_dir` 下。
CACHE_FILENAME = "semantic_cache.pkl"

#: 持久化格式版本号。结构变更时据此拒绝读取旧文件，而不是crash。
CACHE_FORMAT_VERSION = 1


@dataclass
class CacheEntry:
    """一条缓存记录。

    `result_json` 用 JSON 而非直接 pickle `AgentResult`：
    文件可读可 diff，且不会因为 dataclass 字段增删而整个文件失效。
    """

    question: str
    vec: list[float]
    result_json: str
    hit_count: int = 0


# ---------------------------------------------------------------------------
# AgentResult <-> JSON
# ---------------------------------------------------------------------------


def _result_to_json(result: AgentResult) -> str:
    """把 AgentResult 序列化成 JSON 字符串（嵌套 dataclass 一并展开）。"""
    return json.dumps(asdict(result), ensure_ascii=False)


def _json_to_result(payload: str) -> AgentResult:
    """从 JSON 还原 AgentResult，字段缺失时给出安全默认值。

    容错是刻意的：缓存条目可能由旧版本代码写入，缺字段应降级成「字段为空」，
    而不是让反序列化抛异常、把一次本该命中的请求变成 500。
    """
    data = json.loads(payload)
    return AgentResult(
        answer=data.get("answer", ""),
        citations=[Citation(**c) for c in (data.get("citations") or [])],
        steps=[AgentStep(**s) for s in (data.get("steps") or [])],
        tokens=dict(data.get("tokens") or {}),
        latency_ms=dict(data.get("latency_ms") or {}),
        route=data.get("route") or "cache",
        degraded=data.get("degraded"),
    )


def _normalize_rows(vectors: list[list[float]]) -> np.ndarray:
    """按行归一化。归一化后余弦相似度等价于点积，一次矩阵乘法即可扫完全库。"""
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim == 1:  # 单条向量会被 asarray 压成一维，补回一行
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0  # 零向量不产生除零，相似度自然为 0
    return matrix / norms


# ---------------------------------------------------------------------------
# 语义缓存
# ---------------------------------------------------------------------------


class SemanticCache:
    """基于向量相似度的问答缓存。

    用「问题向量」而非「问题字符串」做键：用户换个说法问同一件事时仍能命中，
    这正是语义缓存相对于 `functools.lru_cache` 的全部价值。
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        cache_dir: Path | None = None,
        threshold: float | None = None,
    ) -> None:
        self.embedder = embedder
        self.cache_dir = Path(cache_dir) if cache_dir is not None else settings.cache_dir
        self.threshold = float(
            settings.cache_similarity_threshold if threshold is None else threshold
        )
        self._entries: list[CacheEntry] = []
        self._matrix: np.ndarray | None = None  # 与 _entries 行序严格对应
        self._lock = threading.RLock()  # 一个锁同时保护 entries 与 matrix，顺序天然一致

    # ---- 内部 ----
    @property
    def path(self) -> Path:
        """持久化文件的完整路径。"""
        return self.cache_dir / CACHE_FILENAME

    def _matrix_view(self) -> np.ndarray | None:
        """返回归一化向量矩阵；条目有变动时按需重建（调用方须持有 `self._lock`）。

        条目数在万级以下，整体重建的开销远小于维护增量索引的复杂度；
        缓存本就要求「快」，而这里的快来自矩阵乘法而不是索引结构。
        任何写操作（put / clear / load）都会把 `self._matrix` 置空，
        因此矩阵与 `self._entries` 要么逐行对应，要么就不存在——
        不存在「矩阵还在但某行向量已被改写」的中间状态。
        """
        if not self._entries:
            self._matrix = None
            return None
        if self._matrix is None:
            dims = {len(e.vec) for e in self._entries}
            if len(dims) > 1:
                # 换过 Embedding 模型时缓存里会混入不同维度的向量，矩阵无法对齐：
                # 只保留与最新条目一致的，并明确记录丢弃了多少条。
                keep = len(self._entries[-1].vec)
                dropped = sum(1 for e in self._entries if len(e.vec) != keep)
                self._entries = [e for e in self._entries if len(e.vec) == keep]
                logger.warning(
                    "语义缓存中存在多种向量维度 %s，已丢弃 %d 条不匹配条目",
                    sorted(dims),
                    dropped,
                )
            self._matrix = _normalize_rows([e.vec for e in self._entries])
        return self._matrix

    def _embed(self, question: str) -> list[float] | None:
        """向量化单条问题；失败返回 None，绝不向上抛异常。"""
        try:
            vector = self.embedder.embed_one(question)
        except Exception as exc:  # noqa: BLE001 - 缓存不可用不该影响主流程
            logger.warning("语义缓存向量化失败，本次按未命中处理: %s", exc)
            return None
        return [float(v) for v in vector]

    # ---- 读写 ----
    def get(self, question: str) -> AgentResult | None:
        """按语义相似度查找缓存。

        线性扫描全部条目取最高相似度，`>= threshold` 才算命中。
        空问题直接返回 `None`——空串向量没有语义，不该命中任何条目。
        """
        if not question or not question.strip():
            return None

        with self._lock:
            if not self._entries:
                return None
            matrix = self._matrix_view()
        if matrix is None:
            return None

        # 向量化可能是一次网络调用，放在锁外做，避免阻塞其他读写。
        query = self._embed(question)
        if query is None:
            return None
        query_vec = np.asarray(query, dtype=np.float64)
        if query_vec.shape[0] != matrix.shape[1]:
            logger.warning(
                "查询向量维度 %d 与缓存维度 %d 不一致（可能更换了 Embedding 模型），本次按未命中处理",
                query_vec.shape[0],
                matrix.shape[1],
            )
            return None
        norm = float(np.linalg.norm(query_vec))
        if norm == 0.0:
            return None
        query_vec = query_vec / norm

        with self._lock:
            if matrix is not self._matrix or not self._entries:
                # 查询期间发生了写入（put/clear/load）：矩阵与条目可能已不对齐，
                # 保守放弃本次命中。缓存未命中只会让这次问答多花一次生成，
                # 返回错误答案的代价则高得多。
                return None
            similarities = matrix @ query_vec  # 均已归一化，点积即余弦相似度
            best = int(np.argmax(similarities))
            best_score = float(similarities[best])
            if best_score < self.threshold:
                logger.info(
                    "语义缓存未命中：最高相似度 %.4f < 阈值 %.4f", best_score, self.threshold
                )
                return None
            entry = self._entries[best]
            entry.hit_count += 1
            payload = entry.result_json
            logger.info(
                "语义缓存命中：相似度 %.4f，问题=%r（累计命中 %d 次）",
                best_score,
                entry.question,
                entry.hit_count,
            )

        try:
            return _json_to_result(payload)
        except Exception as exc:  # noqa: BLE001 - 条目损坏时当作未命中
            logger.warning("缓存条目反序列化失败，已按未命中处理: %s", exc)
            return None

    def put(self, question: str, result: AgentResult) -> None:
        """写入缓存。

        **完全相同的问题只原地更新并 `hit_count += 1`，不新增条目**——
        否则反复问同一个问题会把缓存撑成同一问题的多份副本，
        既浪费空间，也让 hit_count 失去「高频问题」的统计意义。
        """
        if not question or not question.strip():
            logger.debug("空问题不入缓存")
            return

        vector = self._embed(question)
        if vector is None:
            return
        try:
            payload = _result_to_json(result)
        except Exception as exc:  # noqa: BLE001 - 无法序列化就放弃缓存这次结果
            logger.warning("AgentResult 序列化失败，跳过缓存写入: %s", exc)
            return

        with self._lock:
            for entry in self._entries:
                if entry.question == question:
                    entry.vec = vector
                    entry.result_json = payload
                    entry.hit_count += 1
                    self._matrix = None  # 向量已改写，矩阵必须重建
                    logger.info(
                        "缓存条目已原地更新: %r（第 %d 次写入）", question, entry.hit_count
                    )
                    return
            self._entries.append(
                CacheEntry(question=question, vec=vector, result_json=payload, hit_count=0)
            )
            self._matrix = None
            logger.info("缓存新增条目: %r（当前 %d 条）", question, len(self._entries))

    # ---- 容量与统计 ----
    def size(self) -> int:
        """当前条目数。"""
        with self._lock:
            return len(self._entries)

    def hit_count(self, question: str) -> int:
        """某条问题的累计命中次数；不存在返回 0。"""
        with self._lock:
            for entry in self._entries:
                if entry.question == question:
                    return entry.hit_count
        return 0

    def top_questions(self, n: int = 10) -> list[tuple[str, int]]:
        """按 hit_count 降序返回高频问题，用于分析真实用户需求分布。"""
        with self._lock:
            ranked = sorted(self._entries, key=lambda e: (-e.hit_count, e.question))
            return [(e.question, e.hit_count) for e in ranked[: max(0, n)]]

    def clear(self) -> None:
        """清空内存条目，并删除已持久化的文件。

        磁盘文件一并删除：否则下次 `load()` 会把刚清掉的数据装回来，
        「清空」就成了只在当前进程内成立的假象。
        """
        with self._lock:
            self._entries = []
            self._matrix = None
        try:
            self.path.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - 文件被占用等边缘情况
            logger.warning("删除缓存文件失败: %s (%s)", self.path, exc)
        logger.info("语义缓存已清空")

    # ---- 持久化 ----
    def save(self) -> None:
        """把缓存刷到磁盘。

        先写临时文件再原子替换：进程写一半被中断时，旧文件仍然完整可用，
        不会留下一个「能读到但反序列化失败」的半截缓存。
        """
        with self._lock:
            payload = {
                "version": CACHE_FORMAT_VERSION,
                "threshold": self.threshold,
                "entries": [asdict(e) for e in self._entries],
            }
            count = len(self._entries)
            try:
                blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as exc:  # noqa: BLE001 - 落盘失败不该中断问答
                logger.warning("语义缓存序列化失败，已跳过保存: %s", exc)
                return

        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(blob)
            tmp.replace(self.path)
        except OSError as exc:
            logger.warning("语义缓存写入失败: %s (%s)", self.path, exc)
            return
        logger.info("语义缓存已保存 %d 条到 %s", count, self.path)

    def load(self) -> bool:
        """从磁盘恢复缓存。

        Returns:
            `True` 表示成功读到并替换了内存条目；文件不存在（首次启动的正常情况）
            或文件损坏时返回 `False`，缓存退化为「空的」，不影响主流程。
        """
        path = self.path
        if not path.exists():
            logger.info("语义缓存文件不存在，从空缓存开始: %s", path)
            return False
        try:
            payload = pickle.loads(path.read_bytes())
        except Exception as exc:  # noqa: BLE001 - 损坏的缓存不该阻断启动
            logger.warning("语义缓存读取失败，已忽略该文件: %s (%s)", path, exc)
            return False
        if not isinstance(payload, dict) or "entries" not in payload:
            logger.warning("语义缓存格式不符合预期，已忽略: %s", path)
            return False
        if payload.get("version") != CACHE_FORMAT_VERSION:
            logger.warning(
                "语义缓存版本 %s 与当前 %d 不一致，已忽略",
                payload.get("version"),
                CACHE_FORMAT_VERSION,
            )
            return False

        entries: list[CacheEntry] = []
        for raw in payload["entries"]:
            try:
                entries.append(
                    CacheEntry(
                        question=raw["question"],
                        vec=[float(v) for v in raw["vec"]],
                        result_json=raw["result_json"],
                        hit_count=int(raw.get("hit_count", 0)),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("跳过一条损坏的缓存条目: %s", exc)

        with self._lock:
            self._entries = entries
            self._matrix = None
        logger.info("语义缓存已载入 %d 条（来自 %s）", len(entries), path)
        return True
