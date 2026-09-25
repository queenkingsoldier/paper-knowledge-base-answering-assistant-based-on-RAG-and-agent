"""三档混合检索编排：`vector` / `hybrid` / `hybrid_rerank`。

档位（与 spec §5.1.4 一致）：

===============  =========================================================  ============
mode             流程                                                       用途
===============  =========================================================  ============
``vector``       向量 top_k                                                  朴素 RAG 基线
``hybrid``       向量 top-20 + BM25 top-20 → RRF → top_k                     验证关键词通道增益
``hybrid_rerank`` 上面两路融合出 top-20 候选 → 交叉编码器精排 → top_k       **生产默认**
===============  =========================================================  ============

**本模块最重要的性质是「不把检索失败变成异常」**：首次启动知识库为空、
embedding 服务挂掉、BM25 索引损坏——这些都必须变成一份可读的
`RetrievalResult.degraded`，而不是抛给前端。理由见 Review Focus #1：
用户第一次打开页面就问问题是这个应用最常见的入口，此刻抛任何异常，
在用户眼里都等同于「应用坏了」。

降级链路的约定：凡是走到「只剩 BM25 一路」的原因里都带 ``bm25`` 字样
（``vector_failed_fallback_bm25`` / ``embedding_failed_fallback_bm25``），
前端与评测据此显示「当前为关键词检索结果」。**向量一路的 try/except 必须
独立**：向量挂了不能把 BM25 一起拖下水，反过来也一样。

依赖方向：retrieval → core；本模块只做编排，不直接发 HTTP。
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

from config import settings as default_settings
from core.logging_utils import get_logger
from core.providers import get_embedder
from core.schemas import RetrievalResult, ScoredChunk
from retrieval.bm25_index import BM25Index
from retrieval.fusion import rrf_fuse
from retrieval.reranker import Reranker
from retrieval.vectorstore import VectorStore

logger = get_logger(__name__)

#: 允许的检索档位。其他值一律 `ValueError`——档位拼错属于调用方 bug，
#: 必须早暴露，绝不能因为「知识库恰好是空的」就静默返回空结果。
MODES: tuple[str, ...] = ("vector", "hybrid", "hybrid_rerank")

#: 默认集合名：`storage/chroma/` 下只有本项目这一个集合。
DEFAULT_COLLECTION = "papers"

#: 降级原因常量（`RetrievalResult.degraded` 的取值）。
#:
#: **`degraded` 只表达「链路真的降级了」，不表达「置信度低」**：低置信由
#: `generation/fallback.decide()` 依据 `chunks[0].score < settings.min_relevance`
#: 判定，本模块不做任何基于 `min_relevance` 的降级分支。取值的全集就是下面
#: 三个（重排序器另有 `rerank_failed: ...` / `rerank_disabled` / `rerank_empty`），
#: 上层据此决定是否在答案前发一个 degraded 事件。
DEGRADED_EMPTY_KB = "empty_knowledge_base"
DEGRADED_VECTOR_FALLBACK = "vector_failed_fallback_bm25"
DEGRADED_EMBEDDING_FALLBACK = "embedding_failed_fallback_bm25"


def _join_degraded(*reasons: str | None) -> str | None:
    """把多个降级原因拼成一行可读文本，去掉重复项（保持出现顺序）。"""
    parts = [reason for reason in reasons if reason]
    return "; ".join(dict.fromkeys(parts)) if parts else None


class HybridRetriever:
    """三档检索的统一入口：持有向量库、BM25 索引与重排序器。"""

    def __init__(
        self,
        vectorstore: VectorStore,
        bm25: BM25Index,
        reranker: Reranker,
        settings=None,
    ) -> None:
        # 三个依赖都公开成属性：前端文档面板与工具层需要直接读
        # `retriever.vectorstore`（如 `list_documents()` / `delete_document()`），
        # 测试也会替换 `retriever.bm25` 与 `retriever.vectorstore.embedder`。
        self.vectorstore = vectorstore
        self.bm25 = bm25
        self.reranker = reranker
        # 参数名与 config 的单例同名：显式传入时以传入者为准（便于测试与多库场景）。
        self.settings = default_settings if settings is None else settings

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def search(self, query: str, mode: str = "hybrid_rerank", top_k: int = 5) -> RetrievalResult:
        """按档位检索，返回 `RetrievalResult`（`degraded` 非空表示走了降级链路）。"""
        if mode not in MODES:
            raise ValueError(f"未知的检索模式：{mode!r}；可选 {' | '.join(MODES)}")

        started = time.perf_counter()
        top_k = max(0, int(top_k))

        # 先判空：知识库为空时直接返回空结果，绝不去碰 Chroma 的空集合
        # 与 BM25 的空语料（两者都会在部分版本上抛除零/越界异常）。
        if self._is_empty():
            logger.info("知识库为空，直接返回空结果（mode=%s）", mode)
            return self._build_result([], mode, started, DEGRADED_EMPTY_KB)

        if mode == "vector":
            chunks, degraded = self._vector_only(query, top_k)
        else:
            candidates, degraded = self._fused_candidates(query)
            if mode == "hybrid":
                chunks = candidates[:top_k]
            else:
                # 注意：这里把**未截断**的 retrieve_top_k 个候选交给重排序，
                # 而不是先把候选截到 top_k——精排的价值正是「先把好东西都捞进来，
                # 再由交叉编码器挑出最相关的 top_k」。
                chunks, rerank_degraded = self.reranker.rerank(query, candidates, top_n=top_k)
                degraded = _join_degraded(degraded, rerank_degraded)

        return self._build_result(chunks, mode, started, degraded)

    # ------------------------------------------------------------------
    # 各档实现
    # ------------------------------------------------------------------
    def _vector_only(self, query: str, top_k: int) -> tuple[list[ScoredChunk], str | None]:
        """`vector` 档：向量一路失败时用 BM25 兜底，而不是把异常抛给调用方。"""
        try:
            chunks = list(self.vectorstore.query(query, top_k))
        except Exception as exc:  # noqa: BLE001 - 见模块 docstring：检索失败必须降级
            logger.warning("向量检索失败，改用 BM25 兜底: %s", exc)
            fallback = self._bm25_only(query, top_k)
            if fallback:
                return fallback, DEGRADED_VECTOR_FALLBACK
            # 兜底也颗粒无收：对「纯向量档」而言等同于该知识库没有可用内容。
            return [], DEGRADED_EMPTY_KB
        if not chunks:
            return [], DEGRADED_EMPTY_KB
        return chunks, None

    def _bm25_only(self, query: str, top_k: int) -> list[ScoredChunk]:
        """BM25 单路（兜底用）；它自己也失败时返回空列表，绝不二次抛出。"""
        try:
            return list(self.bm25.query(query, top_k))
        except Exception as exc:  # noqa: BLE001
            logger.warning("BM25 兜底检索同样失败，返回空结果: %s", exc)
            return []

    def _fused_candidates(self, query: str) -> tuple[list[ScoredChunk], str | None]:
        """两路各召回 `retrieve_top_k` 后做 RRF 融合；单路失败只降级、不牵连另一路。"""
        recall_k = max(1, int(self.settings.retrieve_top_k))
        vector_hits: list[ScoredChunk] = []
        bm25_hits: list[ScoredChunk] = []
        vector_failed = False

        # 向量一路**必须**独立 try/except：embedding 服务挂掉时，
        # 关键词一路仍要给出可用证据（本项目最核心的容错要求）。
        try:
            vector_hits = list(self.vectorstore.query(query, recall_k))
        except Exception as exc:  # noqa: BLE001
            logger.warning("向量检索失败，本次仅用 BM25: %s", exc)
            vector_failed = True

        try:
            bm25_hits = list(self.bm25.query(query, recall_k))
        except Exception as exc:  # noqa: BLE001
            # BM25 挂掉只能记日志、不能进 degraded：`degraded` 的取值是固定集合，
            # 里面没有「关键词一路失败」这一项（见模块顶部常量说明）。用 error 级别
            # 补偿这次无法上报给前端的故障：混合检索静默退化成纯向量了。
            logger.error("BM25 检索失败，本次仅用向量（混合检索已退化为单路）: %s", exc)
            bm25_hits = []

        # 两路等权（不传 weights）；空列表也要占位，故这里固定保留两个位置。
        fused = rrf_fuse([vector_hits, bm25_hits], k=int(self.settings.rrf_k))
        degraded = DEGRADED_EMBEDDING_FALLBACK if vector_failed else None
        return fused, degraded

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _is_empty(self) -> bool:
        """两个索引都为空才算空知识库。"""
        return (
            self._safe_count(self.vectorstore, "向量库") == 0
            and self._safe_count(self.bm25, "BM25") == 0
        )

    @staticmethod
    def _safe_count(store, label: str) -> int:
        """`count()` 不该有能力让「空库判断」失败：异常一律按 0 处理。

        存储真的挂了时，后续那一路的检索会再抛一次并被降级逻辑接住，
        因此这里按 0 处理不会掩盖问题，只是不让它在最外层炸开。
        """
        try:
            return int(store.count())
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s count() 失败，按 0 处理: %s", label, exc)
            return 0

    @staticmethod
    def _build_result(
        chunks: list[ScoredChunk],
        mode: str,
        started: float,
        degraded: str | None,
    ) -> RetrievalResult:
        # latency_ms 必须是 int：前端 st.metric 与评测表直接展示这个字段。
        return RetrievalResult(
            chunks=list(chunks),
            mode=mode,
            latency_ms=int((time.perf_counter() - started) * 1000),
            degraded=degraded,
        )


@lru_cache(maxsize=1)
def build_retriever(storage_dir: Path | None = None) -> HybridRetriever:
    """构造进程级共享的检索器；Streamlit 每次交互都会重跑脚本，必须靠缓存复用。

    路径推导：`storage_dir` 缺省取 `settings.storage_path`，向量库与 BM25
    分别落在其 `chroma/`、`bm25/` 子目录——与 `settings.chroma_dir` /
    `settings.bm25_dir` 的定义完全一致，因此不必写成两条分支。

    BM25 走 `load()`：索引缺失或 pickle 损坏都只返回 `False`，此时用空索引起跑，
    后续入库会重新填充。**绝不因为一个索引文件坏了就让应用起不来。**
    """
    base = Path(storage_dir) if storage_dir is not None else Path(default_settings.storage_path)
    vectorstore = VectorStore(
        persist_dir=base / "chroma",
        collection=DEFAULT_COLLECTION,
        embedder=get_embedder(),
    )
    bm25 = BM25Index(base / "bm25")
    if not bm25.load():
        bm25.build([])
        logger.info("BM25 索引为空（缺失或损坏），已用空索引启动: %s", base / "bm25")
    return HybridRetriever(vectorstore, bm25, Reranker())


__all__ = [
    "DEFAULT_COLLECTION",
    "DEGRADED_EMBEDDING_FALLBACK",
    "DEGRADED_EMPTY_KB",
    "DEGRADED_VECTOR_FALLBACK",
    "MODES",
    "HybridRetriever",
    "build_retriever",
]
