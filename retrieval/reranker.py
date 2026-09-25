"""重排序客户端。

召回与精排分离：向量 + BM25 负责「别漏」（高召回），交叉编码器负责
「排准」（高精度）。交叉编码器把 query 与候选拼在一起过一遍模型，
相关性判断远比双塔余弦相似度可靠，代价是无法离线预计算，只能对小候选集
（`retrieve_top_k`，默认 20）在线调用。

**本模块最重要的性质是「永不失败」**：重排序是锦上添花的环节，它挂了可以
退回上游 RRF 顺序，答案质量略降但用户仍能拿到答案。因此这里捕获一切异常，
把原因塞进 `degraded` 返回给上层展示，绝不向上抛。

依赖方向：retrieval → core。对外 HTTP 调用一律交给 `RerankProvider`。
"""

from __future__ import annotations

from core.logging_utils import get_logger
from core.providers import RerankProvider, get_reranker
from core.schemas import ScoredChunk
from config import settings

logger = get_logger(__name__)


class Reranker:
    """对候选 chunk 做交叉编码器重排，失败时降级为原顺序。"""

    def __init__(
        self,
        provider: RerankProvider | None = None,
        enabled: bool = True,
        min_relevance: float | None = None,
    ) -> None:
        # provider 缺省时用工厂按 settings.rerank_provider 选真实/mock 实现。
        self.provider: RerankProvider = provider if provider is not None else get_reranker()
        self.enabled = enabled
        # 阈值不放字面量默认值：config 是唯一配置来源。
        self.min_relevance = settings.min_relevance if min_relevance is None else float(min_relevance)

    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_n: int
    ) -> tuple[list[ScoredChunk], str | None]:
        """返回 `(重排后的结果, 降级原因)`；正常时降级原因为 `None`。

        - 空候选：直接返回，连 provider 都不调用（省一次无意义的网络/算力）。
        - `enabled=False`：按原顺序截断，`degraded="rerank_disabled"`。
        - provider 抛任何异常：按原顺序截断，`degraded="rerank_failed: ..."`。
        """
        if not candidates:
            return [], None

        limit = max(0, min(int(top_n), len(candidates)))
        if limit == 0:
            return [], None
        if not self.enabled:
            return list(candidates[:limit]), "rerank_disabled"

        documents = [c.chunk.text for c in candidates]
        try:
            ranked = self.provider.rerank(query, documents, limit)
            results: list[ScoredChunk] = []
            for rank, item in enumerate(ranked, start=1):
                # provider 只回 (下标, 分数)：return_documents=False 时没有文本，
                # 因此必须用下标回查本地候选，而不是去解析返回内容。
                idx, score = int(item[0]), float(item[1])
                if not 0 <= idx < len(candidates):
                    raise IndexError(f"provider 返回越界下标: {idx}")
                results.append(
                    ScoredChunk(
                        chunk=candidates[idx].chunk,
                        score=score,
                        source="rerank",
                        rank=rank,
                    )
                )
                if len(results) >= limit:
                    break
        except Exception as exc:  # noqa: BLE001 - 重排失败绝不能让整个问答失败
            logger.warning("重排序失败，已降级为上游顺序: %s", exc)
            return list(candidates[:limit]), f"rerank_failed: {exc}"

        if not results:
            # provider 没吐任何结果（如上游把候选全过滤了）：同样退回上游顺序，
            # 否则问答会因为「没有上下文」而彻底失败。
            logger.warning("重排序返回空结果，已降级为上游顺序")
            return list(candidates[:limit]), "rerank_empty"
        return results, None

    def is_low_confidence(self, chunks: list[ScoredChunk]) -> bool:
        """最高分都低于阈值 → 认为这次检索「没找到像样的证据」。

        上层据此提示用户「知识库中可能没有相关内容」，而不是硬编一个答案。
        """
        if not chunks:
            return False
        return chunks[0].score < self.min_relevance
