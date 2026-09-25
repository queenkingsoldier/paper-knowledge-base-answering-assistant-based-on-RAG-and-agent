"""RRF（Reciprocal Rank Fusion）倒数排名融合——本项目手写算法之一。

公式：

    score(d) = Σ_i  w_i / (k + rank_i(d))

其中 `rank_i(d)` 是文档 d 在第 i 路结果中的名次（从 1 开始），`w_i` 是该路权重，
`k` 是平滑常数（默认 60，来自原论文的经验值）。

**为什么必须有 RRF**：两路检索的分数根本不可比——BM25 分数无上界且随语料
规模漂移，余弦相似度落在 [-1, 1]。想直接加权求和就必须先归一化，而归一化
（min-max / z-score）会引入额外超参，且对异常值极敏感（一条 BM25=999 的
结果就能吃掉整个权重预算）。

**RRF 的价值**：它只读名次、不读原始分数，因此天然免归一化、免调参；
一个文档被越多路召回、名次越靠前，得分就越高。这正是混合检索里
「两路都命中的文档应当排第一」这一直觉的数学表达。

依赖方向：retrieval → core。
"""

from __future__ import annotations

from core.schemas import ScoredChunk

DEFAULT_K = 60


def rrf_fuse(
    result_lists: list[list[ScoredChunk]],
    k: int = DEFAULT_K,
    weights: list[float] | None = None,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """融合多路检索结果，返回 `source="rrf"`、`rank` 从 1 重编号的列表。

    参数
    ----
    result_lists:
        各路结果，**必须已按名次排好序**（名次取列表位置，而非 `ScoredChunk.rank`）。
    k:
        平滑常数。k 越大，头部名次之间的分数差越小（越"民主"）。
    weights:
        每路权重，长度必须与 `result_lists` 一致（含空列表）。
        注意：空列表也要占一个位置，这样调用方传权重时不必先自己清理空路。
    top_k:
        截断长度；`None` 表示不截断。
    """
    lists = list(result_lists or [])
    if weights is None:
        weights = [1.0] * len(lists)
    # 长度校验针对**原始** result_lists：即便某一路为空也要占位，
    # 否则权重会整体错位，静默地把权重加到错误的一路上。
    if len(weights) != len(lists):
        raise ValueError(f"weights 长度 {len(weights)} 与结果列表数 {len(lists)} 不一致")
    if not lists:
        return []

    scores: dict[str, float] = {}
    first_seen: dict[str, ScoredChunk] = {}
    # 先 zip 配对、再跳过空列表：保证 weights[i] 永远对应 result_lists[i]。
    for result, weight in zip(lists, weights):
        if not result:
            continue
        for rank, item in enumerate(result, start=1):
            cid = item.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank)
            # 同一 chunk 出现在多路时保留首次出现的副本，避免下游拿到不同来源的元数据。
            first_seen.setdefault(cid, item)

    if not scores:
        return []
    # 同分时按 chunk_id 升序，保证结果确定、可复现。
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_k is not None:
        ordered = ordered[: max(0, top_k)]
    return [
        ScoredChunk(chunk=first_seen[cid].chunk, score=score, source="rrf", rank=rank)
        for rank, (cid, score) in enumerate(ordered, start=1)
    ]
