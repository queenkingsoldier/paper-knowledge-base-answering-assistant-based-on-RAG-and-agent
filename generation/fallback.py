"""分级降级策略：任何一层故障都不能让整个请求 500。

Spec §5.2.4 把降级分成四档，本模块只负责**判定**（该不该提示、提示哪一条），
具体动作（退到纯 LLM / 退到 BM25 / 返回错误卡片）由调用方执行：

| 触发条件 | 用户可见提示 |
|---|---|
| 检索 0 命中 | `NO_HIT_NOTICE` —— 退到纯 LLM 生成，但要告诉用户 |
| Rerank 最高分 < `settings.min_relevance` | `LOW_CONFIDENCE_NOTICE` —— 仍生成，标注低置信 |
| Embedding API 失败 | `VECTOR_DOWN_NOTICE` —— 退到纯 BM25 |
| LLM API 失败（重试后） | `LLM_DOWN_NOTICE` —— 返回错误卡片 |

判定顺序即优先级：**先看有没有资料，再看上游是否已经降级，最后才看质量与答案**。
上游降级原因直接透传（而不是替换成通用文案），因为「rerank 挂了」和
「向量库挂了」对排查问题意味着完全不同的动作。

依赖约束：只依赖 `config` 与 `core.schemas`。
"""

from __future__ import annotations

from config import settings
from core.logging_utils import get_logger
from core.schemas import RetrievalResult

logger = get_logger(__name__)

# 文案与 Spec §5.2.4 逐字一致：这些字符串会直接出现在前端卡片上。
NO_HIT_NOTICE = "⚠️ 知识库中未找到相关文档，以下回答来自模型自身知识，请谨慎参考。"
LOW_CONFIDENCE_NOTICE = "⚠️ 检索置信度较低，请确认以下片段是否与问题相关。"
VECTOR_DOWN_NOTICE = "ℹ️ 向量服务暂不可用，已降级为关键词检索。"
LLM_DOWN_NOTICE = "❌ 模型服务暂时不可用，请稍后重试。"

#: 有资料、分数也够，但模型什么都没返回——属于「生成本身失败」，提示重试即可。
EMPTY_ANSWER_NOTICE = "⚠️ 模型未返回有效内容，请重试。"


def decide(retrieval: RetrievalResult, answer: str) -> tuple[bool, str | None]:
    """判断这次问答是否需要降级提示。

    优先级（自上而下，先命中先返回）：

    1. 检索结果为空 → `NO_HIT_NOTICE`；
    2. `retrieval.degraded` 非空 → **原样透传**该原因（上游已经知道自己降级了，
       这里再包一层文案只会丢掉排查信息）；
    3. 最高分 `< settings.min_relevance` → `LOW_CONFIDENCE_NOTICE`；
    4. 答案为空 → `EMPTY_ANSWER_NOTICE`；
    5. 否则 `(False, None)`。

    Args:
        retrieval: 检索结果（只读取 `chunks` 与 `degraded` 两个字段）。
        answer: 模型生成的答案。

    Returns:
        `(should_degrade, notice)`。`should_degrade` 为 `True` 时 `notice` 必非空。
    """
    chunks = list(getattr(retrieval, "chunks", None) or [])
    if not chunks:
        return True, NO_HIT_NOTICE

    degraded = getattr(retrieval, "degraded", None)
    if degraded:
        return True, degraded

    top_score = float(getattr(chunks[0], "score", 0.0) or 0.0)
    if top_score < settings.min_relevance:
        # 分数低不拒答：检索可能只是「分块太碎」或「问法绕」，片段仍有参考价值，
        # 交给用户判断比直接不给答案更有用。
        logger.info(
            "检索最高分 %.4f 低于阈值 %.4f，标注低置信", top_score, settings.min_relevance
        )
        return True, LOW_CONFIDENCE_NOTICE

    if not (answer or "").strip():
        logger.warning("检索正常但模型未返回内容")
        return True, EMPTY_ANSWER_NOTICE

    return False, None
