"""RAG 生成流水线：混合检索 → 上下文组装 → 流式生成 → 引用溯源。

本模块把检索层与生成层的零件串成一条**对前端友好**的链路，对外只承诺两件事：

**一、事件序列契约**（`stream_answer`），顺序固定、不可更改：

| # | 事件 | 何时发出 |
|---|---|---|
| 1 | `{"type":"degraded","reason":str}` | 仅降级时；**必须早于任何 token** |
| 2 | `{"type":"citations","citations":[Citation as dict, ...]}` | 恒定发出 |
| 3 | `{"type":"token","text":str}` | 0..N 个，逐段增量 |
| 4 | `{"type":"usage","prompt_tokens":int,"completion_tokens":int}` | 恒定发出 |
| 5 | `{"type":"error","message":str}` | 仅失败时 |
| 6 | `{"type":"done","answer":str,"degraded":str\|None,"latency_ms":dict}` | **永远是最后一个** |

`citations` 排在 token 之前不是审美选择：前端收到第一个 token 时就要按
`[n]` 渲染可点击徽章，引用表必须先到位，否则只能先画纯文本再回头重排。
`done` 永远会出现——它是前端**关闭 loading 态的唯一依据**。因此
`stream_answer` 把 Prompt 组装与 LLM 调用整体包在 `try/except Exception` 中，
失败降级为 `error` 事件而不是异常：生成器一旦向外抛出，调用方的 `for`
循环会直接中断，`done` 永远到不了，界面就永久停在「生成中」。

**二、`degraded` 事件只在「检索链路已经降级」时发出**，刻意**不**包含
「检索到了但最高分偏低」这一档（低置信）。两个理由：低置信要拿到答案才能
定性（`generation.fallback.decide` 的入参含 answer），而 `degraded` 事件按契约
必须早于 token，位置冲突；且低置信时片段本身仍是有效上下文、答案照常生成，
把它塞进「资料为空」的语义里会误导前端。低置信提示改由 `done.degraded` 透出。

依赖方向：generation → retrieval（仅 duck typing，不 import 其内部）→ core。
本模块不直接发任何 HTTP 请求。
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any

from config import settings as default_settings
from core.logging_utils import get_logger
from core.providers import LLMProvider, get_llm
from core.schemas import AgentResult, Citation, RetrievalResult, estimate_tokens
from generation.fallback import decide
from generation.prompts import build_rag_messages

logger = get_logger(__name__)

#: 流式事件。`type` 的取值与顺序见模块 docstring 的表格，那张表就是契约本身。
StreamEvent = dict[str, Any]

#: 检索默认档位。这里显式写出而不依赖 `HybridRetriever.search` 的默认值，
#: 是为了让「引擎跑在哪一档」在调用点一眼可见，换默认值时也不会被静默带走。
DEFAULT_MODE = "hybrid_rerank"

#: 空知识库的降级原因。上游 `HybridRetriever` 在检索为空时会带上同名字段；
#: 这里再兜一个常量，使「检索器只回了空列表、没带 degraded」时前端依然能拿到
#: 稳定的机器可读原因，而不是一个 None。
EMPTY_KNOWLEDGE_BASE = "empty_knowledge_base"

#: 检索器自身抛异常时的降级原因前缀，后面接原始异常描述。
RETRIEVAL_FAILED = "retrieval_failed"


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _elapsed_ms(started: float) -> int:
    """`perf_counter` 起点的毫秒耗时，永不为负。"""
    return max(0, round((time.perf_counter() - started) * 1000))


def _as_int(value: Any) -> int:
    """把 provider 可能给出的 None / 字符串 / float 计数统一成非负整数。"""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _describe(exc: BaseException) -> str:
    """异常 → 单行描述。带上类型名：`401` 与 `TypeError` 的排查动作完全不同。"""
    return f"{type(exc).__name__}: {exc}"


def _chunks_of(retrieval: Any) -> list[Any]:
    """取出检索结果里的 chunk 列表；`None` / 字段缺失一律当空处理。"""
    return list(getattr(retrieval, "chunks", None) or [])


def _usage_of(llm: Any, messages: list[dict], answer: str) -> dict[str, int]:
    """读取 provider 上报的 token 用量；未上报时用 `estimate_tokens` 兜底。

    兜底不是多余的：`last_usage` 是 provider 的「尽力而为」字段，自定义实现
    或异常路径下可能为空。少一个字段就让整个成本面板恒显示 0，代价太大；
    估算值至少量级正确，且与 Prompt 预算用的是同一套估算口径。
    """
    raw = getattr(llm, "last_usage", None)
    if isinstance(raw, dict) and raw:
        return {
            "prompt_tokens": _as_int(raw.get("prompt_tokens")),
            "completion_tokens": _as_int(raw.get("completion_tokens")),
        }
    return {
        "prompt_tokens": estimate_tokens(
            " ".join(str(m.get("content", "")) for m in messages)
        ),
        "completion_tokens": estimate_tokens(answer),
    }


def _with_history(messages: list[dict], history: list[dict] | None) -> list[dict]:
    """把历史轮次插到**当前** user 消息之前。

    位置很关键：历史必须排在当前问题之前，模型才知道「哪个是被追问的问题」；
    插在末尾会让模型把历史当成最新指令，追问场景直接答偏。
    历史里的畸形条目（非 dict、缺 role/content）静默丢弃——上下文拼装
    不应该因为一条脏历史就整轮失败。
    """
    if not history:
        return list(messages)

    turns = [
        m
        for m in history
        if isinstance(m, dict) and m.get("role") and m.get("content")
    ]
    if not turns:
        return list(messages)

    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            return out[:i] + turns + out[i:]
    # 没有 user 消息（理论不可达）：追加在末尾，至少不丢历史。
    return out + turns


def _pre_degraded_reason(retrieval: Any, chunks: list[Any]) -> str | None:
    """生成前就能判定的降级原因（只读检索结果）。

    判定顺序：
    1. 上游 `degraded` 非空 → **原样透传**。「向量库挂了」和「重排挂了」
       对排查意味着完全不同的动作，包一层通用文案只会丢掉这条信息；
    2. 检索结果为空 → `EMPTY_KNOWLEDGE_BASE`。上游通常已经带了这个标记，
       这里兜底是为了让「空」这个事实永远有一个机器可读的原因。
    """
    upstream = getattr(retrieval, "degraded", None)
    if upstream:
        return str(upstream)
    if not chunks:
        return EMPTY_KNOWLEDGE_BASE
    return None


def _final_degraded(pre_reason: str | None, retrieval: Any, answer: str) -> str | None:
    """`done.degraded` 的最终取值。

    生成前的降级原因优先（它是链路级事实）；只有链路本身正常时，才让
    `fallback.decide` 依据答案判断「低置信 / 空答案」这类生成后才可见的降级。
    """
    if pre_reason:
        return pre_reason
    should_degrade, notice = decide(retrieval, answer)
    return notice if should_degrade else None


def _latency(retrieve_ms: int, generate_ms: int, total_ms: int) -> dict[str, int]:
    return {"retrieve": retrieve_ms, "generate": generate_ms, "total": total_ms}


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------


class RAGEngine:
    """检索增强生成引擎：对外只暴露「流式」「非流式」「只检索」三个入口。"""

    def __init__(
        self,
        retriever: Any,
        llm: LLMProvider | None = None,
        settings: Any | None = None,
    ) -> None:
        """构造引擎。

        Args:
            retriever: 任何实现 `search(query, mode=..., top_k=...) -> RetrievalResult`
                的对象（duck typing，不 import 具体实现）。**原样持有、不做任何
                缓存或包装**：`vectorstore` / `bm25` 这类子组件会在运行期被替换
                （删文档、重建索引），引擎每次检索都现取，才不会用到旧引用。
            llm: LLM provider；缺省时走 `get_llm()` 工厂。
            settings: 配置对象；缺省时用全局 `config.settings` 单例。
        """
        self.retriever = retriever
        self.llm: LLMProvider = llm if llm is not None else get_llm()
        self.settings = settings if settings is not None else default_settings

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def retrieve_only(self, question: str, mode: str | None = None) -> RetrievalResult:
        """只做检索，**绝不触碰 LLM**。

        「只检索」是独立能力（文档核对、评测取候选、前端预览），
        调用它不该产生任何生成费用，也不该因为模型服务故障而失败。
        """
        return self._run_search(question, mode)

    def _run_search(self, question: str, mode: str | None) -> RetrievalResult:
        """调用检索器。任何异常都转成「空结果 + degraded」，不向上抛。"""
        requested = mode or DEFAULT_MODE
        started = time.perf_counter()
        try:
            return self.retriever.search(
                question, mode=requested, top_k=self.settings.rerank_top_n
            )
        except Exception as exc:  # noqa: BLE001 - 检索失败要能继续走降级链路
            logger.warning("检索失败，按空结果降级处理: %s", exc)
            return RetrievalResult(
                chunks=[],
                mode=requested,
                latency_ms=_elapsed_ms(started),
                degraded=f"{RETRIEVAL_FAILED}: {exc}",
            )

    # ------------------------------------------------------------------
    # 流式生成
    # ------------------------------------------------------------------
    def stream_answer(
        self,
        question: str,
        mode: str | None = None,
        history: list[dict] | None = None,
    ) -> Iterator[StreamEvent]:
        """流式问答。事件顺序即模块 docstring 中的契约，`done` 永远最后。

        `generate` 段耗时按「发起 LLM 调用 → 流结束」计，**含首 token 等待**：
        用户感知的等待正是这段，只统计首个 token 之后的时长会系统性低估卡顿。
        """
        total_started = time.perf_counter()

        retrieve_started = time.perf_counter()
        retrieval = self._run_search(question, mode)
        retrieve_ms = _elapsed_ms(retrieve_started)

        chunks = _chunks_of(retrieval)
        pre_reason = _pre_degraded_reason(retrieval, chunks)
        if pre_reason:
            # 契约第 1 条：降级必须在 token 之前，前端据此抢先渲染提示条。
            yield {"type": "degraded", "reason": pre_reason}

        error: str | None = None
        try:
            messages, citations = build_rag_messages(
                question, chunks, self.settings.max_context_tokens
            )
            messages = _with_history(messages, history)
        except Exception as exc:  # noqa: BLE001 - 组装失败也要走完事件序列
            logger.warning("组装 RAG 上下文失败，按无资料降级: %s", exc)
            error = _describe(exc)
            messages, citations = [], []

        # 契约第 2 条：引用表恒定发出且早于 token，前端才能在首个 token
        # 到达时就具备渲染 `[n]` 徽章所需的全部信息。
        yield {"type": "citations", "citations": [asdict(c) for c in citations]}

        answer_parts: list[str] = []
        generate_started = time.perf_counter()
        if error is None:
            try:
                # 整个调用（建立连接 + 迭代）都在 try 内：provider 既可能在
                # 调用时抛（鉴权失败），也可能在迭代到一半时抛（连接中断）。
                stream = self.llm.stream(messages, **self._llm_kwargs())
                for piece in stream:
                    if not piece:
                        continue
                    answer_parts.append(piece)
                    yield {"type": "token", "text": piece}
            except Exception as exc:  # noqa: BLE001 - 契约要求失败只发 error 事件
                logger.warning("LLM 流式生成失败，已转为 error 事件: %s", exc)
                error = _describe(exc)
        generate_ms = _elapsed_ms(generate_started)

        answer = "".join(answer_parts)
        usage = _usage_of(self.llm, messages, answer)
        # 契约第 4 条：usage 在 error 之前——失败时也上报（计数为 0），
        # 使「本次失败消耗了多少 prompt token」同样可观测。
        yield {
            "type": "usage",
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
        }

        if error:
            yield {"type": "error", "message": error}

        # 契约第 6 条：done 永远是最后一个事件，且 answer 必须等于
        # 所有 token 的拼接结果（前端以它为准做最终替换与落库）。
        yield {
            "type": "done",
            "answer": answer,
            "degraded": _final_degraded(pre_reason, retrieval, answer),
            "latency_ms": _latency(retrieve_ms, generate_ms, _elapsed_ms(total_started)),
        }

    def _llm_kwargs(self) -> dict[str, Any]:
        """生成参数一律取自配置：业务代码不硬编码默认值。"""
        cfg = self.settings
        return {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_output_tokens,
        }

    # ------------------------------------------------------------------
    # 非流式封装
    # ------------------------------------------------------------------
    def answer(self, question: str, mode: str | None = None) -> AgentResult:
        """非流式问答：消费 `stream_answer` 并汇总成 `AgentResult`。

        刻意**不**另写一条「一次性生成」的代码路径：两条路径一旦分家，
        Citations 编号、降级判定、耗时口径就会逐渐漂移，评测与线上看到的
        也不再是同一个东西。这里只是流式结果的一个折叠（fold）。

        `steps` 恒为 `[]`——路由与 ReAct 轨迹属于 Agent 层，本层不产生步骤。
        """
        citations: list[Citation] = []
        parts: list[str] = []
        degraded: str | None = None
        prompt_tokens = 0
        completion_tokens = 0
        latency_ms = _latency(0, 0, 0)

        for event in self.stream_answer(question, mode=mode):
            kind = event.get("type")
            if kind == "citations":
                # 事件里是 dict（可 JSON 序列化），回到 AgentResult 需要还原成
                # dataclass：asdict 是无损的，字段一一对应。
                citations = [Citation(**item) for item in event["citations"]]
            elif kind == "token":
                parts.append(event["text"])
            elif kind == "usage":
                prompt_tokens = _as_int(event.get("prompt_tokens"))
                completion_tokens = _as_int(event.get("completion_tokens"))
            elif kind == "done":
                degraded = event.get("degraded")
                latency_ms = dict(event.get("latency_ms") or latency_ms)

        return AgentResult(
            answer="".join(parts),
            citations=citations,
            steps=[],
            tokens={"prompt": prompt_tokens, "completion": completion_tokens},
            latency_ms=latency_ms,
            route="rag",
            degraded=degraded,
        )
