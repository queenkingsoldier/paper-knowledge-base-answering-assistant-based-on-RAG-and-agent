"""指标计算、聚合与健康检查。

本模块回答三个问题：

1. **这一次问答花了多少？** —— `compute(AgentResult)` 把一次问答压成一行扁平指标，
   它同时也是 JSONL 轨迹里的 `metrics` 字段，因此「落盘」与「聚合」共用同一套口径。
2. **最近一段时间整体表现如何？** —— `aggregate(records)` 从多条 `metrics` 算出
   P50/P95 延迟、token 总量、工具调用分布、错误率与降级率。
3. **底层组件还活着吗？** —— `health_check(retriever, llm)` 分别探测向量库、
   BM25 与 LLM，任一失败不影响其他探测结果。

另外提供 `render_trace_tree(steps)`：把 ReAct 步骤渲染成纯文本缩进树，
供 Streamlit 右栏与 CLI 复用（不引入任何 UI 依赖，因此可被测试直接断言）。

设计约束：本层属于旁路观测，所有对外函数都必须「不抛异常」——
写入失败、聚合样本为空、组件探测超时，都只降级为返回值里的提示文本。
"""

from __future__ import annotations

import json
import statistics
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from core.logging_utils import get_logger
from core.schemas import AgentResult, AgentStep

logger = get_logger(__name__)

# 健康探测的最长等待时间（秒）。构建书 §5.4.2 明确要求「快速失败」：
# 健康检查本身绝不能把页面拖住，因此给一个短超时，超时即判定为不健康。
PROBE_TIMEOUT_S = 5.0

# 未初始化组件的统一提示文案（前端横幅直接展示）。
_NOT_READY = "未初始化"

# 轨迹树里单段文本的展示上限：Observation 可能包含整段检索结果，
# 不截断会让右栏卡顿；截断并明确标注，避免误以为内容就这么多。
_TREE_TEXT_LIMIT = 500
_TREE_INDENT = "│   "


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _num(value: Any, default: int = 0) -> int:
    """把任意值安全地转成 int。

    轨迹里的 token / 延迟字段来自多个模块，偶尔会是 `None`、字符串或缺失，
    这里统一兜底成 0，避免因为一个脏字段让整条指标链路失败。
    """
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _percentile(values: list[int], percentile: float) -> int:
    """分位数（排序后线性插值，即 `method="inclusive"`）。

    ⚠️ 边界：`statistics.quantiles` 在样本数 < 2 时会抛
    `StatisticsError: must have at least two data points`。
    单样本的「分位数」就是它自己，必须在调用之前直接返回，
    否则延迟看板会在只有一条记录时崩掉。
    """
    if not values:
        return 0
    if len(values) < 2:
        return _num(values[0])
    try:
        cuts = statistics.quantiles(values, n=100, method="inclusive")
        index = max(0, min(len(cuts) - 1, int(percentile) - 1))
        return int(round(cuts[index]))
    except Exception as exc:  # noqa: BLE001 - 分位数算不出来也不该影响看板
        logger.warning("分位数计算失败，退化为最近秩法: %s", exc)
        ordered = sorted(_num(v) for v in values)
        rank = round(percentile / 100 * (len(ordered) - 1))
        return ordered[max(0, min(len(ordered) - 1, rank))]


def _as_pair(value: Any, ok_msg: str = "ok") -> tuple[bool, str]:
    """把组件 `health()` 的返回值统一成 `(bool, str)`。

    真实实现应返回 `tuple[bool, str]`，但探测代码面对的是鸭子类型对象，
    所以既容忍裸 bool，也容忍任意「可判真值」的返回值。
    """
    if isinstance(value, tuple) and len(value) == 2:
        return (bool(value[0]), str(value[1]))
    if isinstance(value, bool):
        return (value, ok_msg if value else "探测失败")
    return (bool(value), ok_msg)


def _safe(probe: Callable[..., tuple[bool, str]], *args: Any) -> tuple[bool, str]:
    """执行单个探测并吞掉所有异常——一个组件坏掉不能连累其他组件。"""
    try:
        ok, msg = probe(*args)
        return (bool(ok), str(msg))
    except Exception as exc:  # noqa: BLE001 - 探测失败就是探测结果的一部分
        logger.warning("健康探测 %s 异常: %s", getattr(probe, "__name__", probe), exc)
        return (False, f"探测异常: {type(exc).__name__}: {exc}")


def _with_timeout(fn: Callable[[], Any], timeout_s: float, label: str) -> Any:
    """在守护线程里跑 `fn`，超过 `timeout_s` 立即放弃等待。

    为什么不用 `ThreadPoolExecutor`：它的 `shutdown(wait=True)`（with 语句退出时）
    会等挂死的探测线程结束，超时就形同虚设。守护线程 + `Event.wait(timeout)`
    才能真正做到「健康检查不拖慢页面」。
    """
    box: dict[str, Any] = {}
    done = threading.Event()

    def _run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - 原始失败原因要带回调用方
            box["error"] = exc
        finally:
            done.set()

    worker = threading.Thread(target=_run, name="dsh-health-probe", daemon=True)
    worker.start()
    if not done.wait(max(0.1, float(timeout_s))):
        raise TimeoutError(f"{label}超时（>{float(timeout_s):g}s）")
    if "error" in box:
        raise box["error"]
    return box.get("value")


# ---------------------------------------------------------------------------
# 单次问答的指标
# ---------------------------------------------------------------------------


def compute(result: AgentResult) -> dict:
    """把一次 `AgentResult` 压成一行可聚合的指标。

    返回键（`steps/total_tokens/tools_used/errors/total_ms/degraded/route/n_citations`）
    是 JSONL 轨迹与前端指标卡片的共同契约；额外附带 `tokens_by_tool`，
    对应构建书 §5.4.2 的「每步 token 消耗按工具分组统计」。
    """
    steps = list(getattr(result, "steps", None) or [])
    tokens = getattr(result, "tokens", None)
    latency = getattr(result, "latency_ms", None)
    citations = getattr(result, "citations", None) or []

    total_tokens = (
        sum(_num(v) for v in tokens.values()) if isinstance(tokens, dict) else 0
    )

    tools_used: list[str] = []
    tokens_by_tool: Counter[str] = Counter()
    errors = 0
    for step in steps:
        action = str(getattr(step, "action", "") or "")
        if action:
            # 同一工具在一轮里被调用多次只算「用过一次」，
            # 这样 aggregate 的 tool_counts 语义是「用到该工具的请求数」。
            if action not in tools_used:
                tools_used.append(action)
            tokens_by_tool[action] += _num(getattr(step, "tokens", 0))
        if getattr(step, "error", None):
            errors += 1

    if isinstance(latency, dict):
        total_ms = latency.get("total")
        total_ms = (
            _num(total_ms)
            if total_ms is not None
            else sum(_num(v) for v in latency.values())
        )
    else:
        total_ms = 0

    return {
        "steps": len(steps),
        "total_tokens": total_tokens,
        "tools_used": tools_used,
        "errors": errors,
        "total_ms": total_ms,
        "degraded": getattr(result, "degraded", None),
        "route": str(getattr(result, "route", "") or ""),
        "n_citations": len(citations),
        "tokens_by_tool": dict(tokens_by_tool),
    }


# ---------------------------------------------------------------------------
# 多条记录聚合
# ---------------------------------------------------------------------------


def aggregate(records: list[dict] | None) -> dict:
    """把多条轨迹记录（`r["metrics"]`）聚合成看板数据。

    空列表 → `count=0` 且各数值为 0：看板在「还没有任何请求」时应当显示 0，
    而不是抛 `ZeroDivisionError`。
    """
    metrics: list[dict] = []
    for record in records or []:
        if isinstance(record, dict):
            item = record.get("metrics")
            if isinstance(item, dict):
                metrics.append(item)

    count = len(metrics)
    totals_ms = [_num(m.get("total_ms")) for m in metrics]
    total_tokens = sum(_num(m.get("total_tokens")) for m in metrics)
    total_steps = sum(_num(m.get("steps")) for m in metrics)
    total_errors = sum(_num(m.get("errors")) for m in metrics)
    degraded_count = sum(1 for m in metrics if m.get("degraded"))

    tool_counts: Counter[str] = Counter()
    tokens_by_tool: Counter[str] = Counter()
    for m in metrics:
        for tool in m.get("tools_used") or []:
            tool_counts[str(tool)] += 1
        per_tool = m.get("tokens_by_tool")
        if isinstance(per_tool, dict):
            for tool, tok in per_tool.items():
                tokens_by_tool[str(tool)] += _num(tok)

    return {
        "count": count,
        "p50_ms": _percentile(totals_ms, 50),
        "p95_ms": _percentile(totals_ms, 95),
        "avg_ms": round(sum(totals_ms) / count, 1) if count else 0.0,
        "total_tokens": total_tokens,
        "tool_counts": dict(tool_counts.most_common()),
        # 错误率按「步」计（错误步数 / 总步数），与 compute 里 errors 的口径一致。
        "error_rate": round(total_errors / total_steps, 4) if total_steps else 0.0,
        # 降级率按「请求」计（发生降级的请求数 / 总请求数）。
        "degraded_rate": round(degraded_count / count, 4) if count else 0.0,
        "total_errors": total_errors,
        "total_steps": total_steps,
        "avg_steps": round(total_steps / count, 2) if count else 0.0,
        "tokens_by_tool": dict(tokens_by_tool),
    }


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------


def _probe_vector_store(retriever: Any) -> tuple[bool, str]:
    """探测向量库：优先用 `VectorStore.health()`，退化为 `count()`。"""
    if retriever is None:
        return (False, _NOT_READY)
    store = getattr(retriever, "vectorstore", None)
    if store is None:
        return (False, _NOT_READY)

    health = getattr(store, "health", None)
    if callable(health):
        return _as_pair(health())

    count = getattr(store, "count", None)
    if callable(count):
        return (True, f"{_num(count())} 个向量")

    return (True, "已就绪（该实现未提供 health/count 接口）")


def _probe_bm25(retriever: Any) -> tuple[bool, str]:
    """探测 BM25：优先 `health()`，其次 `count()`，最后看索引目录是否存在。

    索引为空不算「组件坏了」——知识库本来就可能是空的，
    因此空索引返回 True 但在文案里说明，避免前端误报红色横幅。
    """
    if retriever is None:
        return (False, _NOT_READY)
    index = getattr(retriever, "bm25", None)
    if index is None:
        return (False, _NOT_READY)

    health = getattr(index, "health", None)
    if callable(health):
        return _as_pair(health())

    count = getattr(index, "count", None)
    if callable(count):
        try:
            n = _num(count())
            return (True, f"{n} 个文档块" if n else "0 个文档块（索引为空）")
        except Exception as exc:  # noqa: BLE001 - 未加载的索引回退到目录检查
            logger.debug("BM25 count 探测失败，回退到目录检查: %s", exc)

    index_dir = getattr(index, "index_dir", None)
    if index_dir is not None:
        path = Path(index_dir)
        return (
            (True, f"索引目录已存在: {path}")
            if path.exists()
            else (False, f"索引文件不存在: {path}")
        )

    return (True, "已就绪")


def _probe_llm(llm: Any, timeout_s: float = PROBE_TIMEOUT_S) -> tuple[bool, str]:
    """探测 LLM 连通性。

    优先调用可选的 `health()`；否则发一次 `max_tokens=1` 的最小请求——
    真探一次比只看对象是否存在可靠得多（key 失效、配额耗尽都能立刻暴露）。
    不论哪条路径都在守护线程里限时，保证健康检查不会拖慢页面。
    """
    if llm is None:
        return (False, _NOT_READY)

    health = getattr(llm, "health", None)
    if callable(health):
        return _as_pair(_with_timeout(health, timeout_s, "LLM 健康探测"))

    complete = getattr(llm, "complete", None)
    if not callable(complete):
        return (False, "该 LLM 实现既无 health 也无 complete，无法探测")

    text = _with_timeout(
        lambda: complete([{"role": "user", "content": "ping"}], max_tokens=1),
        timeout_s,
        "LLM 探测",
    )
    snippet = str(text or "").strip().replace("\n", " ")[:24]
    return (True, f"ok: {snippet}" if snippet else "ok")


def health_check(
    retriever: Any, llm: Any, timeout_s: float = PROBE_TIMEOUT_S
) -> dict[str, tuple[bool, str]]:
    """分别探测向量库、LLM 与 BM25，返回 `{组件: (是否健康, 说明)}`。

    三个探测彼此独立：任一抛异常或超时，都只把它自己标记为不健康
    （`retriever`/`llm` 为 None 时对应项为 `(False, "未初始化")`）。
    """
    return {
        "vector_store": _safe(_probe_vector_store, retriever),
        "llm": _safe(_probe_llm, llm, timeout_s),
        "bm25": _safe(_probe_bm25, retriever),
    }


# ---------------------------------------------------------------------------
# 轨迹树渲染
# ---------------------------------------------------------------------------


def _clip(text: Any, limit: int = _TREE_TEXT_LIMIT, indent: str = _TREE_INDENT) -> str:
    """截断过长文本，并把多行内容挂到树形缩进下，保持树的形状可读。"""
    body = str(text if text is not None else "")
    if limit and len(body) > limit:
        body = f"{body[:limit]}…（已截断，共 {len(body)} 字符）"
    return body.replace("\r\n", "\n").replace("\n", f"\n{indent}")


def render_trace_tree(steps: list[AgentStep] | None) -> str:
    """把 ReAct 步骤渲染成纯文本缩进树。

    刻意不返回 HTML/Markdown：Streamlit 的 `st.code`、CLI 与日志都能直接吃纯文本，
    前端要上色可以再包一层。空列表返回一句提示而不是空串，
    这样右栏不会出现「一块什么都没有的空白」。

    形状（与计划书一致）::

        ├─ Step 0 · kb_search (100ms, 10 tok)
        │   Thought: 想法0
        │   Action: kb_search {"q": "x"}
        │   Observation: 观察
    """
    try:
        items = list(steps or [])
        if not items:
            return "（本轮无 Agent 轨迹）"

        lines: list[str] = []
        for step in items:
            action = str(getattr(step, "action", "") or "-")
            head = (
                f"├─ Step {getattr(step, 'index', 0)} · {action} "
                f"({_num(getattr(step, 'latency_ms', 0))}ms, "
                f"{_num(getattr(step, 'tokens', 0))} tok)"
            )
            error = getattr(step, "error", None)
            if error:
                # 用 [错误] 而不是 ✗：Windows 控制台默认 GBK，U+2717 不在 GBK 里，
                # 打印时会直接抛 UnicodeEncodeError。树形符号本身在 GBK 内，无需替换。
                head += f" [错误] {_clip(error)}"
            lines.append(head)

            lines.append(f"│   Thought: {_clip(getattr(step, 'thought', ''))}")

            action_input = getattr(step, "action_input", None)
            if action_input is None:
                lines.append(f"│   Action: {action}")
            else:
                try:
                    rendered = json.dumps(action_input, ensure_ascii=False, default=str)
                except Exception:  # noqa: BLE001 - 参数不可序列化时退回 repr
                    rendered = repr(action_input)
                lines.append(f"│   Action: {action} {_clip(rendered)}")

            lines.append(f"│   Observation: {_clip(getattr(step, 'observation', ''))}")

        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001 - 渲染失败不该让右栏整块崩掉
        logger.warning("轨迹树渲染失败: %s", exc)
        return f"（轨迹渲染失败: {type(exc).__name__}: {exc}）"
