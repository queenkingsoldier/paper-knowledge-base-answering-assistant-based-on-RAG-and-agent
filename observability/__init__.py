"""可观测性层：Agent 轨迹（trace）与指标（metrics）。

对外只暴露四个能力：

- `TraceRecorder`：把每次问答以 JSONL 追加落盘，并可按「最近 n 条」回读；
- `compute`：把一次 `AgentResult` 压成一行可聚合的指标；
- `aggregate`：把多条指标聚合成 P50/P95、工具分布、降级率等看板数据；
- `health_check` / `render_trace_tree`：前端健康横幅与轨迹树的公共实现。

设计约束：本层是「旁路」——任何观测失败都不允许影响主问答流程，
因此写入、读取、探测全部自带兜底，绝不向上抛异常。
"""

from __future__ import annotations

from observability.metrics import aggregate, compute, health_check, render_trace_tree
from observability.trace import TraceRecorder

__all__ = [
    "TraceRecorder",
    "aggregate",
    "compute",
    "health_check",
    "render_trace_tree",
]
