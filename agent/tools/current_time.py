"""当前时间工具（纯函数，无参数）。"""

from __future__ import annotations

from datetime import datetime

from agent.tools.base import register_tool


@register_tool(
    name="current_time",
    description="返回当前日期时间（本地时区），用于回答「今天/现在」这类问题",
    parameters={"type": "object", "properties": {}},
)
def current_time() -> str:
    """返回可读的当前日期时间，例如 `2026-09-25 14:03:07 Friday`。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")
