"""工具抽象与注册表。

两个设计要点：

1. **插件式注册表**：`build_registry()` 逐个 import 全部工具模块来触发
   `@register_tool`。某个工具模块缺失（尚未实现，或它依赖的第三方库没装）只记一条
   debug 日志后跳过，不影响其余工具——某个工具写坏不应该让整个 Agent 起不来，
   也让每个工具模块能独立开发和测试。刻意只捕获 `ImportError`：其他异常是真 bug，
   掩盖它们只会把故障推迟到更难排查的地方。

2. **注册态函数签名固定**：`register_tool` 把函数包成「kwargs 调用 + 返回 str」的
   可调用对象存进注册表。LLM 解析出的 Action Input 天然是 dict（所以按 kwargs 调用），
   而 ReAct 的 Observation 必须是字符串（所以强制 `str()`）。
   装饰器本身返回**原始函数**，这样纯函数（如 `extract_keywords` 返回 `list[str]`）
   保持自己声明的返回类型，只有注册表里的那份做 str 强转。
"""

from __future__ import annotations

import functools
import importlib
from dataclasses import dataclass
from typing import Any, Callable, ClassVar

from core.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Tool:
    """一个可被 ReAct Agent 调用的工具。"""

    name: str
    description: str
    parameters: dict  # JSON Schema（OpenAI function calling 风格）
    func: Callable[..., str]
    timeout_s: float = 30.0

    # 类级注册表：所有工具模块 import 时通过装饰器往里写。
    registry: ClassVar[dict[str, "Tool"]] = {}

    def __call__(self, **kwargs: Any) -> str:
        """等价于 `tool.func(**kwargs)`，方便 executor 直接调用。"""
        return self.func(**kwargs)


# 全部 8 个工具模块。新增工具只需在模块里加装饰器 + 在此登记一行，
# 不需要改动 react.py。
TOOL_MODULES: tuple[str, ...] = (
    "calculator",
    "current_time",
    "keywords",
    "web_search",
    "kb_search",
    "paper_meta",
    "paper_compare",
    "summarize",
)


def register_tool(
    name: str,
    description: str,
    parameters: dict,
    timeout_s: float = 30.0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """把一个函数注册成名为 `name` 的工具。

    Args:
        name: 工具名，同时是 LLM 在 `Action:` 里写的标识符。
        description: 给 LLM 看的功能说明（会进 System Prompt）。
        parameters: 参数的 JSON Schema。
        timeout_s: 单次调用超时，供 executor 使用。

    Returns:
        装饰后的**原始函数**（不含 str 强转）；注册表里存的是强转副本。
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> str:
            return str(func(*args, **kwargs))

        Tool.registry[name] = Tool(
            name=name,
            description=description,
            parameters=parameters,
            func=wrapper,
            timeout_s=timeout_s,
        )
        return func

    return decorator


def build_registry() -> dict[str, Tool]:
    """导入全部工具模块，返回 `{工具名: Tool}` 快照。

    重复调用是幂等的（同名工具覆盖注册）。缺失的工具模块被跳过。
    """
    package = __package__ or "agent.tools"
    for module_name in TOOL_MODULES:
        try:
            importlib.import_module(f"{package}.{module_name}")
        except ImportError as e:
            logger.debug("工具模块 %s 不可用，已跳过：%s", module_name, e)
    return dict(Tool.registry)


def tools_description() -> str:
    """把注册表渲染成 System Prompt 里的「可用工具」清单。

    形如：
        - calculator(expression: string): 计算数学表达式，支持 + - * / ** 与括号
        - current_time(): 返回当前日期时间
    按名字排序，保证同一份注册表每次渲染出的 Prompt 完全一致。
    """
    lines: list[str] = []
    for name, tool in sorted(build_registry().items()):
        properties = tool.parameters.get("properties") or {}
        params = ", ".join(
            f"{key}: {spec.get('type', 'string')}" for key, spec in properties.items()
        )
        lines.append(f"- {name}({params}): {tool.description}")
    return "\n".join(lines)
