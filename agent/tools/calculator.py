"""计算器工具：AST 白名单安全求值。

绝不用裸 `eval` —— 那会让 `__import__('os').system(...)` 直达系统。
两道防线：
1. `ast.parse(mode="eval")` + 白名单遍历，只放行算术表达式需要的节点；
2. `compile` 后用 `{"__builtins__": {}}` 作为 globals 求值，即使白名单被绕过也拿不到内建函数。
"""

from __future__ import annotations

import ast

from agent.tools.base import register_tool

# 白名单：算术表达式所需的全部节点类型。
# Call / Attribute / Name / Subscript / Lambda / 推导式等一律拒绝——
# 它们正是 `__import__('os').system(...)`、`(1).__class__`、`open('x')` 的入口。
_ALLOWED: tuple[type[ast.AST], ...] = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd, ast.FloorDiv,
)


def safe_eval(expression: str) -> float:
    """在 AST 白名单下求值一个数学表达式。

    Raises:
        ValueError: 表达式为空、语法错误、含不允许的语法元素、除零。
    """
    expr = (expression or "").strip()
    if not expr:
        raise ValueError("表达式为空")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"表达式语法错误: {e}") from e
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"表达式含不允许的语法元素: {type(node).__name__}")
    try:
        return float(eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {}))
    except ZeroDivisionError as e:
        raise ValueError("除数不能为零") from e


@register_tool(
    name="calculator",
    description="计算数学表达式，支持 + - * / // % ** 与括号",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "纯数学表达式，例如 (3.14 * 2.56) ** 2",
            }
        },
        "required": ["expression"],
    },
)
def calculator(expression: str) -> str:
    """工具入口：把 `safe_eval` 的异常转成可读 Observation，交给 Agent 自愈。"""
    try:
        return str(safe_eval(expression))
    except Exception as e:  # 工具必须返回字符串，绝不把异常抛给 Agent 循环
        return f"计算失败：{e}"
