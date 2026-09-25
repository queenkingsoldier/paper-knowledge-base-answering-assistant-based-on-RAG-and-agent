"""Agent 工具集：基础抽象与注册表，以及 8 个具体工具模块。

具体工具模块由 `agent.tools.base.build_registry()` 在运行时逐个导入触发注册，
因此这里只导出注册表 API，不导出任何与子模块同名的工具函数
（否则 `from agent.tools import web_search` 会拿到函数而不是模块）。
"""

from agent.tools.base import Tool, build_registry, register_tool, tools_description

__all__ = ["Tool", "build_registry", "register_tool", "tools_description"]
