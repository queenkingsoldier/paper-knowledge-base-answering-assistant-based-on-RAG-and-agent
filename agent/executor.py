"""工具执行器：超时、异常隔离与真并发。

三个设计要点：

1. **一律返回 `str`，绝不抛异常**。ReAct 的自愈能力建立在「工具故障变成
   Observation」之上——Agent 看到 `错误：...` 才知道该换工具、换参数还是直接作答。
   任何异常穿出执行器都会直接打断整个循环，比工具本身失败严重得多。

2. **超时用线程池而不是信号**。`signal.alarm` 只在主线程可用，在 Streamlit 的
   工作线程里会直接抛 `ValueError`；`ThreadPoolExecutor(1) + future.result(timeout)`
   在任意线程都能用。代价是超时后工作线程仍会跑完（Python 无法强杀线程），
   所以这里显式 `shutdown(wait=False)`，不让调用方替别人的超时买单。

3. **并行只负责「顺序」与「隔离」**。是否值得并行由调用方（Agent）决定；
   执行器保证输出顺序与输入严格一致（按下标回填），且单个任务的失败不会
   影响其他任务——否则并行会变成「一个坏工具毁掉整轮推理」。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

from agent.tools.base import Tool
from config import settings
from core.logging_utils import get_logger

logger = get_logger(__name__)


class ToolExecutor:
    """按工具逐个执行的执行器，支持单次超时与批量并行。"""

    def __init__(
        self,
        tools: dict[str, Tool],
        timeout_s: float = settings.tool_timeout_s,
        max_workers: int = settings.parallel_max_workers,
    ) -> None:
        # 刻意不做拷贝：调用方（测试与前端）会直接改 self.tools 换注册表。
        self.tools = tools or {}
        self.timeout_s = timeout_s
        self.max_workers = max(1, int(max_workers))

    # ------------------------------------------------------------------
    # 单个工具
    # ------------------------------------------------------------------
    def run_one(self, tool: Tool, args: dict | None = None) -> str:
        """执行一个工具，永远返回可读字符串。

        工具自身的 `timeout_s` 优先于执行器的默认超时（工具作者最清楚自己多久算慢）。
        """
        name = getattr(tool, "name", "unknown")
        kwargs = dict(args or {})
        timeout = getattr(tool, "timeout_s", None) or self.timeout_s

        raw = self._invoke(tool, kwargs, timeout)
        # 工具返回 None / 42 / list 都不能破坏 Observation 的字符串契约。
        return raw if isinstance(raw, str) else str(raw)

    def run(self, name: str, args: dict | None = None) -> str:
        """按名字执行工具（spec 的循环骨架用的就是这个入口）。"""
        tool = self.tools.get(name)
        if tool is None:
            return f"错误：工具 {name} 不存在。"
        return self.run_one(tool, args)

    def _invoke(self, tool: Tool, kwargs: dict, timeout: float) -> str:
        """在独立线程里跑工具，把超时与异常都转成字符串。"""
        name = getattr(tool, "name", "unknown")
        func = getattr(tool, "func", tool)
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            # 参数不匹配的 TypeError 在 worker 里抛出，由 future.result() 带回。
            future = pool.submit(func, **kwargs)
        except Exception as e:  # noqa: BLE001 - 提交失败（极端情况）也要可读
            logger.warning("工具 %s 提交失败: %s", name, e)
            pool.shutdown(wait=False)
            return f"错误：工具 {name} 执行失败：{e}"
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            logger.warning("工具 %s 超时（>%ss）", name, timeout)
            return f"错误：工具 {name} 超时（>{timeout}s），请尝试其他方式或直接回答。"
        except TypeError as e:
            # 最常见的现实故障：LLM 给的 Action Input 缺参 / 多参 / 名字写错。
            return f"错误：工具 {name} 参数不匹配：{e}。请检查 Action Input。"
        except Exception as e:  # noqa: BLE001 - 工具内部任何异常都不许穿出去
            return f"错误：工具 {name} 执行失败：{e}"
        finally:
            # 超时的线程仍在跑；不等它，否则超时保护形同虚设。
            pool.shutdown(wait=False)

    # ------------------------------------------------------------------
    # 批量并行
    # ------------------------------------------------------------------
    def run_parallel(self, calls: list[tuple[str, dict]]) -> list[str]:
        """并行执行 `[(工具名, 参数), ...]`，返回顺序与输入严格一致。

        未知工具在对应槽位留下可读错误，其余任务照常执行。
        """
        if not calls:
            return []
        results: list[str] = [""] * len(calls)
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(calls))) as pool:
            pending: dict = {}
            for index, (name, args) in enumerate(calls):
                tool = self.tools.get(name)
                if tool is None:
                    results[index] = f"错误：工具 {name} 不存在。"
                    continue
                # 下标随任务一起带走，靠它回填——绝不依赖完成顺序。
                pending[pool.submit(self.run_one, tool, args or {})] = index
            for future in pending:
                index = pending[future]
                try:
                    results[index] = future.result()
                except Exception as e:  # noqa: BLE001 - run_one 已兜底，这里是双保险
                    results[index] = f"错误：工具 {calls[index][0]} 执行失败：{e}"
        return results
