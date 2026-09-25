"""手写 ReAct 核心循环（Thought → Action → Observation）。

刻意**不引入任何 Agent 框架**：循环、解析容错、终止条件、轨迹记录全部显式写出来。
原因是这个循环是「Agent 为什么这么答」的唯一证据链——每一步的思考、动作、
观察都要能被逐行解释，框架会把这段证据藏进它的内部状态里。

四条设计红线：

1. **`run()` 永不抛异常**。LLM 挂掉、工具崩掉、输出格式非法、模型返回空串，
   都必须变成「可读的 answer + 非空 degraded」。前端与评测只消费 `AgentResult`，
   任何异常穿出去等于整个问答功能挂掉。
2. **工具错误是 Observation，不是异常**。`工具不存在 / 参数不匹配 / 超时 / 内部炸了`
   都要交回给 Agent，它看到错误才能换工具、换参数或直接作答——这正是 ReAct
   自愈能力的来源（也是本项目与「一次检索 + 一次生成」的固定管线最大的区别）。
3. **循环一定终止**：显式 `Finish`、重复调用检测、轮次上限，三者必居其一。
   这是防死循环的兜底，不是可选项：真实模型在检索空结果时非常容易原地打转。
4. **每一次失败都留痕**：解析失败、LLM 失败都各记一个 `AgentStep`，
   前端轨迹树里能直接看到「第 2 步：LLM 输出格式非法，已要求重试」。
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from agent.executor import ToolExecutor
from agent.memory import ConversationMemory, Turn
from agent.tools.base import Tool
from config import settings
from core.logging_utils import get_logger
from core.schemas import AgentResult, AgentStep, Citation

logger = get_logger(__name__)

# 连续 3 次「拿不到合法 Action」就放弃重试，转为强制收敛。
MAX_CONSECUTIVE_PARSE_FAILURES = 3

# 同一个 (action, args) 的第 4 次出现判定为死循环。
#
# 为什么不是第 2 次就判：真实对话里「同参数再查一次」常常是合理的（例如上一次
# 工具超时或被限流），过早判定会把可自愈的情况直接掐死；而 3 次重复已经足够说明
# 模型在原地打转，再跑下去只是烧 token。轮次上限（max_iterations）仍是最终保险。
LOOP_REPEAT_LIMIT = 3

_FINISH_ACTIONS = {"finish", "final", "final_answer", "完成", "结束", "最终答案"}

# ---- Action 解析用的正则 -------------------------------------------------
_THOUGHT_RE = re.compile(
    r"(?:Thought|思考)\s*[:：]\s*(.*?)(?=\n(?:Action|行动)\s*[:：]|\Z)", re.S
)
_ACTION_RE = re.compile(r"(?:Action|行动)\s*[:：]\s*(\S+)")
_INPUT_RE = re.compile(r"(?:Action\s*Input|行动输入)\s*[:：]\s*(.*)", re.S)
# 裸键补引号：{expression: 1} → {"expression": 1}
_BARE_KEY_RE = re.compile(r'([{,]\s*)([A-Za-z_]\w*)(\s*:)')
# 裸值补引号：{"expression": 1+1} → {"expression": "1+1"}
_BARE_VALUE_RE = re.compile(r"""(:\s*)([^"'\[\]{},][^,}]*?)(\s*[,}])""")
# 引用格式与 kb_search 的 Observation 一致：`[1] (vit.pdf 第1页) 片段正文`
_CITATION_RE = re.compile(r"\[(\d+)\] \((\S+) 第(\d+)页\)[ \t]*(.*)")

_SYSTEM_PROMPT_TEMPLATE = """你是「科研助理」Agent，通过「思考-行动-观察」循环回答用户关于论文知识库的问题。

可用工具：
{tools}

你必须严格按下面的格式输出，每次只输出一个动作：
Thought: 你的推理过程（可省略，但建议写）
Action: 工具名；如果你想结束并给出答案，就写 Finish
Action Input: 该工具参数的 JSON 对象；Finish 时写成 {{"answer": "最终答案"}}

规则：
1. 涉及论文事实、页码、数据、对比、计算或当前时间时，先调用工具，不要凭记忆作答。
2. 信息足够时必须用 Action: Finish 结束，不要无谓地反复检索同一个查询。
3. Action Input 必须是合法 JSON，键名与字符串值都要用双引号。
4. 观察里出现「错误：」时，换一种工具或参数，或直接 Finish 说明情况。"""


class ActionParseError(Exception):
    """LLM 输出无法解析成 `Thought / Action / Action Input`。"""


# ---------------------------------------------------------------------------
# 解析与容错
# ---------------------------------------------------------------------------
def _quote_bare_values(body: str) -> str:
    """给未加引号的裸值补上引号：`{"expression": 1+1}` → `{"expression": "1+1"}`。

    已经是合法 JSON 字面量的值（数字 / true / false / null）保持原样，
    所以 `{"top_k": 3}` 里的 3 仍然是整数，而不是字符串 "3"。
    """

    def _replace(match: re.Match) -> str:
        value = match.group(2)
        try:
            json.loads(value)
        except ValueError:
            return f'{match.group(1)}"{value}"{match.group(3)}'
        return match.group(0)

    return _BARE_VALUE_RE.sub(_replace, body)


def _load_action_input(body: str) -> dict:
    """把 Action Input 正文解析成 dict。

    容错顺序（每一步都只在前一步失败后才付出代价）：
    1. 直接 `json.loads`；
    2. 从正文里截出第一个 `{...}`——模型常在 JSON 后面继续写自己的话；
    3. 给裸键补双引号（`{query: "ViT"}`）；
    4. 再给裸值补双引号（`{expression: 1+1}`）；
    5. 单引号风格（`{'query': 'ViT'}`，Python dict 打印习惯）。

    Raises:
        ActionParseError: 以上全部失败。异常消息里带原始片段与「JSON」字样，
            方便前端直接展示、也方便 Agent 据此重新生成。
    """
    text = (body or "").strip()
    if not text or not text.strip("{}\t\r\n "):
        return {}  # "{}" / "{ }" / 空 → 无参数

    candidates: list[str] = [text]
    braced = re.search(r"\{.*\}", text, re.S)
    if braced and braced.group(0) != text:
        candidates.append(braced.group(0))

    attempts: list[str] = []
    for candidate in candidates:
        attempts.append(candidate)
        quoted_keys = _BARE_KEY_RE.sub(r'\1"\2"\3', candidate)
        attempts.append(quoted_keys)
        attempts.append(_quote_bare_values(quoted_keys))
        if "'" in candidate:
            single = candidate.replace("'", '"')
            attempts.append(single)
            attempts.append(_quote_bare_values(_BARE_KEY_RE.sub(r'\1"\2"\3', single)))

    for candidate in attempts:
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
        # 模型偶尔只给一个裸值；包一层，保证 Action Input 的契约始终是 dict。
        return {"input": parsed}

    raise ActionParseError(
        f"Action Input 不是合法 JSON：{text!r}。请用双引号包裹键名与字符串值。"
    )


def parse_action(raw: str) -> tuple[str, str, dict]:
    """把一次 LLM 输出解析成 `(thought, action, action_input)`。

    容错点：`Thought` 可缺省（很多模型直接给 Action）、中英文冒号都认、
    裸键裸值的 JSON 会尝试修复。

    Raises:
        ActionParseError: 抽不到 `Action`，或 Action Input 修不好。
    """
    text = raw or ""
    thought_match = _THOUGHT_RE.search(text)
    thought = thought_match.group(1).strip() if thought_match else ""

    action_match = _ACTION_RE.search(text)
    if not action_match:
        raise ActionParseError("未能解析出 Action")
    action = action_match.group(1).strip()

    input_match = _INPUT_RE.search(text)
    body = input_match.group(1) if input_match else ""
    args = _load_action_input(body)
    return thought, action, args


def _is_finish(action: str) -> bool:
    return (action or "").strip().lower() in _FINISH_ACTIONS


def _finish_answer(args: dict) -> str:
    """从 Finish 的 Action Input 里取答案正文。"""
    if not isinstance(args, dict):
        return ""
    for key in ("answer", "final_answer", "response", "result", "text"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float, bool)):
            return str(value)
    # 没按约定给 answer 字段时，退一步找参数里第一个非空字符串。
    for value in args.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_citations(observations: list[str]) -> list[Citation]:
    """从 Observation 里抽 `[n] (文件名 第p页)` 引用，去重后重新编号。

    重编号是必须的：检索工具为每次检索各自编了 [1][2]，而用户看到的答案是一篇
    完整的回复，只能有一套连续的引用编号。
    """
    citations: list[Citation] = []
    seen: set[tuple[str, int]] = set()
    for observation in observations:
        for _number, filename, page_no, snippet in _CITATION_RE.findall(observation or ""):
            key = (filename, int(page_no))
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                Citation(
                    index=len(citations) + 1,
                    filename=filename,
                    page_no=int(page_no),
                    # Observation 里只有文件名与页码，没有 chunk_id；
                    # 留空比编一个对不上任何真实 chunk 的假 ID 更诚实。
                    chunk_id="",
                    snippet=snippet.strip(),
                )
            )
    return citations


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))


def _is_tool(obj: Any) -> bool:
    """注册表里可能混着非工具项（测试用的辅助列表），只认能调用的。"""
    return hasattr(obj, "name") and (callable(getattr(obj, "func", None)) or callable(obj))


def _render_tools(tools: dict) -> str:
    """把工具注册表渲染成 Prompt 里的清单。

    刻意按 `self.tools` 渲染而不是全局注册表：Prompt 里出现 Agent 调不到的工具，
    模型就会去调它，然后拿回一串「工具不存在」——那是最没必要的浪费。
    """
    names = sorted(name for name, tool in tools.items() if _is_tool(tool))
    if not names:
        return "（本次没有可用工具，请思考后直接用 Action: Finish 给出答案。）"
    lines: list[str] = []
    for name in names:
        tool = tools[name]
        properties = (getattr(tool, "parameters", None) or {}).get("properties") or {}
        params = ", ".join(f"{key}: {spec.get('type', 'string')}" for key, spec in properties.items())
        lines.append(f"- {name}({params}): {getattr(tool, 'description', '')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class ReActAgent:
    """手写 ReAct 循环：反复「思考 → 行动 → 观察」，直到给出最终答案。"""

    def __init__(
        self,
        llm,
        tools: dict[str, Tool],
        max_iterations: int = settings.max_iterations,
        tool_timeout_s: float = settings.tool_timeout_s,
        executor: ToolExecutor | None = None,
        memory: ConversationMemory | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools or {}
        self.max_iterations = int(max_iterations)
        self.tool_timeout_s = float(tool_timeout_s)
        # 允许注入 executor（测试与「只读工具并行」的实验都用到）；缺省自建一个。
        self.executor = executor or ToolExecutor(self.tools, timeout_s=self.tool_timeout_s)
        # memory 由 Task 19 的 build_agent 注入：run() 前拼历史，run() 后追加本轮。
        self.memory = memory
        self.system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(tools=_render_tools(self.tools))
        self._default_session_id: str | None = None

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def _direct_route(self, question: str, started: float,
                      session_id: str | None) -> AgentResult | None:
        """规则路由：能用规则判定意图的问题直接调工具，跳过整个 ReAct 循环。

        见构建书 §5.3.3「工具选择路由：根据问题类型快速路由到对应工具，
        减少无效推理轮次」。纯算术、问时间这类问题走完整 ReAct 只是白烧
        token（一次 LLM 调用 + 一轮工具往返），规则既然能判定就该短路。

        返回 None 表示「判定不了，交给完整 ReAct 循环」。
        """
        if not settings.enable_router:
            return None
        try:
            from agent.router import classify

            decision = classify(question)
        except Exception as exc:  # noqa: BLE001 - 路由本身出问题不该影响作答
            logger.debug("规则路由失败，回退 ReAct 循环：%s", exc)
            return None

        if decision.kind != "direct_tool" or not decision.tool:
            return None
        tool = self.tools.get(decision.tool)
        if tool is None:
            # 该工具没注册（例如只注册了子集），老实退回 ReAct，别假装答了
            logger.debug("路由命中 %s 但该工具未注册，回退 ReAct", decision.tool)
            return None

        args = dict(decision.args or {})
        call_start = time.perf_counter()
        observation = self.executor.run_one(tool, args)
        step = AgentStep(
            index=0,
            thought=f"规则路由：{decision.reason}",
            action=decision.tool,
            action_input=args,
            observation=observation,
            latency_ms=_elapsed_ms(call_start),
            tokens=0,
            error=None,
        )
        answer = observation or "（工具没有返回内容）"
        self._remember(session_id, question, answer, [])
        return AgentResult(
            answer=answer,
            citations=[],
            steps=[step],
            tokens={"prompt": 0, "completion": 0},
            latency_ms={"retrieve": 0, "generate": 0, "total": _elapsed_ms(started)},
            route="direct_tool",
            degraded=None,
        )

    def run(
        self,
        question: str,
        history: list[dict] | None = None,
        session_id: str | None = None,
    ) -> AgentResult:
        """回答一个问题。**本方法永不抛异常。**

        Args:
            question: 用户问题。空串也安全：返回一段可读提示并标 degraded。
            history: 额外的历史消息（`[{"role": ..., "content": ...}]`）。
                技术规格书里的骨架写作 `run(question, session_id)`，因此第二个
                位置参数若是字符串，会被当作 session_id 处理。
            session_id: 会话 ID；注入了 memory 时用于拼接与写回多轮上下文。
        """
        started = time.perf_counter()
        if isinstance(history, str) and session_id is None:  # 兼容 run(question, sid)
            history, session_id = None, history
        text = (question or "").strip()

        # ---- 规则路由短路：能用规则判定意图的问题不必进 ReAct 循环 ----
        if text:
            direct = self._direct_route(text, started, self._resolve_session(session_id))
            if direct is not None:
                return direct

        steps: list[AgentStep] = []
        observations: list[str] = []
        messages: list[dict] = []
        answer = ""
        degraded: str | None = None
        prompt_tokens = 0
        completion_tokens = 0
        generate_ms = 0
        seen: dict[tuple[str, str], int] = {}
        consecutive_failures = 0

        try:
            if not text:
                degraded = "empty_question"
                answer = "问题是空的，请告诉我想了解什么。"
            else:
                messages = self._initial_messages(text, history, self._resolve_session(session_id))
                for _iteration in range(self.max_iterations):
                    call_start = time.perf_counter()
                    try:
                        raw = self.llm.complete(messages, stop=["Observation:"])
                    except Exception as e:  # noqa: BLE001 - LLM 故障也要变成 Observation
                        latency = _elapsed_ms(call_start)
                        generate_ms += latency
                        consecutive_failures += 1
                        observation = f"模型调用失败：{e}"
                        observations.append(observation)
                        steps.append(
                            AgentStep(
                                index=len(steps),
                                thought="",
                                action="",
                                action_input={},
                                observation=observation,
                                latency_ms=latency,
                                tokens=0,
                                error=str(e),
                            )
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Observation: {observation}\n"
                                    "请严格按 Thought/Action/Action Input 格式重新输出。"
                                ),
                            }
                        )
                        if consecutive_failures >= MAX_CONSECUTIVE_PARSE_FAILURES:
                            degraded = "llm_error"
                            break
                        continue

                    latency = _elapsed_ms(call_start)
                    generate_ms += latency
                    step_prompt, step_completion = self._last_usage()
                    prompt_tokens += step_prompt
                    completion_tokens += step_completion
                    step_tokens = step_prompt + step_completion
                    raw = raw if isinstance(raw, str) else str(raw or "")
                    messages.append({"role": "assistant", "content": raw})

                    # ---- 解析：失败作为 Observation 交回给模型重试 ----
                    try:
                        thought, action, args = parse_action(raw)
                    except ActionParseError as e:
                        consecutive_failures += 1
                        observation = (
                            f"解析失败：{e}。请严格按 Thought/Action/Action Input 格式重新输出。"
                        )
                        observations.append(observation)
                        steps.append(
                            AgentStep(
                                index=len(steps),
                                thought="",
                                action="",
                                action_input={},
                                observation=observation,
                                latency_ms=latency,
                                tokens=step_tokens,
                                error=str(e),
                            )
                        )
                        messages.append(
                            {"role": "user", "content": f"Observation: {observation}"}
                        )
                        if consecutive_failures >= MAX_CONSECUTIVE_PARSE_FAILURES:
                            degraded = "parse_failed"
                            break
                        continue
                    consecutive_failures = 0

                    # ---- 终止条件 1：显式 Finish ----
                    if _is_finish(action):
                        answer = _finish_answer(args)
                        if not answer:
                            degraded = degraded or "finish_without_answer"
                        steps.append(
                            AgentStep(
                                index=len(steps),
                                thought=thought,
                                action=action,
                                action_input=args,
                                observation=answer,
                                latency_ms=latency,
                                tokens=step_tokens,
                            )
                        )
                        break

                    # ---- 终止条件 2：重复调用检测 ----
                    fingerprint = (
                        action.lower(),
                        json.dumps(args, sort_keys=True, ensure_ascii=False),
                    )
                    repeats = seen.get(fingerprint, 0)
                    seen[fingerprint] = repeats + 1
                    if repeats >= LOOP_REPEAT_LIMIT:
                        observation = (
                            f"错误：检测到重复调用（{action} 已用相同参数调用 "
                            f"{repeats} 次），已中止循环。"
                        )
                        observations.append(observation)
                        steps.append(
                            AgentStep(
                                index=len(steps),
                                thought=thought,
                                action=action,
                                action_input=args,
                                observation=observation,
                                latency_ms=latency,
                                tokens=step_tokens,
                                error="loop_detected",
                            )
                        )
                        degraded = "loop_detected"
                        break

                    # ---- 行动：未知工具与工具故障都只是 Observation ----
                    tool = self._tool(action)
                    if tool is None:
                        observation = (
                            f"错误：工具 {action} 不存在。"
                            f"可用工具：{', '.join(self._available_tools())}"
                        )
                    else:
                        try:
                            observation = self.executor.run_one(tool, args)
                        except Exception as e:  # noqa: BLE001 - executor 已兜底，此处双保险
                            observation = f"错误：工具 {action} 执行失败：{e}"
                    observation = observation if isinstance(observation, str) else str(observation)

                    observations.append(observation)
                    steps.append(
                        AgentStep(
                            index=len(steps),
                            thought=thought,
                            action=action,
                            action_input=args,
                            observation=observation,
                            latency_ms=latency,
                            tokens=step_tokens,
                        )
                    )
                    messages.append({"role": "user", "content": f"Observation: {observation}"})
                else:
                    # ---- 终止条件 3：轮次耗尽（for-else：没被 break 才会走到这里）----
                    degraded = degraded or "max_iterations_reached"

                if not answer:
                    answer = self._force_conclude(text, messages, degraded or "unknown")
        except Exception as e:  # noqa: BLE001 - 最后一道保险：run() 绝不抛异常
            logger.exception("ReAct 循环出现未预期异常")
            degraded = degraded or "unexpected_error"
            if not answer:
                answer = f"抱歉，处理这个问题时出现了未预期的异常：{e}"

        citations = _extract_citations(observations)
        answer = answer or "抱歉，我暂时无法给出答案，请重试或换一种问法。"
        self._remember(session_id, text, answer, citations)
        total_ms = _elapsed_ms(started)
        return AgentResult(
            answer=answer,
            citations=citations,
            steps=steps,
            tokens={"prompt": prompt_tokens, "completion": completion_tokens},
            latency_ms={"retrieve": 0, "generate": generate_ms, "total": total_ms},
            route="agent",
            degraded=degraded,
        )

    # ------------------------------------------------------------------
    # 强制收敛
    # ------------------------------------------------------------------
    def _force_conclude(self, question: str, scratchpad: list[dict], reason: str) -> str:
        """终止条件触发后，用已有观察逼出最终答案。**保证返回非空字符串。**

        措辞刻意不提「Action Input」等字面量：让模型「不要输出 X」有时反而会
        把它诱导出来，最省事的做法是只说「直接给答案」。
        """
        instruction = (
            f"本轮推理已终止（原因：{reason}）。请基于以上全部观察，"
            "直接给出面向用户的最终答案：不要再调用任何工具，不要再输出推理过程的"
            "中间标记，只输出答案正文；信息不足时请明确说明缺少什么。"
        )
        text = ""
        try:
            text = self.llm.complete(
                [*scratchpad, {"role": "user", "content": instruction}],
                stop=["Observation:"],
            )
        except Exception as e:  # noqa: BLE001 - 收敛失败就退到观察兜底
            logger.warning("强制收敛调用失败：%s", e)
        text = (text or "").strip() if isinstance(text, str) else ""

        if text:
            # 模型有时仍会吐一个 Finish 动作出来，这里再解析一次，
            # 免得把 ReAct 脚手架当答案返回给用户。
            try:
                _, action, args = parse_action(text)
                if _is_finish(action):
                    inner = _finish_answer(args)
                    if inner:
                        return inner
            except ActionParseError:
                pass
            return text

        notes = [
            message["content"][len("Observation:") :].strip()
            for message in scratchpad
            if isinstance(message, dict)
            and message.get("role") == "user"
            and str(message.get("content", "")).startswith("Observation:")
        ]
        if notes:
            return "（未能生成正式结论，以下为已有的观察记录）\n" + "\n".join(
                f"- {note}" for note in notes
            )
        return (
            f"抱歉，我未能就「{question}」得出结论（原因：{reason}），"
            "请换个问法或补充更多信息。"
        )

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _tool(self, name: str) -> Any | None:
        candidate = self.tools.get(name)
        return candidate if _is_tool(candidate) else None

    def _available_tools(self) -> list[str]:
        return sorted(name for name, tool in self.tools.items() if _is_tool(tool))

    def _last_usage(self) -> tuple[int, int]:
        usage = getattr(self.llm, "last_usage", None) or {}
        try:
            return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        except (TypeError, ValueError, AttributeError):
            return 0, 0

    def _resolve_session(self, session_id: str | None) -> str | None:
        """确定本次问答归属的会话；注入了 memory 却没有 ID 时自动开一个。"""
        if self.memory is None:
            return session_id
        if session_id:
            try:
                self.memory.new_session(session_id)  # 幂等：已存在则原样返回
            except TypeError:  # 兼容只提供无参 new_session() 的记忆实现
                pass
            return session_id
        if self._default_session_id is None:
            self._default_session_id = self.memory.new_session()
        return self._default_session_id

    def _initial_messages(
        self, question: str, history: list[dict] | None, session_id: str | None
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        if self.memory is not None and session_id:
            try:
                context, summary = self.memory.build_context(session_id)
            except Exception as e:  # noqa: BLE001 - 记忆故障不该影响本次问答
                logger.warning("读取会话记忆失败：%s", e)
                context, summary = [], None
            if summary:
                messages.append({"role": "system", "content": f"此前对话的摘要：{summary}"})
            messages.extend(context)
        for item in history or []:
            if isinstance(item, dict) and item.get("role") and item.get("content") is not None:
                messages.append(
                    {"role": str(item["role"]), "content": str(item["content"])}
                )
        messages.append({"role": "user", "content": question})
        return messages

    def _remember(
        self, session_id: str | None, question: str, answer: str, citations: list[Citation]
    ) -> None:
        if self.memory is None or not session_id or not question:
            return
        try:
            self.memory.append(session_id, Turn(role="user", content=question))
            self.memory.append(
                session_id,
                Turn(
                    role="assistant",
                    content=answer,
                    cited_chunk_ids=[c.chunk_id for c in citations if c.chunk_id],
                ),
            )
        except Exception as e:  # noqa: BLE001 - 写记忆失败不该让回答失败
            logger.warning("写入会话记忆失败：%s", e)
