import pytest

from agent.react import ActionParseError, ReActAgent, parse_action
from agent.tools.base import Tool
from core.providers import MockLLM


def test_parse_action_full_triple():
    raw = ('Thought: 需要检索\n'
           'Action: kb_search\n'
           'Action Input: {"query": "ViT", "top_k": 3}')
    thought, action, args = parse_action(raw)
    assert thought == "需要检索" and action == "kb_search" and args["query"] == "ViT"


def test_parse_action_tolerates_missing_thought():
    _, action, _ = parse_action('Action: current_time\nAction Input: {}')
    assert action == "current_time"


def test_parse_action_tolerates_chinese_colon():
    _, action, _ = parse_action('思考: x\nAction：current_time\nAction Input: {}')
    assert action == "current_time"


def test_parse_action_repairs_unquoted_keys():
    _, _, args = parse_action('Action: calculator\nAction Input: {expression: 1+1}')
    assert args["expression"] == "1+1"


def test_parse_action_missing_action_raises():
    with pytest.raises(ActionParseError):
        parse_action("Thought: 我什么也没做")


def test_parse_action_empty_input_raises():
    with pytest.raises(ActionParseError):
        parse_action("")


def test_parse_action_bad_json_raises_with_readable_message():
    with pytest.raises(ActionParseError) as e:
        parse_action('Action: kb_search\nAction Input: {"query": ')
    assert "JSON" in str(e.value) or "参数" in str(e.value)


class ScriptedLLM(MockLLM):
    """按脚本依次返回，用于驱动确定性 ReAct 测试。"""

    def __init__(self, script):
        super().__init__()
        self.script, self.i = script, 0

    def complete(self, messages, **kw):
        out = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return out


def _tools():
    calls = []

    def make(name, fn):
        return Tool(name=name, description=f"{name} 工具",
                    parameters={"type": "object", "properties": {}}, func=fn)

    return {
        "echo": make("echo", lambda **kw: calls.append(kw) or "回显结果"),
        "_calls": calls,
    }


def test_agent_finishes_immediately_on_finish_action():
    llm = ScriptedLLM(['Thought: 已知\nAction: Finish\nAction Input: {"answer": "答案文本"}'])
    res = ReActAgent(llm=llm, tools={}).run("问题")
    assert res.answer == "答案文本"
    assert len(res.steps) == 1


def test_agent_executes_tool_then_finishes():
    tools = _tools()
    llm = ScriptedLLM([
        'Thought: 先回显\nAction: echo\nAction Input: {"x": 1}',
        'Thought: 完成\nAction: Finish\nAction Input: {"answer": "最终答案"}',
    ])
    res = ReActAgent(llm=llm, tools=tools).run("问题")
    assert res.answer == "最终答案"
    assert "回显结果" in res.steps[0].observation
    assert tools["_calls"] == [{"x": 1}]


def test_agent_stops_after_max_iterations_when_never_finishing():
    llm = ScriptedLLM(['Thought: 再查一次\nAction: echo\nAction Input: {"n": 1}'])
    res = ReActAgent(llm=llm, tools=_tools(), max_iterations=3).run("问题")
    assert len(res.steps) <= 4          # 3 步 + 可能的收敛步
    assert res.degraded == "max_iterations_reached"


def test_repeated_identical_tool_call_is_detected_as_loop():
    llm = ScriptedLLM(['Thought: 重复\nAction: echo\nAction Input: {"n": 1}'])
    res = ReActAgent(llm=llm, tools=_tools(), max_iterations=6).run("问题")
    assert res.degraded == "loop_detected"
    assert len(res.steps) <= 4


def test_malformed_action_is_retried_then_forced_to_conclude():
    llm = ScriptedLLM(['这不是合法输出', '这也不是', '还是不合法'])
    res = ReActAgent(llm=llm, tools=_tools(), max_iterations=6).run("问题")
    assert res.degraded in ("parse_failed", "max_iterations_reached")
    assert isinstance(res.answer, str) and res.answer


def test_unknown_tool_becomes_observation_not_exception():
    tools = _tools()
    llm = ScriptedLLM([
        'Thought: 用不存在的工具\nAction: nope\nAction Input: {}',
        'Thought: 改用 echo\nAction: echo\nAction Input: {}',
        'Thought: 完成\nAction: Finish\nAction Input: {"answer": "ok"}',
    ])
    res = ReActAgent(llm=llm, tools=tools).run("问题")
    assert "不存在" in res.steps[0].observation or "未知" in res.steps[0].observation
    assert res.answer == "ok"


def test_tool_exception_becomes_observation_not_exception():
    def boom(**kw):
        raise RuntimeError("工具内部炸了")

    tools = {"boom": Tool(name="boom", description="d",
                          parameters={"type": "object", "properties": {}}, func=boom)}
    llm = ScriptedLLM([
        'Thought: 调用\nAction: boom\nAction Input: {}',
        'Thought: 完成\nAction: Finish\nAction Input: {"answer": "兜底"}',
    ])
    res = ReActAgent(llm=llm, tools=tools).run("问题")
    assert "工具内部炸了" in res.steps[0].observation
    assert res.answer == "兜底"


def test_empty_question_is_rejected_gracefully():
    res = ReActAgent(llm=ScriptedLLM(['Thought: x\nAction: Finish\nAction Input: {"answer": "a"}']),
                     tools={}).run("")
    assert isinstance(res.answer, str)


def test_steps_carry_latency_and_index():
    llm = ScriptedLLM(['Thought: t\nAction: Finish\nAction Input: {"answer": "a"}'])
    res = ReActAgent(llm=llm, tools={}).run("q")
    assert res.steps[0].index == 0
    assert res.steps[0].latency_ms >= 0


def test_run_never_raises_when_llm_itself_fails():
    class BoomLLM(MockLLM):
        def complete(self, messages, **kw):
            raise RuntimeError("llm 挂了")

    res = ReActAgent(llm=BoomLLM(), tools={}).run("问题")
    assert isinstance(res.answer, str)
    assert res.degraded is not None
