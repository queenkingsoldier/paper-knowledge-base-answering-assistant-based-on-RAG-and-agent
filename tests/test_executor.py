import time

from agent.executor import ToolExecutor
from agent.tools.base import Tool


def _tool(name, fn, timeout=30.0):
    return Tool(name=name, description="d", parameters={"type": "object", "properties": {}},
                func=fn, timeout_s=timeout)


def test_run_one_returns_string():
    ex = ToolExecutor({"echo": _tool("echo", lambda **kw: "ok")}, timeout_s=5, max_workers=2)
    assert ex.run_one(ex.tools["echo"], {}) == "ok"


def test_run_one_wraps_exception_into_readable_string():
    def boom(**kw):
        raise ValueError("坏了")

    ex = ToolExecutor({"b": _tool("b", boom)}, timeout_s=5, max_workers=2)
    assert "坏了" in ex.run_one(ex.tools["b"], {})


def test_run_one_converts_non_string_return_to_string():
    ex = ToolExecutor({"n": _tool("n", lambda **kw: 42)}, timeout_s=5, max_workers=2)
    assert ex.run_one(ex.tools["n"], {}) == "42"


def test_timeout_is_enforced_and_reported():
    def slow(**kw):
        time.sleep(2)
        return "too late"

    ex = ToolExecutor({"s": _tool("s", slow, timeout=0.3)}, timeout_s=0.3, max_workers=2)
    out = ex.run_one(ex.tools["s"], {})
    assert "超时" in out


def test_bad_kwargs_are_reported_not_raised():
    ex = ToolExecutor({"t": _tool("t", lambda required_arg: "x")}, timeout_s=5, max_workers=2)
    out = ex.run_one(ex.tools["t"], {"wrong": 1})
    assert isinstance(out, str) and out


def test_run_parallel_preserves_input_order():
    ex = ToolExecutor({}, timeout_s=5, max_workers=4)
    ex.tools = {f"t{i}": _tool(f"t{i}", (lambda i: (lambda **kw: f"r{i}"))(i)) for i in range(4)}
    outs = ex.run_parallel([(f"t{i}", {}) for i in range(4)])
    assert outs == ["r0", "r1", "r2", "r3"]


def test_run_parallel_is_actually_concurrent():
    def slow(**kw):
        time.sleep(0.4)
        return "done"

    ex = ToolExecutor({}, timeout_s=5, max_workers=4)
    ex.tools = {f"t{i}": _tool(f"t{i}", slow) for i in range(4)}
    t0 = time.time()
    ex.run_parallel([(f"t{i}", {}) for i in range(4)])
    assert time.time() - t0 < 1.2      # 串行需 1.6s


def test_run_parallel_isolates_failures():
    def boom(**kw):
        raise RuntimeError("bad")

    ex = ToolExecutor({}, timeout_s=5, max_workers=4)
    ex.tools = {"ok": _tool("ok", lambda **kw: "fine"), "bad": _tool("bad", boom)}
    outs = ex.run_parallel([("ok", {}), ("bad", {})])
    assert outs[0] == "fine" and "bad" in outs[1]


def test_run_parallel_empty_returns_empty():
    assert ToolExecutor({}, 5, 2).run_parallel([]) == []


def test_unknown_tool_in_parallel_is_reported():
    outs = ToolExecutor({}, 5, 2).run_parallel([("ghost", {})])
    assert "不存在" in outs[0]
