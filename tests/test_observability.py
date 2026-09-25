import json
from pathlib import Path

from core.schemas import AgentResult, AgentStep, Citation, RetrievalResult
from observability.metrics import aggregate, compute, health_check, render_trace_tree
from observability.trace import TraceRecorder


def _step(i, action, tokens=10, err=None):
    return AgentStep(index=i, thought=f"想法{i}", action=action, action_input={"q": "x"},
                     observation="观察", latency_ms=100, tokens=tokens, error=err)


def _res(steps=None, degraded=None):
    return AgentResult(answer="答案", citations=[Citation(1, "f.pdf", 1, "c", "s")],
                       steps=steps or [_step(0, "kb_search")], tokens={"prompt": 100, "completion": 50},
                       latency_ms={"retrieve": 10, "generate": 200, "total": 300},
                       route="agent", degraded=degraded)


def test_record_appends_jsonl(tmp_path: Path):
    r = TraceRecorder(tmp_path)
    r.record(_res(), "s1")
    r.record(_res(), "s2")
    lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["session_id"] == "s1"


def test_record_never_raises_on_unserializable(tmp_path: Path):
    bad = _res()
    bad.steps[0].action_input = {"obj": object()}
    TraceRecorder(tmp_path).record(bad, "s")     # 不应抛异常


def test_read_recent_returns_last_n_newest_first(tmp_path: Path):
    r = TraceRecorder(tmp_path)
    for i in range(5):
        r.record(_res(), f"s{i}")
    got = r.read_recent(n=2)
    assert len(got) == 2 and got[0]["session_id"] == "s4"


def test_read_recent_on_missing_file_returns_empty(tmp_path: Path):
    assert TraceRecorder(tmp_path / "none").read_recent() == []


def test_compute_extracts_core_metrics():
    m = compute(_res())
    assert m["steps"] == 1
    assert m["total_tokens"] == 150
    assert m["tools_used"] == ["kb_search"]
    assert m["total_ms"] == 300


def test_compute_counts_errors():
    m = compute(_res(steps=[_step(0, "kb_search", err="boom")]))
    assert m["errors"] == 1


def test_aggregate_computes_p50_and_p95():
    records = [{"metrics": {"total_ms": v, "total_tokens": 10, "tools_used": [], "errors": 0,
                            "steps": 1, "degraded": None}} for v in (100, 200, 300, 400, 500)]
    a = aggregate(records)
    assert a["count"] == 5
    assert a["p50_ms"] == 300
    assert a["p95_ms"] >= 400


def test_aggregate_on_empty_returns_zeros():
    a = aggregate([])
    assert a["count"] == 0 and a["p50_ms"] == 0


def test_aggregate_counts_tool_usage():
    records = [{"metrics": {"total_ms": 1, "total_tokens": 1, "tools_used": ["kb_search", "calculator"],
                            "errors": 0, "steps": 1, "degraded": None}}]
    assert aggregate(records)["tool_counts"]["kb_search"] == 1


def test_render_trace_tree_contains_all_three_phases():
    text = render_trace_tree([_step(0, "kb_search")])
    assert "Thought" in text and "Action" in text and "Observation" in text


def test_render_trace_tree_on_empty_list():
    assert isinstance(render_trace_tree([]), str)


def test_health_check_reports_each_component():
    class Retr:
        vectorstore = type("V", (), {"health": staticmethod(lambda: (True, "ok")), "count": staticmethod(lambda: 3)})()

    class LLM:
        def complete(self, *a, **k):
            return "pong"

    h = health_check(Retr(), LLM())
    assert "vector_store" in h and "llm" in h
    assert all(isinstance(v, tuple) and len(v) == 2 for v in h.values())


def test_health_check_reports_failure_without_raising():
    class BoomLLM:
        def complete(self, *a, **k):
            raise RuntimeError("401 unauthorized")

    h = health_check(None, BoomLLM())
    assert h["llm"][0] is False
