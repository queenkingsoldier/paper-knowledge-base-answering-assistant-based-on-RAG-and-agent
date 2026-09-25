from pathlib import Path

from agent.memory import ConversationMemory, Turn
from core.providers import MockLLM


def test_new_session_ids_are_unique():
    m = ConversationMemory()
    assert m.new_session() != m.new_session()


def test_append_and_get_roundtrip():
    m = ConversationMemory()
    sid = m.new_session()
    m.append(sid, Turn(role="user", content="你好", ts="t"))
    assert m.get(sid)[0].content == "你好"


def test_sessions_are_isolated():
    m = ConversationMemory()
    a, b = m.new_session(), m.new_session()
    m.append(a, Turn(role="user", content="A 的内容", ts="t"))
    assert m.get(b) == []


def test_get_unknown_session_returns_empty_list():
    assert ConversationMemory().get("nope") == []


def test_sliding_window_drops_oldest_but_keeps_pairs():
    m = ConversationMemory(max_tokens=60, summarize_after_turns=999)
    sid = m.new_session()
    for i in range(10):
        m.append(sid, Turn(role="user", content=f"问题{i}" * 10, ts="t"))
        m.append(sid, Turn(role="assistant", content=f"答案{i}" * 10, ts="t"))
    ctx, _ = m.build_context(sid)
    assert len(ctx) < 20
    assert ctx[-1]["role"] == "assistant"      # 末尾整轮完整


def test_summary_compression_triggered_after_threshold():
    m = ConversationMemory(max_tokens=100000, summarize_after_turns=3, llm=MockLLM())
    sid = m.new_session()
    for i in range(5):
        m.append(sid, Turn(role="user", content=f"q{i}", ts="t"))
        m.append(sid, Turn(role="assistant", content=f"a{i}", ts="t"))
    _, summary = m.build_context(sid)
    assert summary is not None and len(summary) > 0


def test_no_summary_when_below_threshold():
    m = ConversationMemory(max_tokens=100000, summarize_after_turns=10, llm=MockLLM())
    sid = m.new_session()
    m.append(sid, Turn(role="user", content="q", ts="t"))
    _, summary = m.build_context(sid)
    assert summary is None


def test_clear_removes_session():
    m = ConversationMemory()
    sid = m.new_session()
    m.append(sid, Turn(role="user", content="x", ts="t"))
    m.clear(sid)
    assert m.get(sid) == []


def test_build_context_on_empty_session_returns_empty():
    m = ConversationMemory()
    ctx, summary = m.build_context(m.new_session())
    assert ctx == [] and summary is None


def test_save_and_load_roundtrip(tmp_path: Path):
    m = ConversationMemory(store_dir=tmp_path)
    sid = m.new_session()
    m.append(sid, Turn(role="user", content="持久化", ts="t"))
    m.save(sid)
    fresh = ConversationMemory(store_dir=tmp_path)
    assert fresh.load(sid) is True
    assert fresh.get(sid)[0].content == "持久化"


def test_load_returns_false_for_unknown_session(tmp_path: Path):
    assert ConversationMemory(store_dir=tmp_path).load("ghost") is False


def test_list_sessions_includes_created(tmp_path: Path):
    m = ConversationMemory(store_dir=tmp_path)
    sid = m.new_session()
    m.append(sid, Turn(role="user", content="x", ts="t"))
    m.save(sid)
    assert any(s["session_id"] == sid for s in m.list_sessions())
