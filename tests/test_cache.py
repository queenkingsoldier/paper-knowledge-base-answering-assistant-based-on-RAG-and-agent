from pathlib import Path

import pytest

from core.providers import MockEmbedding
from core.schemas import AgentResult, Citation
from generation.cache import SemanticCache


@pytest.fixture
def cache(tmp_path: Path) -> SemanticCache:
    return SemanticCache(MockEmbedding(), tmp_path, threshold=0.95)


def _res(text: str = "答案") -> AgentResult:
    return AgentResult(answer=text, citations=[Citation(1, "f.pdf", 1, "c", "snip")],
                       steps=[], tokens={}, latency_ms={}, route="rag", degraded=None)


def test_exact_question_hits(cache):
    cache.put("BERT 用了什么优化器", _res())
    assert cache.get("BERT 用了什么优化器") is not None


def test_different_question_misses_at_high_threshold(cache):
    cache.put("BERT 用了什么优化器", _res())
    assert cache.get("今天天气怎么样") is None


def test_near_duplicate_misses_when_below_threshold(tmp_path: Path):
    c = SemanticCache(MockEmbedding(), tmp_path, threshold=1.01)   # 苛刻阈值
    c.put("问题甲", _res())
    assert c.get("问题甲") is None


def test_empty_cache_get_returns_none(cache):
    assert cache.get("任意") is None


def test_put_same_question_twice_increments_hit_count(cache):
    cache.put("q", _res())
    cache.put("q", _res())
    assert cache.size() == 1


def test_save_and_load_roundtrip(tmp_path: Path):
    c = SemanticCache(MockEmbedding(), tmp_path, threshold=0.95)
    c.put("持久化问题", _res("持久化答案"))
    c.save()
    fresh = SemanticCache(MockEmbedding(), tmp_path, threshold=0.95)
    assert fresh.load() is True
    got = fresh.get("持久化问题")
    assert got is not None and got.answer == "持久化答案"


def test_load_returns_false_when_no_file(tmp_path: Path):
    assert SemanticCache(MockEmbedding(), tmp_path / "none", 0.95).load() is False


def test_clear_empties_cache(cache):
    cache.put("q", _res())
    cache.clear()
    assert cache.size() == 0


def test_empty_question_returns_none(cache):
    assert cache.get("") is None
