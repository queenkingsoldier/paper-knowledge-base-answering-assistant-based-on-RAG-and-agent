import math

import pytest

from core.providers import (MockEmbedding, MockLLM, MockRerank, get_embedder,
                            get_llm, get_reranker)


def test_mock_embedding_is_deterministic():
    e = MockEmbedding()
    assert e.embed(["你好"]) == e.embed(["你好"])


def test_mock_embedding_dim_is_1024_and_normalized():
    e = MockEmbedding()
    v = e.embed_one("任意文本")
    assert len(v) == 1024
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-6)


def test_mock_embedding_differs_for_different_text():
    e = MockEmbedding()
    assert e.embed_one("苹果") != e.embed_one("香蕉")


def test_mock_embedding_handles_empty_batch():
    assert MockEmbedding().embed([]) == []


def test_mock_embedding_handles_empty_string():
    v = MockEmbedding().embed_one("")
    assert len(v) == 1024


def test_mock_rerank_orders_by_overlap_with_query():
    r = MockRerank()
    out = r.rerank("transformer vision", ["完全无关的内容", "transformer for vision tasks"], top_n=2)
    assert out[0][0] == 1
    assert out[0][1] > out[1][1]


def test_mock_llm_stream_yields_multiple_tokens_and_records_usage():
    llm = MockLLM()
    text = "".join(llm.stream([{"role": "user", "content": "hi"}]))
    assert len(text) > 0
    assert llm.last_usage["completion_tokens"] > 0


def test_getters_return_provider_instances():
    assert isinstance(get_llm(), MockLLM)
    assert isinstance(get_embedder(), MockEmbedding)
    assert isinstance(get_reranker(), MockRerank)


def test_missing_key_raises_at_settings_construction(monkeypatch):
    from pydantic import ValidationError

    from config import Settings

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises((ValueError, ValidationError)) as exc:
        Settings(_env_file=None, llm_provider="deepseek", deepseek_api_key="")
    assert "DEEPSEEK_API_KEY" in str(exc.value)
