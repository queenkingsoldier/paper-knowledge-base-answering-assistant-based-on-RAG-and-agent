from pathlib import Path

import pytest

from core.providers import MockEmbedding, MockLLM, MockRerank
from core.schemas import Chunk
from generation.rag import RAGEngine
from retrieval.bm25_index import BM25Index
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.vectorstore import VectorStore

TEXT = "vision transformer 使用 ImageNet-21k 预训练，随后在 ImageNet-1k 上微调。"


@pytest.fixture
def engine(tmp_path: Path) -> RAGEngine:
    vs = VectorStore(tmp_path / "chroma", "t", MockEmbedding())
    bm = BM25Index(tmp_path / "bm25")
    c = Chunk(chunk_id="d1::3::0", doc_id="d1", filename="vit.pdf", page_no=3, seq=0,
              text=TEXT, token_count=30)
    vs.upsert([c]); bm.build([c])
    r = HybridRetriever(vs, bm, Reranker(provider=MockRerank()))
    return RAGEngine(retriever=r, llm=MockLLM())


def test_stream_emits_citations_first_then_tokens_then_done(engine):
    events = list(engine.stream_answer("用了什么数据集"))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "citations"
    assert "token" in kinds
    assert kinds[-1] == "done"


def test_citations_point_to_real_filename_and_page(engine):
    events = list(engine.stream_answer("用了什么数据集"))
    cites = events[0]["citations"]
    assert cites and cites[0]["filename"] == "vit.pdf" and cites[0]["page_no"] == 3


def test_streamed_text_reassembles_to_full_answer(engine):
    events = list(engine.stream_answer("q"))
    joined = "".join(e["text"] for e in events if e["type"] == "token")
    assert len(joined) > 0
    assert events[-1]["answer"] == joined


def test_usage_event_is_emitted_and_has_both_token_counts(engine):
    events = list(engine.stream_answer("q"))
    usage = [e for e in events if e["type"] == "usage"]
    assert usage and "prompt_tokens" in usage[0] and "completion_tokens" in usage[0]


def test_answer_on_empty_knowledge_base_marks_degraded(engine):
    engine.retriever.vectorstore.delete_document("d1")
    engine.retriever.bm25 = BM25Index(Path(engine.retriever.bm25.index_dir))
    events = list(engine.stream_answer("任意"))
    degraded = [e for e in events if e["type"] == "degraded"]
    assert degraded and degraded[0]["reason"] == "empty_knowledge_base"
    assert events[-1]["type"] == "done"


def test_non_stream_answer_returns_agent_result(engine):
    res = engine.answer("用了什么数据集")
    assert res.answer
    assert res.citations and res.steps == []
    assert res.latency_ms["total"] >= 0


def test_llm_failure_emits_error_event_not_exception(engine):
    class BoomLLM(MockLLM):
        def stream(self, messages, **kw):
            raise RuntimeError("api 401")

    engine.llm = BoomLLM()
    events = list(engine.stream_answer("q"))
    assert any(e["type"] == "error" for e in events)
    assert events[-1]["type"] == "done"


def test_history_is_prepended_to_messages(engine):
    captured = {}

    class SpyLLM(MockLLM):
        def stream(self, messages, **kw):
            captured["messages"] = messages
            return super().stream(messages, **kw)

    engine.llm = SpyLLM()
    list(engine.stream_answer("追问", history=[{"role": "user", "content": "上一轮问题"}]))
    roles = [m["role"] for m in captured["messages"]]
    assert roles.count("user") >= 2


def test_retrieve_only_does_not_call_llm(engine):
    class ExplodingLLM(MockLLM):
        def stream(self, messages, **kw):
            raise AssertionError("不应调用 LLM")

    engine.llm = ExplodingLLM()
    r = engine.retrieve_only("ImageNet")
    assert r.chunks
