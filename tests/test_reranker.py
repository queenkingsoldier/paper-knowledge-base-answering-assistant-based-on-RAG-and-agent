from core.providers import MockRerank
from core.schemas import Chunk, ScoredChunk
from retrieval.reranker import Reranker


class BoomRerank:
    def rerank(self, query, documents, top_n):
        raise RuntimeError("上游 503")


def _s(cid: str, text: str, rank: int) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(chunk_id=cid, doc_id="d", filename="f.pdf", page_no=1,
                                   seq=0, text=text, token_count=1),
                       score=0.5, source="rrf", rank=rank)


def test_rerank_reorders_and_relabels():
    r = Reranker(provider=MockRerank())
    cands = [_s("a::1::0", "完全无关的内容", 1), _s("b::1::0", "transformer vision tasks", 2)]
    out, degraded = r.rerank("transformer vision", cands, top_n=2)
    assert degraded is None
    assert out[0].chunk.chunk_id == "b::1::0"
    assert all(s.source == "rerank" for s in out)
    assert [s.rank for s in out] == [1, 2]


def test_empty_candidates_returns_empty_without_calling_provider():
    out, degraded = Reranker(provider=MockRerank()).rerank("q", [], top_n=5)
    assert out == [] and degraded is None


def test_disabled_reranker_passes_through_truncated():
    r = Reranker(provider=MockRerank(), enabled=False)
    cands = [_s(f"c{i}::1::0", f"t{i}", i + 1) for i in range(10)]
    out, degraded = r.rerank("q", cands, top_n=3)
    assert len(out) == 3
    assert degraded == "rerank_disabled"


def test_provider_failure_degrades_to_rrf_order():
    r = Reranker(provider=BoomRerank())
    cands = [_s("a::1::0", "t1", 1), _s("b::1::0", "t2", 2)]
    out, degraded = r.rerank("q", cands, top_n=2)
    assert degraded is not None and "503" in degraded
    assert [s.chunk.chunk_id for s in out] == ["a::1::0", "b::1::0"]
    assert all(s.source == "rrf" for s in out)


def test_low_relevance_is_reported_via_min_relevance_check():
    r = Reranker(provider=MockRerank(), min_relevance=2.0)   # 阈值高到不可能满足
    out, _ = r.rerank("q", [_s("a::1::0", "unrelated", 1)], top_n=1)
    assert r.is_low_confidence(out) is True


def test_is_low_confidence_false_for_empty_list():
    assert Reranker(provider=MockRerank()).is_low_confidence([]) is False


def test_top_n_larger_than_candidates_is_clamped():
    r = Reranker(provider=MockRerank())
    out, _ = r.rerank("q", [_s("a::1::0", "x", 1)], top_n=10)
    assert len(out) == 1
