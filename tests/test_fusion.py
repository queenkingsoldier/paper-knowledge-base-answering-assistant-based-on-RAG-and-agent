import pytest

from core.schemas import Chunk, ScoredChunk
from retrieval.fusion import rrf_fuse


def _s(cid: str, score: float, source: str, rank: int) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(chunk_id=cid, doc_id=cid.split("::")[0], filename="f.pdf",
                                   page_no=1, seq=0, text=cid, token_count=1),
                       score=score, source=source, rank=rank)


def test_single_list_preserves_order():
    a = [_s("a::1::0", 0.9, "vector", 1), _s("b::1::0", 0.8, "vector", 2)]
    out = rrf_fuse([a])
    assert [s.chunk.chunk_id for s in out] == ["a::1::0", "b::1::0"]


def test_document_in_both_lists_outranks_document_in_one():
    va = [_s("x::1::0", 0.9, "vector", 1), _s("y::1::0", 0.5, "vector", 2)]
    bm = [_s("y::1::0", 9.0, "bm25", 1), _s("z::1::0", 5.0, "bm25", 2)]
    out = rrf_fuse([va, bm])
    assert out[0].chunk.chunk_id == "y::1::0"   # 两路都命中 -> 排第一


def test_source_is_rrf_and_ranks_are_renumbered():
    out = rrf_fuse([[_s("a::1::0", 1.0, "vector", 1)], [_s("b::1::0", 2.0, "bm25", 1)]])
    assert all(s.source == "rrf" for s in out)
    assert [s.rank for s in out] == [1, 2]


def test_scores_are_inversely_rank_based_not_raw():
    # raw score 差异巨大也不应影响 RRF：只用排名
    out = rrf_fuse([[_s("a::1::0", 0.001, "vector", 1), _s("b::1::0", 999.0, "vector", 2)]])
    assert out[0].chunk.chunk_id == "a::1::0"


def test_k_smoothing_reduces_gap_between_top_ranks():
    lists = [[_s("a::1::0", 1, "v", 1), _s("b::1::0", 1, "v", 2)]]
    gap_low_k = rrf_fuse(lists, k=1)[0].score - rrf_fuse(lists, k=1)[1].score
    gap_high_k = rrf_fuse(lists, k=60)[0].score - rrf_fuse(lists, k=60)[1].score
    assert gap_high_k < gap_low_k


def test_weights_bias_toward_one_channel():
    va = [_s("a::1::0", 1, "vector", 1)]
    bm = [_s("b::1::0", 1, "bm25", 1)]
    out = rrf_fuse([va, bm], weights=[1.0, 0.1])
    assert out[0].chunk.chunk_id == "a::1::0"


def test_deduplicates_same_chunk_across_lists():
    same = "a::1::0"
    out = rrf_fuse([[_s(same, 1, "vector", 1)], [_s(same, 1, "bm25", 1)]])
    assert len(out) == 1


def test_empty_input_returns_empty_list():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_top_k_truncates():
    out = rrf_fuse([[_s(f"c{i}::1::0", 1, "v", i + 1) for i in range(10)]], top_k=3)
    assert len(out) == 3


def test_mismatched_weights_length_raises():
    with pytest.raises(ValueError):
        rrf_fuse([[_s("a::1::0", 1, "v", 1)]], weights=[1.0, 2.0])
