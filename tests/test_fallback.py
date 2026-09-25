from core.schemas import Chunk, ScoredChunk
from generation.fallback import (LOW_CONFIDENCE_NOTICE, NO_HIT_NOTICE, decide)


def _s(score: float) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(chunk_id="a::1::0", doc_id="a", filename="f.pdf", page_no=1,
                                   seq=0, text="t", token_count=1),
                       score=score, source="rerank", rank=1)


class R:
    def __init__(self, chunks, degraded=None):
        self.chunks, self.degraded = chunks, degraded


def test_no_chunks_triggers_no_hit_notice():
    should, notice = decide(R([]), "答案")
    assert should is True and notice == NO_HIT_NOTICE


def test_low_top1_score_triggers_low_confidence():
    should, notice = decide(R([_s(0.05)]), "答案")
    assert should is True and notice == LOW_CONFIDENCE_NOTICE


def test_high_score_no_degradation():
    should, notice = decide(R([_s(0.85)]), "答案")
    assert should is False and notice is None


def test_upstream_degradation_is_surfaced():
    should, notice = decide(R([_s(0.85)], degraded="rerank_failed: 503"), "答案")
    assert should is True and "503" in notice


def test_empty_answer_with_chunks_is_flagged():
    should, notice = decide(R([_s(0.85)]), "")
    assert should is True
