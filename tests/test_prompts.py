from core.schemas import Chunk, ScoredChunk
from generation.prompts import (build_rag_messages, extract_citation_indices,
                                validate_citations)


def _s(cid: str, text: str, page: int, rank: int, score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(chunk_id=cid, doc_id="d", filename="vit.pdf", page_no=page,
                                   seq=0, text=text, token_count=10),
                       score=score, source="rerank", rank=rank)


def test_messages_have_system_and_user_roles():
    msgs, _ = build_rag_messages("问题", [_s("a::1::0", "内容甲", 1, 1)], max_context_tokens=1000)
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"
    assert "问题" in msgs[-1]["content"]


def test_citations_are_numbered_from_one_and_carry_page():
    _, cites = build_rag_messages("q", [_s("a::3::0", "甲", 3, 1), _s("b::7::0", "乙", 7, 2)],
                                  max_context_tokens=1000)
    assert [c.index for c in cites] == [1, 2]
    assert cites[0].page_no == 3 and cites[1].page_no == 7
    assert cites[0].filename == "vit.pdf"


def test_context_block_contains_inline_source_labels():
    msgs, _ = build_rag_messages("q", [_s("a::3::0", "独特内容ABC", 3, 1)], max_context_tokens=1000)
    user = msgs[-1]["content"]
    assert "独特内容ABC" in user and "vit.pdf" in user and "3" in user


def test_higher_ranked_chunk_appears_first():
    msgs, _ = build_rag_messages("q", [_s("a::1::0", "第一", 1, 1), _s("b::1::0", "第二", 1, 2)],
                                 max_context_tokens=1000)
    body = msgs[-1]["content"]
    assert body.index("第一") < body.index("第二")


def test_token_budget_truncates_later_chunks():
    long = "字" * 3000
    _, cites = build_rag_messages("q", [_s("a::1::0", "短", 1, 1), _s("b::1::0", long, 1, 2)],
                                  max_context_tokens=200)
    assert len(cites) == 1


def test_no_chunks_produces_no_context_prompt():
    msgs, cites = build_rag_messages("孤立问题", [], max_context_tokens=1000)
    assert cites == []
    assert "未找到" in msgs[-1]["content"] or "没有" in msgs[-1]["content"]


def test_extract_citation_indices_handles_adjacent_and_spaced():
    assert extract_citation_indices("答案[1][2]。") == [1, 2]
    assert extract_citation_indices("答案 [1] [3]") == [1, 3]
    assert extract_citation_indices("无引用") == []


def test_extract_citation_indices_deduplicates_and_sorts():
    assert extract_citation_indices("a[2] b[1] c[2]") == [1, 2]


def test_extract_ignores_malformed_markers():
    assert extract_citation_indices("a[abc] b[] c[") == []


def test_validate_citations_flags_out_of_range_indices():
    cleaned, bad = validate_citations("答案[1][9]", valid={1, 2})
    assert bad == [9]
    assert "[9]" in cleaned


def test_validate_citations_returns_empty_bad_when_all_valid():
    _, bad = validate_citations("答案[1][2]", valid={1, 2})
    assert bad == []
