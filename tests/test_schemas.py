from core.schemas import Chunk, Document, Page, ScoredChunk, estimate_tokens


def test_document_holds_pages_with_page_numbers():
    doc = Document(doc_id="abc123", filename="vit.pdf", filetype="pdf",
                   pages=[Page(text="hello", page_no=1)], meta={}, ingested_at="2026-01-01")
    assert doc.pages[0].page_no == 1


def test_chunk_id_composes_from_doc_page_seq():
    c = Chunk(chunk_id="abc123::3::2", doc_id="abc123", filename="vit.pdf",
              page_no=3, seq=2, text="text", token_count=1)
    assert c.chunk_id.split("::") == ["abc123", "3", "2"]


def test_estimate_tokens_counts_cjk_as_one_token_each():
    assert estimate_tokens("你好世界") == 4


def test_estimate_tokens_counts_latin_by_word_groups():
    # 12 个英文单词 -> 向上取整(12/0.75)=16
    text = "one two three four five six seven eight nine ten eleven twelve"
    assert estimate_tokens(text) == 16


def test_estimate_tokens_empty_string_is_zero():
    assert estimate_tokens("") == 0


def test_scored_chunk_keeps_source_and_rank():
    c = Chunk(chunk_id="a::1::0", doc_id="a", filename="f.pdf", page_no=1,
              seq=0, text="t", token_count=1)
    s = ScoredChunk(chunk=c, score=0.9, source="rerank", rank=1)
    assert (s.source, s.rank) == ("rerank", 1)
