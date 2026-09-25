from core.schemas import Document, Page
from ingest.chunkers import (FixedChunker, RecursiveChunker, SemanticChunker,
                             get_chunker, is_reference_section, strip_boilerplate)


def _doc(text: str, pages: int = 1) -> Document:
    return Document(doc_id="d1", filename="f.pdf", filetype="pdf",
                    pages=[Page(text=text, page_no=i + 1) for i in range(pages)],
                    meta={}, ingested_at="2026-01-01")


def test_fixed_chunker_respects_size_and_produces_ids():
    chunks = FixedChunker(chunk_size=100, overlap=0).split(_doc("字" * 1000))
    assert len(chunks) >= 10
    assert all(c.chunk_id.startswith("d1::") for c in chunks)


def test_fixed_chunker_produces_no_empty_chunks():
    chunks = FixedChunker(chunk_size=50, overlap=10).split(_doc("a b c d e f"))
    assert all(c.text.strip() for c in chunks)


def test_chunk_ids_are_unique_within_document():
    chunks = RecursiveChunker(chunk_size=60, overlap=10).split(_doc("段落。\n\n" * 50))
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_recursive_chunker_prefers_paragraph_boundary():
    text = "甲" * 80 + "\n\n" + "乙" * 80
    chunks = RecursiveChunker(chunk_size=100, overlap=0).split(_doc(text))
    assert chunks[0].text.strip().endswith("甲")


def test_recursive_chunker_preserves_page_number():
    chunks = RecursiveChunker(chunk_size=100, overlap=0).split(_doc("内容" * 100, pages=3))
    assert {c.page_no for c in chunks} == {1, 2, 3}


def test_recursive_chunker_handles_single_very_long_sentence():
    chunks = RecursiveChunker(chunk_size=100, overlap=0).split(_doc("啊" * 5000))
    assert len(chunks) > 1
    assert all(c.text.strip() for c in chunks)


def test_recursive_chunker_handles_cjk_fullwidth_punctuation():
    text = "第一句。第二句！第三句？第四句；第五句：" * 40
    chunks = RecursiveChunker(chunk_size=100, overlap=0).split(_doc(text))
    assert len(chunks) > 1
    assert all(c.text.strip() for c in chunks)


def test_semantic_chunker_without_embed_fn_falls_back_to_recursive():
    a = SemanticChunker(chunk_size=100, threshold=0.5, embed_fn=None).split(_doc("甲" * 300))
    b = RecursiveChunker(chunk_size=100, overlap=0).split(_doc("甲" * 300))
    assert len(a) == len(b)


def test_semantic_chunker_with_embed_fn_splits_at_low_similarity():
    def fake_embed(texts):
        # 含"切换"的句子向量指向另一方向，制造语义断点
        return [[0.0, 1.0] if "切换" in t else [1.0, 0.0] for t in texts]

    text = "主题甲内容。主题甲继续。\n切换\n主题乙内容。主题乙继续。"
    chunks = SemanticChunker(chunk_size=1000, threshold=0.9, embed_fn=fake_embed).split(_doc(text))
    assert len(chunks) >= 2


def test_chunker_factory_rejects_unknown_strategy():
    import pytest
    with pytest.raises(ValueError):
        get_chunker("nope")


def test_chunker_factory_returns_expected_types():
    assert isinstance(get_chunker("fixed"), FixedChunker)
    assert isinstance(get_chunker("recursive"), RecursiveChunker)
    assert isinstance(get_chunker("semantic"), SemanticChunker)


def test_strip_boilerplate_removes_repeated_header_lines():
    pages = [Page(text="IEEE TRANSACTIONS ON PAMI\n正文一", page_no=i) for i in (1, 2, 3, 4)]
    pages.append(Page(text="IEEE TRANSACTIONS ON PAMI\n正文五", page_no=5))
    out = strip_boilerplate(pages)
    assert "IEEE TRANSACTIONS ON PAMI" not in out[0].text
    assert "正文一" in out[0].text


def test_is_reference_section_detects_both_languages():
    assert is_reference_section("References")
    assert is_reference_section("参考文献")
    assert not is_reference_section("实验结果与分析")
