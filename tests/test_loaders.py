from pathlib import Path

import pytest

from ingest.loaders import SUPPORTED_EXTS, LoadError, load_document, make_doc_id, safe_filename


def test_load_txt_splits_into_single_page(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("第一段\n\n第二段", encoding="utf-8")
    doc = load_document(p)
    assert doc.filetype == "txt"
    assert len(doc.pages) == 1
    assert "第二段" in doc.pages[0].text


def test_load_markdown(tmp_path: Path):
    p = tmp_path / "a.md"
    p.write_text("# 标题\n\n正文", encoding="utf-8")
    assert load_document(p).filetype == "md"


def test_doc_id_is_content_addressed_and_stable(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("same", encoding="utf-8")
    assert make_doc_id(p) == make_doc_id(p)
    q = tmp_path / "b.txt"
    q.write_text("same", encoding="utf-8")
    assert make_doc_id(p) == make_doc_id(q)   # 内容相同则 ID 相同 -> 天然幂等


def test_doc_id_changes_when_content_changes(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("v1", encoding="utf-8")
    first = make_doc_id(p)
    p.write_text("v2", encoding="utf-8")
    assert make_doc_id(p) != first


def test_unsupported_extension_raises_load_error(tmp_path: Path):
    p = tmp_path / "a.xyz"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(LoadError):
        load_document(p)


def test_missing_file_raises_load_error(tmp_path: Path):
    with pytest.raises(LoadError):
        load_document(tmp_path / "nope.pdf")


def test_corrupt_pdf_raises_load_error_not_crash(tmp_path: Path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"not a real pdf at all")
    with pytest.raises(LoadError):
        load_document(p)


def test_corrupt_docx_raises_load_error(tmp_path: Path):
    p = tmp_path / "bad.docx"
    p.write_bytes(b"PK\x03\x04 garbage")
    with pytest.raises(LoadError):
        load_document(p)


def test_empty_txt_yields_zero_pages_not_exception(tmp_path: Path):
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    assert load_document(p).pages == []


def test_safe_filename_strips_path_separators():
    assert "/" not in safe_filename("../../etc/passwd")
    assert "\\" not in safe_filename("..\\..\\win.ini")


def test_supported_exts_contract():
    assert SUPPORTED_EXTS == {".pdf", ".docx", ".txt", ".md"}
