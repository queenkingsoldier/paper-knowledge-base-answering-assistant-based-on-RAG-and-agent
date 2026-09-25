from pathlib import Path

from core.providers import MockEmbedding, MockRerank
from ingest.chunkers import RecursiveChunker
from ingest.pipeline import IngestPipeline
from retrieval.bm25_index import BM25Index
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.vectorstore import VectorStore


def _pipeline(tmp_path: Path) -> IngestPipeline:
    vs = VectorStore(tmp_path / "chroma", "t", MockEmbedding())
    bm = BM25Index(tmp_path / "bm25")
    r = HybridRetriever(vs, bm, Reranker(provider=MockRerank()))
    return IngestPipeline(retriever=r, chunker=RecursiveChunker(chunk_size=200, overlap=20),
                          data_dir=tmp_path / "raw")


def test_ingest_single_txt(tmp_path: Path):
    (tmp_path / "raw").mkdir()
    p = tmp_path / "raw" / "a.txt"
    p.write_text("注意力机制是 Transformer 的核心。\n\n" * 20, encoding="utf-8")
    rep = _pipeline(tmp_path).ingest_paths([p])
    assert rep.files_ok == 1 and rep.chunks_added > 0


def test_ingest_is_idempotent_for_same_content(tmp_path: Path):
    (tmp_path / "raw").mkdir()
    p = tmp_path / "raw" / "a.txt"
    p.write_text("内容" * 200, encoding="utf-8")
    pipe = _pipeline(tmp_path)
    pipe.ingest_paths([p])
    rep2 = pipe.ingest_paths([p])
    assert rep2.files_skipped == 1 and rep2.chunks_added == 0


def test_corrupt_file_fails_but_batch_continues(tmp_path: Path):
    (tmp_path / "raw").mkdir()
    good = tmp_path / "raw" / "good.txt"
    good.write_text("好内容" * 200, encoding="utf-8")
    bad = tmp_path / "raw" / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    rep = _pipeline(tmp_path).ingest_paths([good, bad])
    assert rep.files_ok == 1 and rep.files_failed == 1
    assert len(rep.errors) == 1 and "bad.pdf" in rep.errors[0]["filename"]


def test_empty_file_is_skipped_gracefully(tmp_path: Path):
    (tmp_path / "raw").mkdir()
    p = tmp_path / "raw" / "empty.txt"
    p.write_text("", encoding="utf-8")
    rep = _pipeline(tmp_path).ingest_paths([p])
    assert rep.files_failed == 0
    assert rep.chunks_added == 0


def test_progress_callback_is_invoked(tmp_path: Path):
    (tmp_path / "raw").mkdir()
    p = tmp_path / "raw" / "a.txt"
    p.write_text("内容" * 100, encoding="utf-8")
    seen = []
    _pipeline(tmp_path).ingest_paths([p], progress_cb=lambda name, frac: seen.append((name, frac)))
    assert seen and seen[-1][1] == 1.0


def test_missing_path_fails_gracefully(tmp_path: Path):
    rep = _pipeline(tmp_path).ingest_paths([tmp_path / "ghost.txt"])
    assert rep.files_failed == 1
