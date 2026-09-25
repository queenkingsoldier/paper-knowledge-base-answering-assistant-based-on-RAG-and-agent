from pathlib import Path

import pytest

from core.providers import MockEmbedding, MockRerank
from core.schemas import Chunk
from retrieval.bm25_index import BM25Index
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.vectorstore import VectorStore


@pytest.fixture
def retriever(tmp_path: Path) -> HybridRetriever:
    vs = VectorStore(tmp_path / "chroma", "t", MockEmbedding())
    bm = BM25Index(tmp_path / "bm25")
    vs.upsert([
        Chunk(chunk_id="d1::1::0", doc_id="d1", filename="vit.pdf", page_no=3, seq=0,
              text="vision transformer 使用 ImageNet 预训练", token_count=10),
        Chunk(chunk_id="d2::1::0", doc_id="d2", filename="rnn.pdf", page_no=1, seq=0,
              text="循环神经网络用于序列建模", token_count=10),
    ])
    bm.build([
        Chunk(chunk_id="d1::1::0", doc_id="d1", filename="vit.pdf", page_no=3, seq=0,
              text="vision transformer 使用 ImageNet 预训练", token_count=10),
        Chunk(chunk_id="d2::1::0", doc_id="d2", filename="rnn.pdf", page_no=1, seq=0,
              text="循环神经网络用于序列建模", token_count=10),
    ])
    return HybridRetriever(vs, bm, Reranker(provider=MockRerank()))


def test_vector_mode_returns_retrieval_result(retriever):
    r = retriever.search("vision transformer", mode="vector", top_k=2)
    assert r.mode == "vector"
    assert r.chunks and all(c.source == "vector" for c in r.chunks)


def test_hybrid_mode_uses_rrf(retriever):
    r = retriever.search("ImageNet", mode="hybrid", top_k=2)
    assert r.mode == "hybrid"
    assert all(c.source == "rrf" for c in r.chunks)


def test_hybrid_rerank_mode_labels_source_rerank(retriever):
    r = retriever.search("vision transformer", mode="hybrid_rerank", top_k=2)
    assert r.mode == "hybrid_rerank"
    assert r.chunks[0].source == "rerank"


def test_unknown_mode_raises_value_error(retriever):
    with pytest.raises(ValueError):
        retriever.search("q", mode="nope")


def test_empty_knowledge_base_returns_empty_not_crash(retriever):
    retriever.vectorstore.delete_document("d1")
    retriever.vectorstore.delete_document("d2")
    retriever.bm25 = BM25Index(Path(retriever.bm25.index_dir))
    r = retriever.search("任意问题", mode="hybrid_rerank")
    assert r.chunks == []
    assert r.degraded == "empty_knowledge_base"


def test_embedding_failure_degrades_to_bm25_only(retriever):
    def boom(_):
        raise RuntimeError("embedding api down")

    retriever.vectorstore.embedder = type("X", (), {"embed_one": staticmethod(boom), "dim": 1024})()
    r = retriever.search("ImageNet", mode="hybrid_rerank", top_k=2)
    assert r.degraded is not None and "bm25" in r.degraded
    assert r.chunks and r.chunks[0].chunk.doc_id == "d1"


def test_latency_is_recorded(retriever):
    assert retriever.search("vision", mode="hybrid").latency_ms >= 0
