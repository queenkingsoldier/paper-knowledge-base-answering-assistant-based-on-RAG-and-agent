from pathlib import Path

import pytest

from core.providers import MockEmbedding
from core.schemas import Chunk
from retrieval.vectorstore import VectorStore


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(persist_dir=tmp_path / "chroma", collection="t", embedder=MockEmbedding())


def _chunk(cid: str, doc_id: str, text: str, page: int = 1) -> Chunk:
    return Chunk(chunk_id=cid, doc_id=doc_id, filename=f"{doc_id}.pdf",
                 page_no=page, seq=0, text=text, token_count=5)


def test_upsert_then_count(store):
    assert store.upsert([_chunk("d1::1::0", "d1", "transformer attention")]) == 1
    assert store.count() == 1


def test_upsert_is_idempotent_for_same_chunk_id(store):
    c = _chunk("d1::1::0", "d1", "transformer attention")
    store.upsert([c])
    store.upsert([c])
    assert store.count() == 1


def test_query_returns_scored_chunks_with_source_vector(store):
    store.upsert([_chunk("d1::1::0", "d1", "vision transformer"), _chunk("d2::1::0", "d2", "循环神经网络")])
    out = store.query("vision transformer", top_k=2)
    assert len(out) == 2
    assert all(s.source == "vector" for s in out)
    assert out[0].rank == 1


def test_query_on_empty_store_returns_empty_list_not_error(store):
    assert store.query("任意问题", top_k=5) == []


def test_document_exists_and_delete(store):
    store.upsert([_chunk("d1::1::0", "d1", "a"), _chunk("d1::2::0", "d1", "b")])
    assert store.document_exists("d1")
    assert store.delete_document("d1") == 2
    assert not store.document_exists("d1")
    assert store.count() == 0


def test_delete_nonexistent_document_returns_zero(store):
    assert store.delete_document("ghost") == 0


def test_list_documents_aggregates_chunk_counts(store):
    store.upsert([_chunk("d1::1::0", "d1", "a"), _chunk("d1::2::0", "d1", "b"), _chunk("d2::1::0", "d2", "c")])
    docs = {d["doc_id"]: d for d in store.list_documents()}
    assert docs["d1"]["chunk_count"] == 2
    assert docs["d2"]["chunk_count"] == 1


def test_persistence_across_instances(tmp_path: Path):
    d = tmp_path / "chroma"
    VectorStore(persist_dir=d, collection="t", embedder=MockEmbedding()).upsert([_chunk("d1::1::0", "d1", "x")])
    assert VectorStore(persist_dir=d, collection="t", embedder=MockEmbedding()).count() == 1


def test_health_reports_ok(store):
    ok, msg = store.health()
    assert ok is True and isinstance(msg, str)


def test_page_no_survives_roundtrip(store):
    store.upsert([_chunk("d1::7::0", "d1", "第七页内容", page=7)])
    assert store.query("第七页内容", top_k=1)[0].chunk.page_no == 7
