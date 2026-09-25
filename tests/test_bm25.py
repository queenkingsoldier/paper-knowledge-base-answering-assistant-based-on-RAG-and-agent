from pathlib import Path

from core.schemas import Chunk
from retrieval.bm25_index import BM25Index, tokenize


def _chunk(cid: str, doc_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id=doc_id, filename=f"{doc_id}.pdf",
                 page_no=1, seq=0, text=text, token_count=5)


def test_tokenize_lowercases_english_words():
    assert "transformer" in tokenize("Transformer 模型")


def test_tokenize_emits_cjk_bigrams():
    toks = tokenize("注意力机制")
    assert "注意" in toks and "意力" in toks and "机制" in toks


def test_tokenize_handles_pure_numbers():
    assert tokenize("3.14") != []


def test_tokenize_empty_string_returns_empty_list():
    assert tokenize("") == []


def test_tokenize_never_raises_on_fullwidth_punctuation():
    assert isinstance(tokenize("你好。世界！测试？；：（）"), list)


def test_build_and_query_finds_keyword_match(tmp_path: Path):
    idx = BM25Index(tmp_path)
    idx.build([_chunk("d1::1::0", "d1", "BatchNormalization 加速训练收敛"),
               _chunk("d2::1::0", "d2", "循环神经网络用于序列建模")])
    out = idx.query("BatchNormalization", top_k=2)
    assert out[0].chunk.chunk_id == "d1::1::0"
    assert out[0].source == "bm25"
    assert out[0].rank == 1


def test_query_on_empty_index_returns_empty_list(tmp_path: Path):
    assert BM25Index(tmp_path).query("任意", top_k=5) == []


def test_query_with_all_out_of_vocab_terms_returns_empty(tmp_path: Path):
    idx = BM25Index(tmp_path)
    idx.build([_chunk("d1::1::0", "d1", "苹果香蕉")])
    assert idx.query("zzzz qqqq", top_k=5) == []


def test_incremental_add_does_not_lose_earlier_chunks(tmp_path: Path):
    idx = BM25Index(tmp_path)
    idx.build([_chunk("d1::1::0", "d1", "alpha beta")])
    idx.add([_chunk("d2::1::0", "d2", "gamma delta")])
    assert idx.count() == 2
    assert idx.query("gamma", top_k=1)[0].chunk.doc_id == "d2"


def test_add_is_idempotent_by_chunk_id(tmp_path: Path):
    idx = BM25Index(tmp_path)
    c = _chunk("d1::1::0", "d1", "alpha")
    idx.build([c])
    idx.add([c])
    assert idx.count() == 1


def test_remove_document(tmp_path: Path):
    idx = BM25Index(tmp_path)
    idx.build([_chunk("d1::1::0", "d1", "alpha"), _chunk("d2::1::0", "d2", "beta")])
    assert idx.remove_document("d1") == 1
    assert idx.count() == 1


def test_save_then_load_roundtrip(tmp_path: Path):
    idx = BM25Index(tmp_path)
    idx.build([_chunk("d1::1::0", "d1", "transformer attention")])
    idx.save()
    fresh = BM25Index(tmp_path)
    assert fresh.load() is True
    assert fresh.query("transformer", top_k=1)[0].chunk.chunk_id == "d1::1::0"


def test_load_returns_false_when_no_index_file(tmp_path: Path):
    assert BM25Index(tmp_path).load() is False
