from pathlib import Path

import pytest

from agent import build_agent
from core.providers import MockEmbedding, MockLLM, MockRerank
from core.schemas import Chunk
from generation.rag import RAGEngine
from ingest.chunkers import RecursiveChunker
from ingest.pipeline import IngestPipeline
from retrieval.bm25_index import BM25Index
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.vectorstore import VectorStore


@pytest.fixture
def wired(tmp_path: Path):
    vs = VectorStore(tmp_path / "chroma", "t", MockEmbedding())
    bm = BM25Index(tmp_path / "bm25")
    chunks = [Chunk(chunk_id="d1::1::0", doc_id="d1", filename="vit.pdf", page_no=1, seq=0,
                    text="ViT 使用 ImageNet-21k 预训练。", token_count=20)]
    vs.upsert(chunks); bm.build(chunks)
    retriever = HybridRetriever(vs, bm, Reranker(provider=MockRerank()))
    engine = RAGEngine(retriever=retriever, llm=MockLLM())
    return retriever, engine


def test_end_to_end_upload_to_answer(tmp_path: Path, wired):
    retriever, engine = wired
    raw = tmp_path / "raw"; raw.mkdir()
    (raw / "paper.txt").write_text("注意力机制是 Transformer 的核心组件。\n\n" * 30, encoding="utf-8")
    pipe = IngestPipeline(retriever=retriever, chunker=RecursiveChunker(200, 20), data_dir=raw)
    rep = pipe.ingest_paths([raw / "paper.txt"])
    assert rep.files_ok == 1 and rep.chunks_added > 0

    agent = build_agent(retriever=retriever, llm=MockLLM())
    result = agent.run("注意力机制是什么")
    assert isinstance(result.answer, str) and result.answer
    assert result.steps


def test_agent_has_all_eight_tools_wired(wired):
    retriever, _ = wired
    agent = build_agent(retriever=retriever, llm=MockLLM())
    assert len(agent.tools) == 8


def test_agent_retrieval_tool_actually_returns_kb_content(wired):
    """Agent 真正调用 kb_search 时，Observation 里应带回知识库的片段与来源。"""
    retriever, _ = wired
    agent = build_agent(retriever=retriever, llm=MockLLM())
    result = agent.run("ViT 预训练数据集")
    obs = " ".join(s.observation for s in result.steps)
    assert "vit.pdf" in obs, f"Observation 未带回来源：{obs[:200]}"


def test_answer_once_works_without_streamlit_runtime(wired, monkeypatch):
    import streamlit_app

    retriever, _ = wired
    monkeypatch.setattr(streamlit_app, "_get_retriever", lambda: retriever)
    monkeypatch.setattr(streamlit_app, "_get_llm", lambda: MockLLM())
    res = streamlit_app.answer_once("ViT 预训练数据集", "sess-1")
    assert isinstance(res.answer, str)


def test_full_chain_survives_empty_knowledge_base(tmp_path: Path, wired, monkeypatch):
    import streamlit_app

    retriever, _ = wired
    retriever.vectorstore.delete_document("d1")
    retriever.bm25 = BM25Index(tmp_path / "bm25")
    monkeypatch.setattr(streamlit_app, "_get_retriever", lambda: retriever)
    monkeypatch.setattr(streamlit_app, "_get_llm", lambda: MockLLM())
    res = streamlit_app.answer_once("任意问题", "sess-2")
    assert "空" in res.answer or "未找到" in res.answer or res.degraded is not None


def test_empty_kb_still_answers_pure_arithmetic(wired, monkeypatch):
    """空知识库不该妨碍不需要检索的工具类问题。"""
    import streamlit_app

    retriever, _ = wired
    retriever.vectorstore.delete_document("d1")
    monkeypatch.setattr(streamlit_app, "_get_retriever", lambda: retriever)
    monkeypatch.setattr(streamlit_app, "_get_llm", lambda: MockLLM())
    res = streamlit_app.answer_once("3.14 * 2.56 等于多少", "sess-3")
    assert res.degraded != "empty_knowledge_base", "算术问题不该被空知识库短路"
    assert res.answer
