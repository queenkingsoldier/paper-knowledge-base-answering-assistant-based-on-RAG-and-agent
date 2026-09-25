from pathlib import Path

import pytest

from agent.tools import build_registry
from agent.tools import kb_search as kb_mod
from agent.tools.kb_search import kb_search
from agent.tools.paper_compare import paper_compare
from agent.tools.paper_meta import paper_meta
from core.providers import MockEmbedding, MockLLM, MockRerank
from core.schemas import Chunk
from generation.rag import RAGEngine
from retrieval.bm25_index import BM25Index
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.vectorstore import VectorStore


@pytest.fixture
def engine(tmp_path: Path) -> RAGEngine:
    vs = VectorStore(tmp_path / "chroma", "t", MockEmbedding())
    bm = BM25Index(tmp_path / "bm25")
    chunks = [
        Chunk(chunk_id="d1::1::0", doc_id="d1", filename="vit.pdf", page_no=1, seq=0,
              text="ViT 将图像切成 16x16 patch。", token_count=20),
        Chunk(chunk_id="d2::1::0", doc_id="d2", filename="swin.pdf", page_no=2, seq=0,
              text="Swin 使用层次化窗口注意力。", token_count=20),
    ]
    vs.upsert(chunks); bm.build(chunks)
    return RAGEngine(retriever=HybridRetriever(vs, bm, Reranker(provider=MockRerank())), llm=MockLLM())


@pytest.fixture(autouse=True)
def _inject(engine):
    """注入**全部**知识库类工具的依赖。

    必须走生产路径（`agent._inject_dependencies`）而不是只调用
    `kb_mod.set_engine(engine)`：工具依赖是模块级全局状态，若这里只注入
    kb_search，paper_meta / paper_compare / summarize 会保留上一个测试文件里
    `build_agent()` 注入的别的 VectorStore，表现为「单独跑通过、整套跑失败」
    的跨文件状态泄漏。复用生产注入函数还有一个好处：它本身也被测到了。
    """
    from agent import _inject_dependencies

    llm = getattr(engine, "llm", None) or MockLLM()
    _inject_dependencies(engine, engine.retriever, llm)


def test_kb_search_returns_formatted_citation_block(engine):
    out = kb_search(query="patch", top_k=2)
    assert "vit.pdf" in out or "swin.pdf" in out
    assert isinstance(out, str)


def test_kb_search_on_empty_kb_returns_readable_notice(engine):
    engine.retriever.vectorstore.delete_document("d1")
    engine.retriever.vectorstore.delete_document("d2")
    engine.retriever.bm25 = BM25Index(Path(engine.retriever.bm25.index_dir))
    assert "空" in kb_search(query="任意") or "未找到" in kb_search(query="任意")


def test_kb_search_without_engine_returns_error_string_not_raise():
    kb_mod._ENGINE = None
    try:
        out = kb_search(query="q")
        assert "未初始化" in out or "错误" in out
    finally:
        kb_mod.set_engine(None)


def test_paper_meta_unknown_doc_returns_not_found_message():
    assert "未找到" in paper_meta(doc_id="ghost")


def test_paper_meta_lists_known_documents_when_no_arg(engine):
    out = paper_meta()
    assert "vit.pdf" in out


def test_paper_compare_missing_doc_reports_clearly(engine):
    out = paper_compare(doc_id_a="d1", doc_id_b="ghost")
    assert "未找到" in out


def test_all_eight_tools_registered():
    reg = build_registry()
    expected = {"kb_search", "paper_meta", "paper_compare", "extract_keywords",
                "summarize_paper", "calculator", "current_time", "web_search"}
    assert expected.issubset(set(reg))
