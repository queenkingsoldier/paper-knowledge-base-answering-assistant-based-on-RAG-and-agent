"""无 API key 的全链路冒烟自检。

用法：
    python scripts/smoke_test.py

任一步失败即打印 [FAIL] <步骤>: <原因> 并以退出码 1 结束。
全部通过打印 [OK] 并以退出码 0 结束。

设计意图：在没有真实密钥、不花钱、不联网的前提下，证明
「入库 → 三档检索 → RRF → 流式生成 → ReAct Agent → 观测 → 评测指标」
这条链路是**真的能跑**，而不是只通过了单元测试。
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 冒烟测试必须离线，强制 mock，避免误用真实密钥产生费用。
import os  # noqa: E402

os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "mock"
os.environ["RERANK_PROVIDER"] = "mock"

# 让临时目录落在工作区内（本机沙箱只允许写工作区）
_TMP = PROJECT_ROOT / ".tmp" / "smoke"
_TMP.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(_TMP)
for _v in ("TMPDIR", "TEMP", "TMP"):
    os.environ[_v] = str(_TMP)

FAILURES: list[str] = []
STEP = 0


def step(name: str):
    """装饰器：把每一步包成「失败不中断、最后统一汇报」。"""
    def deco(fn):
        def wrapper():
            global STEP
            STEP += 1
            label = f"{STEP}. {name}"
            try:
                detail = fn()
                print(f"[ OK ] {label}" + (f" — {detail}" if detail else ""))
                return True
            except Exception as exc:  # noqa: BLE001 - 冒烟测试就是要抓全部异常
                print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=3)
                FAILURES.append(label)
                return False
        return wrapper
    return deco


@step("导入全部模块")
def check_imports():
    import agent.react  # noqa: F401
    import agent.memory  # noqa: F401
    import agent.router  # noqa: F401
    import generation.rag  # noqa: F401
    import ingest.pipeline  # noqa: F401
    import observability.metrics  # noqa: F401
    import retrieval.hybrid  # noqa: F401
    from agent.tools import build_registry
    tools = build_registry()
    return f"工具注册表 {len(tools)} 个: {', '.join(sorted(tools))}"


@step("构造检索器与 GenAI 引擎")
def check_build():
    from core.providers import MockEmbedding
    from generation.rag import RAGEngine
    from retrieval.bm25_index import BM25Index
    from retrieval.hybrid import HybridRetriever
    from retrieval.reranker import Reranker
    from retrieval.vectorstore import VectorStore

    global _VS, _BM, _RETRIEVER, _ENGINE
    workdir = _TMP / "kb"
    _VS = VectorStore(workdir / "chroma", "smoke", MockEmbedding())
    _BM = BM25Index(workdir / "bm25")
    _RETRIEVER = HybridRetriever(_VS, _BM, Reranker())
    _ENGINE = RAGEngine(retriever=_RETRIEVER)
    return "ok"


@step("写入假数据并建索引")
def check_index():
    from ingest.chunkers import RecursiveChunker
    from ingest.pipeline import IngestPipeline

    corpus = _TMP / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "vit.txt").write_text(
        "ViT 将图像切成 16x16 的 patch，并使用 ImageNet-21k 预训练。\n\n"
        "Vision Transformer 直接把 Transformer 用在图像分类上。\n" * 6,
        encoding="utf-8",
    )
    (corpus / "swin.txt").write_text(
        "Swin Transformer 使用层次化窗口注意力，复杂度与图像尺寸成线性。\n\n"
        "Swin-L 在 ImageNet-1k 上达到 87.3% top-1 准确率。\n" * 6,
        encoding="utf-8",
    )
    pipeline = IngestPipeline(retriever=_RETRIEVER, chunker=RecursiveChunker(256, 32),
                              data_dir=corpus)
    rep = pipeline.ingest_paths(sorted(corpus.glob("*.txt")))
    if rep.files_ok != 2 or rep.chunks_added <= 0:
        raise AssertionError(f"入库异常: ok={rep.files_ok} chunks={rep.chunks_added} err={rep.errors}")
    return f"{rep.files_ok} 个文件 / {rep.chunks_added} 个 chunk"


@step("三档检索各跑一次")
def check_retrieval():
    details = []
    for mode in ("vector", "hybrid", "hybrid_rerank"):
        res = _RETRIEVER.search("ImageNet 预训练", mode=mode, top_k=3)
        if not res.chunks:
            raise AssertionError(f"模式 {mode} 未返回任何结果 (degraded={res.degraded})")
        details.append(f"{mode}:{len(res.chunks)}")
    return " ".join(details)


@step("空知识库提问不崩溃")
def check_empty_kb():
    from core.providers import MockEmbedding
    from retrieval.bm25_index import BM25Index
    from retrieval.hybrid import HybridRetriever
    from retrieval.reranker import Reranker
    from retrieval.vectorstore import VectorStore

    empty_dir = _TMP / "empty"
    vs = VectorStore(empty_dir / "chroma", "empty", MockEmbedding())
    bm = BM25Index(empty_dir / "bm25")
    res = HybridRetriever(vs, bm, Reranker()).search("任意问题")
    if res.chunks:
        raise AssertionError("空库竟然返回了结果")
    if res.degraded != "empty_knowledge_base":
        raise AssertionError(f"降级标记不是 empty_knowledge_base: {res.degraded}")
    return f"degraded={res.degraded}"


@step("RAG 流式生成事件序列")
def check_stream():
    events = list(_ENGINE.stream_answer("ViT 用什么数据集预训练"))
    kinds = [e["type"] for e in events]
    if not events:
        raise AssertionError("没有产生任何事件")
    if kinds[0] != "citations":
        raise AssertionError(f"首个事件应为 citations，实际 {kinds[0]}")
    if kinds[-1] != "done":
        raise AssertionError(f"最后事件应为 done，实际 {kinds[-1]}")
    text = "".join(e.get("text", "") for e in events if e["type"] == "token")
    if not text.strip():
        raise AssertionError("没有产生任何 token")
    if events[-1].get("answer") != text:
        raise AssertionError("done.answer 与 token 拼接不一致")
    return f"{len(events)} 个事件 / {len(text)} 字"


@step("ReAct Agent 端到端（mock LLM）")
def check_agent():
    from agent import build_agent

    agent = build_agent(retriever=_RETRIEVER)
    result = agent.run("ViT 用什么数据集预训练")
    if not result.answer.strip():
        raise AssertionError("Agent 返回了空答案")
    if not result.steps:
        raise AssertionError("Agent 没有产生任何轨迹步")
    return f"{len(result.steps)} 步 / answer {len(result.answer)} 字"


@step("纯算术问题不触发检索")
def check_router():
    from agent.router import classify

    d = classify("3.14 * 2.56 等于多少")
    if d.kind != "direct_tool" or d.tool != "calculator":
        raise AssertionError(f"算术未被路由到 calculator: kind={d.kind} tool={d.tool}")
    d2 = classify("3.14")
    if d2.kind != "agent":
        raise AssertionError("纯数字被误判为算术表达式")
    return f"算术→{d.tool}，纯数字→{d2.kind}"


@step("可观测层写入与聚合")
def check_observability():
    from observability.metrics import aggregate, compute, render_trace_tree
    from observability.trace import TraceRecorder

    rec = TraceRecorder(_TMP / "logs")
    from agent import build_agent

    result = build_agent(retriever=_RETRIEVER).run("Swin 的复杂度是多少")
    rec.record(result, "smoke-session")
    rows = rec.read_recent(10)
    if not rows:
        raise AssertionError("轨迹没有落盘")
    agg = aggregate(rows)
    if agg["count"] != len(rows):
        raise AssertionError(f"聚合数量不符: {agg['count']} != {len(rows)}")
    tree = render_trace_tree(result.steps)
    for kw in ("Thought", "Action", "Observation"):
        if kw not in tree:
            raise AssertionError(f"轨迹树缺少 {kw}")
    return f"{len(rows)} 条轨迹 / 聚合 count={agg['count']}"


@step("评测指标函数可调用")
def check_eval_metrics():
    from scripts.eval_retrieval import hit_at_k, recall_at_k, reciprocal_rank

    ids = [c.chunk.chunk_id for c in _RETRIEVER.search("ViT", mode="hybrid_rerank").chunks]
    if not ids:
        raise AssertionError("检索无结果，无法计算指标")
    gold = {ids[0]}
    h, rr, rc = hit_at_k(ids, gold, 5), reciprocal_rank(ids, gold), recall_at_k(ids, gold, 5)
    if h != 1 or rr != 1.0 or rc != 1.0:
        raise AssertionError(f"指标计算异常: hit={h} rr={rr} recall={rc}")
    return f"hit@5={h} mrr={rr:.2f} recall@5={rc:.2f}"


def main() -> int:
    print("=" * 68)
    print("智能科研助理 · 全链路冒烟自检（mock 模式，不联网、不产生费用）")
    print("=" * 68)
    for fn in (check_imports, check_build, check_index, check_retrieval, check_empty_kb,
               check_stream, check_agent, check_router, check_observability,
               check_eval_metrics):
        fn()
    print("-" * 68)
    if FAILURES:
        print(f"[FAIL] {len(FAILURES)} 个步骤失败：")
        for f in FAILURES:
            print(f"   - {f}")
        return 1
    print("[OK] 全链路自检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
