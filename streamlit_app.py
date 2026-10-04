"""智能科研助理 · Streamlit 前端入口。

    streamlit run streamlit_app.py

设计说明：
- `answer_once(question, session_id)` 是**不依赖 Streamlit 运行时**的纯逻辑入口，
  集成测试直接调它，无需启动 Streamlit。
- 重对象（Chroma 连接、BM25 索引、LLM 客户端）的复用靠 `retrieval.hybrid.build_retriever`
  与 `core.providers.get_*` 上的 `lru_cache`。Streamlit 每次交互都会重跑整个脚本，
  但模块本身保持加载，所以缓存跨重跑有效 —— 这是 Streamlit 应用最常见的性能陷阱。
- 所有可变状态放 `st.session_state`，否则对话历史会在重跑时丢失。
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from core.logging_utils import mask_secret  # noqa: E402
from core.schemas import AgentResult, Citation  # noqa: E402
from ingest.chunkers import get_chunker  # noqa: E402
from ingest.loaders import SUPPORTED_EXTS, safe_filename  # noqa: E402
from ingest.pipeline import IngestPipeline  # noqa: E402
from observability.metrics import compute, health_check, render_trace_tree  # noqa: E402
from observability.trace import TraceRecorder  # noqa: E402
from retrieval.hybrid import build_retriever  # noqa: E402

# ---------------------------------------------------------------------------
# 布局常量
# ---------------------------------------------------------------------------

# 三栏各自独立滚动，所以每栏需要一个确定的高度。
# `"stretch"` 让容器自适应视口剩余高度（Streamlit >= 1.29 支持 int | "stretch"）。
# 若在某些环境下 stretch 表现不对（比如容器塌成 0 高），把这里改成整数像素值
# （例如 640）即可，不需要动其它任何代码。
PANEL_HEIGHT: int | str = "stretch"


# ---------------------------------------------------------------------------
# 上传区状态管理
# ---------------------------------------------------------------------------


def _uploader_key() -> str:
    """当前这一轮要用的上传控件 key。

    Streamlit 按 key 归置控件状态，因此**换 key 就等于换一个全新的空控件**。
    `st.file_uploader` 没有官方的清空接口，这是让上传区在入库成功后恢复空白的
    干净做法：入库完把 seq 加一，下一轮渲染出来的就是一个没挂任何文件的新控件。
    """
    return f"uploader_{int(st.session_state.get('uploader_seq', 0))}"


def _reset_uploader() -> None:
    """让下一次渲染使用全新的上传控件（旧控件连同已选文件一并被丢弃）。"""
    # 用下标赋值而不是属性赋值：两者在真实 session_state 上等价，
    # 但下标写法在普通 dict 上也成立，便于脱离 Streamlit 运行时单测这段逻辑。
    st.session_state["uploader_seq"] = int(st.session_state.get("uploader_seq", 0)) + 1


def _flash_ingest_report() -> None:
    """显示上一轮入库的结果，只显示一次。

    必须在入库之后的**下一轮**才显示：入库函数末尾会 `st.rerun()` 去刷新文档
    列表，如果就地 `st.success()`，那条消息会被紧接着的重跑冲掉——用户其实
    从来没看到过入库结果。所以结果先暂存到 session_state，重跑后再取出来显示。
    """
    flash = st.session_state.pop("ingest_flash", None)
    if not flash:
        return
    st.success(
        f"入库完成：成功 {flash['ok']} · 跳过 {flash['skipped']}（内容未变）· "
        f"失败 {flash['failed']} · 新增 {flash['chunks']} 块 · {flash['ms']} ms"
    )
    for e in flash.get("errors", []):
        st.error(f"{e['filename']}：{e['error']}")


# ---------------------------------------------------------------------------
# 资源获取（Streamlit 无关，便于测试替换）
# ---------------------------------------------------------------------------


def _get_retriever():
    """返回全局唯一的检索器。build_retriever 自带 lru_cache。"""
    return build_retriever()


def _get_llm():
    from core.providers import get_llm

    return get_llm()


def _get_memory():
    from agent.memory import ConversationMemory

    return ConversationMemory(
        max_tokens=settings.memory_max_tokens,
        summarize_after_turns=settings.summarize_after_turns,
        store_dir=settings.session_dir,
    )


def _get_recorder():
    return TraceRecorder(settings.log_dir)


# ---------------------------------------------------------------------------
# 纯逻辑入口（无 Streamlit 依赖）
# ---------------------------------------------------------------------------


def answer_once(question: str, session_id: str) -> AgentResult:
    """跑一次完整的 Agent 问答，落轨迹，写记忆，返回结果。

    刻意不碰任何 st.* API —— 这样集成测试与 CLI 都能直接调用。
    """
    from agent import build_agent

    question = (question or "").strip()
    if not question:
        return AgentResult(answer="请输入问题。", citations=[], steps=[], tokens={},
                           latency_ms={"total": 0}, route="empty", degraded="empty_question")

    retriever = _get_retriever()

    # 空知识库 + 需要检索的问题 → 直接给明确提示。
    # 这是作业明确要求的边缘场景（「空知识库提问 → 提示知识库为空，请先上传文档，
    # 不抛异常」）。纯算术 / 问时间这类不需要知识库的问题仍照常走到工具路由，
    # 所以这里必须先问一次 router。
    from agent.router import classify

    if retriever.vectorstore.count() == 0 and retriever.bm25.count() == 0:
        if classify(question).kind == "agent":
            return AgentResult(
                answer=(
                    "⚠️ 知识库为空，请先在左侧上传论文（支持 PDF / DOCX / TXT / MD）。\n\n"
                    "上传后系统会先建立索引，再基于你的文献回答问题。"
                ),
                citations=[], steps=[], tokens={}, latency_ms={"total": 0},
                route="empty_kb", degraded="empty_knowledge_base",
            )

    llm = _get_llm()
    memory = _get_memory()
    recorder = _get_recorder()

    try:
        memory.load(session_id)
    except Exception:  # noqa: BLE001 - 历史读不到不该阻断提问
        pass

    agent = build_agent(retriever=retriever, llm=llm, memory=memory)

    try:
        result = agent.run(question)
    except Exception as exc:  # noqa: BLE001 - 任何异常都要变成可读回答，不能让界面崩
        result = AgentResult(
            answer=f"❌ 处理问题时出错：{type(exc).__name__}: {exc}\n请稍后重试。",
            citations=[], steps=[], tokens={}, latency_ms={"total": 0},
            route="error", degraded=f"exception: {type(exc).__name__}",
        )

    try:
        recorder.record(result, session_id)
    except Exception:  # noqa: BLE001 - 观测失败绝不影响主流程
        pass

    return result


# ---------------------------------------------------------------------------
# 界面辅助
# ---------------------------------------------------------------------------


def _render_citations(citations: list) -> None:
    """渲染引用来源。同时接受 Citation 对象与 dict（两种路径都会传进来）。"""
    if not citations:
        return
    with st.expander(f"📎 引用来源（{len(citations)} 条）", expanded=False):
        for c in citations:
            item = Citation(**c) if isinstance(c, dict) else c
            st.markdown(
                f"**[{item.index}]** `{item.filename}` · 第 {item.page_no} 页  \n"
                f"<span style='color:#666;font-size:0.85em'>{item.snippet[:280]}</span>",
                unsafe_allow_html=True,
            )


def _render_trace(result: AgentResult) -> None:
    if not result.steps:
        st.caption("本轮无 Agent 轨迹（可能走了快速 RAG 路径）")
        return
    with st.expander(f"🧠 推理轨迹（{len(result.steps)} 步）", expanded=True):
        st.code(render_trace_tree(result.steps), language="text")


def _render_metrics(result: AgentResult) -> None:
    m = compute(result)
    c1, c2 = st.columns(2)
    c1.metric("Token", m["total_tokens"])
    c2.metric("耗时", f"{m['total_ms']} ms")
    c3, c4 = st.columns(2)
    c3.metric("推理步数", m["steps"])
    c4.metric("引用数", m["n_citations"])
    st.caption(f"路径：`{result.route}`")
    if result.degraded:
        st.warning(f"降级：{result.degraded}")

    if m["tools_used"]:
        st.caption("调用工具：" + "、".join(f"`{t}`" for t in m["tools_used"]))


def _doc_panel() -> None:
    retriever = _get_retriever()
    docs = retriever.vectorstore.list_documents()

    st.subheader("📚 知识库")
    if not docs:
        st.info("知识库为空，请先上传文档。")
    else:
        for d in docs:
            row = st.columns([4, 1])
            row[0].markdown(
                f"**{d['filename']}**  \n"
                f"<span style='color:#888;font-size:0.8em'>{d['chunk_count']} 块 · "
                f"{d['ingested_at'][:19]}</span>",
                unsafe_allow_html=True,
            )
            if row[1].button("删除", key=f"del_{d['doc_id']}", help="删除该文档的向量与关键词索引"):
                n = retriever.vectorstore.delete_document(d["doc_id"])
                n += retriever.bm25.remove_document(d["doc_id"])
                retriever.bm25.save()
                st.success(f"已删除 {d['filename']}（{n} 条索引）")
                st.rerun()

    st.caption(f"向量 {retriever.vectorstore.count()} 块 · 关键词 {retriever.bm25.count()} 块")


def _upload_panel() -> None:
    st.subheader("⬆️ 上传论文")
    files = st.file_uploader(
        "支持 PDF / DOCX / TXT / MD，可多选",
        type=[e.lstrip(".") for e in sorted(SUPPORTED_EXTS)],
        accept_multiple_files=True,
        key=_uploader_key(),
    )
    if not files:
        return

    if not st.button("开始入库", type="primary"):
        st.caption(f"已选择 {len(files)} 个文件，点击上方按钮开始。")
        return

    retriever = _get_retriever()
    chunker = get_chunker(settings.chunk_strategy, chunk_size=settings.chunk_size,
                          overlap=settings.chunk_overlap)
    pipeline = IngestPipeline(retriever=retriever, chunker=chunker, data_dir=settings.raw_dir)

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for f in files:
        dest = settings.raw_dir / safe_filename(f.name)
        dest.write_bytes(f.getbuffer())
        paths.append(dest)

    bar = st.progress(0.0, text="解析中…")

    def on_progress(name: str, frac: float) -> None:
        bar.progress(min(max(frac, 0.0), 1.0), text=f"{name or '完成'} ({frac * 100:.0f}%)")

    report = pipeline.ingest_paths(paths, progress_cb=on_progress)
    bar.empty()

    # 结果暂存到 session_state：下面紧接着的 st.rerun() 会立刻重跑脚本，
    # 就地 st.success() 的话消息会被冲掉，用户看不到任何反馈。
    st.session_state.ingest_flash = {
        "ok": report.files_ok,
        "skipped": report.files_skipped,
        "failed": report.files_failed,
        "chunks": report.chunks_added,
        "ms": report.elapsed_ms,
        "errors": list(report.errors),
    }
    # 换掉上传控件 key，使下一轮渲染出一个没挂任何文件的新控件（上传区复位）
    _reset_uploader()
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="智能科研助理", page_icon="🔬", layout="wide")
    st.title("🔬 智能科研助理")
    st.caption("基于 RAG + Agent 的论文知识库问答系统 · 混合检索 + 重排序 + 可解释推理轨迹")

    # ---- 健康状态横幅 ----
    retriever = _get_retriever()
    health = health_check(retriever, _get_llm())
    bad = {k: v[1] for k, v in health.items() if not v[0]}
    if bad:
        st.error("⚠️ 组件异常：" + "；".join(f"{k} → {msg}" for k, msg in bad.items()) +
                 "　（检查 .env 中的密钥，或把三个 *_PROVIDER 设为 mock 离线运行）")
    else:
        key_hint = mask_secret(settings.deepseek_api_key) or "mock"
        st.caption(f"✅ 所有组件正常 · LLM={settings.llm_provider}/{key_hint} · "
                   f"Embedding={settings.embedding_provider} · Rerank={settings.rerank_provider}")

    # ---- 会话状态初始化 ----
    if "session_id" not in st.session_state:
        st.session_state.session_id = _get_memory().new_session()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "uploader_seq" not in st.session_state:
        # 上传控件的代次。入库成功后自增，从而换出一个空的上传控件。
        st.session_state.uploader_seq = 0

    # ---- 侧栏：会话管理 ----
    with st.sidebar:
        st.subheader("💬 会话")
        if st.button("➕ 新建会话", use_container_width=True):
            st.session_state.session_id = _get_memory().new_session()
            st.session_state.messages = []
            st.session_state.last_result = None
            st.rerun()
        st.caption(f"当前会话：`{st.session_state.session_id[:8]}`")

        mode = st.radio(
            "问答模式",
            ["🤖 Agent 智能体", "⚡ RAG 快速流式"],
            help="Agent 模式会自主决策调用工具并展示推理轨迹；RAG 模式直接检索生成，速度更快。",
        )

        st.divider()
        st.caption("配置")
        st.caption(f"分块：{settings.chunk_strategy} / {settings.chunk_size}")
        st.caption(f"检索：{settings.retrieve_top_k} → rerank {settings.rerank_top_n}")
        st.caption(f"上下文预算：{settings.max_context_tokens} tokens")

    left, center, right = st.columns([1, 2.4, 1.4])

    with left:
        # 独立滚动区：中栏对话再长，左栏也始终留在屏幕内，不会被落在页面上方
        with st.container(height=PANEL_HEIGHT):
            _flash_ingest_report()
            _upload_panel()
            st.divider()
            _doc_panel()

    with center:
        st.subheader("💬 对话")
        # 只有消息列表进滚动区。`st.chat_input` 必须留在容器**外面**，
        # 否则输入框会跟着对话一起被滚走。
        with st.container(height=PANEL_HEIGHT):
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("citations"):
                        _render_citations(msg["citations"])
            # 占位子容器：本轮对话要写在滚动区**内部**，但问题得等
            # chat_input 返回后才知道，所以先占位、后填充。
            new_turn = st.container()

        question = st.chat_input("就你的论文库提问，例如：ViT 用了什么数据集预训练？")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with new_turn:
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    if mode.startswith("⚡"):
                        from generation.rag import RAGEngine

                        engine = RAGEngine(retriever=_get_retriever(), llm=_get_llm())
                        text_holder = st.empty()
                        acc = ""
                        raw_citations: list = []
                        for ev in engine.stream_answer(question):
                            if ev["type"] == "citations":
                                raw_citations = ev["citations"]
                            elif ev["type"] == "token":
                                acc += ev["text"]
                                text_holder.markdown(acc + "▌")
                            elif ev["type"] == "error":
                                st.error(ev["message"])
                        text_holder.markdown(acc or "（无输出）")
                        # 事件里的引用是 dict，转成 Citation 后再渲染与入历史，
                        # 保证两条路径（Agent / RAG）存进 messages 的类型一致。
                        citations = [Citation(**c) if isinstance(c, dict) else c
                                     for c in raw_citations]
                        _render_citations(citations)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": acc, "citations": citations}
                        )
                    else:
                        with st.status("Agent 正在推理…", expanded=False) as status:
                            result = answer_once(question, st.session_state.session_id)
                            status.update(
                                label=f"完成（{len(result.steps)} 步推理，{result.latency_ms.get('total', 0)} ms）",
                                state="complete",
                            )
                        st.markdown(result.answer)
                        _render_citations(result.citations)
                        st.session_state.last_result = result
                        st.session_state.messages.append(
                            {"role": "assistant", "content": result.answer,
                             "citations": result.citations}
                        )

    with right:
        # 与左栏同理：独立滚动区，长对话下不会把轨迹面板甩到视口外
        with st.container(height=PANEL_HEIGHT):
            st.subheader("🔍 可观测")
            if st.session_state.last_result is not None:
                res = st.session_state.last_result
                _render_trace(res)
                st.divider()
                _render_metrics(res)
            else:
                st.info("提一个问题后，这里会显示 Agent 的完整推理轨迹与各项指标。")
            st.divider()
            st.subheader("🩺 健康检查")
            for name, (ok, msg) in health.items():
                st.caption(("✅ " if ok else "❌ ") + f"{name}: {msg}")


if __name__ == "__main__":
    main()
