# 智能科研助理（RAG+Agent 论文知识库问答）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个可交互的论文知识库问答 Web 应用——上传论文后，用「混合检索 + 重排序」精确定位证据，由手写 ReAct Agent 自主决策调用 8 个工具，生成带引用溯源、且推理轨迹可见的答案。

**Architecture:** 五层单向依赖（ui → agent → generation → retrieval → ingest → core），层间只通过 `core/schemas.py` 的 6 个数据结构通信。RAG 被封装成 Agent 的一个 Tool，使简单问题可跳过检索。所有 LLM/Embedding/Rerank 调用经 `core/providers.py` 抽象，支持 `mock` 实现以便无 API key 时验证全链路。

**Tech Stack:** Python 3.12 / DeepSeek `deepseek-chat`（生成）/ SiliconFlow `BAAI/bge-m3`（向量化）/ SiliconFlow `BAAI/bge-reranker-v2-m3`（重排序）/ ChromaDB / rank_bm25 / PyMuPDF / python-docx / Streamlit / pytest。**不含 LangChain、LlamaIndex、sentence-transformers、torch。**

**Spec:** `docs/项目构建书.md`

## Global Constraints

- Python `>=3.10`（实测环境 3.12.7，Anaconda）。所有路径用 `pathlib.Path`，禁止裸字符串拼接。
- 依赖仅允许 `requirements.txt` 中列出的包。**严禁**引入 `langchain`、`llama_index`、`sentence_transformers`、`torch`、`jieba`。
- 所有配置项必须定义在 `config.py` 的 `Settings` 中，禁止在业务代码里硬编码默认值。敏感值只从环境变量/`.env` 读取。
- `.env` 必须在 `.gitignore` 内；仓库只提交 `.env.example`。日志中 API key 一律脱敏为 `sk-***`。
- 三个 Provider 开关 `LLM_PROVIDER` / `EMBEDDING_PROVIDER` / `RERANK_PROVIDER` 取值为 `deepseek|mock`、`siliconflow|mock`、`siliconflow|mock`。
- 若 provider ≠ `mock` 且对应 key 为空 → **模块导入期**即抛 `ValueError`，不得延迟到首次请求。
- Embedding 维度固定 1024（`BAAI/bge-m3`）。Mock 实现必须同样返回 1024 维，且对相同文本确定性地返回相同向量。
- 所有对外部 API 的调用必须有超时（默认 30s）与指数退避重试（最多 3 次，间隔 1s/2s/4s）。
- 代码注释与文档用中文；变量名、函数名用英文。
- 每个 Task 结束必须 `git commit`，提交信息用 `feat:` / `test:` / `docs:` 前缀。

## Review Focus

以下是 spec 隐含、但没有任何 Task 的测试直接覆盖、且最容易在使用中咬人的 5 类输入/故障。每一行都在其归属 Task 中被一条测试钉住：

1. **首次启动、知识库为空时提问** —— 合理预期：提示"知识库为空，请先上传文档"，而不是 `ZeroDivisionError` / 空列表索引崩溃。（Task 9 / Task 16 覆盖）
2. **`pip install` 后未配置任何 API key 就运行** —— 合理预期：启动即报明确可读的配置错误，而不是运行到一半抛 `KeyError` 或静默发出空 Authorization 头。（Task 2 覆盖）
3. **上传同一文件两次 / 上传损坏的 PDF** —— 合理预期：前者幂等跳过并提示已存在；后者该文件标失败但整批继续，不中断。（Task 3 / Task 5 覆盖）
4. **LLM 返回格式非法的 Action（缺 Action Input、JSON 坏引号、输出空串）** —— 合理预期：解析器给出可读错误并让 Agent 重试，连续 3 次后才强制收敛，而不是抛 `json.JSONDecodeError` 穿到前端。（Task 16 覆盖）
5. **含中文全角标点、超长单句、以及只含数字的"问题"** —— 合理预期：分块不产生空 chunk、BM25 分词不崩、路由不把 `"3.14"` 误判成需要检索。（Task 4 / Task 6 / Task 17 覆盖）

---

### Task 1: 工程骨架、配置中心与统一数据结构

**Files:**
- Create: `requirements.txt`, `.env.example`, `.gitignore`, `README.md`
- Create: `config.py`
- Create: `core/__init__.py`, `core/schemas.py`
- Create: `tests/__init__.py`, `tests/test_schemas.py`
- Create: `pytest.ini`

**Interfaces:**
- Consumes: 无（项目起点）
- Produces:
  - `config.Settings`（pydantic-settings 模型，全部字段见 §6.1）与模块级单例 `config.settings`
  - `core.schemas.Page(text: str, page_no: int)`
  - `core.schemas.Document(doc_id: str, filename: str, filetype: str, pages: list[Page], meta: dict, ingested_at: str)`
  - `core.schemas.Chunk(chunk_id: str, doc_id: str, filename: str, page_no: int, seq: int, text: str, token_count: int)`
  - `core.schemas.ScoredChunk(chunk: Chunk, score: float, source: str, rank: int)`
  - `core.schemas.Citation(index: int, filename: str, page_no: int, chunk_id: str, snippet: str)`
  - `core.schemas.Message(role: str, content: str, citations: list[Citation])`
  - `core.schemas.AgentStep(index: int, thought: str, action: str, action_input: dict, observation: str, latency_ms: int, tokens: int, error: str | None)`
  - `core.schemas.AgentResult(answer: str, citations: list[Citation], steps: list[AgentStep], tokens: dict, latency_ms: dict, route: str, degraded: str | None)`
  - `core.schemas.RetrievalResult(chunks: list[ScoredChunk], mode: str, latency_ms: int, degraded: str | None)`
  - `core.schemas.estimate_tokens(text: str) -> int`

- [ ] **Step 1: 写失败测试 `tests/test_schemas.py`**

```python
from core.schemas import Chunk, Document, Page, ScoredChunk, estimate_tokens


def test_document_holds_pages_with_page_numbers():
    doc = Document(doc_id="abc123", filename="vit.pdf", filetype="pdf",
                   pages=[Page(text="hello", page_no=1)], meta={}, ingested_at="2026-01-01")
    assert doc.pages[0].page_no == 1


def test_chunk_id_composes_from_doc_page_seq():
    c = Chunk(chunk_id="abc123::3::2", doc_id="abc123", filename="vit.pdf",
              page_no=3, seq=2, text="text", token_count=1)
    assert c.chunk_id.split("::") == ["abc123", "3", "2"]


def test_estimate_tokens_counts_cjk_as_one_token_each():
    assert estimate_tokens("你好世界") == 4


def test_estimate_tokens_counts_latin_by_word_groups():
    # 12 个英文单词 -> 向上取整(12/0.75)=16
    text = "one two three four five six seven eight nine ten eleven twelve"
    assert estimate_tokens(text) == 16


def test_estimate_tokens_empty_string_is_zero():
    assert estimate_tokens("") == 0


def test_scored_chunk_keeps_source_and_rank():
    c = Chunk(chunk_id="a::1::0", doc_id="a", filename="f.pdf", page_no=1,
              seq=0, text="t", token_count=1)
    s = ScoredChunk(chunk=c, score=0.9, source="rerank", rank=1)
    assert (s.source, s.rank) == ("rerank", 1)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core'`

- [ ] **Step 3: 写 `requirements.txt`、`.gitignore`、`.env.example`、`pytest.ini`**

`requirements.txt` 内容与 Spec §2.3 完全一致（openai/httpx/tenacity/chromadb/rank-bm25/numpy/pymupdf/python-docx/streamlit/pydantic/pydantic-settings/python-dotenv/pandas/matplotlib/pytest）。

`.gitignore` 必须包含：`.env`、`__pycache__/`、`*.pyc`、`.pytest_cache/`、`storage/`、`data/raw/*`、`data/processed/*`、`!data/raw/.gitkeep`、`eval/reports/*`、`!eval/reports/.gitkeep`。

`.env.example` 内容与 Spec §6.2 一致。

`pytest.ini`：
```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q
```

- [ ] **Step 4: 写 `config.py`**

按 Spec §6.1 逐字段实现 `Settings`，并加校验：

```python
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    # ...（§6.1 全部字段，类型与默认值一字不差）

    @model_validator(mode="after")
    def _check_keys_present(self):
        missing = []
        if self.llm_provider != "mock" and not self.deepseek_api_key:
            missing.append("DEEPSEEK_API_KEY")
        if self.embedding_provider != "mock" and not self.siliconflow_api_key:
            missing.append("SILICONFLOW_API_KEY")
        if self.rerank_provider != "mock" and not self.siliconflow_api_key:
            missing.append("SILICONFLOW_API_KEY")
        if missing:
            raise ValueError(
                f"缺少必填配置: {', '.join(sorted(set(missing)))}。"
                f"请在 .env 中填写，或将 *_PROVIDER 设为 mock 以离线运行。"
            )
        return self


settings = Settings()
```

- [ ] **Step 5: 写 `core/schemas.py` 与 `core/__init__.py`**

全部用 `@dataclass`。`estimate_tokens` 规则：CJK 字符每个计 1 token，其余按 `ceil(英文单词数 / 0.75)`，CJK 与英文分别统计后相加。

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m pytest tests/test_schemas.py -v`
Expected: 6 passed

- [ ] **Step 7: 提交**

```bash
git init
git add -A
git commit -m "feat: 工程骨架、配置中心与统一数据结构"
```

---

### Task 2: Provider 抽象层（LLM / Embedding / Rerank）含 Mock 实现

**Files:**
- Create: `core/providers.py`
- Create: `core/logging_utils.py`
- Test: `tests/test_providers.py`

**Interfaces:**
- Consumes: `config.settings`、`core.schemas`
- Produces:
  - `core.providers.LLMProvider`（ABC）：`complete(messages: list[dict], **kw) -> str`、`stream(messages: list[dict], **kw) -> Iterator[str]`、`last_usage -> dict`
  - `core.providers.EmbeddingProvider`（ABC）：`embed(texts: list[str]) -> list[list[float]]`、`dim: int`、`embed_one(text: str) -> list[float]`
  - `core.providers.RerankProvider`（ABC）：`rerank(query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]`（返回 `(原下标, 分数)`）
  - `core.providers.DeepSeekLLM`、`SiliconFlowEmbedding`、`SiliconFlowRerank`
  - `core.providers.MockLLM`、`MockEmbedding`、`MockRerank`
  - `core.providers.get_llm() -> LLMProvider`、`get_embedder() -> EmbeddingProvider`、`get_reranker() -> RerankProvider`（带 `functools.lru_cache`）
  - `core.logging_utils.mask_secret(s: str) -> str`、`get_logger(name: str) -> Logger`

- [ ] **Step 1: 写失败测试 `tests/test_providers.py`**

```python
import math

import pytest

from core.providers import MockEmbedding, MockLLM, MockRerank, get_embedder, get_llm, get_reranker


def test_mock_embedding_is_deterministic():
    e = MockEmbedding()
    assert e.embed(["你好"]) == e.embed(["你好"])


def test_mock_embedding_dim_is_1024_and_normalized():
    e = MockEmbedding()
    v = e.embed_one("任意文本")
    assert len(v) == 1024
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-6)


def test_mock_embedding_differs_for_different_text():
    e = MockEmbedding()
    assert e.embed_one("苹果") != e.embed_one("香蕉")


def test_mock_embedding_handles_empty_batch():
    assert MockEmbedding().embed([]) == []


def test_mock_embedding_handles_empty_string():
    v = MockEmbedding().embed_one("")
    assert len(v) == 1024


def test_mock_rerank_orders_by_overlap_with_query():
    r = MockRerank()
    out = r.rerank("transformer vision", ["完全无关的内容", "transformer for vision tasks"], top_n=2)
    assert out[0][0] == 1
    assert out[0][1] > out[1][1]


def test_mock_llm_stream_yields_multiple_tokens_and_records_usage():
    llm = MockLLM()
    text = "".join(llm.stream([{"role": "user", "content": "hi"}]))
    assert len(text) > 0
    assert llm.last_usage["completion_tokens"] > 0


def test_getters_return_provider_instances():
    assert isinstance(get_llm(), MockLLM)
    assert isinstance(get_embedder(), MockEmbedding)
    assert isinstance(get_reranker(), MockRerank)


def test_missing_key_raises_at_settings_construction(monkeypatch):
    from pydantic import ValidationError

    from config import Settings

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises((ValueError, ValidationError)) as exc:
        Settings(_env_file=None, llm_provider="deepseek", deepseek_api_key="")
    assert "DEEPSEEK_API_KEY" in str(exc.value)
```

> 测试通过 conftest 注入 `LLM_PROVIDER=mock` 等环境变量，见 Step 2。

- [ ] **Step 2: 写 `tests/conftest.py`**

```python
import os

os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("EMBEDDING_PROVIDER", "mock")
os.environ.setdefault("RERANK_PROVIDER", "mock")
```

- [ ] **Step 3: 运行测试确认失败**

Run: `python -m pytest tests/test_providers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.providers'`

- [ ] **Step 4: 实现 `core/logging_utils.py`**

```python
def mask_secret(value: str) -> str:
    """把 API key 脱敏为 sk-***abcdef 形式，用于日志。"""
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:3]}***{value[-4:]}"
```

`get_logger` 返回配置了 `StreamHandler`、格式 `%(asctime)s %(levelname)s %(name)s %(message)s` 的 logger，并用 `logging.getLogger` 缓存避免重复 handler。

- [ ] **Step 5: 实现 `core/providers.py`**

关键实现约束：

**MockEmbedding —— 必须确定性且归一化**，用 `hashlib.sha256` 播种：

```python
class MockEmbedding(EmbeddingProvider):
    dim = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # 32 字节 -> 反复扩展成 1024 个 [-1,1) 浮点数
        raw = (digest * 32)[: self.dim]
        v = [(b - 127.5) / 127.5 for b in raw]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]
```

**MockRerank** 用查询与文档的词元重合度打分（英文按词、中文按 bigram）。

**MockLLM**：`complete()` 返回固定可用话术；`stream()` 按字符切分 yield，并在结束时把 `last_usage` 设为估算的 prompt/completion token 数。当 prompt 中检测到 `"Action:"` 协议要求时（ReAct 场景），返回合法的 `Action: Finish` 结构，使 Agent 在全 mock 下也能收敛。

**DeepSeekLLM**：用 `openai.OpenAI(api_key=..., base_url=...)`；`stream()` 必须传 `stream_options={"include_usage": True}`，并在流结束后从末帧 `chunk.usage` 写入 `last_usage`；用 `tenacity.retry` 包裹（`stop_after_attempt(3)`、`wait_exponential(multiplier=1, min=1, max=4)`、仅对 `APIConnectionError`/`RateLimitError`/`InternalServerError` 重试）。

**SiliconFlowEmbedding**：`POST {base}/embeddings`，body `{"model": ..., "input": texts, "encoding_format": "float"}`，按 `data[i]["index"]` **排序后**返回（不能假定响应顺序）。分批 `EMBED_BATCH_SIZE=32`。**磁盘缓存**：key 为 `sha256(text)`，存 `storage/cache/emb/{key}.npy`（用 `numpy.save`），命中直接读，未命中的批量请求后写回。

**SiliconFlowRerank**：`POST {base}/rerank`，body `{"model":..., "query":..., "documents":[...], "top_n":..., "return_documents": False}`，返回 `[(r["index"], r["relevance_score"]) for r in resp["results"]]`。

三个 `get_*` 工厂读 `settings.*_provider` 分派，用 `functools.lru_cache(maxsize=1)` 保证单例。

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m pytest tests/test_providers.py -v`
Expected: 9 passed

- [ ] **Step 7: 提交**

```bash
git add -A
git commit -m "feat: Provider 抽象层与 Mock 实现，支持无 key 全链路运行"
```

---

### Task 3: 多格式文档加载器

**Files:**
- Create: `ingest/__init__.py`, `ingest/loaders.py`
- Test: `tests/test_loaders.py`

**Interfaces:**
- Consumes: `core.schemas.{Document, Page}`
- Produces:
  - `ingest.loaders.LoadError(Exception)`
  - `ingest.loaders.load_document(path: Path) -> Document`
  - `ingest.loaders.SUPPORTED_EXTS: set[str]` = `{".pdf", ".docx", ".txt", ".md"}`
  - `ingest.loaders.make_doc_id(path: Path) -> str`（内容 sha256 前 16 位）
  - `ingest.loaders.safe_filename(name: str) -> str`

- [ ] **Step 1: 写失败测试 `tests/test_loaders.py`**

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_loaders.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ingest'`

- [ ] **Step 3: 实现 `ingest/loaders.py`**

要点：
- `make_doc_id`：**分块读取**文件（`read(1 << 20)` 循环）算 sha256，避免把大 PDF 整个读进内存。
- `load_document` 按后缀分派；任何解析异常包成 `LoadError(f"{filename}: {原始错误}")`，**不泄漏第三方堆栈到前端**。
- **PyMuPDF**：`fitz.open(path)`，逐页 `page.get_text("text")`；`try/finally` 确保 `doc.close()`。空页只记 `page_no` 不产生空 `Page`。
- **DOCX**：`python-docx` 读 `doc.paragraphs`，全部段落合并为 `page_no=1`（DOCX 无页概念，诚实标注为 1 页）。
- **TXT/MD**：`read_text(encoding="utf-8", errors="replace")`，`errors="replace"` 防止 GBK 文件直接崩。
- **元信息抽取 `_extract_meta`**：尽力从首两页正则匹配标题（首页最长非空行）、年份（`(19|20)\d{2}`）、DOI（`10\.\d{4,}/\S+`）；抽不到就留空，**绝不编造**。
- `safe_filename`：`Path(name).name` 后替换 `<>:"|?*` 为空，并限制长度 120。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_loaders.py -v`
Expected: 11 passed

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "feat: 多格式文档加载器（PDF/DOCX/TXT/MD）与内容寻址 doc_id"
```

---

### Task 4: 文本分块策略（三种）

**Files:**
- Create: `ingest/chunkers.py`
- Test: `tests/test_chunkers.py`

**Interfaces:**
- Consumes: `core.schemas.{Chunk, Document, Page}`
- Produces:
  - `ingest.chunkers.BaseChunker`（ABC）：`split(doc: Document) -> list[Chunk]`
  - `ingest.chunkers.FixedChunker(chunk_size: int, overlap: int)`
  - `ingest.chunkers.RecursiveChunker(chunk_size: int, overlap: int)`
  - `ingest.chunkers.SemanticChunker(chunk_size: int, threshold: float, embed_fn: Callable[[list[str]], list[list[float]]] | None)`
  - `ingest.chunkers.get_chunker(strategy: str, **kw) -> BaseChunker`（`"fixed"|"recursive"|"semantic"`）
  - `ingest.chunkers.strip_boilerplate(pages: list[Page]) -> list[Page]`
  - `ingest.chunkers.is_reference_section(text: str) -> bool`

- [ ] **Step 1: 写失败测试 `tests/test_chunkers.py`**

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_chunkers.py -v`
Expected: FAIL — `ImportError: cannot import name 'FixedChunker'`

- [ ] **Step 3: 实现 `ingest/chunkers.py`**

要点：
- 共享 `_emit(doc, page_no, texts)` 辅助函数统一生成 `Chunk` 与递增 `seq`，并**过滤空白 chunk**（对应 Review Focus #5）。
- `FixedChunker`：按 `estimate_tokens` 滑动窗口，`step = chunk_size - overlap`，`overlap` 用 `max(0, ...)` 保护，`step <= 0` 时强制为 `chunk_size`（防止死循环）。
- `RecursiveChunker`：分隔符优先级 `["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", " ", ""]`；对超长片段递归降级到下一级分隔符；空字符串分隔符作为最终兜底做硬切。**这是默认策略**。
- `SemanticChunker`：`embed_fn is None` 时**直接委托 `RecursiveChunker`**（不报错，降级友好）；否则按句切分→批量 embed→相邻余弦 < `threshold` 处断开→再按 `chunk_size` 二次合并超长块。
- `strip_boilerplate`：统计每页首行/末行在所有页中的出现频次，出现率 > 60% 且长度 < 100 的行视为页眉页脚并从各页删除。
- `is_reference_section`：对文本做小写化后判断是否以 `references` / `bibliography` / `参考文献` 开头。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_chunkers.py -v`
Expected: 14 passed

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "feat: 三种文本分块策略（固定/递归/语义）与页眉页脚剔除"
```

---

### Task 5: 向量库封装（Chroma）与增量索引

**Files:**
- Create: `retrieval/__init__.py`, `retrieval/vectorstore.py`
- Test: `tests/test_vectorstore.py`

**Interfaces:**
- Consumes: `core.schemas.{Chunk, ScoredChunk}`、`core.providers.EmbeddingProvider`
- Produces:
  - `retrieval.vectorstore.VectorStore(persist_dir: Path, collection: str, embedder: EmbeddingProvider)`
  - `VectorStore.upsert(chunks: list[Chunk]) -> int`（返回实际写入数）
  - `VectorStore.document_exists(doc_id: str) -> bool`
  - `VectorStore.delete_document(doc_id: str) -> int`
  - `VectorStore.query(query: str, top_k: int) -> list[ScoredChunk]`
  - `VectorStore.list_documents() -> list[dict]`（`{doc_id, filename, chunk_count, ingested_at}`）
  - `VectorStore.count() -> int`
  - `VectorStore.health() -> tuple[bool, str]`

- [ ] **Step 1: 写失败测试 `tests/test_vectorstore.py`**

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_vectorstore.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'retrieval'`

- [ ] **Step 3: 实现 `retrieval/vectorstore.py`**

要点：
- `chromadb.PersistentClient(path=str(persist_dir))` + `get_or_create_collection(name, metadata={"hnsw:space": "cosine"})`。**用 cosine 空间**，与归一化向量的点积等价。
- `upsert` 先把 chunk 的 `doc_id`/`filename`/`page_no`/`seq`/`text` 全部写进 metadata（Chroma 的 `documents` 存正文，metadata 存坐标）。**`page_no` 必须为 int**，这是引用溯源的关键。
- Chroma 的 `where` 过滤要求值类型一致；`delete_document` 用 `collection.delete(where={"doc_id": doc_id})`，删除前先 `get(where=...)` 数出 `ids` 长度作为返回值。
- `query`：调 `embedder.embed_one(query)`，`collection.query(query_embeddings=[...], n_results=top_k, include=["documents","metadatas","distances"])`。cosine 距离转相似度 **`score = 1 - distance`**。**空集合时 Chroma 会抛异常，必须先用 `count()==0` 短路返回 `[]`**（对应 Review Focus #1）。
- `list_documents`：`collection.get(include=["metadatas"])` 后在 Python 侧聚合。
- `health()`：`count()` 可正常调用即 `(True, "ok")`，异常返回 `(False, str(e))`。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_vectorstore.py -v`
Expected: 10 passed

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "feat: Chroma 向量库封装，支持增量 upsert 与文档级删除"
```

---

### Task 6: BM25 关键词索引

**Files:**
- Create: `retrieval/bm25_index.py`
- Test: `tests/test_bm25.py`

**Interfaces:**
- Consumes: `core.schemas.{Chunk, ScoredChunk}`
- Produces:
  - `retrieval.bm25_index.tokenize(text: str) -> list[str]`
  - `retrieval.bm25_index.BM25Index(index_dir: Path)`
  - `BM25Index.build(chunks: list[Chunk]) -> None`
  - `BM25Index.add(chunks: list[Chunk]) -> None`（增量）
  - `BM25Index.remove_document(doc_id: str) -> int`
  - `BM25Index.query(query: str, top_k: int) -> list[ScoredChunk]`
  - `BM25Index.save() -> None` / `BM25Index.load() -> bool`
  - `BM25Index.count() -> int`

- [ ] **Step 1: 写失败测试 `tests/test_bm25.py`**

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_bm25.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'retrieval.bm25_index'`

- [ ] **Step 3: 实现 `retrieval/bm25_index.py`**

要点：
- `tokenize`：`re.findall(r"[a-zA-Z][a-zA-Z0-9\-]*", text.lower())` 取英文词（长度 ≥ 2），再对**所有 CJK 连续片段**生成 bigram：`re.findall(r"[\u4e00-\u9fff]+", text)` → 每段生成 `seg[i:i+2]`；单字段则直接保留该字。数字用 `re.findall(r"\d+(?:\.\d+)?", text)` 单独保留。
- `build`：用 `rank_bm25.BM25Okapi([tokenize(c.text) for c in chunks])`；同时维护 `self.chunks: dict[chunk_id, Chunk]` 与 `self.order: list[chunk_id]`（下标即 BM25 语料下标）。
- `add`：把新 chunk 并入 `self.order` 后**整体重建** BM25（BM25Okapi 不支持增量；由于索引仅存 token 列表，全量重建在万级 chunk 下仍在秒级）。已存在的 `chunk_id` 跳过（对应幂等性测试）。
- `query`：`scores = self.bm25.get_scores(tokenize(query))`；**过滤 `score <= 0`**（否则会返回大量无关 chunk）；按分数降序取 `top_k`；若全为 0 则返回 `[]`。
- `save`：`pickle.dump({"chunks": self.chunks, "order": self.order}, f)` 到 `index_dir/bm25.pkl`；`load` 时重建 `BM25Okapi`。文件不存在返回 `False`。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_bm25.py -v`
Expected: 13 passed

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "feat: BM25 关键词索引，含 CJK bigram 分词与持久化"
```

---

### Task 7: RRF 倒数排名融合（手写）

**Files:**
- Create: `retrieval/fusion.py`
- Test: `tests/test_fusion.py`

**Interfaces:**
- Consumes: `core.schemas.ScoredChunk`
- Produces:
  - `retrieval.fusion.rrf_fuse(result_lists: list[list[ScoredChunk]], k: int = 60, weights: list[float] | None = None, top_k: int | None = None) -> list[ScoredChunk]`
  - 返回的 `ScoredChunk.source == "rrf"`，`rank` 从 1 重新编号。

- [ ] **Step 1: 写失败测试 `tests/test_fusion.py`**

```python
import pytest

from core.schemas import Chunk, ScoredChunk
from retrieval.fusion import rrf_fuse


def _s(cid: str, score: float, source: str, rank: int) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(chunk_id=cid, doc_id=cid.split("::")[0], filename="f.pdf",
                                   page_no=1, seq=0, text=cid, token_count=1),
                       score=score, source=source, rank=rank)


def test_single_list_preserves_order():
    a = [_s("a::1::0", 0.9, "vector", 1), _s("b::1::0", 0.8, "vector", 2)]
    out = rrf_fuse([a])
    assert [s.chunk.chunk_id for s in out] == ["a::1::0", "b::1::0"]


def test_document_in_both_lists_outranks_document_in_one():
    va = [_s("x::1::0", 0.9, "vector", 1), _s("y::1::0", 0.5, "vector", 2)]
    bm = [_s("y::1::0", 9.0, "bm25", 1), _s("z::1::0", 5.0, "bm25", 2)]
    out = rrf_fuse([va, bm])
    assert out[0].chunk.chunk_id == "y::1::0"   # 两路都命中 -> 排第一


def test_source_is_rrf_and_ranks_are_renumbered():
    out = rrf_fuse([[_s("a::1::0", 1.0, "vector", 1)], [_s("b::1::0", 2.0, "bm25", 1)]])
    assert all(s.source == "rrf" for s in out)
    assert [s.rank for s in out] == [1, 2]


def test_scores_are_inversely_rank_based_not_raw():
    # raw score 差异巨大也不应影响 RRF：只用排名
    out = rrf_fuse([[_s("a::1::0", 0.001, "vector", 1), _s("b::1::0", 999.0, "vector", 2)]])
    assert out[0].chunk.chunk_id == "a::1::0"


def test_k_smoothing_reduces_gap_between_top_ranks():
    lists = [[_s("a::1::0", 1, "v", 1), _s("b::1::0", 1, "v", 2)]]
    gap_low_k = rrf_fuse(lists, k=1)[0].score - rrf_fuse(lists, k=1)[1].score
    gap_high_k = rrf_fuse(lists, k=60)[0].score - rrf_fuse(lists, k=60)[1].score
    assert gap_high_k < gap_low_k


def test_weights_bias_toward_one_channel():
    va = [_s("a::1::0", 1, "vector", 1)]
    bm = [_s("b::1::0", 1, "bm25", 1)]
    out = rrf_fuse([va, bm], weights=[1.0, 0.1])
    assert out[0].chunk.chunk_id == "a::1::0"


def test_deduplicates_same_chunk_across_lists():
    same = "a::1::0"
    out = rrf_fuse([[_s(same, 1, "vector", 1)], [_s(same, 1, "bm25", 1)]])
    assert len(out) == 1


def test_empty_input_returns_empty_list():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_top_k_truncates():
    out = rrf_fuse([[_s(f"c{i}::1::0", 1, "v", i + 1) for i in range(10)]], top_k=3)
    assert len(out) == 3


def test_mismatched_weights_length_raises():
    with pytest.raises(ValueError):
        rrf_fuse([[_s("a::1::0", 1, "v", 1)]], weights=[1.0, 2.0])
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_fusion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'retrieval.fusion'`

- [ ] **Step 3: 实现 `retrieval/fusion.py`**

```python
def rrf_fuse(result_lists, k=60, weights=None, top_k=None):
    """倒数排名融合：score(d) = Σ_i w_i / (k + rank_i(d))

    只用排名的好处：BM25 分数无上界、余弦相似度在 [-1,1]，
    两者量纲不可比。若做加权求和必须先归一化，而归一化会引入
    额外超参且对异常值敏感。RRF 天然免归一化、免调参。
    """
    lists = [lst for lst in (result_lists or []) if lst]
    if not lists:
        return []
    if weights is None:
        weights = [1.0] * len(lists)
    if len(weights) != len(result_lists or []):
        raise ValueError(f"weights 长度 {len(weights)} 与结果列表数 {len(result_lists or [])} 不一致")

    scores: dict[str, float] = {}
    first_seen: dict[str, ScoredChunk] = {}
    for lst, w in zip(lists, [w for w in weights if True][: len(lists)]):
        for rank, sc in enumerate(lst, start=1):
            cid = sc.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
            first_seen.setdefault(cid, sc)

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_k:
        ordered = ordered[:top_k]
    return [ScoredChunk(chunk=first_seen[cid].chunk, score=score, source="rrf", rank=i)
            for i, (cid, score) in enumerate(ordered, start=1)]
```

> 注意 `weights` 与 `lists`（已过滤空列表）的下标对齐问题：实现时先按 `zip(result_lists, weights)` 配对，再对 `lst` 为空的项跳过。测试 `test_mismatched_weights_length_raises` 要求对 **原始** `result_lists` 长度校验。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_fusion.py -v`
Expected: 10 passed

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "feat: 手写 RRF 倒数排名融合算法"
```

---

### Task 8: 重排序客户端

**Files:**
- Create: `retrieval/reranker.py`
- Test: `tests/test_reranker.py`

**Interfaces:**
- Consumes: `core.providers.RerankProvider` / `get_reranker()`、`core.schemas.ScoredChunk`
- Produces:
  - `retrieval.reranker.Reranker(provider: RerankProvider | None = None, enabled: bool = True, min_relevance: float = 0.2)`
  - `Reranker.rerank(query: str, candidates: list[ScoredChunk], top_n: int) -> tuple[list[ScoredChunk], str | None]`（第二项为降级原因，正常为 `None`）
  - 返回项的 `source == "rerank"`，`rank` 从 1 重编号。

- [ ] **Step 1: 写失败测试 `tests/test_reranker.py`**

```python
from core.providers import MockRerank
from core.schemas import Chunk, ScoredChunk
from retrieval.reranker import Reranker


class BoomRerank:
    def rerank(self, query, documents, top_n):
        raise RuntimeError("上游 503")


def _s(cid: str, text: str, rank: int) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(chunk_id=cid, doc_id="d", filename="f.pdf", page_no=1,
                                   seq=0, text=text, token_count=1),
                       score=0.5, source="rrf", rank=rank)


def test_rerank_reorders_and_relabels():
    r = Reranker(provider=MockRerank())
    cands = [_s("a::1::0", "完全无关的内容", 1), _s("b::1::0", "transformer vision tasks", 2)]
    out, degraded = r.rerank("transformer vision", cands, top_n=2)
    assert degraded is None
    assert out[0].chunk.chunk_id == "b::1::0"
    assert all(s.source == "rerank" for s in out)
    assert [s.rank for s in out] == [1, 2]


def test_empty_candidates_returns_empty_without_calling_provider():
    out, degraded = Reranker(provider=MockRerank()).rerank("q", [], top_n=5)
    assert out == [] and degraded is None


def test_disabled_reranker_passes_through_truncated():
    r = Reranker(provider=MockRerank(), enabled=False)
    cands = [_s(f"c{i}::1::0", f"t{i}", i + 1) for i in range(10)]
    out, degraded = r.rerank("q", cands, top_n=3)
    assert len(out) == 3
    assert degraded == "rerank_disabled"


def test_provider_failure_degrades_to_rrf_order():
    r = Reranker(provider=BoomRerank())
    cands = [_s("a::1::0", "t1", 1), _s("b::1::0", "t2", 2)]
    out, degraded = r.rerank("q", cands, top_n=2)
    assert degraded is not None and "503" in degraded
    assert [s.chunk.chunk_id for s in out] == ["a::1::0", "b::1::0"]
    assert all(s.source == "rrf" for s in out)


def test_low_relevance_is_reported_via_min_relevance_check():
    r = Reranker(provider=MockRerank(), min_relevance=2.0)   # 阈值高到不可能满足
    out, _ = r.rerank("q", [_s("a::1::0", "unrelated", 1)], top_n=1)
    assert r.is_low_confidence(out) is True


def test_is_low_confidence_false_for_empty_list():
    assert Reranker(provider=MockRerank()).is_low_confidence([]) is False


def test_top_n_larger_than_candidates_is_clamped():
    r = Reranker(provider=MockRerank())
    out, _ = r.rerank("q", [_s("a::1::0", "x", 1)], top_n=10)
    assert len(out) == 1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_reranker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'retrieval.reranker'`

- [ ] **Step 3: 实现 `retrieval/reranker.py`**

要点：
- `rerank` 把 `candidates` 的 `chunk.text` 作为 `documents` 传入；provider 返回 `(原下标, 分数)`，据此回查 `candidates[idx]`（**用下标回查而非解析返回文本**，因为 `return_documents=False` 不返回文本）。
- 任何 `Exception` 都被捕获，降级为原顺序截断，`degraded` 返回 `f"rerank_failed: {e}"`。**重排序失败绝不能让整个问答失败**。
- `enabled=False` 时直接截断返回，`degraded="rerank_disabled"`。
- `is_low_confidence(chunks)`：`chunks` 非空且 `chunks[0].score < min_relevance` 时返回 `True`。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_reranker.py -v`
Expected: 7 passed

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "feat: 重排序客户端，含失败降级与低置信判定"
```

---

### Task 9: 混合检索编排与批量入库管线

**Files:**
- Create: `retrieval/hybrid.py`
- Create: `ingest/pipeline.py`
- Test: `tests/test_hybrid.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `VectorStore`、`BM25Index`、`Reranker`、`rrf_fuse`、`get_chunker`、`load_document`、`settings`
- Produces:
  - `retrieval.hybrid.HybridRetriever(vectorstore, bm25, reranker, settings)`
  - `HybridRetriever.search(query: str, mode: str = "hybrid_rerank", top_k: int = 5) -> RetrievalResult`
  - `retrieval.hybrid.build_retriever(storage_dir: Path | None = None) -> HybridRetriever`（带 `lru_cache`，供 Streamlit 复用）
  - `ingest.pipeline.IngestReport(files_ok: int, files_skipped: int, files_failed: int, chunks_added: int, errors: list[dict], elapsed_ms: int)`
  - `ingest.pipeline.IngestPipeline(retriever, chunker, data_dir)`
  - `IngestPipeline.ingest_paths(paths: list[Path], progress_cb: Callable[[str, float], None] | None = None) -> IngestReport`

- [ ] **Step 1: 写失败测试 `tests/test_hybrid.py`**

```python
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
```

- [ ] **Step 2: 写失败测试 `tests/test_pipeline.py`**

```python
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
```

- [ ] **Step 3: 运行测试确认失败**

Run: `python -m pytest tests/test_hybrid.py tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'retrieval.hybrid'`

- [ ] **Step 4: 实现 `retrieval/hybrid.py`**

`search` 流程：
1. 计时开始。`if self.vectorstore.count() == 0 and self.bm25.count() == 0` → 立即返回空 `RetrievalResult(degraded="empty_knowledge_base")`（**Review Focus #1**）。
2. `mode == "vector"`：`vectorstore.query(query, top_k)`；若抛异常 → 尝试 BM25 兜底，`degraded="vector_failed_fallback_bm25"`；两者都空 → `degraded="empty_knowledge_base"`。
3. `mode == "hybrid"`：两路各取 `settings.retrieve_top_k`（默认 20）。**向量一路用 `try/except` 包裹**，失败则只用 BM25 且 `degraded="embedding_failed_fallback_bm25"`。然后 `rrf_fuse` → 截断 `top_k`。
4. `mode == "hybrid_rerank"`：先做步骤 3 得到候选（不截断到 `top_k`，保留 `retrieve_top_k` 个），再 `reranker.rerank(query, cands, top_n=top_k)`；把 reranker 的 `degraded` 透传。
5. 其他 mode → `raise ValueError`。
6. 填充 `RetrievalResult(chunks, mode, latency_ms, degraded)`。

`build_retriever` 用 `@lru_cache(maxsize=1)`，内部从 `settings` 读取 `storage/` 路径并 `bm25.load()`（失败则 `build([])`）。

- [ ] **Step 5: 实现 `ingest/pipeline.py`**

要点：
- 对每个 path：`load_document` → `strip_boilerplate` → `chunker.split` → 若 `vectorstore.document_exists(doc.doc_id)` 且 `bm25` 中已有该 doc → **计入 `files_skipped` 并 continue**（幂等，**Review Focus #3**）。
- 空 `chunks`（如空文件）→ 跳过，不计失败。
- 入库：`vectorstore.upsert(chunks)` + `bm25.add(chunks)`，成功后累加 `chunks_added`。
- **单文件异常必须 catch 并记入 `errors`，不中断整批**（**Review Focus #3**）。错误结构 `{"filename":..., "error": str(e)}`。
- `progress_cb(filename, fraction)`：每文件开始前回调 `(name, i/len)`，全部结束后回调 `("", 1.0)`。
- 结束后若 `chunks_added > 0` 则 `bm25.save()`。

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m pytest tests/test_hybrid.py tests/test_pipeline.py -v`
Expected: 13 passed

- [ ] **Step 7: 提交**

```bash
git add -A
git commit -m "feat: 三档混合检索编排与批量入库管线（幂等、单文件失败不中断）"
```

---

### Task 10: Prompt 模板与引用溯源

**Files:**
- Create: `generation/__init__.py`, `generation/prompts.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: `core.schemas.{Citation, ScoredChunk}`、`core.schemas.estimate_tokens`
- Produces:
  - `generation.prompts.RAG_SYSTEM_PROMPT: str`
  - `generation.prompts.build_rag_messages(question: str, chunks: list[ScoredChunk], max_context_tokens: int) -> tuple[list[dict], list[Citation]]`
  - `generation.prompts.extract_citation_indices(answer: str) -> list[int]`
  - `generation.prompts.validate_citations(answer: str, valid: set[int]) -> tuple[str, list[int]]`
  - `generation.prompts.NO_CONTEXT_PROMPT: str`

- [ ] **Step 1: 写失败测试 `tests/test_prompts.py`**

```python
from core.schemas import Chunk, ScoredChunk
from generation.prompts import (build_rag_messages, extract_citation_indices,
                                validate_citations)


def _s(cid: str, text: str, page: int, rank: int, score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(chunk_id=cid, doc_id="d", filename="vit.pdf", page_no=page,
                                   seq=0, text=text, token_count=10),
                       score=score, source="rerank", rank=rank)


def test_messages_have_system_and_user_roles():
    msgs, _ = build_rag_messages("问题", [_s("a::1::0", "内容甲", 1, 1)], max_context_tokens=1000)
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"
    assert "问题" in msgs[-1]["content"]


def test_citations_are_numbered_from_one_and_carry_page():
    _, cites = build_rag_messages("q", [_s("a::3::0", "甲", 3, 1), _s("b::7::0", "乙", 7, 2)],
                                  max_context_tokens=1000)
    assert [c.index for c in cites] == [1, 2]
    assert cites[0].page_no == 3 and cites[1].page_no == 7
    assert cites[0].filename == "vit.pdf"


def test_context_block_contains_inline_source_labels():
    msgs, _ = build_rag_messages("q", [_s("a::3::0", "独特内容ABC", 3, 1)], max_context_tokens=1000)
    user = msgs[-1]["content"]
    assert "独特内容ABC" in user and "vit.pdf" in user and "3" in user


def test_higher_ranked_chunk_appears_first():
    msgs, _ = build_rag_messages("q", [_s("a::1::0", "第一", 1, 1), _s("b::1::0", "第二", 1, 2)],
                                 max_context_tokens=1000)
    body = msgs[-1]["content"]
    assert body.index("第一") < body.index("第二")


def test_token_budget_truncates_later_chunks():
    long = "字" * 3000
    _, cites = build_rag_messages("q", [_s("a::1::0", "短", 1, 1), _s("b::1::0", long, 1, 2)],
                                  max_context_tokens=200)
    assert len(cites) == 1


def test_no_chunks_produces_no_context_prompt():
    msgs, cites = build_rag_messages("孤立问题", [], max_context_tokens=1000)
    assert cites == []
    assert "未找到" in msgs[-1]["content"] or "没有" in msgs[-1]["content"]


def test_extract_citation_indices_handles_adjacent_and_spaced():
    assert extract_citation_indices("答案[1][2]。") == [1, 2]
    assert extract_citation_indices("答案 [1] [3]") == [1, 3]
    assert extract_citation_indices("无引用") == []


def test_extract_citation_indices_deduplicates_and_sorts():
    assert extract_citation_indices("a[2] b[1] c[2]") == [1, 2]


def test_extract_ignores_malformed_markers():
    assert extract_citation_indices("a[abc] b[] c[") == []


def test_validate_citations_flags_out_of_range_indices():
    cleaned, bad = validate_citations("答案[1][9]", valid={1, 2})
    assert bad == [9]
    assert "[9]" in cleaned


def test_validate_citations_returns_empty_bad_when_all_valid():
    _, bad = validate_citations("答案[1][2]", valid={1, 2})
    assert bad == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_prompts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'generation'`

- [ ] **Step 3: 实现 `generation/prompts.py`**

要点：
- `RAG_SYSTEM_PROMPT` 用 Spec §5.2.1 的四段式，包含「只能依据参考资料」「不足时必须说明」两条硬约束。
- `build_rag_messages`：先按 `rank` 升序（即相关性降序）排列；逐条累加 `estimate_tokens`，超 `max_context_tokens` 则停止（**不截断单条**——保留完整语义块比塞入半句更有价值，且测试要求 `len(cites)==1`）。编号从 1 开始，与 `cites` 下标一致。
- 空 `chunks` 时返回 `NO_CONTEXT_PROMPT` 作为 user 消息、`cites=[]`。
- `extract_citation_indices`：`re.findall(r"\[(\d{1,3})\]", answer)`，去重排序。非法标记（`[abc]`、`[]`、裸 `[`）自然被正则排除。
- `validate_citations`：把不在 `valid` 中的编号替换为 `[n?]` 形式并返回坏编号列表，供前端标注"未验证引用"。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_prompts.py -v`
Expected: 11 passed

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "feat: RAG Prompt 模板、token 预算截断与引用溯源校验"
```

---

### Task 11: RAG 生成流水线（流式 + 引用）

**Files:**
- Create: `generation/rag.py`
- Test: `tests/test_rag.py`

**Interfaces:**
- Consumes: `HybridRetriever`、`build_rag_messages`、`get_llm()`、`settings`
- Produces:
  - `generation.rag.StreamEvent`（`dict`，`type ∈ {"citations","token","usage","degraded","error","done"}`）
  - `generation.rag.RAGEngine(retriever, llm=None, settings=None)`
  - `RAGEngine.stream_answer(question: str, mode: str | None = None, history: list[dict] | None = None) -> Iterator[StreamEvent]`
  - `RAGEngine.answer(question: str, mode: str | None = None) -> AgentResult`
  - `RAGEngine.retrieve_only(question: str, mode: str | None = None) -> RetrievalResult`

- [ ] **Step 1: 写失败测试 `tests/test_rag.py`**

```python
from pathlib import Path

import pytest

from core.providers import MockEmbedding, MockLLM, MockRerank
from core.schemas import Chunk
from generation.rag import RAGEngine
from retrieval.bm25_index import BM25Index
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.vectorstore import VectorStore

TEXT = "vision transformer 使用 ImageNet-21k 预训练，随后在 ImageNet-1k 上微调。"


@pytest.fixture
def engine(tmp_path: Path) -> RAGEngine:
    vs = VectorStore(tmp_path / "chroma", "t", MockEmbedding())
    bm = BM25Index(tmp_path / "bm25")
    c = Chunk(chunk_id="d1::3::0", doc_id="d1", filename="vit.pdf", page_no=3, seq=0,
              text=TEXT, token_count=30)
    vs.upsert([c]); bm.build([c])
    r = HybridRetriever(vs, bm, Reranker(provider=MockRerank()))
    return RAGEngine(retriever=r, llm=MockLLM())


def test_stream_emits_citations_first_then_tokens_then_done(engine):
    events = list(engine.stream_answer("用了什么数据集"))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "citations"
    assert "token" in kinds
    assert kinds[-1] == "done"


def test_citations_point_to_real_filename_and_page(engine):
    events = list(engine.stream_answer("用了什么数据集"))
    cites = events[0]["citations"]
    assert cites and cites[0]["filename"] == "vit.pdf" and cites[0]["page_no"] == 3


def test_streamed_text_reassembles_to_full_answer(engine):
    events = list(engine.stream_answer("q"))
    joined = "".join(e["text"] for e in events if e["type"] == "token")
    assert len(joined) > 0
    assert events[-1]["answer"] == joined


def test_usage_event_is_emitted_and_has_both_token_counts(engine):
    events = list(engine.stream_answer("q"))
    usage = [e for e in events if e["type"] == "usage"]
    assert usage and "prompt_tokens" in usage[0] and "completion_tokens" in usage[0]


def test_answer_on_empty_knowledge_base_marks_degraded(engine):
    engine.retriever.vectorstore.delete_document("d1")
    engine.retriever.bm25 = BM25Index(Path(engine.retriever.bm25.index_dir))
    events = list(engine.stream_answer("任意"))
    degraded = [e for e in events if e["type"] == "degraded"]
    assert degraded and degraded[0]["reason"] == "empty_knowledge_base"
    assert events[-1]["type"] == "done"


def test_non_stream_answer_returns_agent_result(engine):
    res = engine.answer("用了什么数据集")
    assert res.answer
    assert res.citations and res.steps == []
    assert res.latency_ms["total"] >= 0


def test_llm_failure_emits_error_event_not_exception(engine):
    class BoomLLM(MockLLM):
        def stream(self, messages, **kw):
            raise RuntimeError("api 401")

    engine.llm = BoomLLM()
    events = list(engine.stream_answer("q"))
    assert any(e["type"] == "error" for e in events)
    assert events[-1]["type"] == "done"


def test_history_is_prepended_to_messages(engine):
    captured = {}

    class SpyLLM(MockLLM):
        def stream(self, messages, **kw):
            captured["messages"] = messages
            return super().stream(messages, **kw)

    engine.llm = SpyLLM()
    list(engine.stream_answer("追问", history=[{"role": "user", "content": "上一轮问题"}]))
    roles = [m["role"] for m in captured["messages"]]
    assert roles.count("user") >= 2


def test_retrieve_only_does_not_call_llm(engine):
    class ExplodingLLM(MockLLM):
        def stream(self, messages, **kw):
            raise AssertionError("不应调用 LLM")

    engine.llm = ExplodingLLM()
    r = engine.retrieve_only("ImageNet")
    assert r.chunks
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_rag.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'generation.rag'`

- [ ] **Step 3: 实现 `generation/rag.py`**

事件序列契约（前端依赖，**顺序不可变**）：
1. `{"type":"degraded","reason":str}` —— 仅在有降级时，**必须在 token 之前**
2. `{"type":"citations","citations":[Citation as dict, ...]}`
3. `{"type":"token","text":str}` —— 0..N 个
4. `{"type":"usage","prompt_tokens":int,"completion_tokens":int}`
5. `{"type":"error","message":str}` —— 仅失败时
6. `{"type":"done","answer":str,"degraded":str|None,"latency_ms":dict}` —— **永远是最后一个**

要点：
- `retrieve_only` 只调 `retriever.search`，**绝不触碰 LLM**（测试用 ExplodingLLM 钉住）。
- LLM 调用整体包在 `try/except Exception` 中，异常转成 `error` 事件后仍发 `done`（**任何失败都不能让生成器抛出**）。
- `answer()` 是非流式便捷封装：消费 `stream_answer`，拼接 token，返回 `AgentResult(answer, citations, steps=[], tokens, latency_ms, route="rag", degraded)`。
- `latency_ms` 分三段：`retrieve`（检索）、`generate`（首个 token 到结束）、`total`。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_rag.py -v`
Expected: 9 passed

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "feat: RAG 流式生成流水线，事件序列契约与引用溯源"
```

---

### Task 12: 语义缓存与降级策略

**Files:**
- Create: `generation/cache.py`, `generation/fallback.py`
- Test: `tests/test_cache.py`, `tests/test_fallback.py`

**Interfaces:**
- Consumes: `EmbeddingProvider`、`AgentResult`、`settings`
- Produces:
  - `generation.cache.SemanticCache(embedder, cache_dir: Path, threshold: float)`
  - `SemanticCache.get(question: str) -> AgentResult | None`
  - `SemanticCache.put(question: str, result: AgentResult) -> None`
  - `SemanticCache.save() / load() -> bool`、`SemanticCache.size() -> int`、`SemanticCache.clear() -> None`
  - `generation.fallback.decide(retrieval: RetrievalResult, answer: str) -> tuple[bool, str | None]`（是否降级、提示语）
  - `generation.fallback.NO_HIT_NOTICE: str`、`LOW_CONFIDENCE_NOTICE: str`、`VECTOR_DOWN_NOTICE: str`、`LLM_DOWN_NOTICE: str`

- [ ] **Step 1: 写失败测试 `tests/test_cache.py`**

```python
from pathlib import Path

import pytest

from core.providers import MockEmbedding
from core.schemas import AgentResult, Citation
from generation.cache import SemanticCache


@pytest.fixture
def cache(tmp_path: Path) -> SemanticCache:
    return SemanticCache(MockEmbedding(), tmp_path, threshold=0.95)


def _res(text: str = "答案") -> AgentResult:
    return AgentResult(answer=text, citations=[Citation(1, "f.pdf", 1, "c", "snip")],
                       steps=[], tokens={}, latency_ms={}, route="rag", degraded=None)


def test_exact_question_hits(cache):
    cache.put("BERT 用了什么优化器", _res())
    assert cache.get("BERT 用了什么优化器") is not None


def test_different_question_misses_at_high_threshold(cache):
    cache.put("BERT 用了什么优化器", _res())
    assert cache.get("今天天气怎么样") is None


def test_near_duplicate_misses_when_below_threshold(tmp_path: Path):
    c = SemanticCache(MockEmbedding(), tmp_path, threshold=1.01)   # 苛刻阈值
    c.put("问题甲", _res())
    assert c.get("问题甲") is None


def test_empty_cache_get_returns_none(cache):
    assert cache.get("任意") is None


def test_put_same_question_twice_increments_hit_count(cache):
    cache.put("q", _res())
    cache.put("q", _res())
    assert cache.size() == 1


def test_save_and_load_roundtrip(tmp_path: Path):
    c = SemanticCache(MockEmbedding(), tmp_path, threshold=0.95)
    c.put("持久化问题", _res("持久化答案"))
    c.save()
    fresh = SemanticCache(MockEmbedding(), tmp_path, threshold=0.95)
    assert fresh.load() is True
    got = fresh.get("持久化问题")
    assert got is not None and got.answer == "持久化答案"


def test_load_returns_false_when_no_file(tmp_path: Path):
    assert SemanticCache(MockEmbedding(), tmp_path / "none", 0.95).load() is False


def test_clear_empties_cache(cache):
    cache.put("q", _res())
    cache.clear()
    assert cache.size() == 0


def test_empty_question_returns_none(cache):
    assert cache.get("") is None
```

- [ ] **Step 2: 写失败测试 `tests/test_fallback.py`**

```python
from core.schemas import Chunk, ScoredChunk
from generation.fallback import (LOW_CONFIDENCE_NOTICE, NO_HIT_NOTICE, decide)


def _s(score: float) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(chunk_id="a::1::0", doc_id="a", filename="f.pdf", page_no=1,
                                   seq=0, text="t", token_count=1),
                       score=score, source="rerank", rank=1)


class R:
    def __init__(self, chunks, degraded=None):
        self.chunks, self.degraded = chunks, degraded


def test_no_chunks_triggers_no_hit_notice():
    should, notice = decide(R([]), "答案")
    assert should is True and notice == NO_HIT_NOTICE


def test_low_top1_score_triggers_low_confidence():
    should, notice = decide(R([_s(0.05)]), "答案")
    assert should is True and notice == LOW_CONFIDENCE_NOTICE


def test_high_score_no_degradation():
    should, notice = decide(R([_s(0.85)]), "答案")
    assert should is False and notice is None


def test_upstream_degradation_is_surfaced():
    should, notice = decide(R([_s(0.85)], degraded="rerank_failed: 503"), "答案")
    assert should is True and "503" in notice


def test_empty_answer_with_chunks_is_flagged():
    should, notice = decide(R([_s(0.85)]), "")
    assert should is True
```

- [ ] **Step 3: 运行测试确认失败**

Run: `python -m pytest tests/test_cache.py tests/test_fallback.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'generation.cache'`

- [ ] **Step 4: 实现 `generation/cache.py`**

要点：内存 `list[dict]`（`{question, vec, result_json, hit_count}`）+ `pickle` 持久化到 `cache_dir/semantic_cache.pkl`。`get` 用**余弦相似度**（向量已归一化则等价于点积）线性扫描；万级以下条目线性扫描耗时可忽略。`put` 对完全相同 question **原地更新并 `hit_count += 1`**，不新增条目。`get("")` 直接返回 `None`。

- [ ] **Step 5: 实现 `generation/fallback.py`**

```python
NO_HIT_NOTICE = "⚠️ 知识库中未找到相关文档，以下回答来自模型自身知识，请谨慎参考。"
LOW_CONFIDENCE_NOTICE = "⚠️ 检索置信度较低，请确认以下片段是否与问题相关。"
VECTOR_DOWN_NOTICE = "ℹ️ 向量服务暂不可用，已降级为关键词检索。"
LLM_DOWN_NOTICE = "❌ 模型服务暂时不可用，请稍后重试。"
```

`decide` 优先级：无 chunks → `NO_HIT_NOTICE`；有 `retrieval.degraded` → 透传该原因；`chunks[0].score < 0.2` → `LOW_CONFIDENCE_NOTICE`；答案为空 → `"⚠️ 模型未返回有效内容，请重试。"`；否则 `(False, None)`。

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m pytest tests/test_cache.py tests/test_fallback.py -v`
Expected: 14 passed

- [ ] **Step 7: 提交**

```bash
git add -A
git commit -m "feat: 语义缓存与分级降级策略"
```

---

### Task 13: 工具抽象、注册表与三个纯函数工具

**Files:**
- Create: `agent/__init__.py`, `agent/tools/__init__.py`, `agent/tools/base.py`
- Create: `agent/tools/calculator.py`, `agent/tools/current_time.py`, `agent/tools/keywords.py`
- Test: `tests/test_tools_pure.py`

**Interfaces:**
- Consumes: 无外部依赖（三个工具均为纯函数）
- Produces:
  - `agent.tools.base.Tool(name: str, description: str, parameters: dict, func: Callable[..., str], timeout_s: float = 30.0)`
  - `agent.tools.base.Tool.registry: dict[str, Tool]`
  - `agent.tools.base.register_tool(name, description, parameters, timeout_s=30.0)` —— 装饰器
  - `agent.tools.base.build_registry() -> dict[str, Tool]`
  - `agent.tools.base.tools_description() -> str`（用于拼 System Prompt）
  - `agent.tools.calculator.safe_eval(expression: str) -> float`
  - `agent.tools.keywords.extract_keywords(text: str, top_n: int = 8) -> list[str]`
  - 三个已注册工具名：`calculator`、`current_time`、`extract_keywords`

- [ ] **Step 1: 写失败测试 `tests/test_tools_pure.py`**

```python
import pytest

from agent.tools import build_registry
from agent.tools.calculator import safe_eval
from agent.tools.keywords import extract_keywords


def test_safe_eval_basic_arithmetic():
    assert safe_eval("3.14 * 2.56") == pytest.approx(8.0384)


def test_safe_eval_respects_precedence_and_parens():
    assert safe_eval("2 + 3 * 4") == 14
    assert safe_eval("(2 + 3) * 4") == 20


def test_safe_eval_supports_power_and_unary():
    assert safe_eval("2 ** 10") == 1024
    assert safe_eval("-5 + 3") == -2


def test_safe_eval_rejects_function_calls():
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('echo hi')")


def test_safe_eval_rejects_attribute_access():
    with pytest.raises(ValueError):
        safe_eval("(1).__class__")


def test_safe_eval_rejects_names():
    with pytest.raises(ValueError):
        safe_eval("open('x')")


def test_safe_eval_rejects_empty_and_syntax_error():
    with pytest.raises(ValueError):
        safe_eval("")
    with pytest.raises(ValueError):
        safe_eval("2 +")


def test_safe_eval_rejects_division_by_zero_with_clear_message():
    with pytest.raises(ValueError) as e:
        safe_eval("1 / 0")
    assert "零" in str(e.value) or "zero" in str(e.value).lower()


def test_extract_keywords_returns_top_n_distinct_terms():
    kws = extract_keywords("Transformer 注意力机制 Transformer 注意力机制 自监督", top_n=3)
    assert len(kws) <= 3 and len(set(kws)) == len(kws)


def test_extract_keywords_ranks_repeated_term_first():
    assert extract_keywords("扩散模型 扩散模型 扩散模型 图像生成", top_n=1) == ["扩散模型"] or \
           "扩散" in extract_keywords("扩散模型 扩散模型 扩散模型 图像生成", top_n=1)[0]


def test_extract_keywords_empty_text_returns_empty():
    assert extract_keywords("", top_n=5) == []


def test_extract_keywords_handles_pure_numbers():
    assert isinstance(extract_keywords("3.14 2.71 1.41", top_n=2), list)


def test_registry_has_the_three_pure_tools():
    reg = build_registry()
    for name in ("calculator", "current_time", "extract_keywords"):
        assert name in reg


def test_every_registered_tool_has_valid_json_schema():
    for name, tool in build_registry().items():
        assert tool.parameters["type"] == "object"
        assert isinstance(tool.parameters["properties"], dict)
        assert tool.description


def test_registry_tools_are_callable_and_return_str():
    reg = build_registry()
    assert isinstance(reg["calculator"].func(expression="1+1"), str)
    assert isinstance(reg["current_time"].func(), str)
    assert isinstance(reg["extract_keywords"].func(text="甲 乙 甲"), str)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_tools_pure.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent'`

- [ ] **Step 3: 实现 `agent/tools/base.py`**

- `Tool` 用 `@dataclass`；`registry: ClassVar[dict[str, Tool]] = {}` 定义在类上但在 `build_registry` 中通过导入所有工具模块触发注册。
- `register_tool` 装饰器把 `func` 包成 **kwargs 调用**并将返回值强制 `str()`（工具必须返回字符串 Observation）。
- `build_registry()` 内部 `import agent.tools.calculator, agent.tools.current_time, agent.tools.keywords, ...`（全部 8 个），因此**每个工具模块都必须被显式 import 一次**，否则装饰器不触发。
- `tools_description()` 输出格式：
  ```
  - calculator(expression: string): 计算数学表达式，支持 + - * / ** 与括号
  - current_time(): 返回当前日期时间
  ...
  ```

- [ ] **Step 4: 实现 `agent/tools/calculator.py`**

```python
_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
            ast.USub, ast.UAdd, ast.FloorDiv)


def safe_eval(expression: str) -> float:
    """AST 白名单求值。绝不使用裸 eval —— 那会让 `__import__('os').system(...)` 直达系统。"""
    expr = (expression or "").strip()
    if not expr:
        raise ValueError("表达式为空")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"表达式语法错误: {e}") from e
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"表达式含不允许的语法元素: {type(node).__name__}")
    try:
        return float(eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {}))
    except ZeroDivisionError as e:
        raise ValueError("除数不能为零") from e
```

> `compile` + 空 `__builtins__` 是第二道防线；白名单遍历是第一道。两层都必须有。

- [ ] **Step 5: 实现 `agent/tools/keywords.py` 与 `current_time.py`**

- `extract_keywords`：复用 `retrieval.bm25_index.tokenize` 的英文/CJK 切分思路，但**输出可读词**——英文取原词，中文用 2-4 字滑窗并按频次排序；用 `Counter` 取 `most_common(top_n)`；空文本返回 `[]`。
- `current_time`：返回 `datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")`，无参数（`parameters = {"type":"object","properties":{}}`）。

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m pytest tests/test_tools_pure.py -v`
Expected: 15 passed

- [ ] **Step 7: 提交**

```bash
git add -A
git commit -m "feat: 工具抽象与注册表，含 AST 白名单安全计算器"
```

---

### Task 14: 知识库类工具（kb_search / paper_meta / paper_compare / summarize_paper）

**Files:**
- Create: `agent/tools/kb_search.py`, `agent/tools/paper_meta.py`, `agent/tools/paper_compare.py`, `agent/tools/summarize.py`
- Modify: `agent/tools/base.py`（在 `build_registry` 中 import 这四个模块）
- Test: `tests/test_tools_kb.py`

**Interfaces:**
- Consumes: `RAGEngine`、`VectorStore.list_documents`、`get_llm()`
- Produces:
  - `agent.tools.kb_search.set_engine(engine: RAGEngine) -> None` 与 `kb_search(query: str, top_k: int = 5, mode: str = "hybrid_rerank") -> str`
  - `agent.tools.paper_meta.paper_meta(doc_id: str = "", filename: str = "") -> str`
  - `agent.tools.paper_compare.paper_compare(doc_id_a: str, doc_id_b: str, aspect: str = "method") -> str`
  - `agent.tools.summarize.summarize_paper(doc_id: str) -> str`
  - 上述 4 个工具名已注册

> 依赖注入原因：工具需要访问检索器与 LLM，但不能在 import 期构造（否则 Streamlit 每次重跑都重建）。统一用模块级 `set_*` 注入，由 `agent/__init__.py` 的 `build_agent()` 完成。

- [ ] **Step 1: 写失败测试 `tests/test_tools_kb.py`**

```python
from pathlib import Path

import pytest

from agent.tools import build_registry
from agent.tools import kb_search as kb_mod
from agent.tools.kb_search import kb_search
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
    kb_mod.set_engine(engine)


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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_tools_kb.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.tools.kb_search'`

- [ ] **Step 3: 实现四个工具**

- `kb_search`：模块级 `_ENGINE: RAGEngine | None`，`set_engine()` 设置它。`_ENGINE is None` → 返回 `"错误：检索引擎未初始化"`（**不抛异常**——Observation 要能交还给 Agent）。`engine.retrieve_only(query, mode)` → 格式化为：
  ```
  [1] (vit.pdf 第1页) ViT 将图像切成 16x16 patch。
  [2] (swin.pdf 第2页) Swin 使用层次化窗口注意力。
  ```
  空结果 → `"知识库中未找到与该问题相关的内容。"`。注意 `top_k` 与 `mode` 要为每个工具提供默认值，因为 LLM 常常省略参数。
- `paper_meta`：从 `vectorstore.list_documents()` 拿文档清单；给 `doc_id`/`filename` 时匹配并输出该文档的 `doc_id/filename/chunk_count/page 范围`；无参时列出全部文档名。
- `paper_compare`：对两个 doc_id 各做一次 `engine.retrieve_only(f"{aspect} 方法 数据集 结果", ...)`，但**按 doc_id 过滤**——实现方式是对检索结果做 `chunk.doc_id == target` 筛选；若某篇无命中则明确说明。
- `summarize_paper`：取该文档前 N 个 chunk（按 `seq`），拼成 prompt 调 `get_llm().complete()`，要求输出「背景-方法-结果-结论」四段。**必须标注"以下摘要由模型基于前 N 个片段生成，可能不完整"**，避免用户误以为是全文摘要。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_tools_kb.py -v`
Expected: 7 passed

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "feat: 知识库类 Agent 工具（检索/元信息/对比/摘要）"
```

---

### Task 15: 联网搜索工具

**Files:**
- Create: `agent/tools/web_search.py`
- Modify: `agent/tools/base.py`（import 该模块）
- Test: `tests/test_tools_web.py`

**Interfaces:**
- Consumes: `httpx`、`settings`
- Produces:
  - `agent.tools.web_search.web_search(query: str, max_results: int = 3) -> str`
  - 已注册工具名 `web_search`

- [ ] **Step 1: 写失败测试 `tests/test_tools_web.py`**

```python
import pytest

from agent.tools import web_search as ws


def test_network_failure_returns_readable_string(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(ws.httpx, "get", boom)
    out = ws.web_search("transformer")
    assert isinstance(out, str)
    assert "失败" in out or "不可用" in out


def test_timeout_returns_readable_string(monkeypatch):
    def boom(*a, **k):
        raise ws.httpx.TimeoutException("timed out")

    monkeypatch.setattr(ws.httpx, "get", boom)
    assert isinstance(ws.web_search("q"), str)


def test_parses_results_from_html(monkeypatch):
    html = ('<a class="result__a" href="https://x.com">标题一</a>'
            '<a class="result__a" href="https://y.com">标题二</a>')
    monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: type("R", (), {"text": html, "status_code": 200})())
    out = ws.web_search("q", max_results=2)
    assert "标题一" in out and "https://x.com" in out


def test_no_results_returns_readable_message(monkeypatch):
    monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: type("R", (), {"text": "<html></html>", "status_code": 200})())
    out = ws.web_search("q")
    assert isinstance(out, str) and len(out) > 0


def test_empty_query_is_rejected_gracefully():
    assert "为空" in ws.web_search("")


def test_respects_max_results(monkeypatch):
    html = "".join(f'<a class="result__a" href="https://s{i}.com">T{i}</a>' for i in range(10))
    monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: type("R", (), {"text": html, "status_code": 200})())
    out = ws.web_search("q", max_results=2)
    assert out.count("https://") <= 2
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_tools_web.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.tools.web_search'`

- [ ] **Step 3: 实现 `agent/tools/web_search.py`**

要点：用 DuckDuckGo 的 HTML 端点 `https://html.duckduckgo.com/html/?q=...`（无需 API key），`httpx.get(url, timeout=10.0, headers={"User-Agent": "Mozilla/5.0 ..."})`，用 `re.findall` 抽 `<a class="result__a" href="(...)">(...)</a>`（**不引入 BeautifulSoup，避免新增依赖**）。任何异常（网络、超时、解析）都返回可读字符串 `"联网搜索失败：{e}。请基于知识库内容回答，或告知用户当前无法联网。"`。空 query 返回 `"错误：搜索关键词为空。"`。结果格式 `"[1] 标题一\n    https://x.com"`。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_tools_web.py -v`
Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "feat: 联网搜索工具（DuckDuckGo，失败可读降级）"
```

---

### Task 16: ReAct 核心循环（手写）

**Files:**
- Create: `agent/react.py`
- Test: `tests/test_react.py`

**Interfaces:**
- Consumes: `build_registry`、`tools_description`、`LLMProvider`、`core.schemas.{AgentResult, AgentStep, Citation}`
- Produces:
  - `agent.react.parse_action(raw: str) -> tuple[str, str, dict]`（`(thought, action, action_input)`，解析失败抛 `ActionParseError`）
  - `agent.react.ActionParseError(Exception)`
  - `agent.react.ReActAgent(llm, tools: dict[str, Tool], max_iterations: int = 6, tool_timeout_s: float = 30.0, executor=None)`
  - `ReActAgent.run(question: str, history: list[dict] | None = None) -> AgentResult`
  - `ReActAgent.system_prompt: str`

- [ ] **Step 1: 写失败测试 `tests/test_react.py`**

```python
import pytest

from agent.react import ActionParseError, ReActAgent, parse_action
from agent.tools.base import Tool
from core.providers import MockLLM


def test_parse_action_full_triple():
    raw = ('Thought: 需要检索\n'
           'Action: kb_search\n'
           'Action Input: {"query": "ViT", "top_k": 3}')
    thought, action, args = parse_action(raw)
    assert thought == "需要检索" and action == "kb_search" and args["query"] == "ViT"


def test_parse_action_tolerates_missing_thought():
    _, action, _ = parse_action('Action: current_time\nAction Input: {}')
    assert action == "current_time"


def test_parse_action_tolerates_chinese_colon():
    _, action, _ = parse_action('思考: x\nAction：current_time\nAction Input: {}')
    assert action == "current_time"


def test_parse_action_repairs_unquoted_keys():
    _, _, args = parse_action('Action: calculator\nAction Input: {expression: 1+1}')
    assert args["expression"] == "1+1"


def test_parse_action_missing_action_raises():
    with pytest.raises(ActionParseError):
        parse_action("Thought: 我什么也没做")


def test_parse_action_empty_input_raises():
    with pytest.raises(ActionParseError):
        parse_action("")


def test_parse_action_bad_json_raises_with_readable_message():
    with pytest.raises(ActionParseError) as e:
        parse_action('Action: kb_search\nAction Input: {"query": ')
    assert "JSON" in str(e.value) or "参数" in str(e.value)


class ScriptedLLM(MockLLM):
    """按脚本依次返回，用于驱动确定性 ReAct 测试。"""

    def __init__(self, script):
        super().__init__()
        self.script, self.i = script, 0

    def complete(self, messages, **kw):
        out = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return out


def _tools():
    calls = []

    def make(name, fn):
        return Tool(name=name, description=f"{name} 工具",
                    parameters={"type": "object", "properties": {}}, func=fn)

    return {
        "echo": make("echo", lambda **kw: calls.append(kw) or "回显结果"),
        "_calls": calls,
    }


def test_agent_finishes_immediately_on_finish_action():
    llm = ScriptedLLM(['Thought: 已知\nAction: Finish\nAction Input: {"answer": "答案文本"}'])
    res = ReActAgent(llm=llm, tools={}).run("问题")
    assert res.answer == "答案文本"
    assert len(res.steps) == 1


def test_agent_executes_tool_then_finishes():
    tools = _tools()
    llm = ScriptedLLM([
        'Thought: 先回显\nAction: echo\nAction Input: {"x": 1}',
        'Thought: 完成\nAction: Finish\nAction Input: {"answer": "最终答案"}',
    ])
    res = ReActAgent(llm=llm, tools=tools).run("问题")
    assert res.answer == "最终答案"
    assert "回显结果" in res.steps[0].observation
    assert tools["_calls"] == [{"x": 1}]


def test_agent_stops_after_max_iterations_when_never_finishing():
    llm = ScriptedLLM(['Thought: 再查一次\nAction: echo\nAction Input: {"n": 1}'])
    res = ReActAgent(llm=llm, tools=_tools(), max_iterations=3).run("问题")
    assert len(res.steps) <= 4          # 3 步 + 可能的收敛步
    assert res.degraded == "max_iterations_reached"


def test_repeated_identical_tool_call_is_detected_as_loop():
    llm = ScriptedLLM(['Thought: 重复\nAction: echo\nAction Input: {"n": 1}'])
    res = ReActAgent(llm=llm, tools=_tools(), max_iterations=6).run("问题")
    assert res.degraded == "loop_detected"
    assert len(res.steps) <= 4


def test_malformed_action_is_retried_then_forced_to_conclude():
    llm = ScriptedLLM(['这不是合法输出', '这也不是', '还是不合法'])
    res = ReActAgent(llm=llm, tools=_tools(), max_iterations=6).run("问题")
    assert res.degraded in ("parse_failed", "max_iterations_reached")
    assert isinstance(res.answer, str) and res.answer


def test_unknown_tool_becomes_observation_not_exception():
    tools = _tools()
    llm = ScriptedLLM([
        'Thought: 用不存在的工具\nAction: nope\nAction Input: {}',
        'Thought: 改用 echo\nAction: echo\nAction Input: {}',
        'Thought: 完成\nAction: Finish\nAction Input: {"answer": "ok"}',
    ])
    res = ReActAgent(llm=llm, tools=tools).run("问题")
    assert "不存在" in res.steps[0].observation or "未知" in res.steps[0].observation
    assert res.answer == "ok"


def test_tool_exception_becomes_observation_not_exception():
    def boom(**kw):
        raise RuntimeError("工具内部炸了")

    tools = {"boom": Tool(name="boom", description="d",
                          parameters={"type": "object", "properties": {}}, func=boom)}
    llm = ScriptedLLM([
        'Thought: 调用\nAction: boom\nAction Input: {}',
        'Thought: 完成\nAction: Finish\nAction Input: {"answer": "兜底"}',
    ])
    res = ReActAgent(llm=llm, tools=tools).run("问题")
    assert "工具内部炸了" in res.steps[0].observation
    assert res.answer == "兜底"


def test_empty_question_is_rejected_gracefully():
    res = ReActAgent(llm=ScriptedLLM(['Thought: x\nAction: Finish\nAction Input: {"answer": "a"}']),
                     tools={}).run("")
    assert isinstance(res.answer, str)


def test_steps_carry_latency_and_index():
    llm = ScriptedLLM(['Thought: t\nAction: Finish\nAction Input: {"answer": "a"}'])
    res = ReActAgent(llm=llm, tools={}).run("q")
    assert res.steps[0].index == 0
    assert res.steps[0].latency_ms >= 0


def test_run_never_raises_when_llm_itself_fails():
    class BoomLLM(MockLLM):
        def complete(self, messages, **kw):
            raise RuntimeError("llm 挂了")

    res = ReActAgent(llm=BoomLLM(), tools={}).run("问题")
    assert isinstance(res.answer, str)
    assert res.degraded is not None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_react.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.react'`

- [ ] **Step 3: 实现 `agent/react.py`**

**`parse_action` 容错策略**（Review Focus #4）：
1. 用 `re.search(r"(?:Thought|思考)\s*[:：]\s*(.*?)(?=\n(?:Action|行动)\s*[:：]|\Z)", raw, re.S)` 抽 thought（可缺省为空串）。
2. 用 `re.search(r"(?:Action|行动)\s*[:：]\s*(\S+)", raw)` 抽 action，**找不到即抛 `ActionParseError("未能解析出 Action")`**。
3. 用 `re.search(r"(?:Action\s*Input|行动输入)\s*[:：]\s*(.*)", raw, re.S)` 抽参数体，`strip()`。
4. 参数体为 `{}` 或空 → `{}`。
5. `json.loads` 失败时尝试修复：给裸键加引号（正则 `([{,]\s*)([A-Za-z_]\w*)(\s*:)` → `\1"\2"\3`），再试一次。
6. 仍失败 → 抛 `ActionParseError(f"Action Input 不是合法 JSON: {原始片段}")`。

**`run` 循环要点**：
- 全程 `try/except Exception` 包裹 LLM 调用与解析；解析失败时 `failed_parses += 1`，把错误信息作为 Observation 追加（`"解析失败：{e}。请严格按 Thought/Action/Action Input 格式重新输出。"`）并 `continue`。**连续 3 次解析失败** → `degraded="parse_failed"` 并走 `_force_conclude`。
- 循环指纹 `seen: set[tuple[str, str]]`，key 为 `(action, json.dumps(args, sort_keys=True))`；**第二次出现即判定 loop**，`degraded="loop_detected"`，走 `_force_conclude`。
- 未知工具 → Observation 为 `f"错误：工具 {action} 不存在。可用工具：{', '.join(self.tools)}"`。
- 工具异常 → Observation 为 `f"错误：工具 {action} 执行失败：{e}"`。
- 工具调用经 `self.executor.run_one(tool, args)`，内部 `concurrent.futures.ThreadPoolExecutor(1)` + `future.result(timeout=tool_timeout_s)` 实现超时，超时抛 `TimeoutError` 并被转成 Observation。
- `max_iterations` 耗尽 → `degraded="max_iterations_reached"` → `_force_conclude`。
- `_force_conclude(question, scratchpad, reason)`：用一个明确要求「基于以上观察直接给出最终答案，不要再调用工具」的 prompt 调 `llm.complete`；**若这一步也失败，返回已收集的 Observation 拼接作为兜底答案**，保证 `answer` 永远是非空字符串。
- `citation` 提取：从所有 Observation 中用 `re.findall(r"\[(\d+)\] \((\S+) 第(\d+)页\)")` 抽引用，去重后编号为 `Citation`。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_react.py -v`
Expected: 15 passed

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "feat: 手写 ReAct 循环，含解析容错、循环检测与三个终止条件"
```

---

### Task 17: 工具路由、并行执行器与会话记忆

**Files:**
- Create: `agent/router.py`, `agent/executor.py`, `agent/memory.py`
- Test: `tests/test_router.py`, `tests/test_executor.py`, `tests/test_memory.py`

**Interfaces:**
- Consumes: `Tool` 注册表、`settings`
- Produces:
  - `agent.router.RouteDecision(kind: str, tool: str | None, args: dict, reason: str)`
  - `agent.router.classify(question: str) -> RouteDecision`（`kind ∈ {"direct_tool","agent"}`）
  - `agent.executor.ToolExecutor(tools: dict[str, Tool], timeout_s: float, max_workers: int)`
  - `ToolExecutor.run_one(tool: Tool, args: dict) -> str`
  - `ToolExecutor.run_parallel(calls: list[tuple[str, dict]]) -> list[str]`
  - `agent.memory.ConversationMemory(max_tokens: int, summarize_after_turns: int, llm=None, store_dir: Path | None = None)`
  - `ConversationMemory.new_session() -> str`、`get(sid) -> list[Turn]`、`append(sid, turn)`、`build_context(sid) -> tuple[list[dict], str | None]`、`clear(sid)`、`list_sessions() -> list[dict]`、`save(sid)`、`load(sid) -> bool`

- [ ] **Step 1: 写失败测试 `tests/test_router.py`**

```python
from agent.router import classify


def test_pure_arithmetic_routes_to_calculator():
    d = classify("3.14 * 2.56 等于多少")
    assert d.kind == "direct_tool" and d.tool == "calculator"


def test_bare_number_is_not_treated_as_arithmetic():
    d = classify("3.14")
    assert d.kind == "agent"          # 只有数字，不是算式


def test_time_question_routes_to_current_time():
    for q in ("现在几点了", "今天日期是多少", "当前时间"):
        d = classify(q)
        assert d.kind == "direct_tool" and d.tool == "current_time"


def test_comparison_question_routes_to_agent_or_compare():
    d = classify("对比论文 A 和论文 B 的方法区别")
    assert d.kind == "agent" or d.tool == "paper_compare"


def test_normal_research_question_goes_to_agent():
    d = classify("ViT 用了什么数据集预训练")
    assert d.kind == "agent"


def test_empty_question_goes_to_agent_without_crash():
    assert classify("").kind == "agent"


def test_fullwidth_punctuation_question_goes_to_agent():
    assert classify("你好。世界！") .kind == "agent"


def test_decision_carries_human_readable_reason():
    assert classify("2+2").reason
```

- [ ] **Step 2: 写失败测试 `tests/test_executor.py`**

```python
import time

from agent.executor import ToolExecutor
from agent.tools.base import Tool


def _tool(name, fn, timeout=30.0):
    return Tool(name=name, description="d", parameters={"type": "object", "properties": {}},
                func=fn, timeout_s=timeout)


def test_run_one_returns_string():
    ex = ToolExecutor({"echo": _tool("echo", lambda **kw: "ok")}, timeout_s=5, max_workers=2)
    assert ex.run_one(ex.tools["echo"], {}) == "ok"


def test_run_one_wraps_exception_into_readable_string():
    def boom(**kw):
        raise ValueError("坏了")

    ex = ToolExecutor({"b": _tool("b", boom)}, timeout_s=5, max_workers=2)
    assert "坏了" in ex.run_one(ex.tools["b"], {})


def test_run_one_converts_non_string_return_to_string():
    ex = ToolExecutor({"n": _tool("n", lambda **kw: 42)}, timeout_s=5, max_workers=2)
    assert ex.run_one(ex.tools["n"], {}) == "42"


def test_timeout_is_enforced_and_reported():
    def slow(**kw):
        time.sleep(2)
        return "too late"

    ex = ToolExecutor({"s": _tool("s", slow, timeout=0.3)}, timeout_s=0.3, max_workers=2)
    out = ex.run_one(ex.tools["s"], {})
    assert "超时" in out


def test_bad_kwargs_are_reported_not_raised():
    ex = ToolExecutor({"t": _tool("t", lambda required_arg: "x")}, timeout_s=5, max_workers=2)
    out = ex.run_one(ex.tools["t"], {"wrong": 1})
    assert isinstance(out, str) and out


def test_run_parallel_preserves_input_order():
    ex = ToolExecutor({}, timeout_s=5, max_workers=4)
    ex.tools = {f"t{i}": _tool(f"t{i}", (lambda i: (lambda **kw: f"r{i}"))(i)) for i in range(4)}
    outs = ex.run_parallel([(f"t{i}", {}) for i in range(4)])
    assert outs == ["r0", "r1", "r2", "r3"]


def test_run_parallel_is_actually_concurrent():
    def slow(**kw):
        time.sleep(0.4)
        return "done"

    ex = ToolExecutor({}, timeout_s=5, max_workers=4)
    ex.tools = {f"t{i}": _tool(f"t{i}", slow) for i in range(4)}
    t0 = time.time()
    ex.run_parallel([(f"t{i}", {}) for i in range(4)])
    assert time.time() - t0 < 1.2      # 串行需 1.6s


def test_run_parallel_isolates_failures():
    def boom(**kw):
        raise RuntimeError("bad")

    ex = ToolExecutor({}, timeout_s=5, max_workers=4)
    ex.tools = {"ok": _tool("ok", lambda **kw: "fine"), "bad": _tool("bad", boom)}
    outs = ex.run_parallel([("ok", {}), ("bad", {})])
    assert outs[0] == "fine" and "bad" in outs[1]


def test_run_parallel_empty_returns_empty():
    assert ToolExecutor({}, 5, 2).run_parallel([]) == []


def test_unknown_tool_in_parallel_is_reported():
    outs = ToolExecutor({}, 5, 2).run_parallel([("ghost", {})])
    assert "不存在" in outs[0]
```

- [ ] **Step 3: 写失败测试 `tests/test_memory.py`**

```python
from pathlib import Path

from agent.memory import ConversationMemory, Turn
from core.providers import MockLLM


def test_new_session_ids_are_unique():
    m = ConversationMemory()
    assert m.new_session() != m.new_session()


def test_append_and_get_roundtrip():
    m = ConversationMemory()
    sid = m.new_session()
    m.append(sid, Turn(role="user", content="你好", ts="t"))
    assert m.get(sid)[0].content == "你好"


def test_sessions_are_isolated():
    m = ConversationMemory()
    a, b = m.new_session(), m.new_session()
    m.append(a, Turn(role="user", content="A 的内容", ts="t"))
    assert m.get(b) == []


def test_get_unknown_session_returns_empty_list():
    assert ConversationMemory().get("nope") == []


def test_sliding_window_drops_oldest_but_keeps_pairs():
    m = ConversationMemory(max_tokens=60, summarize_after_turns=999)
    sid = m.new_session()
    for i in range(10):
        m.append(sid, Turn(role="user", content=f"问题{i}" * 10, ts="t"))
        m.append(sid, Turn(role="assistant", content=f"答案{i}" * 10, ts="t"))
    ctx, _ = m.build_context(sid)
    assert len(ctx) < 20
    assert ctx[-1]["role"] == "assistant"      # 末尾整轮完整


def test_summary_compression_triggered_after_threshold():
    m = ConversationMemory(max_tokens=100000, summarize_after_turns=3, llm=MockLLM())
    sid = m.new_session()
    for i in range(5):
        m.append(sid, Turn(role="user", content=f"q{i}", ts="t"))
        m.append(sid, Turn(role="assistant", content=f"a{i}", ts="t"))
    _, summary = m.build_context(sid)
    assert summary is not None and len(summary) > 0


def test_no_summary_when_below_threshold():
    m = ConversationMemory(max_tokens=100000, summarize_after_turns=10, llm=MockLLM())
    sid = m.new_session()
    m.append(sid, Turn(role="user", content="q", ts="t"))
    _, summary = m.build_context(sid)
    assert summary is None


def test_clear_removes_session():
    m = ConversationMemory()
    sid = m.new_session()
    m.append(sid, Turn(role="user", content="x", ts="t"))
    m.clear(sid)
    assert m.get(sid) == []


def test_build_context_on_empty_session_returns_empty():
    m = ConversationMemory()
    ctx, summary = m.build_context(m.new_session())
    assert ctx == [] and summary is None


def test_save_and_load_roundtrip(tmp_path: Path):
    m = ConversationMemory(store_dir=tmp_path)
    sid = m.new_session()
    m.append(sid, Turn(role="user", content="持久化", ts="t"))
    m.save(sid)
    fresh = ConversationMemory(store_dir=tmp_path)
    assert fresh.load(sid) is True
    assert fresh.get(sid)[0].content == "持久化"


def test_load_returns_false_for_unknown_session(tmp_path: Path):
    assert ConversationMemory(store_dir=tmp_path).load("ghost") is False


def test_list_sessions_includes_created(tmp_path: Path):
    m = ConversationMemory(store_dir=tmp_path)
    sid = m.new_session()
    m.append(sid, Turn(role="user", content="x", ts="t"))
    m.save(sid)
    assert any(s["session_id"] == sid for s in m.list_sessions())
```

- [ ] **Step 4: 运行测试确认失败**

Run: `python -m pytest tests/test_router.py tests/test_executor.py tests/test_memory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.router'`

- [ ] **Step 5: 实现三个模块**

**`router.py`**：
- 算术判定用**严格正则** `^\s*[-+*/().\d\s*]+$` 且必须至少含一个运算符、至少两个数字 —— 这样 `"3.14"` 不会被误判（**Review Focus #5**）。若问题中夹带中文（如 `"3.14 * 2.56 等于多少"`），先 `re.findall(r"[-+*/().\d\s]+", q)` 抽出最长数字表达式片段，若其长度 ≥ 3 且含运算符则判为 calculator，`args={"expression": 片段}`。
- 时间判定：问题含 `现在|今天|当前|几点|日期|时间|current time|today` 且长度 < 30。
- 其余 → `RouteDecision(kind="agent", tool=None, args={}, reason="需要检索与推理")`。
- 永远不抛异常，空串返回 `agent`。

**`executor.py`**：
- `run_one`：`ThreadPoolExecutor(max_workers=1)` 提交 `tool.func(**args)`，`future.result(timeout=tool.timeout_s)`；捕获 `TimeoutError` → `f"错误：工具 {name} 超时（>{t}s），请尝试其他方式或直接回答。"`；捕获 `TypeError`（参数不匹配）→ `f"错误：工具 {name} 参数不匹配：{e}。请检查 Action Input。"`；其他 `Exception` → `f"错误：工具 {name} 执行失败：{e}"`。返回值一律 `str()`。
- `run_parallel`：`ThreadPoolExecutor(max_workers=max_workers)` 提交全部任务，用 `as_completed` 收集到按输入下标的数组，保证**输出顺序与输入一致**。未知工具在该槽位写 `f"错误：工具 {name} 不存在。"`。单任务失败不影响其他任务。

**`memory.py`**：
- `Turn` 为 dataclass（`role, content, ts, cited_chunk_ids`）。
- `build_context(sid)`：若 `len(turns)/2 > summarize_after_turns` 且 `llm` 不为 None → 取前半轮次压缩成摘要（**缓存摘要，避免每次调用都重新摘要**：仅在轮次数增长超过阈值时才重算），返回 `(近期轮次转成的 messages, summary)`；否则返回 `(全部轮次 messages, None)`。
- 滑动窗口：从最新往旧累加 `estimate_tokens(content)`，超 `max_tokens` 停止；**按整轮（user+assistant）为最小单位**裁剪，并在 `ctx[-1]` 保证是完整轮次（测试断言 `ctx[-1]["role"] == "assistant"`）。
- `store_dir` 为 None 时 `save/load` 静默 no-op（内存模式，便于测试与 Streamlit 从简）。

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m pytest tests/test_router.py tests/test_executor.py tests/test_memory.py -v`
Expected: 30 passed

- [ ] **Step 7: 提交**

```bash
git add -A
git commit -m "feat: 工具路由、并行执行器（超时/隔离）与会话记忆管理"
```

---

### Task 18: 可观测性（轨迹与指标）

**Files:**
- Create: `observability/__init__.py`, `observability/trace.py`, `observability/metrics.py`
- Test: `tests/test_observability.py`

**Interfaces:**
- Consumes: `core.schemas.{AgentResult, AgentStep}`、`Settings`
- Produces:
  - `observability.trace.TraceRecorder(log_dir: Path)`
  - `TraceRecorder.record(result: AgentResult, session_id: str) -> None`（追加 JSONL）
  - `TraceRecorder.read_recent(n: int = 50) -> list[dict]`
  - `observability.metrics.compute(result: AgentResult) -> dict`
  - `observability.metrics.aggregate(records: list[dict]) -> dict`
  - `observability.metrics.health_check(retriever, llm) -> dict[str, tuple[bool, str]]`
  - `observability.metrics.render_trace_tree(steps: list[AgentStep]) -> str`

- [ ] **Step 1: 写失败测试 `tests/test_observability.py`**

```python
import json
from pathlib import Path

from core.schemas import AgentResult, AgentStep, Citation, RetrievalResult
from observability.metrics import aggregate, compute, health_check, render_trace_tree
from observability.trace import TraceRecorder


def _step(i, action, tokens=10, err=None):
    return AgentStep(index=i, thought=f"想法{i}", action=action, action_input={"q": "x"},
                     observation="观察", latency_ms=100, tokens=tokens, error=err)


def _res(steps=None, degraded=None):
    return AgentResult(answer="答案", citations=[Citation(1, "f.pdf", 1, "c", "s")],
                       steps=steps or [_step(0, "kb_search")], tokens={"prompt": 100, "completion": 50},
                       latency_ms={"retrieve": 10, "generate": 200, "total": 300},
                       route="agent", degraded=degraded)


def test_record_appends_jsonl(tmp_path: Path):
    r = TraceRecorder(tmp_path)
    r.record(_res(), "s1")
    r.record(_res(), "s2")
    lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["session_id"] == "s1"


def test_record_never_raises_on_unserializable(tmp_path: Path):
    bad = _res()
    bad.steps[0].action_input = {"obj": object()}
    TraceRecorder(tmp_path).record(bad, "s")     # 不应抛异常


def test_read_recent_returns_last_n_newest_first(tmp_path: Path):
    r = TraceRecorder(tmp_path)
    for i in range(5):
        r.record(_res(), f"s{i}")
    got = r.read_recent(n=2)
    assert len(got) == 2 and got[0]["session_id"] == "s4"


def test_read_recent_on_missing_file_returns_empty(tmp_path: Path):
    assert TraceRecorder(tmp_path / "none").read_recent() == []


def test_compute_extracts_core_metrics():
    m = compute(_res())
    assert m["steps"] == 1
    assert m["total_tokens"] == 150
    assert m["tools_used"] == ["kb_search"]
    assert m["total_ms"] == 300


def test_compute_counts_errors():
    m = compute(_res(steps=[_step(0, "kb_search", err="boom")]))
    assert m["errors"] == 1


def test_aggregate_computes_p50_and_p95():
    records = [{"metrics": {"total_ms": v, "total_tokens": 10, "tools_used": [], "errors": 0,
                            "steps": 1, "degraded": None}} for v in (100, 200, 300, 400, 500)]
    a = aggregate(records)
    assert a["count"] == 5
    assert a["p50_ms"] == 300
    assert a["p95_ms"] >= 400


def test_aggregate_on_empty_returns_zeros():
    a = aggregate([])
    assert a["count"] == 0 and a["p50_ms"] == 0


def test_aggregate_counts_tool_usage():
    records = [{"metrics": {"total_ms": 1, "total_tokens": 1, "tools_used": ["kb_search", "calculator"],
                            "errors": 0, "steps": 1, "degraded": None}}]
    assert aggregate(records)["tool_counts"]["kb_search"] == 1


def test_render_trace_tree_contains_all_three_phases():
    text = render_trace_tree([_step(0, "kb_search")])
    assert "Thought" in text and "Action" in text and "Observation" in text


def test_render_trace_tree_on_empty_list():
    assert isinstance(render_trace_tree([]), str)


def test_health_check_reports_each_component():
    class Retr:
        vectorstore = type("V", (), {"health": staticmethod(lambda: (True, "ok")), "count": staticmethod(lambda: 3)})()

    class LLM:
        def complete(self, *a, **k):
            return "pong"

    h = health_check(Retr(), LLM())
    assert "vector_store" in h and "llm" in h
    assert all(isinstance(v, tuple) and len(v) == 2 for v in h.values())


def test_health_check_reports_failure_without_raising():
    class BoomLLM:
        def complete(self, *a, **k):
            raise RuntimeError("401 unauthorized")

    h = health_check(None, BoomLLM())
    assert h["llm"][0] is False
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_observability.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'observability'`

- [ ] **Step 3: 实现 `observability/trace.py`**

要点：`record` 用 `json.dumps(payload, ensure_ascii=False, default=str)` —— **`default=str` 是关键**，它让任何不可序列化对象降级成字符串而不是抛 `TypeError`（测试钉住）。整体包在 `try/except Exception` 中并 `logger.warning`，**观测失败绝不能影响主流程**。`storage/logs/` 目录 `mkdir(parents=True, exist_ok=True)`。

- [ ] **Step 4: 实现 `observability/metrics.py`**

- `compute(result)` → `{"steps":int, "total_tokens":int, "tools_used":[...], "errors":int, "total_ms":int, "degraded":str|None, "route":str, "n_citations":int}`。
- `aggregate(records)` → 从 `r["metrics"]` 列表算 `count / p50_ms / p95_ms / avg_ms / total_tokens / tool_counts / error_rate / degraded_rate`。P50/P95 用**排序后线性插值**（`statistics.quantiles` 在 n<2 时会抛异常，需先判 `len < 2` 直接返回该值）。空列表 → 全 0。
- `health_check(retriever, llm)` → `{"vector_store": (bool,str), "llm": (bool,str), "bm25": (bool,str)}`；**每个探测独立 try/except**，任一失败不影响其他；`retriever` 为 None 时返回 `(False, "未初始化")`。
- `render_trace_tree(steps)` → 纯文本缩进树，供 Streamlit 与 CLI 复用：
  ```
  ├─ Step 0 · kb_search (100ms, 10 tok)
  │   Thought: 想法0
  │   Action: kb_search {"q": "x"}
  │   Observation: 观察
  ```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_observability.py -v`
Expected: 13 passed

- [ ] **Step 6: 提交**

```bash
git add -A
git commit -m "feat: 可观测性层（JSONL 轨迹、指标聚合、健康检查）"
```

---

### Task 19: 组装入口与 Streamlit 三栏前端

**Files:**
- Create: `agent/__init__.py`（补全 `build_agent()`）
- Create: `streamlit_app.py`
- Test: `tests/test_integration.py`

**Interfaces:**
- Consumes: 全部下层模块
- Produces:
  - `agent.build_agent(retriever=None, memory=None, recorder=None) -> ReActAgent`（内部调各工具的 `set_*` 完成注入）
  - `streamlit_app.main() -> None`
  - `streamlit_app.answer_once(question: str, session_id: str) -> AgentResult`（**无 Streamlit 依赖的纯逻辑入口**，供集成测试与 CLI 复用）

- [ ] **Step 1: 写失败测试 `tests/test_integration.py`**

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_integration.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_agent' from 'agent'`

- [ ] **Step 3: 实现 `agent/__init__.py` 的 `build_agent()`**

```python
def build_agent(retriever=None, llm=None, memory=None, recorder=None):
    from agent.memory import ConversationMemory
    from agent.react import ReActAgent
    from agent.tools import build_registry
    from agent.tools import kb_search as kb_mod
    from generation.rag import RAGEngine

    retriever = retriever or build_retriever()
    llm = llm or get_llm()
    engine = RAGEngine(retriever=retriever, llm=llm)
    kb_mod.set_engine(engine)
    paper_mod.set_vectorstore(retriever.vectorstore)
    paper_mod.set_llm(llm)
    ...
    return ReActAgent(llm=llm, tools=build_registry(), memory=memory)
```

`ReActAgent` 增加可选 `memory` 参数：`run()` 时若 `memory` 存在，先 `build_context(session_id)` 拼历史，结束后 `append` 本轮。

- [ ] **Step 4: 实现 `streamlit_app.py`**

结构（**所有重对象必须走 `@st.cache_resource`**，否则 Streamlit 每次交互重跑脚本会反复重建 Chroma 连接）：
- `_get_retriever()` / `_get_llm()` / `_get_memory()` / `_get_recorder()` 均带 `@st.cache_resource`。
- `answer_once(question, session_id)`：GET agent → `agent.run(...)` → `recorder.record(...)` → 返回 `AgentResult`。**这是不依赖 Streamlit runtime 的纯逻辑入口**，集成测试直接调它。
- `main()`：
  - `st.set_page_config(layout="wide")`；顶部 `st.title("智能科研助理")` + 健康状态横幅（`health_check` 失败则 `st.error`）。
  - `left, center, right = st.columns([1, 2.4, 1.4])`。
  - 左栏：`st.file_uploader(accept_multiple_files=True, type=["pdf","docx","txt","md"])` → 保存到 `data/raw/` → `IngestPipeline.ingest_paths(..., progress_cb=...)` 用 `st.progress` 更新 → `st.success` 显示 `IngestReport` 摘要；文档列表 `st.dataframe` + 每行删除按钮；刷新按钮。
  - 中栏：`st.chat_message` 渲染 `st.session_state.messages`；`st.chat_input` → `st.write_stream(engine.stream_answer(...))`；引用 `Citation` 用 `st.expander` 展示 `filename + page_no + snippet`。
  - 右栏：`st.expander("推理轨迹")` 内用 `st.code(render_trace_tree(steps))`；指标 `st.metric` 展示 token/延迟/路由/降级；`st.session_state.last_result` 供渲染。
  - 侧栏 `st.sidebar`：新建会话、历史会话列表与切换、清空。

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_integration.py -v`
Expected: 4 passed

- [ ] **Step 6: 手工验证前端可启动**

Run: `python -m streamlit run streamlit_app.py --server.headless true --server.port 8599`
Expected: 输出 `You can now view your Streamlit app in your browser.` 且无 traceback。验证后 Ctrl-C 停止。

- [ ] **Step 7: 提交**

```bash
git add -A
git commit -m "feat: Agent 组装入口与 Streamlit 三栏前端"
```

---

### Task 20: 评测脚本、评测集与冒烟测试

**Files:**
- Create: `scripts/build_index.py`, `scripts/smoke_test.py`
- Create: `scripts/eval_retrieval.py`, `scripts/eval_agent.py`
- Create: `eval/qa_set.jsonl`, `eval/qa_set_demo.jsonl`
- Test: `tests/test_eval_metrics.py`

**Interfaces:**
- Consumes: 全部下层模块
- Produces:
  - `scripts.eval_retrieval.hit_at_k(ranked_ids: list[str], gold: set[str], k: int) -> int`
  - `scripts.eval_retrieval.reciprocal_rank(ranked_ids: list[str], gold: set[str]) -> float`
  - `scripts.eval_retrieval.recall_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float`
  - `scripts.eval_retrieval.evaluate(retriever, qa_pairs, modes, ks) -> pd.DataFrame`
  - 可执行：`python scripts/smoke_test.py`（退出码 0/1）

- [ ] **Step 1: 写失败测试 `tests/test_eval_metrics.py`**

```python
import pytest

from scripts.eval_retrieval import hit_at_k, recall_at_k, reciprocal_rank


def test_hit_at_k_finds_relevant_within_k():
    assert hit_at_k(["a", "b", "c"], {"c"}, k=3) == 1


def test_hit_at_k_misses_relevant_outside_k():
    assert hit_at_k(["a", "b", "c"], {"c"}, k=2) == 0


def test_hit_at_k_empty_gold_is_zero():
    assert hit_at_k(["a"], set(), k=5) == 0


def test_reciprocal_rank_uses_first_hit_position():
    assert reciprocal_rank(["x", "y", "gold"], {"gold"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_zero_when_absent():
    assert reciprocal_rank(["x"], {"gold"}) == 0.0


def test_recall_at_k_fraction_of_gold_found():
    assert recall_at_k(["a", "b", "c"], {"a", "c", "z"}, k=3) == pytest.approx(2 / 3)


def test_recall_at_k_empty_gold_is_zero_not_division_error():
    assert recall_at_k(["a"], set(), k=3) == 0.0


def test_metrics_are_order_sensitive_for_rr_but_not_hit():
    assert hit_at_k(["a", "b"], {"a"}, 2) == hit_at_k(["b", "a"], {"a"}, 2)
    assert reciprocal_rank(["a", "b"], {"a"}) > reciprocal_rank(["b", "a"], {"a"})
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_eval_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts'`

- [ ] **Step 3: 实现三个指标函数与 `scripts/eval_retrieval.py`**

- 三个指标函数必须是**纯函数**（`scripts/__init__.py` 需存在，且 `eval_retrieval.py` 的 `main()` 要放在 `if __name__ == "__main__":` 内，避免 import 时执行）。
- `evaluate()`：遍历 `qa_pairs` × `modes`，对每对调 `retriever.search(q, mode, top_k=max(ks))`，取 `chunk_id` 序列，算三个指标，汇总成 DataFrame（列：`mode / n / hit@1 / hit@3 / hit@5 / mrr / recall@5`）。写 CSV 到 `eval/reports/retrieval_<timestamp>.csv`，并用 matplotlib 画柱状图 `retrieval_compare.png`（**matplotlib 需要 `matplotlib.use("Agg")` 以避免无显示环境报错**）。
- CLI：`python scripts/eval_retrieval.py --qa eval/qa_set.jsonl --modes vector hybrid hybrid_rerank`。

- [ ] **Step 4: 实现 `scripts/smoke_test.py`**

无 key 全链路自检（**对应成功标准 S9**），任一步失败打印 `[FAIL] <步骤>: <原因>` 并 `sys.exit(1)`：
1. import 全部模块（含 `streamlit_app`）不报错
2. 造 3 个假 chunk 入库
3. 三档检索各跑一次，断言 `chunks` 非空
4. `RAGEngine.stream_answer` 消费完，断言最后一个事件是 `done`
5. `build_agent().run("2+2")` 断言 `answer` 非空
6. 断言 `eval_retrieval` 三个指标函数可调用
7. 打印 `[OK] 全链路自检通过` 并 `exit(0)`

- [ ] **Step 5: 实现 `scripts/eval_agent.py`**

对每个评测问题跑 `agent.run()`，记录：`expected_tool` vs `实际首个工具`（准确率）、`len(steps)`（平均轮次）、`degraded` 比例、端到端延迟、token 消耗。输出 CSV + 汇总打印。

- [ ] **Step 6: 构造评测集**

`eval/qa_set_demo.jsonl`：**5 条**手工构造、`gold_chunk_ids` 与 demo 文档严格对应的样例（用于跑通流程与 CI）。
`eval/qa_set.jsonl`：**50 条**，覆盖事实性/对比性/归纳性/推理性四类各约 25%，`gold_chunk_ids` 需在真实论文入库后由 `scripts/build_index.py --dump-chunks` 输出的 chunk 清单核对填写。**未核对 gold 的条目不得计入正式评测。**

- [ ] **Step 7: 运行测试确认通过**

Run: `python -m pytest tests/test_eval_metrics.py -v`
Expected: 8 passed

- [ ] **Step 8: 运行冒烟测试**

Run: `python scripts/smoke_test.py`
Expected: `[OK] 全链路自检通过`，退出码 0

- [ ] **Step 9: 提交**

```bash
git add -A
git commit -m "feat: 评测脚本、评测集与无 key 全链路冒烟自检"
```

---

### Task 21: 全套文档

**Files:**
- Create: `README.md`（覆盖 Task 1 的占位版）
- Create: `docs/技术设计文档.md`, `docs/用户使用手册.md`
- Create: `docs/分块策略对比实验报告.md`, `docs/Bad Case分析报告.md`
- Create: `docs/评测报告.md`
- Modify: `docs/项目构建书.md`（回填实测数据）

**Interfaces:**
- Consumes: `eval/reports/` 的真实产出数据
- Produces: 交付物清单第 2-7 项

- [ ] **Step 1: 写 `README.md`**

必须包含：项目一句话简介、架构图（复用构建书 §3.1）、**三步快速开始**（`pip install -r requirements.txt` → `cp .env.example .env` 并填 key → `streamlit run streamlit_app.py`）、mock 模式说明、目录结构、常见问题。

- [ ] **Step 2: 写 `docs/技术设计文档.md`**

面向开发者，包含架构图、每个模块的职责与接口签名、API 调用说明（DeepSeek/SiliconFlow 的请求响应示例）、**全部配置参数表**（参数名/类型/默认值/作用，从 `config.py` 逐条誊写）、数据流图。

- [ ] **Step 3: 写 `docs/用户使用手册.md`**

面向非技术用户：安装部署步骤、界面三栏功能说明、引用阅读方法、常见问题（key 报错怎么办、上传失败怎么办、回答不准怎么办）。

- [ ] **Step 4: 跑实验并写两份报告**

```bash
python scripts/eval_retrieval.py --qa eval/qa_set_demo.jsonl \
  --modes vector hybrid hybrid_rerank --chunk-sizes 256 512 1024
python scripts/eval_agent.py --qa eval/qa_set_demo.jsonl
```

把**真实产出的数字**填进 `docs/分块策略对比实验报告.md` 与 `docs/评测报告.md`，并把三档对比结论回填到 `docs/项目构建书.md` §5.1.4 的验收段落。**禁止使用估计值或编造数据**；若某配置跑不出差异，如实记录"在该评测集规模下差异不显著"。

- [ ] **Step 5: 写 `docs/Bad Case分析报告.md`**

按检索失败/生成错误/引用错误/Agent 决策错误四类，每类列出**实际观察到的**案例（含问题原文、检索结果、答案、错在哪）、根因分析、改进方案、以及至少一轮优化前后的对比数据。

- [ ] **Step 6: 提交**

```bash
git add -A
git commit -m "docs: 技术设计文档、用户手册、评测报告与 Bad Case 分析"
```

---

## Self-Review

**1. Spec coverage**

| 构建书章节 | 覆盖 Task |
|---|---|
| §5.1.1 多格式加载器 | Task 3 |
| §5.1.2 分块策略 + 对比实验 | Task 4、Task 20、Task 21 |
| §5.1.3 向量化存储 + 增量索引 | Task 2、Task 5 |
| §5.1.4 BM25 / RRF / Rerank / 三档编排 | Task 6、Task 7、Task 8、Task 9 |
| §5.2.1 Prompt 工程 + 引用溯源 | Task 10 |
| §5.2.2 流式输出 | Task 11、Task 19 |
| §5.2.3 语义缓存 | Task 12 |
| §5.2.4 降级策略 | Task 12 |
| §5.2.5 日志与可观测性 | Task 18 |
| §5.3.1 手写 ReAct 循环 | Task 16 |
| §5.3.2 八个工具 | Task 13、Task 14、Task 15 |
| §5.3.3 路由 / 并行 / 错误恢复 | Task 17 |
| §5.3.4 记忆管理 | Task 17 |
| §5.4.1 RAG+Agent 融合 | Task 14、Task 19 |
| §5.4.2 可观测性 + 健康检查 | Task 18 |
| §5.4.3 Streamlit 三栏 | Task 19 |
| §5.4.4 端到端与边缘测试 | Task 19、Task 20 |
| §5.5.1 评测集 | Task 20 |
| §5.5.2 系统级评估 | Task 20 |
| §5.5.3 Bad Case 分析 | Task 21 |
| §6 配置与密钥 | Task 1、Task 2 |
| §10 交付物清单 | Task 21 |
| 成功标准 S1–S9 | S1→T9/T20，S2/S3→T20，S4→T10/T11，S5→T17，S6→T16，S7→T12/T19，S8→T19，S9→T20 |

无遗漏。

**2. Placeholder scan**

已移除全部 "TODO/TBD"。Task 21 的两份实验报告以「用真实产出填充」的方式指定，并明确禁止编造数据——这是对数据依赖型交付物的正确处理，不是占位符。

**3. Type consistency**

逐条核对跨 Task 的签名：
- `estimate_tokens`（T1 定义）→ T10 使用 ✓
- `ScoredChunk.source` 取值集：`vector`(T5) / `bm25`(T6) / `rrf`(T7) / `rerank`(T8) → T9/T11 断言一致 ✓
- `RetrievalResult.degraded` 取值：`empty_knowledge_base` / `embedding_failed_fallback_bm25` / `rerank_failed: ...` / `rerank_disabled` → T9 与 T12 一致 ✓
- `build_rag_messages` 返回 `(messages, citations)`（T10）→ T11 消费一致 ✓
- `Reranker.rerank` 返回 `(list, str|None)`（T8）→ T9 消费一致 ✓
- `ToolExecutor.run_one(tool, args)`（T17）→ T16 的 `self.executor.run_one` 一致 ✓
- `AgentStep` 字段（T1）→ T16 构造、T18 `render_trace_tree` 消费一致 ✓
- 工具注入函数名：`set_engine`(T14) / `set_vectorstore` / `set_llm` → T19 `build_agent` 一致 ✓
- `build_agent(retriever, llm, memory, recorder)`：T19 定义，测试用 `build_agent(retriever=..., llm=...)` 一致 ✓

**4. Review Focus**

5 条均已落测：① 空知识库 → T5 `test_query_on_empty_store_returns_empty_list_not_error`、T9 `test_empty_knowledge_base_returns_empty_not_crash`、T11 `test_answer_on_empty_knowledge_base_marks_degraded`、T19 `test_full_chain_survives_empty_knowledge_base`；② 缺 key → T2 `test_missing_key_raises_at_settings_construction`；③ 重复上传/损坏文件 → T3 `test_corrupt_pdf_raises_load_error_not_crash`、T9 `test_ingest_is_idempotent_for_same_content`、T9 `test_corrupt_file_fails_but_batch_continues`；④ 非法 Action → T16 `test_parse_action_bad_json_raises_with_readable_message`、`test_malformed_action_is_retried_then_forced_to_conclude`；⑤ 中文标点/超长句/纯数字 → T4 `test_recursive_chunker_handles_cjk_fullwidth_punctuation`、`test_recursive_chunker_handles_single_very_long_sentence`、T6 `test_tokenize_never_raises_on_fullwidth_punctuation`、T17 `test_bare_number_is_not_treated_as_arithmetic`。

无未覆盖项。
