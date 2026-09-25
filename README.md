# 智能科研助理 —— 基于 RAG+Agent 的论文知识库问答系统

上传论文（PDF / DOCX / TXT / MD），系统用**混合检索 + 重排序**精确定位证据，由**手写 ReAct Agent** 自主决策调用 8 个工具，生成**带引用溯源、推理轨迹可见**的答案。

南京农业大学课程实践 1 项目。

---

## 特色

| 能力 | 说明 |
|---|---|
| 混合检索 | 向量检索（语义）+ BM25（关键词）双路，**手写 RRF** 融合 |
| 重排序 | bge-reranker-v2-m3 交叉编码器精排，Top-5 命中率显著优于朴素 RAG |
| 三档可切换 | `vector` / `hybrid` / `hybrid_rerank`，支持消融对比实验 |
| 手写 Agent | ReAct 循环约 200 行核心代码，**不依赖 LangChain** |
| 8 个工具 | 知识库检索、论文元信息、论文对比、关键词提取、结构化摘要、计算器、当前时间、联网搜索 |
| 引用溯源 | 答案中 `[n]` 可点击，回跳到真实文档名 + 页码 |
| 全程可观测 | Thought → Action → Observation 轨迹树 + Token / 延迟 / 命中率指标 |
| 降级容错 | 无命中、低置信、向量服务故障、LLM 故障各有对应降级路径，**任何一环故障都不会让请求 500** |
| 零本地模型 | 生成与向量化全部走云端 API，不需要下载任何模型权重 |
| Mock 模式 | 无 API key 也能跑通全链路与评测脚本，便于验收与教学演示 |

## 技术栈

| 层次 | 选型 |
|---|---|
| 生成模型 | DeepSeek `deepseek-chat`（OpenAI 兼容） |
| 向量化 | SiliconFlow `BAAI/bge-m3`（1024 维 / 8192 上下文） |
| 重排序 | SiliconFlow `BAAI/bge-reranker-v2-m3` |
| 关键词检索 | `rank_bm25` |
| 向量数据库 | ChromaDB（本地持久化，支持增量 upsert） |
| PDF 解析 | PyMuPDF（对比 PyPDF2 / pdfplumber 后的最优方案） |
| 前端 | Streamlit |
| 测试 | pytest |

**刻意不使用** LangChain / LlamaIndex / sentence-transformers / torch —— 前者与「手写 ReAct 与 RRF」的教学目标冲突，后者与「不做本地部署」冲突。

## 架构

```
┌─────────────────────────────────────────────┐
│  UI 层 (Streamlit)                           │
│  左栏 文档管理 │ 中栏 对话+流式 │ 右栏 轨迹树  │
└───────────────────┬─────────────────────────┘
                    │ Message / AgentStep
┌───────────────────▼─────────────────────────┐
│  Agent 决策层 (手写 ReAct)                    │
│  react ── router ── executor ── memory       │
│              │ 工具注册表 (8 tools)           │
└───────────────────┬─────────────────────────┘
                    │ Tool 调用
┌───────────────────▼─────────────────────────┐
│  生成层 (RAG)  prompts ─ rag ─ cache ─ fallback│
└───────────────────┬─────────────────────────┘
                    │ top-k ScoredChunk
┌───────────────────▼─────────────────────────┐
│  检索层   vector ┐                            │
│           bm25   ├─► RRF ─► Rerank           │
└───────────────────┬─────────────────────────┘
                    │ Chunk
┌───────────────────▼─────────────────────────┐
│  数据层  loaders ─ chunkers ─ pipeline        │
└─────────────────────────────────────────────┘
        observability（横切所有层）
```

层间依赖单向，只通过 `core/schemas.py` 的数据结构通信；每层内部实现可独立替换。

## 快速开始

### 1. 安装依赖

```bash
# 方式 A：用项目自带的虚拟环境（推荐，已随仓库创建）
.venv\Scripts\activate
pip install -r requirements.txt

# 方式 B：用你自己的 Python 环境
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
copy .env.example .env
```

编辑 `.env`，填入两个 key：

```dotenv
DEEPSEEK_API_KEY=sk-...          # https://platform.deepseek.com/api_keys
SILICONFLOW_API_KEY=sk-...       # https://cloud.siliconflow.cn/account/ak
```

> **为什么要两个 key？** DeepSeek 目前不提供 embedding 接口，因此向量化与重排序需要 SiliconFlow。SiliconFlow 注册即有免费额度，bge 系列模型与作业文档指定的方案一致。

### 3. 启动

```bash
streamlit run streamlit_app.py
```

浏览器打开 `http://localhost:8501`，在左栏上传论文，等索引进度条走完，即可在中栏提问。

## 没有 API Key？用 Mock 模式

```dotenv
LLM_PROVIDER=mock
EMBEDDING_PROVIDER=mock
RERANK_PROVIDER=mock
```

Mock 模式用确定性伪向量 + 固定话术，能跑通**除真实语义质量外的全部链路**（入库 → 三档检索 → 流式生成 → ReAct Agent → 轨迹 → 评测脚本）。适合先验收工程完整性，之后改回真实 provider 只需改这三行。

```bash
python scripts/smoke_test.py     # 全链路自检，退出码 0 表示通过
```

## 目录结构

```
├── config.py                 # 全局配置中心（唯一参数来源）
├── streamlit_app.py          # 前端入口
├── core/                     # 统一数据结构 + 外部服务 Provider 抽象
│   ├── schemas.py            # ★ 层间接口契约
│   └── providers.py          # ★ LLM / Embedding / Rerank，含 Mock
├── ingest/                   # 文档加载、分块、入库管线
├── retrieval/                # 向量库、BM25、RRF 融合、重排序、混合检索编排
├── generation/               # Prompt、流式生成、引用溯源、语义缓存、降级
├── agent/                    # 手写 ReAct 循环、8 个工具、路由、执行器、记忆
├── observability/            # 轨迹记录与指标聚合
├── scripts/                  # 建索引、评测、冒烟自检
├── tests/                    # pytest 单元测试
├── eval/                     # 评测集与评测报告
├── data/                     # 原始文档与中间产物
├── storage/                  # 向量库 / BM25 索引 / 缓存 / 日志（已 gitignore）
└── docs/                     # 构建书、技术设计、实验报告、用户手册
```

## 文档

| 文档 | 内容 |
|---|---|
| `docs/项目构建书.md` | 技术选型理由、架构、五大模块搭建方案、验收标准 |
| `docs/技术设计文档.md` | 模块接口、API 说明、全部配置参数 |
| `docs/分块策略对比实验报告.md` | 三种分块策略的检索效果对比 |
| `docs/评测报告.md` | 三档检索的 Hit@5 / MRR / Recall，Agent 指标 |
| `docs/Bad Case分析报告.md` | 失败案例归因与优化前后对比 |
| `docs/用户使用手册.md` | 面向非技术用户的安装与使用说明 |

## 常见问题

**Q: 报错说缺少 `DEEPSEEK_API_KEY`？**
这是启动期的配置校验。在 `.env` 填好 key，或把三个 `*_PROVIDER` 设为 `mock`。

**Q: 上传 PDF 失败？**
若 PDF 是扫描件（图片型），PyMuPDF 抽不出文字。请先用 OCR 工具转成文本型 PDF，或直接上传 TXT。

**Q: 回答不准 / 检索不到？**
先看右栏的检索轨迹和 Top-1 置信度。若是分块过碎导致语义不完整，在 `.env` 调大 `CHUNK_SIZE`；若是专业术语匹配不上，用 `hybrid` 模式（关键词通道）会更好。

**Q: 重复上传同一个文件会怎样？**
`doc_id` 是文件内容的 sha256，相同文件会被识别并跳过，不会重复向量化、不会重复计费。

## 许可

课程实践项目，仅供学习使用。
