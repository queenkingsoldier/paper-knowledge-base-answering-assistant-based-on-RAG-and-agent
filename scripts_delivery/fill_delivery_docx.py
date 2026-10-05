r"""把《项目交付.docx》模板填充为完整课程报告。

设计要点
--------
* 就地编辑：在模板段落之间插入内容，保留封面、标题、分节符等原有排版。
* 格式继承：新增段落直接沿用模板中同类型段落的 pPr/rPr（正文→仿宋4号，
  小节标题→黑体加粗），避免字号字体漂移。
* 所有数字均有出处：来自 eval/reports/*.csv、docs/*.md、
  scripts/smoke_test.py 的实际输出，未编造。

用法：
    python scripts_delivery/fill_delivery_docx.py
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "项目交付.docx"
BAK = ROOT / "scripts_delivery" / "项目交付.原模板备份.docx"
ARCH_PNG = ROOT / "docs" / "architecture.png"

# ---------------------------------------------------------------- 页面与表格宽度

# 模板原始页边距（上/下 0.79in，左/右 1.25in）只留出 14.66cm 可用宽度，
# 6 列表格必然溢出页面。故统一改为中文课程报告通用的 2.5cm 左右边距，
# 得到 16.0cm 可用宽度，并让所有表格按此宽度配比列宽。
MARGIN_LR_CM = 2.5
MARGIN_TB_CM = 2.54
USABLE_CM = 21.0 - 2 * MARGIN_LR_CM          # A4 宽 21cm → 16.0cm

# 模板正文继承 Normal 的 14pt（四号）。中文课程报告常用 12pt（小四），
# 且 14pt 正文在 16cm 版心内会因字体度量差异溢出到页面边缘
# （原模板未改动时即已如此），故正文显式落到 12pt。
BODY_PT = 12
CELL_PT = 10.5
TABLE_PAD_CM = 0.30          # 单元格左右内边距合计，配列宽时要扣掉


def fit(widths_cm: list[float]) -> list[float]:
    """把列宽按比例缩到「版心 − 单元格内边距」，消除表格横向溢出。

    单元格默认左右各留 0.19cm 内边距，列宽合计等于版心时表格实际会宽出
    n×0.38cm，因此必须预留这部分。
    """
    budget = USABLE_CM - TABLE_PAD_CM
    total = sum(widths_cm)
    if total <= budget:
        return widths_cm
    k = budget / total
    return [round(w * k, 2) for w in widths_cm]


def normalize_margins(doc: Document) -> None:
    for s in doc.sections:
        s.left_margin = Cm(MARGIN_LR_CM)
        s.right_margin = Cm(MARGIN_LR_CM)
        s.top_margin = Cm(MARGIN_TB_CM)
        s.bottom_margin = Cm(MARGIN_TB_CM)


# ---------------------------------------------------------------- 模板定位锚点

ANCHORS = {
    "h_toc": "目录(",
    "h1": "一、项目概述",
    "p_bg_title": "1.1 项目背景",
    "p_bg_hint": "（简述项目背景",
    "p_goal": "1.2 项目目标",
    "p_stack": "1.3 技术栈概览",
    "h2": "二、团队分工说明",
    "p_team_hint": "包括：角色，成员姓名",
    "p_team_sum": "贡献比例四项之和应为100%",
    "h3": "三、模块交付物清单",
    "p_mod_hint": "（说明完成情况，每个模块制作一个表格说明）",
    "m1": "模块一：文档处理与检索层",
    "m2": "模块二：RAG生成层",
    "m3": "模块三：Agent决策层",
    "m4": "模块四：系统集成与前端展示",
    "m5": "模块五：评测与交付",
    "h4": "技术实现记录",
    "p_impl_hint": "（详细说明技术是如何实现的）",
    "impl41": "4.1 模块一：文档处理与检索层",
    "impl42": "4.2 模块二：RAG生成层",
    "impl43": "4.3 模块三：Agent决策层",
    "impl44": "4.4 模块四：系统集成与前端展示",
    "impl441": "4.4.1 系统架构图",
    "p_arch_hint": "（插入系统架构图图片）",
    "p_arch_text": "文字描述：系统分为五层",
    "impl442": "4.4.2 界面功能截图",
    "p_shot_hint": "（附上3-5张界面截图",
    "impl45": "4.5 模块五：评测",
    "h5": "五、技术设计文档",
    "d51": "5.1 系统架构图",
    "p_d51_hint": "（插入最终的架构图",
    "d52": "5.2 模块接口定义",
    "d53": "5.3 核心数据流向",
    "d54": "5.4 配置文件说明",
    "h6": "六、总结与展望",
    "s61": "6.1 项目总结",
    "s62": "6.2 主要困难与解决方案",
    "s63": "6.3 待改进方向",
    "h7": "七、附录",
    "a_title": "附录A：项目代码目录结构",
    "a_fence": "```",
    "b_title": "附录B：环境配置说明",
    "b_hint": "（附上环境配置步骤",
    "b_fence_line": "1. 创建虚拟环境",
}


# ---------------------------------------------------------------- 基础工具

def find(doc: Document, key: str) -> Paragraph:
    needle = ANCHORS[key]
    for p in doc.paragraphs:
        if needle in p.text:
            return p
    raise KeyError(f"模板中找不到锚点 {key!r}（{needle!r}）")


def set_text(par: Paragraph, text: str) -> None:
    """保留首个 run 的格式，改写整段文字；若为空段则写入 rPr 为空的 run。"""
    runs = par.runs
    if not runs:
        par.add_run(text)
        return
    runs[0].text = text
    for r in runs[1:]:
        r.text = ""


def clear_text(par: Paragraph) -> None:
    set_text(par, "")


def delete(par: Paragraph) -> None:
    el = par._element
    el.getparent().remove(el)


def remove_range_between(doc: Document, start_el, end_el, inclusive=0) -> int:
    """删除 body 中 [start_el, end_el) 之间的段落/表格。

    用 XML 元素身份做区间判断，而不是 `list.index(Paragraph)`——后者永远失败，
    因为每次访问 doc.paragraphs 都会新建 Paragraph 包装对象。
    inclusive 表示末尾额外多删 n 个元素（用于把结束锚点自身也删掉）。
    """
    body = doc.element.body
    children = list(body)
    try:
        i = children.index(start_el)
        j = children.index(end_el)
    except ValueError:
        return 0
    if j < i:
        return 0
    targets = children[i:j + inclusive]
    for el in targets:
        body.remove(el)
    return len(targets)


def remove_all_fenced_blocks(doc: Document) -> int:
    """删除文档中所有围栏块（``` / ```bash … 闭合 ```）。

    模板里是**两个独立**的围栏块：附录A 的目录树示例、附录B 的 bash 示例。
    它们被「附录B 标题 + 提示语」隔开，因此不能用一个 [首个围栏, 最后一个围栏]
    的大区间去删——那会把中间的标题和提示语一并删掉。
    这里按「遇到开栏 → 找下一个闭栏」逐块删除。
    """
    tick = "`" * 3
    removed = 0
    while True:
        ps = doc.paragraphs
        lo = None
        for i, p in enumerate(ps):
            if p.text.strip().startswith(tick):
                lo = i
                break
        if lo is None:
            break
        hi = None
        for j in range(lo + 1, len(ps)):
            if ps[j].text.strip().startswith(tick):
                hi = j
                break
        if hi is None:            # 只有开栏没有闭栏，不删，避免吞掉后续内容
            break
        for p in ps[lo:hi + 1]:
            p._element.getparent().remove(p._element)
            removed += 1
    return removed


def clone_after(anchor_el, template_el):
    """在 anchor_el 之后插入 template_el 的副本，返回其 CT_P/CT_Tbl。"""
    new = copy.deepcopy(template_el)
    anchor_el.addnext(new)
    return new


# ---- 样式来源（在模板里各取一个代表段，作为格式母版）----

def grab_style_source(doc: Document) -> dict:
    return {
        "h_section": find(doc, "h1")._element,      # 一级小节标题
        "h_module": find(doc, "m1")._element,        # 模块小标题
        "h_sub": find(doc, "p_goal")._element,       # 1.2 / 5.1 这类子标题
        "body": find(doc, "p_bg_hint")._element,     # 正文
        "body2": find(doc, "p_team_hint")._element,
        "table": doc.tables[0]._element,             # Table Grid 母版
    }


class Builder:
    """按顺序在文档中插入段落 / 表格，维护“当前插入位置”。"""

    def __init__(self, doc: Document, styles: dict):
        self.doc = doc
        self.styles = styles

    # -- 插入并返回新 Paragraph（可链式调用）--
    def para(self, after_el, text: str = "", kind: str = "body", *,
             bold: bool = False, eastasia: str | None = None,
             size_pt: float | None = None):
        new_el = clone_after(after_el, self.styles[kind])
        par = Paragraph(new_el, self.doc)
        set_text(par, text)
        # 正文与小标题统一显式设定字号（不依赖模板 Normal 的 14pt），
        # 否则中文字体度量会把行撑出版心。
        if size_pt is None and kind in ("body", "body2", "h_module", "h_sub"):
            size_pt = BODY_PT
        if bold or eastasia or size_pt:
            for run in par.runs:
                if bold:
                    run.font.bold = True
                if size_pt:
                    run.font.size = Pt(size_pt)
                if eastasia:
                    rPr = run._element.get_or_add_rPr()
                    rFonts = rPr.get_or_add_rFonts()
                    rFonts.set(qn("w:eastAsia"), eastasia)
                    rFonts.set(qn("w:ascii"), eastasia)
                    rFonts.set(qn("w:hAnsi"), eastasia)
        return new_el

    def paras(self, after_el, items, kind: str = "body"):
        """连续插入若干正文段，items 为字符串序列。返回最后一个元素。"""
        cur = after_el
        for item in items:
            cur = self.para(cur, item, kind)
        return cur

    def table(self, after_el, header, rows, col_widths_cm=None):
        new_el = clone_after(after_el, self.styles["table"])
        # 清空母版带来的旧内容，重建行列
        from docx.table import Table

        tbl = Table(new_el, self.doc)
        n_cols = len(header)

        # 补足行（母版行数不定，先补后删更稳）
        while len(tbl.rows) < len(rows) + 1:
            tbl.add_row()
        # 删除多余行
        while len(tbl.rows) > len(rows) + 1:
            tbl._tbl.remove(tbl.rows[-1]._tr)

        # 删除多余列：必须同时删 tblGrid 的 gridCol 与各行的 tc。
        # 只删行内 tc 的话，tbl._column_count 仍按 tblGrid 返回旧列数，
        # 于是 tbl.cell(r, c) 算出的扁平索引会越界（IndexError）。
        while tbl._column_count > n_cols:
            for row in tbl.rows:
                cells = row.cells
                if len(cells) > n_cols:
                    row._tr.remove(cells[-1]._tc)
            grid = tbl._tbl.find(qn("w:tblGrid"))
            if grid is None:
                break
            cols = grid.findall(qn("w:gridCol"))
            if len(cols) <= n_cols:
                break
            for col in cols[n_cols:]:
                grid.remove(col)

        # 补足列：母版只有 4 列，宽度更大的表需要先扩 tblGrid 与各行 tc
        if tbl._column_count < n_cols:
            grid = tbl._tbl.find(qn("w:tblGrid"))
            if grid is not None:
                for _ in range(n_cols - len(grid.findall(qn("w:gridCol")))):
                    grid.append(OxmlElement("w:gridCol"))
            for row in tbl.rows:
                tr = row._tr
                while len(row.cells) < n_cols:
                    # 以本行最后一个 tc 为原型复制，保证段落/字体格式一致
                    tcs = tr.findall(qn("w:tc"))
                    if tcs:
                        tr.append(copy.deepcopy(tcs[-1]))
                    else:
                        tr.append(OxmlElement("w:tc"))

        # 写入表头与数据
        # 注意：Table._cells 是带缓存的 property，直接改 _tr 后缓存会失真，
        # 因此统一走 _cell_at（先失效缓存再取），而不是 tbl.cell()。
        for j, text in enumerate(header):
            self._fill_cell(self._cell_at(tbl, 0, j), text, bold=True)
        for i, row in enumerate(rows, start=1):
            for j, text in enumerate(row):
                self._fill_cell(self._cell_at(tbl, i, j), text, bold=False)

        # 固定列宽：同时更新 tblGrid 与每个单元格，否则 Word 会自动均分
        if col_widths_cm:
            tbl.autofit = False
            grid = tbl._tbl.find(qn("w:tblGrid"))
            if grid is not None:
                cols = grid.findall(qn("w:gridCol"))
                for j, width in enumerate(col_widths_cm):
                    if j < len(cols):
                        cols[j].set(qn("w:w"), str(int(Cm(width).twips)))
            for j, width in enumerate(col_widths_cm):
                for i in range(len(tbl.rows)):
                    if j < len(tbl.rows[i].cells):
                        tbl.rows[i].cells[j].width = Cm(width)
        return new_el

    @staticmethod
    def _cell_at(tbl, row_idx: int, col_idx: int):
        """取单元格，并先失效 Table._cells 缓存。

        python-docx 的 Table._cells 是 cached_property，一旦在增删行/列之前被
        访问过，之后 tbl.cell(r, c) 会按旧的行列数索引而抛 IndexError。
        """
        tbl.__dict__.pop("_cells", None)
        return tbl.cell(row_idx, col_idx)

    @staticmethod
    def _fill_cell(cell, text: str, bold: bool) -> None:
        # 清掉模板残留的多余段落，只保留第一段
        while len(cell.paragraphs) > 1:
            cell.paragraphs[-1]._element.getparent().remove(
                cell.paragraphs[-1]._element)
        par = cell.paragraphs[0]
        for r in list(par.runs):
            r.text = ""
        run = par.add_run(text)
        run.font.bold = bold
        run.font.size = Pt(CELL_PT)
        rPr = run._element.get_or_add_rPr()
        rFonts = rPr.get_or_add_rFonts()
        rFonts.set(qn("w:eastAsia"), "仿宋")
        rFonts.set(qn("w:ascii"), "仿宋")
        rFonts.set(qn("w:hAnsi"), "仿宋")
        setattr(par.paragraph_format, "space_before", Pt(0))
        par.paragraph_format.space_after = Pt(0)
        par.paragraph_format.line_spacing = 1.0


# ================================================================ 内容

BACKGROUND = (
    "科研文献数量激增，研究者面对私有论文库时存在三重障碍：通用大模型无法覆盖最新"
    "论文与内部资料，且存在幻觉风险；传统关键词检索缺乏语义理解，专有名词与同义改写"
    "之间难以匹配；文献阅读与多篇对比耗时巨大，严重打断研究思路。检索增强生成（RAG）"
    "通过“先检索、后生成”把私有知识安全接入大模型，并让答案可溯源；Agent 则通过"
    "ReAct 范式自主规划工具调用，把系统能力从“被动问答”扩展到“多步主动服务”。"
    "本项目即基于这一技术路径，构建面向个人论文库的科研问答助手。"
)

OBJECTIVES = [
    ("O1", "多格式文档入库与索引",
     "支持 PDF / DOCX / TXT / MD 批量入库，自动解析、分块、向量化并建立向量与关键词双索引，"
     "同一文件重复上传自动去重跳过。"),
    ("O2", "高精度混合检索与重排序",
     "向量检索与 BM25 关键词检索双路并行，用自研 RRF 融合后交由交叉编码器精排，"
     "提供 vector / hybrid / hybrid_rerank 三档可切换，支持消融对比。"),
    ("O3", "带引用溯源的流式问答",
     "答案中每个事实标注 [n] 引用，可回跳到真实文档名与页码；逐字流式输出；"
     "无命中、低相关、服务故障均有明确降级路径，不返回 500。"),
    ("O4", "手写 ReAct Agent 与工具集",
     "不依赖 LangChain 手写 ReAct 循环，注册 8 个可调用工具，具备工具路由、并行调用、"
     "错误恢复与会话记忆管理能力。"),
    ("O5", "全过程可观测与系统评测",
     "完整记录 Thought→Action→Observation 轨迹及 Token / 延迟 / 命中率指标并前端可视化；"
     "建立评测集与自动化评测脚本，产出评测报告与 Bad Case 分析。"),
]

STACK = [
    ("生成模型", "DeepSeek deepseek-chat", "OpenAI 兼容接口，中文科研问答表现稳定，成本低。"),
    ("向量化", "BAAI/bge-m3（SiliconFlow）", "1024 维、8192 上下文，中英文与长文本兼顾。"),
    ("重排序", "BAAI/bge-reranker-v2-m3", "交叉编码器，对 (query, passage) 联合编码后精排。"),
    ("向量数据库", "ChromaDB", "本地持久化，支持增量 upsert 与按 doc_id 删除。"),
    ("关键词检索", "rank_bm25", "纯 Python 实现，配合自研 RRF 完成双路融合。"),
    ("文本分块", "自研 fixed / recursive / semantic", "三种策略可切换，用于分块消融实验。"),
    ("PDF 解析", "PyMuPDF", "对比 PyPDF2、pdfplumber 后选定的最优方案。"),
    ("前端", "Streamlit", "单页三栏布局，原生支持流式输出与 Markdown 渲染。"),
    ("配置管理", "pydantic-settings", "环境变量 / .env / 默认值三级优先级，启动期即校验密钥。"),
    ("测试", "pytest", "228 个单元测试，含集成断言与 Streamlit AppTest 无头测试。"),
]

TEAM = [
    ("后端开发工程师", "Backend", "_________________",
     "模块一：文档处理与检索层；模块二：RAG 生成层",
     "多格式加载与三种分块策略；向量库与 BM25 双索引；自研 RRF 融合与重排序接入；"
     "Prompt 工程、流式生成、引用溯源、语义缓存与四级降级。", "30%"),
    ("Agent 开发工程师", "Agent Dev", "_________________",
     "模块三：Agent 决策层",
     "手写 ReAct 核心循环；8 个工具的设计与注册；规则路由与并行执行；"
     "超时 / 异常 / 死循环的错误恢复；会话隔离、滑动窗口与摘要压缩的记忆管理。", "25%"),
    ("前端开发工程师", "Frontend", "_________________",
     "模块四：系统集成与前端展示",
     "Streamlit 三栏界面与流式对话；推理轨迹树与指标面板可视化；文档管理与会话历史；"
     "全系统端到端联调与架构图绘制。", "25%"),
    ("测试与质量保证工程师", "QA", "_________________",
     "模块五：评测与交付",
     "评测集构建与 gold 标注自动校验；三档检索消融实验；Agent 决策指标评测；"
     "Bad Case 归因与迭代验证；技术设计文档、用户手册与评测报告撰写。", "20%"),
]

MODULES: dict[str, list[tuple[str, str, str]]] = {
    "m1": [
        ("多格式文档加载器（PDF/Word/TXT/MD）", "✔", "已实现",
         "ingest/loaders.py；PDF 用 PyMuPDF，对比 PyPDF2 与 pdfplumber 后选定。"),
        ("3 种以上文本分块策略", "✔", "已实现",
         "ingest/chunkers.py；fixed / recursive / semantic 三种，均支持 chunk_size 与 overlap。"),
        ("分块策略对比实验", "✔", "已实现",
         "5 种配置（recursive 128/256/512、fixed 256、semantic 256）详见《分块策略对比实验报告》。"),
        ("Embedding 模型接入与向量化存储", "✔", "已实现",
         "retrieval/vectorstore.py；bge-m3 + ChromaDB，doc_id 为文件内容 sha256，支持增量 upsert。"),
        ("混合检索（向量+BM25+RRF 融合）", "✔", "已实现",
         "retrieval/fusion.py 手写 RRF（k=60），融合向量与 BM25 两路排名。"),
        ("接入 Reranker 重排序", "✔", "已实现",
         "retrieval/reranker.py；bge-reranker-v2-m3 对 Top-20 精排取 Top-5，失败自动降级。"),
        ("检索质量评估（Hit@5 / MRR）", "✔", "已实现",
         "三档对比数据见 eval/reports/*.csv 与《评测报告》第 2 节。"),
        ("学术 PDF 版式优化（双栏/跨页表格/公式）", "—", "未实现",
         "当前依赖 PyMuPDF 默认阅读顺序，未专门处理双栏与跨页表格，已列入待改进项。"),
    ],
    "m2": [
        ("RAG 专用 Prompt 模板", "✔", "已实现",
         "generation/prompts.py；含系统角色、检索上下文、用户问题、输出格式四段结构。"),
        ("上下文智能拼接（排序与动态截断）", "✔", "已实现",
         "按相关性排序拼接，单块超预算截断，总预算 max_context_tokens=3000。"),
        ("引用溯源机制（答案标注文档名+页码）", "✔", "已实现",
         "generation/rag.py；[n] 标记映射到真实文档名与页码，越界编号由 validate_citations 过滤。"),
        ("流式输出（打字机效果）", "✔", "已实现",
         "RAG 流式生成产出 13 个事件；前端用 st.write_stream 逐字渲染。"),
        ("语义缓存机制", "✔", "已实现",
         "generation/cache.py；bge-m3 embedding 余弦相似度匹配，阈值 0.95/0.90/0.85。"),
        ("降级策略（无结果/低相关性/API失败）", "✔", "已实现",
         "generation/fallback.py 覆盖空知识库、低相关、重排序故障、LLM 异常四类，均返回 degraded 标记。"),
        ("RAG 请求日志记录（Token/耗时/命中率）", "✔", "已实现",
         "observability/trace.py 按 session 落 storage/logs/trace.jsonl，可观测页可聚合查看。"),
    ],
    "m3": [
        ("ReAct 核心循环手写实现", "✔", "已实现",
         "agent/react.py 约 520 行，Thought→Action→Observation 闭环，不依赖 LangChain。"),
        ("Agent System Prompt 设计", "✔", "已实现",
         "含角色定义、工具描述（运行时动态注入）、输出格式约束；新增工具无需改 Prompt 结构。"),
        ("工具集开发（标明几个工具）", "✔", "已实现（8 个）",
         "kb_search / paper_meta / paper_compare / keywords / summarize / calculator / "
         "current_time / web_search，冒烟测试断言“工具注册表 8 个”。"),
        ("工具选择路由", "✔", "已实现",
         "agent/router.py 规则路由，8 条单测覆盖；命中 direct_tool 直接执行并跳过整个 ReAct 循环。"),
        ("并行工具调用", "✔", "已实现",
         "agent/executor.py 用 ThreadPoolExecutor(max_workers=4) 并发执行互不依赖的工具。"),
        ("错误恢复（超时/异常/死循环）", "✔", "已实现",
         "工具级超时 tool_timeout_s=30；tenacity 指数退避重试 3 次；max_iterations=6 兜底强制终止。"),
        ("多轮对话记忆管理", "✔", "已实现",
         "agent/memory.py；session_id 会话隔离、memory_max_tokens=2000 滑动窗口、"
         "超过 summarize_after_turns=8 轮触发摘要压缩。"),
    ],
    "m4": [
        ("全系统端到端打通", "✔", "已实现",
         "scripts/smoke_test.py 10 步全链路自检退出码 0；Streamlit AppTest 无头执行异常 0。"),
        ("对话历史管理（新建/切换/删除会话）", "✔", "已实现",
         "streamlit_app.py 侧栏会话管理，历史按 session_id 持久化到 storage/sessions。"),
        ("可观测性前端展示（轨迹树+指标）", "✔", "已实现",
         "右栏展开 Thought/Action/Observation 轨迹树，并展示 Token、分级延迟、路由类型与降级标记。"),
        ("知识库管理面板（文档列表与状态）", "✔", "已实现",
         "左栏显示文件名、页数、切分块数、入库时间，支持删除与刷新。"),
        ("健康检查接口", "✔", "已实现",
         "检查 LLM 服务、向量库与本地索引连通性，以横幅形式给出绿/红状态。"),
        ("多用户并发测试", "—", "未实现",
         "Streamlit 单进程单用户模型，未做并发压测，见《评测报告》第 6 节局限说明。"),
        ("界面截图归档", "—", "部分实现",
         "功能已完成，截图需本地运行 streamlit run streamlit_app.py 后自行补入 4.4.2。"),
    ],
    "m5": [
        ("评测集（50+ 问答对，覆盖 4 类问题）", "—", "部分实现（22 条）",
         "eval/qa_set_demo.jsonl 现有 22 条（事实 10 / 对比 4 / 归纳 3 / 推理 5），未达 50+；"
         "gold 标注经 scripts/validate_qa_set.py 自动校验，36 个 gold 全部通过。"),
        ("检索性能评估（Hit@5 / MRR / Recall）", "✔", "已实现",
         "三档 × 两口径（片段级、文档级）指标已产出，原始数据见 eval/reports/。"),
        ("Agent 决策评估（工具准确率/推理轮次）", "✔", "已实现",
         "工具调用准确率 1.0000（22/22），平均推理轮次 1.909。"),
        ("Bad Case 分类与根因分析", "✔", "已实现",
         "《Bad Case 分析报告》记录 BC-01～BC-07 共 7 个真实缺陷，均含根因与修复前后实测数据。"),
        ("迭代优化前后对比", "✔", "已实现",
         "工具准确率 0.9091→1.0000；Prompt token 44,055→40,003（−9.2%）；单元测试 215→228。"),
        ("技术设计文档", "✔", "已实现", "docs/技术设计文档.md（模块接口、API、全部配置参数）。"),
        ("用户使用手册", "✔", "已实现", "docs/用户使用手册.md（安装部署、界面说明、10 条常见问题）。"),
        ("答案质量人工评分", "—", "未实现",
         "Mock 模式答案无意义，需真实 API 下人工 1-5 分双人评分，已列入待办。"),
        ("Docker 一键部署", "—", "未实现",
         "当前以虚拟环境方式交付，Docker 化列入待改进方向。"),
        ("演示 PPT", "—", "未实现", "尚无 PPT 交付物，列入待办。"),
    ],
}

IMPL_OVERVIEW = [
    ("模块一", "文档处理与检索层", "ingest/ + retrieval/", "★★★☆☆"),
    ("模块二", "RAG 生成层", "generation/", "★★★☆☆"),
    ("模块三", "Agent 决策层", "agent/", "★★★★★"),
    ("模块四", "系统集成与前端", "streamlit_app.py + observability/", "★★★★☆"),
    ("模块五", "评测与交付", "eval/ + scripts/ + docs/", "★★★★☆"),
]

IMPL_M1 = [
    "「加载」：ingest/loaders.py 统一入口，按扩展名分派到 PyMuPDF / python-docx / 纯文本三条路径，"
    "逐页保留页码（Page.page_no 从 1 起），页码是后续引用溯源的坐标基础。doc_id 取文件内容的 "
    "sha256 前 16 位，实现内容寻址——同一文件重复上传会被识别并跳过，不重复向量化、不重复计费。",
    "「分块」：ingest/chunkers.py 实现三种策略并可工厂化切换。fixed 按 token 数硬切；"
    "recursive 按「段落 → 换行 → 句号 → 空格」优先级递归下切；semantic 用句子向量断点切分，"
    "在未注入 embed_fn 时自动降级为 recursive 而不报错。每个 chunk 记录 "
    "chunk_id = f\"{doc_id}::{page_no}::{seq}\"，把来源坐标固化进索引，"
    "因此任一检索结果都能反查回原文页码。",
    "「索引」：向量侧用 ChromaDB 持久化，按 doc_id 支持增量 upsert 与整篇删除；"
    "关键词侧用 rank_bm25 维护内存索引并以 pickle 落盘。两套物理索引由 "
    "scripts/build_index.py 统一维护，--rebuild 会同时清空向量库与 BM25，"
    "并在日志中打印两边清掉的 chunk 数以便核对（该一致性检查源于一次真实的残留缺陷，见 BC-06）。",
    "「检索」：retrieval/fusion.py 手写 RRF，对向量与 BM25 两路排名按 1/(k+rank) 累加融合，"
    "k 取 60。融合后由 bge-reranker-v2-m3 交叉编码器对 Top-20 候选精排取 Top-5。"
    "三档模式（vector / hybrid / hybrid_rerank）通过统一接口暴露，便于消融对比。",
    "「评测口径「：分块消融必须在」文档级」口径下比较。因为 chunk 级 gold 与分块方式强绑定——"
    "一旦改动 chunk_size，chunk_id 中的 seq 全变，原标注立即失效。文档级以"
    "「是否召回正确文档中的任意片段」计分，与分块方式无关，是跨配置比较唯一公平的基准。",
]

IMPL_M2 = [
    "「Prompt 结构」：generation/prompts.py 把模板拆成四段——系统角色（只依据给定上下文作答、"
    "无法回答时明确说明而非编造）、检索上下文（带编号与来源标记）、用户问题、输出格式约束"
    "（事实后必须紧跟 [n] 引用）。检索片段以 [1]《文件名》第 N 页 的结构化文件头分隔，"
    "使模型能稳定地按编号引用。",
    "「动态截断」：拼接按相关性排序（重排序后的顺序），单块超过预算即截断，"
    "总预算由 max_context_tokens=3000 控制，并按 estimate_tokens() 预留输出空间——"
    "该函数对 CJK 按一字一 token、其余按「英文单词数 / 0.75」估算，宁可略微高估以避免"
    "超模型上限。",
    "「引用溯源」：答案中的 [n] 不是让模型自由发挥的装饰，而是由 _extract_citations 从工具 "
    "Observation 中按正则抽取、重编号后生成的 Citation 对象（含 filename / page_no / chunk_id / "
    "snippet），并做越界校验。前端把 [n] 渲染为可展开标记，点开即见原文片段与页码。",
    "「流式输出」：采用事件序列而非裸文本流（text / citation / done / error），"
    "使引用标记能在答案生成的对应位置动态插入，而不是等全部生成完再统一附加。",
    "「语义缓存」：generation/cache.py 以 bge-m3 embedding 的余弦相似度匹配历史问题，"
    "阈值刻意取高（默认 0.95）以抑制「语义相近但答案不同」的误命中；"
    "命中则直接返回缓存答案，省去完整的检索与生成开销。",
    "「四级降级」：generation/fallback.py 分别处理空知识库（明确提示未找到并回退为模型自身知识，"
    "答案上方加⚠️警示）、低相关性（低于 min_relevance=0.2 时提示并展示检索到的内容供用户判断）、"
    "重排序服务故障（退回上游 RRF 顺序）、LLM 调用异常（重试后返回友好错误与重试建议）。"
    "关键在于：「任何一环故障都不会让请求 500」，请求仍以 200 返回并携带 degraded 标记。",
    "「可观测」：observability/trace.py 为每次请求记录完整日志（问题、Top-k 文档、回答、"
    "分级耗时、prompt/completion token）并按 session 落盘为 JSONL；"
    "同时记录每次检索的 Top-1 置信度分数，用于观察分数分布。",
]

IMPL_M3 = [
    "「ReAct 循环」：agent/react.py 手写实现，循环体为「构造 Prompt → 调用 LLM → 解析 "
    "Thought/Action → 执行工具 → 把 Observation 回填为下一轮输入」。终止条件有二："
    "模型输出 final_answer 动作，或达到 max_iterations=6 强制终止并如实告知已达上限。"
    "整条链路不引入 LangChain，其 .tools 即工具注册表，便于单测直接替换。",
    "「System Prompt」：分静态与动态两部分。静态部分是角色定义、工具选择准则与输出格式约束；"
    "动态部分每轮注入当前可用工具清单（由 tools_description() 生成）与对话记忆摘要。"
    "这样新增工具只需注册，无需改动 Prompt 文本结构。",
    "「工具集」：agent/tools/ 下每个工具继承 Tool 基类，声明 name / description / 参数 Schema / "
    "超时；base.build_registry() 通过装饰器注册并在导入期发现。8 个工具为：知识库检索 "
    "（封装 RAG 引擎，Agent 的核心工具）、论文元信息、论文对比、关键词提取、结构化摘要、"
    "计算器、当前时间、联网搜索。",
    "「路由短路「：agent/router.py 用规则匹配识别」无需检索」的问题类型（纯算术、查时间等），"
    "命中即直接执行对应工具并返回，跳过整个 ReAct 循环。该短路放在 ReActAgent.run() 入口"
    "而非前端，使 CLI、评测脚本与前端「三个入口同时受益」；实测带来 9.2% 的 token 下降。"
    "中文口语运算符（乘以/除以/加上/减去）在抽取算式前会先归一化为符号。",
    "「并行执行」：agent/executor.py 在一轮中出现多个互不依赖的工具调用时，用 "
    "ThreadPoolExecutor(max_workers=4) 并发执行，再按原始顺序回填 Observation。",
    "「错误恢复」：三层防护——工具级超时（tool_timeout_s=30，超时记为失败 Observation 交还模型"
    "由模型决定跳过或换工具）、API 级重试（tenacity 指数退避，间隔 1/2/4 秒，最多 3 次）、"
    "循环级兜底（max_iterations 上限 + 无进展检测，触发即强制终止并提示用户）。"
    "所有异常都被收敛为 Observation 或 degraded 标记，不让异常穿出到前端。",
    "「记忆管理」：agent/memory.py 以 session_id 做会话隔离，每个会话独立维护消息列表；"
    "按 memory_max_tokens=2000 做滑动窗口裁剪（从最旧开始丢弃）；"
    "当轮次超过 summarize_after_turns=8 时，先把早期对话压缩成一段阶段性摘要，"
    "再用摘要替换原始历史，从而在控制 token 的同时保持语义连贯。",
]

IMPL_M4 = [
    "「端到端链路」：streamlit_app.py 作为唯一前端入口，把检索器、LLM、RAG 引擎与记忆"
    "注入到 Agent 组装入口 agent.build_agent()。重资源（Chroma 连接、BM25 索引）经 "
    "lru_cache 只构造一次——因为 Streamlit 每次交互都会重跑整个脚本，若在 import 期构造"
    "会把几秒的启动开销变成常态。",
    "「界面布局」：左栏为文档管理（多文件上传、解析与向量化进度条、文档列表、删除、刷新）；"
    "中栏为对话区（流式打字机输出、Markdown 渲染、可展开的引用标记）；"
    "右栏为可观测面板（轨迹树、Token 与分级延迟指标、路由类型、降级标记、健康状态横幅）。",
    "「轨迹可视化」：把 AgentStep 序列渲染成可展开的 Thought / Action / Observation 三色"
    "轨迹树，让「AI 为什么给出这个答案「与」AI 给出了什么答案」同样可见。"
    "答案不准确时，看轨迹通常能立刻区分是检索没找到还是模型理解错了。",
    "「会话管理」：支持新建、切换、删除会话，历史按 session_id 持久化，"
    "切换会话即切换记忆上下文与轨迹视图。",
    "「端到端验证「：不用」HTTP 200」当作通过标准——服务器返回 200 只说明 HTML 外壳吐出来了，"
    "脚本内部异常要到 websocket 渲染阶段才暴露。因此使用 Streamlit 官方无头测试框架 AppTest "
    "真正执行了一遍前端脚本：异常 0、标题 1 个、子标题 6 个、聊天输入 1 个、文件上传组件 1 个、"
    "警告与错误均为空。",
]

IMPL_M5 = [
    "「评测集与标注可信度」：eval/qa_set_demo.jsonl 的每条 gold 都带 verify_key"
    "（答案必然出现的字面片段），由 scripts/validate_qa_set.py 自动校验"
    "「gold chunk 真实存在于索引中「且」关键事实确实能在 gold 集合里找到」。"
    "构建过程中该校验器实际抓到 3 处标注错误（把不含该事实的 chunk 标成了 gold）并已修正——"
    "这说明人工核对若只走过场是会出错的，标注必须机器可验证。",
    "「检索评测」：scripts/eval_retrieval.py 同时支持片段级与文档级两种 gold 口径，"
    "输出 Hit@1/3/5、Recall@1/3/5、MRR 与降级率，并可跨模式批量对比。",
    "「Agent 评测」：scripts/eval_agent.py 逐题比对实际调用工具与 expected_tool，"
    "统计工具调用准确率、平均推理轮次、崩溃率、降级率、P95 延迟与 prompt/completion token 总量。",
    "「Bad Case 闭环「：本项目记录了 7 个」开发过程中真实观察到」的缺陷（BC-01～BC-07），"
    "而非「设想中可能出什么问题」的清单。每条均给出：现象（含原始输出）→ 复现方式 → 根因 → "
    "修复 → 修复前后实测数据。其中 5 条已修复并验证，2 条（重排序真实增益、分块策略结论）"
    "已定性但需真实 API 与更大语料复核；未观测到的类别也如实标注，不编造案例。",
    "「三条可迁移的教训」：（1）单元测试全绿 ≠ 功能接通——BC-01 与 BC-05 都是"
    "「模块自身测试通过、但集成层完全没接线或签名不匹配」，必须有集成层断言；"
    "（2）规格里的示例就是必须支持的测试用例——BC-02 直接源于规格示例"
    "「3.14乘以2.56」未被当作验收条件；（3）一个逻辑索引只要存在多个物理存储，"
    "任何重建/删除操作都必须覆盖全部副本，否则会出现不报错、不崩溃的静默数据污染。",
]

UI_DESC = [
    ("左栏 · 文档管理",
     "多文件上传（PDF / DOCX / TXT / MD）、逐文件解析与向量化进度条、"
     "已入库文档列表（文件名 / 页数 / 切分块数 / 入库时间）、单篇删除、目录刷新。"),
    ("中栏 · 对话区",
     "底部输入框提问，答案逐字流式输出；句中 [1][2] 为引用标记，可展开查看对应的"
     "文件名、页码与原文片段；支持 Markdown 渲染。"),
    ("右栏 · 可观测面板",
     "推理轨迹树（Thought → Action → Observation 完整链条）、Token 消耗与分级延迟、"
     "本轮路由类型（直接作答 / 检索 / 调用工具）、降级标记、以及顶部健康状态横幅。"),
    ("侧栏 · 会话管理",
     "新建会话、切换历史会话、删除会话；每个会话的记忆上下文与轨迹视图相互隔离。"),
]

MERMAID = (
    "graph TD\n"
    "    UI[\"展示层 Streamlit<br/>文档管理 | 对话+流式 | 轨迹树\"] --> AG\n"
    "    AG[\"调度层 手写 ReAct Agent<br/>react | router | executor | memory<br/>"
    "工具注册表 8 tools\"] --> GEN\n"
    "    GEN[\"生成层 RAG<br/>prompts | rag | cache | fallback\"] --> RET\n"
    "    RET[\"检索层<br/>vector + bm25 → 手写 RRF → Reranker\"] --> DATA\n"
    "    DATA[\"数据层<br/>loaders | chunkers | pipeline\"]\n"
    "    OBS[\"可观测层 trace + metrics（横切所有层）\"]\n"
    "    UI -.-> OBS\n"
    "    AG -.-> OBS\n"
    "    GEN -.-> OBS\n"
    "    RET -.-> OBS\n"
    "    DATA -.-> OBS"
)

INTERFACES = [
    ("ingest", "loaders.load_file(path) -> Document",
     "Document（pages:[Page]）", "PyMuPDF / python-docx"),
    ("ingest", "chunkers.get_chunker(strategy, **kw).split(doc) -> [Chunk]",
     "[Chunk]（含 doc_id / page_no / seq）", "core.schemas.estimate_tokens"),
    ("ingest", "pipeline.ingest_paths(paths) -> IngestReport",
     "IngestReport（成功/跳过/失败计数）", "loaders + chunkers + vectorstore + bm25"),
    ("retrieval", "vectorstore.VectorStore.add(chunks) / .search(q, k)",
     "[ScoredChunk]（source=\"vector\"）", "ChromaDB + bge-m3"),
    ("retrieval", "bm25_index.BM25Index.add(chunks) / .search(q, k)",
     "[ScoredChunk]（source=\"bm25\"）", "rank_bm25"),
    ("retrieval", "fusion.reciprocal_rank_fusion(lists, k=60) -> [ScoredChunk]",
     "[ScoredChunk]（source=\"rrf\"）", "无（纯算法）"),
    ("retrieval", "reranker.Reranker.rerank(q, candidates, top_n) -> [ScoredChunk]",
     "[ScoredChunk]（source=\"rerank\"）", "bge-reranker-v2-m3"),
    ("retrieval", "hybrid.HybridRetriever.retrieve(q, mode) -> RetrievalResult",
     "RetrievalResult（含 degraded 标记）", "上述四者编排"),
    ("generation", "rag.RAGEngine.answer(q, ...) -> AgentResult",
     "AgentResult（answer + citations + tokens）", "prompts + llm"),
    ("generation", "rag.RAGEngine.stream(q, ...) -> Iterator[事件]",
     "事件序列 text / citation / done / error", "prompts + llm"),
    ("generation", "cache.SemanticCache.get(q) / .put(q, answer)",
     "命中答案或 None", "bge-m3 embedding"),
    ("agent", "agent.build_agent(retriever, llm, memory, recorder) -> ReActAgent",
     "ReActAgent（.tools 为注册表）", "agent + generation + retrieval"),
    ("agent", "react.ReActAgent.run(question, session_id) -> AgentResult",
     "AgentResult（steps + route + degraded）", "tools + memory + llm"),
    ("agent", "tools.base.build_registry() -> dict[str, Tool]",
     "工具注册表（8 个工具）", "各工具模块"),
    ("agent", "memory.ConversationMemory.build_context(session_id) -> str",
     "供 Prompt 使用的历史上下文", "无（纯内存 + 落盘）"),
    ("observability", "trace.TraceRecorder.record(...) / metrics.aggregate()",
     "JSONL 轨迹与聚合指标", "storage/logs/trace.jsonl"),
]

DATAFLOWS = [
    ("① 入库流",
     "本地文件 → 解析（按扩展名分派）→ 逐页文本 → 分块（chunk_size / overlap）→ "
     "bge-m3 向量化 → ChromaDB upsert；同时分词 → BM25 索引落盘。",
     "doc_id = sha256(文件内容)[:16]，故重复上传被幂等跳过。"),
    ("② 检索流",
     "用户问题 → 向量 Top-K 与 BM25 Top-K 并行 → 手写 RRF 融合 → "
     "交叉编码器对 Top-20 精排取 Top-5 → [ScoredChunk]。",
     "三档模式在此处分叉：vector 只走向量，hybrid 加 RRF，hybrid_rerank 再加精排。"),
    ("③ 生成流",
     "Top-5 片段按相关性拼接（动态截断至 3000 token）→ RAG Prompt → LLM 流式生成 → "
     "抽取并校验 [n] 引用 → AgentResult。",
     "无有效片段时走 fallback：回退为模型自身知识并加⚠️警示，不编造引用。"),
    ("④ Agent 决策流",
     "问题 → 构造 Message（含记忆摘要）→ 规则路由判定；命中 direct_tool 则直接执行工具并返回，"
     "否则进入 ReAct 循环：LLM 推理 → 选择工具 → 执行 → Observation 回填 → 循环，"
     "直至 final_answer 或达到 max_iterations。",
     "每步产生一个 AgentStep（含 latency_ms 与 tokens），供轨迹树渲染与评测消费。"),
    ("⑤ 观测流",
     "各层在关键节点回调 TraceRecorder → 按 session_id 落盘 JSONL → "
     "metrics 聚合为 Token / 延迟 / 命中率 / 降级率指标。",
     "观测层横切所有层，但不参与数据流，也不改变任何层的返回值。"),
]

CONFIG_TABLE = [
    ("Provider 开关", "llm_provider / embedding_provider / rerank_provider", "deepseek / siliconflow",
     "设为 mock 可在无 API Key 时离线跑通全链路。"),
    ("生成", "deepseek_model", "deepseek-flash / deepseek-chat", "生成模型名。"),
    ("生成", "temperature / top_p", "0.3 / 0.95", "较低温度以抑制科研问答中的发挥。"),
    ("生成", "max_context_tokens / max_output_tokens", "3000 / 1500", "上下文与输出预算。"),
    ("向量化", "embedding_model / embed_batch_size", "BAAI/bge-m3 / 32", "向量化模型与批大小。"),
    ("重排序", "rerank_model / rerank_top_n", "BAAI/bge-reranker-v2-m3 / 5", "精排模型与保留条数。"),
    ("检索", "chunk_strategy / chunk_size / chunk_overlap", "recursive / 512 / 64", "分块策略与粒度。"),
    ("检索", "retrieve_top_k / rrf_k", "20 / 60", "召回条数与 RRF 平滑常数。"),
    ("检索", "min_relevance", "0.2", "低于此分数触发低相关性降级。"),
    ("Agent", "max_iterations / tool_timeout_s", "6 / 30.0", "循环上限与单工具超时。"),
    ("Agent", "parallel_max_workers / enable_router", "4 / True", "并行度与规则路由开关。"),
    ("记忆", "memory_max_tokens / summarize_after_turns", "2000 / 8", "滑动窗口预算与摘要触发轮次。"),
    ("缓存", "semantic_cache_enabled / cache_similarity_threshold", "True / 0.95",
     "阈值刻意偏高，以抑制语义相近但答案不同的误命中。"),
    ("存储", "storage_dir / data_dir", "storage / data", "索引、缓存、日志与原始文档目录。"),
]

CONFIG_RULES = [
    "加载优先级：构造参数 > 环境变量 > .env 文件 > config.py 默认值。"
    "所有可调参数集中在 config.py，业务代码不硬编码默认值。",
    ".env 以绝对路径加载（PROJECT_ROOT / \".env\"）。相对路径会相对于「当前工作目录」解析，"
    "从其他目录启动 streamlit 或脚本时会静默读不到配置。",
    "密钥校验发生在「配置构造期」而非首次请求时：缺少必要 key 会立刻抛出带修复指引的异常。"
    "刻意不做「延迟到第一次调用才失败」——那样用户会先看到一堆正常输出，"
    "然后在某个深埋的调用点撞上 401，排查成本高得多。",
    "派生路径（chroma_dir / bm25_dir / cache_dir / log_dir / session_dir / raw_dir）"
    "均由 storage_dir、data_dir 计算得出，避免路径散落在各处。",
]

SUMMARY = [
    "本项目完整走通了「文档入库 → 分块索引 → 混合检索 → 引用生成 → Agent 决策 → 前端可视化 → "
    "系统评测」的全链路，产出一个可交互、可复现、可观测的论文知识库问答系统，"
    "而非一个仅能跑通的演示脚本。",    "「工程规模」：73 个 Python 源文件、27 个测试文件；pytest 全量 228 个用例通过；"
    "10 步全链路冒烟自检退出码 0；Streamlit AppTest 无头执行异常 0。",
    "「核心能力已手写实现」：ReAct 循环（约 520 行，不依赖 LangChain）、RRF 倒数排名融合、"
    "三种分块策略、四级降级链路、语义缓存、会话记忆（滑动窗口 + 摘要压缩），"
    "共 8 个可调用工具。",
    "「关键实验结论」：混合检索相对朴素向量检索有大幅且稳定的提升——文档级 MRR 从 0.5542 "
    "提升到 0.9167（+65%），Hit@1 从 0.30 提升到 0.85；加入重排序后 MRR 进一步升至 0.9667、"
    "Hit@1 升至 0.95。该提升在 5 种分块配置下稳定复现（MRR 提升 +33%～+64%），"
    "混合检索的架构选择得到实验支持。",
    "「Agent 指标」：工具调用准确率 1.0000（22/22），平均推理轮次 1.909，崩溃率与降级率均为 0。"
    "路由短路带来 Prompt token 从 44,055 降至 40,003（−9.2%）。",
    "「方法论收获」：建立了「可机器校验的标注 → 自动化评测 → Bad Case 归因 → 修复后回归验证」"
    "的闭环。7 个缺陷中有 5 个完成修复并用实测数据验证，另 2 个已定性并明确标注了复核条件。",
]

DIFFICULTIES = [
    ("D1", "路由模块已实现却从未被调用",
     "工具调用准确率停在 0.9091，纯算术问题照走完整检索。根因是 agent/router.py 实现完整"
     "且 8 条单测全绿，但 ReActAgent.run() 从未调用它——只有前端用它做过空库判断，"
     "路由结果被丢弃。「单元测试全绿不等于功能接通」：模块级测试只能证明模块自身正确，"
     "证明不了它被调用。",
     "把 _direct_route() 接入 ReActAgent.run() 入口，命中时直接执行工具并返回，"
     "跳过整个 ReAct 循环。放在 run() 而非前端，使 CLI、评测脚本、前端三个入口同时受益。"
     "准确率 0.9091→1.0000，并意外带来 token 下降 9.2%。"),
    ("D2", "重排序在片段级口径下反而使检索变差",
     "片段级评测中 hybrid_rerank 的 MRR（0.7933）低于 hybrid（0.8167），与直觉相反。"
     "根因是当时使用的是 MockRerank——按词元重合度打分的单路匹配器，"
     "信息量低于融合了双路信号的 RRF，用它重排必然丢信息。",
     "判定为「打分器能力缺陷而非架构缺陷」，未改代码。反向结论本身有价值："
     "它证明了重排序必须用交叉编码器，不能用简单相似度函数凑数。"
     "真实 bge-reranker-v2-m3 的效果已列入真实 API 复核清单。"),
    ("D3", "评测集标注错误未被人工核对发现",
     "人工标注的 gold 中混入了 3 处错误——把不含该事实的 chunk 标成了正确答案来源，"
     "而人工复核时未察觉。",
     "构建 scripts/validate_qa_set.py，为每条 gold 增加 verify_key（答案必然出现的字面片段），"
     "自动断言「gold chunk 真实存在于索引中「且」关键事实确实能在 gold 集合里找到」。"
     "该校验器上线时即抓出全部 3 处错误。"),
    ("D4", "一个逻辑索引存在两套物理存储导致静默数据污染",
     "用 --rebuild 换 chunk_size 重建后，索引里残留了本应被删除的旧 chunk，"
     "检索会召回到不该存在的片段，且引用页码可能与当前分块方案不一致——"
     "不报错、不崩溃，极难察觉。根因是 --rebuild 只清了向量库，没清 BM25。",
     "为 BM25Index 增加 clear()，令 --rebuild 同时清空两套索引，"
     "并在日志中同时打印两边清掉的 chunk 数便于核对。连续用两种 chunk_size 验证，"
     "两边计数始终一致、无残留。"),
    ("D5", "测试之间因模块级全局状态互相污染",
     "tests/test_tools_kb.py 单独运行为 7 passed，与 tests/test_integration.py 同跑则 2 failed。"
     "根因是工具依赖用模块级全局变量注入，而 fixture 只重新注入了其中一个模块，"
     "另一个仍持有上个测试文件留下的空库。",
     "把 fixture 改为走「生产注入路径」 agent._inject_dependencies()，一次注入全部工具。"
     "既修正了隔离问题，又让生产注入函数本身被测试覆盖。"
     "结果：单独 7 passed、组合 13 passed、全量 228 passed。"),
    ("D6", "中文口语运算符不被路由识别",
     "「3.14 乘以 2.56 等于多少？」被判为需要检索。根因是算式抽取正则只认 + - * / 符号，"
     "而中文用户习惯写「乘以」，导致文本被切成不含完整运算符的片段。"
     "关键在于规格书举的例子恰好就是「3.14乘以2.56」——这是必须支持的主路径，不是边缘情况。",
     "抽算式前先把中文运算符词归一化为符号（长词优先匹配，避免「乘以「被」乘」拆掉），"
     "同时保留「片段内至少两个数字「的约束以防」加速度」这类词被误判。"
     "准确率 0.9545→1.0000，且纯数字与「论文1和论文2的区别」均不误判。"),
]

IMPROVEMENTS = [
    ("扩充评测集至 50+ 条", "高",
     "现为 22 条（事实 10 / 对比 4 / 归纳 3 / 推理 5），单样本权重达 4.5%，置信区间过宽。"
     "计划按作业要求收集 10-20 篇同方向论文，把语料扩到 200+ chunk，评测集补足 50+ 条。"),
    ("在真实 API 下复核全部指标", "高",
     "当前全部数字来自 Mock Provider：MockEmbedding 的语义能力为零、MockLLM 返回固定话术，"
     "因此本报告中的检索下界结论可参考，但「不能外推为系统真实性能」。"
     "需填入真实 key 后重跑三档检索与 Agent 评测，并特别验证 hybrid_rerank 是否在两个口径下"
     "都优于 hybrid。"),
    ("答案质量人工评分", "高",
     "面向正确性、完整性、引用准确性三个维度做 1-5 分双人评分并取均值。"
     "Mock 模式下答案内容无意义，此项只能在真实 API 下开展。"),
    ("学术 PDF 版式优化", "中",
     "当前依赖 PyMuPDF 默认阅读顺序，未处理双栏排版、跨页表格与公式保留，"
     "这直接影响双栏论文的召回质量。计划引入版面分析按栏切分并做表格结构还原。"),
    ("Docker 一键部署", "中",
     "当前以 Python 虚拟环境方式交付。计划补充 Dockerfile 与 docker-compose.yml，"
     "把依赖安装、环境变量注入与索引持久化卷一次性封装，降低复现成本。"),
    ("并发与性能优化", "中",
     "Streamlit 单进程单用户模型，未做多用户并发压测。计划引入多用户会话隔离与"
     "检索结果的缓存复用，并测量并发下的 P95 延迟。"),
    ("重排序与分块的正式结论", "中",
     "在扩充后的语料上重跑分块消融，补充 chunk_size=384 与 overlap 消融，"
     "确认最优配置后再给出正式的分块策略结论。"),
    ("演示 PPT 与界面截图归档", "低",
     "补齐演示 PPT 交付物，并在本地运行前端后补入 4.4.2 所需的界面截图。"),
]

TREE = """RAG+agent的论文知识库问答助手/
├── README.md                       项目总览与快速开始
├── requirements.txt                依赖清单（刻意保持 ASCII）
├── pytest.ini                      测试配置
├── .env.example                    配置模板
├── config.py                       ★ 全局配置中心（唯一参数来源）
├── streamlit_app.py                ★ 前端入口（三栏界面）
│
├── core/                           ★ 统一数据结构与外部服务抽象
│   ├── schemas.py                  层间接口契约（Document/Chunk/AgentResult…）
│   ├── providers.py                LLM / Embedding / Rerank（含 Mock 实现）
│   └── logging_utils.py            日志工厂
│
├── ingest/                         模块一：文档处理
│   ├── loaders.py                  多格式加载（PyMuPDF / python-docx / 纯文本）
│   ├── chunkers.py                 三种分块策略（fixed / recursive / semantic）
│   └── pipeline.py                 批量入库管线与状态追踪
│
├── retrieval/                      模块一：检索层
│   ├── vectorstore.py              ChromaDB 封装（增量 upsert / 按文档删除）
│   ├── bm25_index.py               BM25 关键词索引（支持 clear 与落盘）
│   ├── fusion.py                   ★ 手写 RRF 倒数排名融合
│   ├── reranker.py                 交叉编码器重排序（含降级）
│   └── hybrid.py                   混合检索编排（三档模式）
│
├── generation/                     模块二：RAG 生成层
│   ├── prompts.py                  RAG Prompt 模板与工具描述生成
│   ├── rag.py                      ★ 流式生成引擎与引用溯源
│   ├── cache.py                    语义缓存（embedding 相似度匹配）
│   └── fallback.py                 四级降级策略
│
├── agent/                          模块三：Agent 决策层
│   ├── react.py                    ★ 手写 ReAct 核心循环
│   ├── router.py                   规则路由（direct_tool 短路）
│   ├── executor.py                 并行工具执行
│   ├── memory.py                   会话隔离 / 滑动窗口 / 摘要压缩
│   └── tools/                      8 个工具模块
│       ├── base.py                 Tool 基类与注册表
│       ├── kb_search.py            知识库检索（核心工具）
│       ├── paper_meta.py           论文元信息提取
│       ├── paper_compare.py        论文对比
│       ├── keywords.py             关键词提取
│       ├── summarize.py            结构化摘要生成
│       ├── calculator.py           计算器
│       ├── current_time.py         当前时间
│       └── web_search.py           联网搜索
│
├── observability/                  可观测层（横切所有层）
│   ├── trace.py                    轨迹记录（JSONL 落盘）
│   └── metrics.py                  指标聚合（Token/延迟/命中率/降级率）
│
├── scripts/                        命令行工具
│   ├── build_index.py              建索引（支持 --rebuild / --dump-chunks）
│   ├── eval_retrieval.py           检索评测（两种 gold 口径）
│   ├── eval_agent.py               Agent 决策评测
│   ├── validate_qa_set.py          评测集 gold 标注自动校验
│   ├── smoke_test.py               全链路自检（无需 API Key）
│   └── export_docx.py              文档导出
│
├── tests/                          单元测试与集成测试（228 passed）
│   ├── test_loaders.py             test_chunkers.py     test_vectorstore.py
│   ├── test_bm25.py                test_fusion.py       test_hybrid.py
│   ├── test_reranker.py            test_rag.py          test_prompts.py
│   ├── test_cache.py               test_fallback.py     test_react.py
│   ├── test_router.py              test_executor.py     test_memory.py
│   ├── test_tools_kb.py            test_tools_pure.py   test_tools_web.py
│   ├── test_observability.py       test_integration.py  test_ui_upload.py
│   └── test_eval_metrics.py        test_schemas.py      test_providers.py
│
├── eval/                           评测集与评测产物
│   ├── qa_set_demo.jsonl           评测集（22 条，含 verify_key 标注）
│   ├── demo_corpus/                演示语料（ViT / Swin / DETR / MAE）
│   └── reports/                    评测 CSV 与图表
│
├── docs/                           项目文档
│   ├── 项目构建书.md / .docx        技术选型理由与五大模块搭建方案
│   ├── 技术设计文档.md              模块接口、API 与全部配置参数
│   ├── 分块策略对比实验报告.md       三种分块策略的检索效果对比
│   ├── 评测报告.md                  三档检索指标与 Agent 指标
│   ├── Bad Case分析报告.md          7 个真实缺陷的归因与修复验证
│   ├── 用户使用手册.md              面向非技术用户的安装与使用说明
│   └── architecture.png            系统架构图
│
├── data/                           原始文档（raw）与中间产物（processed）
└── storage/                        向量库 / BM25 索引 / 缓存 / 日志（已 gitignore）"""

ENV_A = """1. 前置条件
   - Python 3.10 或更高版本（本项目在 Python 3.12.7 上开发与验证）
   - Windows / macOS / Linux 均可
   - 两个云端 API Key（本项目不下载任何本地模型权重）"""

ENV_B = """2. 创建虚拟环境并安装依赖

   cd RAG+agent的论文知识库问答助手

   python -m venv .venv
   .venv\\Scripts\\activate          # Windows
   # source .venv/bin/activate       # macOS / Linux

   pip install -r requirements.txt"""

ENV_C = """3. 配置 API Key

   复制配置模板：
   copy .env.example .env            # Windows
   # cp .env.example .env            # macOS / Linux

   用编辑器打开 .env，填入两个 Key：

   DEEPSEEK_API_KEY=sk-...           # 生成，https://platform.deepseek.com/api_keys
   SILICONFLOW_API_KEY=sk-...        # 向量化 + 重排序，https://cloud.siliconflow.cn/account/ak

   为什么需要两家？DeepSeek 目前不提供 embedding 接口，因此向量化与重排序
   由 SiliconFlow 提供；其 bge 系列模型与作业指定方案一致，注册即有免费额度。"""

ENV_D = """4. 离线验收（无需任何 API Key）

   若暂时没有 Key，可在 .env 中开启 Mock 模式跑通除语义质量外的全部链路：

   LLM_PROVIDER=mock
   EMBEDDING_PROVIDER=mock
   RERANK_PROVIDER=mock

   然后执行全链路自检（退出码 0 表示通过）：

   python scripts/smoke_test.py
   python -m pytest tests/"""

ENV_E = """5. 启动应用

   streamlit run streamlit_app.py

   浏览器访问 http://localhost:8501，在左栏上传论文，待索引进度条走完即可提问。

6. 常用命令行操作（可选）

   python scripts/build_index.py data/raw --strategy recursive --chunk-size 512 --rebuild
   python scripts/validate_qa_set.py eval/qa_set_demo.jsonl
   python scripts/eval_retrieval.py --qa eval/qa_set_demo.jsonl --gold-level doc --modes vector hybrid hybrid_rerank
   python scripts/eval_agent.py --qa eval/qa_set_demo.jsonl"""


# ================================================================ 主流程

def main() -> None:
    if not BAK.exists():
        shutil.copy2(SRC, BAK)
        print(f"[backup] {BAK.name}")
    doc = Document(SRC)
    normalize_margins(doc)
    styles = grab_style_source(doc)
    B = Builder(doc, styles)

    # ---------- 封面 ----------
    cover = next(p for p in doc.paragraphs if "题    目" in p.text)
    set_text(cover, "题    目：  智能科研助理——基于 RAG+Agent 的论文知识库问答系统")
    print("[1] 封面题目已填")

    # ---------- 目录标题清理 ----------
    clear_text(find(doc, "h_toc"))

    # ================= 一、项目概述 =================
    h1 = find(doc, "h1")
    set_text(h1, "一、项目概述")

    el_bg = find(doc, "p_bg_title")._element
    set_text(Paragraph(el_bg, doc), "1.1 项目背景")
    hint = find(doc, "p_bg_hint")._element
    set_text(Paragraph(hint, doc), BACKGROUND)

    el_goal = find(doc, "p_goal")._element
    set_text(Paragraph(el_goal, doc), "1.2 项目目标")
    cur = B.paras(el_goal, [
        "围绕“让研究者用自己的论文库直接提问、并拿到可核查的答案”这一目标，本项目设定五项可验收目标："
    ])
    cur = B.table(cur, ["编号", "目标", "说明"],
                  [[k, v, d] for k, v, d in OBJECTIVES], fit([1.6, 3.6, 10.4]))

    el_stack = find(doc, "p_stack")._element
    set_text(Paragraph(el_stack, doc), "1.3 技术栈概览")
    cur = B.paras(el_stack, [
        "本项目不使用 LangChain / LlamaIndex / sentence-transformers / torch。"
        "前者的高层封装会遮蔽 ReAct 与 RRF 的实现细节，与“手写核心逻辑”的教学目标冲突；"
        "后者意味着本地模型部署，与“不下载模型权重、全部走云端 API”的工程约束冲突。"
        "因此仅在工程性较强的环节（HTTP 调用、向量库、前端框架）使用成熟库，"
        "而把 ReAct 循环、RRF 融合、分块策略、降级链路留作手写实现。"
    ])
    cur = B.table(cur, ["层次", "选型", "说明"],
                  [[a, b, c] for a, b, c in STACK], fit([2.4, 4.2, 9.0]))
    print("[2] 第一章完成")

    # ================= 二、团队分工 =================
    h2 = find(doc, "h2")
    set_text(h2, "二、团队分工说明")
    el_th = find(doc, "p_team_hint")._element
    set_text(Paragraph(el_th, doc),
             "小组共 4 人，按作业建议的角色模型分工，每人独立负责一个模块的端到端交付"
             "（设计 → 实现 → 自测 → 文档）。")
    el_ts = find(doc, "p_team_sum")._element
    set_text(Paragraph(el_ts, doc),
             "贡献比例合计 100%。模块之间通过 core/schemas.py 的数据结构通信，"
             "接口先行使四人可并行开发；每位成员同时也是其他模块的评审人，"
             "采用交叉验收而非各自验收，以降低“模块自测全绿、集成却未接通”的风险。")
    cur = B.table(el_ts, ["角色", "成员姓名", "主要负责模块", "核心职责", "贡献比例"],
                  [[role, name, mod, duty, pct] for role, _code, name, mod, duty, pct in TEAM],
                  fit([2.6, 2.4, 3.4, 6.6, 1.6]))
    print("[3] 第二章完成")

    # ================= 三、模块交付物清单 =================
    h3 = find(doc, "h3")
    set_text(h3, "三、模块交付物清单")
    hint3 = find(doc, "p_mod_hint")._element
    set_text(Paragraph(hint3, doc),
             "下表逐模块说明完成情况。“已实现”指功能可用且有测试或实测数据支撑；"
             "“部分实现”指已具备主体能力但未达作业要求的量化标准（如评测集条数）；"
             "“未实现”指尚未开展并已列入待改进方向。")
    HEADER6 = ["功能项", "已实现", "未实现", "部分实现", "说明"]
    WIDTHS6 = fit([4.7, 1.35, 1.35, 1.65, 6.95])
    for key, title in (("m1", "模块一：文档处理与检索层"),
                       ("m2", "模块二：RAG 生成层"),
                       ("m3", "模块三：Agent 决策层"),
                       ("m4", "模块四：系统集成与前端展示"),
                       ("m5", "模块五：评测与交付")):
        par = find(doc, key)
        set_text(par, title)
        for run in par.runs:
            run.font.bold = True
        el = par._element
        rows = []
        for item, _mark, status, note in MODULES[key]:
            rows.append([item,
                         "✔" if status.startswith("已实现") else "",
                         "✔" if status.startswith("未实现") else "",
                         "✔" if status.startswith("部分实现") else "",
                         note])
        B.table(el, HEADER6, rows, WIDTHS6)

    # 删除模块下原有的条目行（已被表格取代）
    delete_marks = ("引用溯源", "流式输出", "语义缓存", "降级策略", "RAG请求日志",
                    "ReAct核心循环", "System Prompt", "工具集开发", "错误恢复",
                    "多轮对话记忆", "全系统端到端", "对话历史管理", "评测集",
                    "Bad Case", "技术设计文档", "用户使用手册")
    prefixes = ("2.1", "2.2", "2.3", "2.4", "2.5", "3.1", "3.2", "3.3",
                "3.4", "3.5", "4.1", "4.2", "5.1", "5.2", "5.3", "5.4")
    for p in list(doc.paragraphs):
        t = p.text.strip()
        if (t.startswith(prefixes) and len(t) < 40
                and any(k in t for k in delete_marks)):
            delete(p)

    # 删除模板自带的「已实现/未实现/部分实现」空白勾选表（其表头无首列标题），
    # 该表已被 5 张模块交付物表取代。按「无表头首列 + 三列状态名」精确识别，
    # 避免误删同名表头的自建表。
    for tbl in list(doc.tables):
        if len(tbl.columns) != 4:
            continue
        head = [c.text.strip() for c in tbl.rows[0].cells]
        if head[0] == "" and head[1:] == ["已实现", "未实现", "部分实现"]:
            tbl._tbl.getparent().remove(tbl._tbl)
            print("    删除模板原有空白勾选表")
    print("[4] 第三章完成")

    # ================= 四、技术实现记录 =================
    h4 = find(doc, "h4")
    set_text(h4, "四、技术实现记录")
    el_ih = find(doc, "p_impl_hint")._element
    set_text(Paragraph(el_ih, doc),
             "本章说明各模块的「具体实现方式」——关键文件、核心算法选择、以及为什么这样做。"
             "所有结论均标注了可核查的出处（源码路径、测试或评测数据）。")
    cur = B.table(el_ih, ["模块", "名称", "主要代码", "工作量占比"],
                  [[a, b, c, d] for a, b, c, d in IMPL_OVERVIEW], fit([1.7, 4.2, 6.6, 2.5]))

    # 4.1
    el41 = find(doc, "impl41")._element
    set_text(Paragraph(el41, doc), "4.1 模块一：文档处理与检索层")
    B.paras(el41, IMPL_M1)
    # 4.2
    el42 = find(doc, "impl42")._element
    set_text(Paragraph(el42, doc), "4.2 模块二：RAG 生成层")
    B.paras(el42, IMPL_M2)
    # 4.3
    el43 = find(doc, "impl43")._element
    set_text(Paragraph(el43, doc), "4.3 模块三：Agent 决策层")
    B.paras(el43, IMPL_M3)
    # 4.4
    el44 = find(doc, "impl44")._element
    set_text(Paragraph(el44, doc), "4.4 模块四：系统集成与前端展示")
    B.paras(el44, IMPL_M4)

    # 4.4.1 系统架构图
    el441 = find(doc, "impl441")._element
    set_text(Paragraph(el441, doc), "4.4.1 系统架构图")
    hint_arch = find(doc, "p_arch_hint")._element
    txt_arch = find(doc, "p_arch_text")._element
    set_text(Paragraph(hint_arch, doc), "图 4-1 系统五层架构")
    arch_par = Paragraph(txt_arch, doc)
    set_text(arch_par, "")
    run = arch_par.add_run()
    run.add_picture(str(ARCH_PNG), width=Cm(15.5))
    cur = B.paras(txt_arch, [
        "图 4-1 说明：系统分为五层，自上而下依次为展示层（Streamlit 三栏界面）、"
        "调度层（手写 ReAct Agent）、生成层（RAG 流水线）、检索层（混合检索 + 重排序）、"
        "数据层（向量数据库与文档存储）。箭头上的文字是相邻两层之间「唯一」的数据契约，"
        "全部定义在 core/schemas.py 中。",
        "两个设计约束值得强调：其一，层间依赖「单向」（自上而下），"
        "下层不知道上层的存在，因此每层内部实现可独立替换——"
        "例如更换 Embedding 模型只需改 core/providers.py，检索层以上代码无需改动。"
        "其二，可观测层「横切」所有层，各层在关键节点回调 TraceRecorder，"
        "但可观测层不参与数据流，也不改变任何层的返回值，因此它可以被整体关闭而不影响功能。",
    ])

    # 4.4.2 界面功能截图
    el442 = find(doc, "impl442")._element
    set_text(Paragraph(el442, doc), "4.4.2 界面功能说明")
    hint_shot = find(doc, "p_shot_hint")._element
    set_text(Paragraph(hint_shot, doc),
             "说明：界面功能已全部实现并通过 Streamlit 官方无头测试框架 AppTest 的自动化验证"
             "（异常 0、子标题 6 个、聊天输入 1 个、文件上传组件 1 个、警告与错误均为空）。"
             "真实截图需在本地运行 streamlit run streamlit_app.py 后补入，下表先给出各区域的"
             "功能与交互说明。")
    B.table(hint_shot, ["界面区域", "功能说明"], UI_DESC, fit([3.6, 12.0]))

    # 4.5
    el45 = find(doc, "impl45")._element
    set_text(Paragraph(el45, doc), "4.5 模块五：评测与交付")
    B.paras(el45, IMPL_M5)
    print("[5] 第四章完成")

    # ================= 五、技术设计文档 =================
    h5 = find(doc, "h5")
    set_text(h5, "五、技术设计文档")

    el51 = find(doc, "d51")._element
    set_text(Paragraph(el51, doc), "5.1 系统架构图")
    hint51 = find(doc, "p_d51_hint")._element
    set_text(Paragraph(hint51, doc),
             "最终架构图见 4.4.1（图 4-1）。为便于二次编辑，此处同时给出等价的 Mermaid 源码，"
             "可直接粘贴到支持 Mermaid 的编辑器渲染：")
    mp = B.para(hint51, MERMAID, "body")
    B.paras(mp, [
        "架构要点：检索、生成、Agent 决策、前端展示四个模块之间只通过清晰定义的接口交互，"
        "每个模块可独立替换、升级和测试。例如若未来出现更优的 Embedding 模型，"
        "只需替换向量化模块，其余代码无需改动；若要把 ChromaDB 换成 FAISS，"
        "只需实现同样的 add / search 接口。这一约束既便于分阶段推进，"
        "也为后续迭代扩展奠定了架构基础。"
    ])

    el52 = find(doc, "d52")._element
    set_text(Paragraph(el52, doc), "5.2 模块接口定义")
    cur = B.paras(el52, [
        "下表列出各模块的对外接口。所有跨模块通信的数据结构统一定义在 core/schemas.py"
        "（Document / Chunk / ScoredChunk / Citation / Message / AgentStep / AgentResult / "
        "RetrievalResult），该模块只能被依赖、不得依赖任何其他项目模块。"
    ])
    cur = B.table(cur, ["模块", "接口", "输入 / 输出", "依赖"],
                  [[a, b, c, d] for a, b, c, d in INTERFACES], fit([2.1, 6.0, 4.4, 3.1]))

    el53 = find(doc, "d53")._element
    set_text(Paragraph(el53, doc), "5.3 核心数据流向")
    cur = B.paras(el53, [
        "系统共有五条主要数据流，其中①～③构成 RAG 主链路，④是 Agent 决策链路，"
        "⑤是可观测的旁路。"
    ])
    cur = B.table(cur, ["数据流", "路径", "关键设计"],
                  [[a, b, c] for a, b, c in DATAFLOWS], fit([2.1, 8.3, 5.2]))
    cur = B.paras(cur, [
        "贯穿五条流的统一原则：「每一环都有明确的降级出口」。检索无命中回退为模型自身知识并"
        "加警示；相关性低于阈值时提示用户并展示已检索内容；重排序服务故障时退回上游 RRF 顺序；"
        "LLM 调用异常经重试后返回友好错误。因此任何单点故障都不会让请求以 500 结束——"
        "仍以 200 返回，并通过 degraded 字段如实标记降级原因。"
    ])

    el54 = find(doc, "d54")._element
    set_text(Paragraph(el54, doc), "5.4 配置文件说明")
    cur = B.paras(el54, [
        "全部可配置参数集中在 config.py（pydantic-settings 的 Settings 类），"
        "业务代码不硬编码任何默认值。下表为分组说明，完整清单见 docs/技术设计文档.md。"
    ])
    cur = B.table(cur, ["分组", "参数", "默认值", "说明"],
                  [[a, b, c, d] for a, b, c, d in CONFIG_TABLE], fit([1.8, 4.6, 3.6, 5.6]))
    B.paras(cur, ["配置加载规则："] + CONFIG_RULES)
    print("[6] 第五章完成")

    # ================= 六、总结与展望 =================
    h6 = find(doc, "h6")
    set_text(h6, "六、总结与展望")

    el61 = find(doc, "s61")._element
    set_text(Paragraph(el61, doc), "6.1 项目总结")
    B.paras(el61, SUMMARY)

    el62 = find(doc, "s62")._element
    set_text(Paragraph(el62, doc), "6.2 主要困难与解决方案")
    cur = B.paras(el62, [
        "本节记录的 6 个问题都是开发过程中「真实遇到并定位到根因」的，"
        "每个都给出原因分析与处置方式，而非泛泛而谈的经验之谈。"
    ])
    cur = B.table(cur, ["编号", "问题", "原因分析", "解决方案与验证"],
                  [[a, b, c, d] for a, b, c, d in DIFFICULTIES], fit([1.3, 3.2, 5.8, 5.3]))

    el63 = find(doc, "s63")._element
    set_text(Paragraph(el63, doc), "6.3 待改进方向")
    cur = B.paras(el63, [
        "以下事项按优先级排列。其中标注“高”优先级的三项，直接关系到本报告数字的"
        "「可外推性」，应最优先处理。"
    ])
    cur = B.table(cur, ["改进方向", "优先级", "说明"],
                  [[a, b, c] for a, b, c in IMPROVEMENTS], fit([3.9, 1.5, 10.2]))
    print("[7] 第六章完成")

    # ================= 七、附录 =================
    h7 = find(doc, "h7")
    set_text(h7, "七、附录")

    elA = find(doc, "a_title")._element
    set_text(Paragraph(elA, doc), "附录A：项目代码目录结构")

    # 模板里有两个独立围栏块（附录A 目录树示例、附录B bash 示例），一次全部删掉。
    # 不能只删「第一对」，否则第二块的闭合围栏会被第一块的区间吞掉（实测结果）。
    n = remove_all_fenced_blocks(doc)
    if n:
        print(f"    删除模板围栏示例块 {n} 段")

    # 目录树逐行插入。缩进用半角空格模拟，折行后用悬挂缩进使续行对齐，
    # 避免续行跑到行首破坏树形观感。
    cur = elA
    for line in TREE.split("\n"):
        el = B.para(cur, line, "body")
        indent = len(line) - len(line.lstrip(" "))
        if indent:
            par = Paragraph(el, doc)
            par.paragraph_format.left_indent = Pt(indent * 6)
            par.paragraph_format.first_line_indent = Pt(-indent * 6)
        cur = el

    elB = find(doc, "b_title")._element
    set_text(Paragraph(elB, doc), "附录B：环境配置说明")
    hintB = find(doc, "b_hint")._element
    set_text(Paragraph(hintB, doc),
             "以下步骤在 Windows 上实测通过。「注意：本项目不使用 Ollama 等本地模型」，"
             "生成与向量化全部走云端 API，因此无需下载任何模型权重；"
             "若暂时没有 API Key，可切换到 Mock 模式完成离线验收（见第 4 步）。")

    cur = hintB
    for block in (ENV_A, ENV_B, ENV_C, ENV_D, ENV_E):
        for line in block.split("\n"):
            cur = B.para(cur, line, "body")
    print("[8] 附录完成")

    doc.save(SRC)
    print(f"\n已写入：{SRC}")


if __name__ == "__main__":
    main()
