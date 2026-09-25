"""把《项目构建书》Markdown 转成 .docx。

Markdown 仍是唯一的事实来源；本脚本只是把它渲染成便于提交/打印的 docx。
支持：多级标题、表格、围栏代码块、有序/无序列表、引用块、粗体、行内代码、分隔线。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "项目构建书.md"
DST = ROOT / "docs" / "项目构建书.docx"

CN_FONT = "微软雅黑"
MONO_FONT = "Consolas"
BODY_SIZE = Pt(10.5)
MONO_SIZE = Pt(9)


def _set_cn_font(run, name: str) -> None:
    """python-docx 只设 latin 字体，中文要走 w:eastAsia，否则会退回宋体。"""
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), name)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)


_INLINE_RE = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")


def add_inline(paragraph, text: str) -> None:
    """处理行内 **粗体** 与 `代码`，其余按普通文本。"""
    for piece in _INLINE_RE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            run = paragraph.add_run(piece[2:-2])
            run.bold = True
            _set_cn_font(run, CN_FONT)
        elif piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
            run = paragraph.add_run(piece[1:-1])
            run.font.name = MONO_FONT
            run.font.size = MONO_SIZE
            run.font.color.rgb = RGBColor(0xC7, 0x25, 0x4E)
        else:
            run = paragraph.add_run(piece)
            _set_cn_font(run, CN_FONT)
            run.font.size = BODY_SIZE


def add_code_block(doc: Document, lines: list[str]) -> None:
    para = doc.add_paragraph()
    para.paragraph_format.left_indent = Pt(14)
    para.paragraph_format.space_before = Pt(4)
    para.paragraph_format.space_after = Pt(8)
    run = para.add_run("\n".join(lines))
    run.font.name = MONO_FONT
    run.font.size = MONO_SIZE
    # 淡灰底色，让代码块在 Word 里一眼可辨
    shd = para._element.get_or_add_pPr().makeelement(qn("w:shd"), {})
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), "F5F5F5")
    para._element.get_or_add_pPr().append(shd)


def add_table(doc: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    header, body = rows[0], rows[1:]
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, cell_text in enumerate(header):
        cell = table.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(cell_text)
        run.bold = True
        run.font.size = Pt(9.5)
        _set_cn_font(run, CN_FONT)

    for row in body:
        cells = table.add_row().cells
        for i in range(len(header)):
            value = row[i] if i < len(row) else ""
            cells[i].text = ""
            add_inline(cells[i].paragraphs[0], value)
            for r in cells[i].paragraphs[0].runs:
                r.font.size = Pt(9.5)

    doc.add_paragraph()


def convert(src: Path, dst: Path) -> int:
    text = src.read_text(encoding="utf-8")
    lines = text.splitlines()
    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = CN_FONT
    style.font.size = BODY_SIZE
    style.element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)

    i = 0
    tables = 0
    codes = 0
    while i < len(lines):
        line = lines[i]

        # 围栏代码块
        if line.strip().startswith("```"):
            i += 1
            buf: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            add_code_block(doc, buf)
            codes += 1
            i += 1
            continue

        # 表格：连续以 | 开头的行
        if line.strip().startswith("|") and i + 1 < len(lines) and re.match(
            r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]
        ):
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                raw = lines[i].strip()
                if not re.match(r"^\|[\s:|-]+\|$", raw):
                    cells = [c.strip() for c in raw.strip("|").split("|")]
                    rows.append(cells)
                i += 1
            add_table(doc, rows)
            tables += 1
            continue

        stripped = line.strip()

        if stripped in ("---", "***", "___"):
            doc.add_paragraph().add_run("─" * 40).font.color.rgb = RGBColor(0xBB, 0xBB, 0xBB)
            i += 1
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level = min(len(m.group(1)), 4)
            heading = doc.add_heading(level=level)
            add_inline(heading, m.group(2))
            for r in heading.runs:
                _set_cn_font(r, CN_FONT)
                r.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
            i += 1
            continue

        if stripped.startswith(">"):
            para = doc.add_paragraph()
            para.paragraph_format.left_indent = Pt(18)
            add_inline(para, stripped.lstrip("> ").strip())
            for r in para.runs:
                r.italic = True
                r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
            i += 1
            continue

        m = re.match(r"^[-*+]\s+(.*)$", stripped)
        if m:
            para = doc.add_paragraph(style="List Bullet")
            add_inline(para, m.group(1))
            i += 1
            continue

        m = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if m:
            para = doc.add_paragraph(style="List Number")
            add_inline(para, m.group(1))
            i += 1
            continue

        if not stripped:
            i += 1
            continue

        para = doc.add_paragraph()
        add_inline(para, stripped)
        i += 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    doc.save(dst)
    print(f"wrote {dst}")
    print(f"  headings/tables/code-blocks: {tables} tables, {codes} code blocks")
    return 0


if __name__ == "__main__":
    sys.exit(convert(SRC, DST))
