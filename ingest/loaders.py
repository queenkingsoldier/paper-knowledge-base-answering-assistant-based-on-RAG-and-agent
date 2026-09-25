"""多格式文档加载器：文件 → `Document(pages=[Page, ...])`。

设计要点：

- **内容寻址的 `doc_id`**：`sha256(文件内容)[:16]`。同一份内容永远得到同一个 ID，
  因此增量索引天然幂等——重复上传直接按 ID 跳过，不需要额外的去重表。
  计算哈希时**分块读取**（1 MiB/次），几百 MB 的 PDF 也不会整个进内存。
- **异常收口**：PyMuPDF / python-docx 的任何解析异常都在这里被包成 `LoadError`。
  第三方库的堆栈不允许泄漏到前端——用户需要知道"哪个文件坏了"，
  不需要知道 `fitz` 内部哪一行抛了什么。
- **诚实标注**：抽不到的元信息留空字符串，不猜、不编造。DOCX 没有页的概念，
  就诚实标成 1 页，而不是伪造页码。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from core.schemas import Document, Page

#: 支持的文件后缀（小写，含点）。契约：与 UI 端 accept 列表保持一致。
SUPPORTED_EXTS: set[str] = {".pdf", ".docx", ".txt", ".md"}

#: 计算 sha256 时的读取块大小：1 MiB。
_READ_BLOCK = 1 << 20

#: 文件名长度上限，避免超长名字打爆文件系统与前端布局。
_MAX_FILENAME_LEN = 120

#: Windows 下非法的文件名字符，统一替换为空。
_ILLEGAL_FILENAME_CHARS = '<>:"|?*'

#: 元信息只在文档开头这几页里找——后面的页是正文，找标题没有意义。
_META_SCAN_PAGES = 2

_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_DOI_RE = re.compile(r"10\.\d{4,}/\S+")

_UNSUPPORTED_HINT = "支持的类型：" + ", ".join(sorted(SUPPORTED_EXTS))


class LoadError(Exception):
    """文档加载失败。

    对外只暴露这一种异常：调用方（入库管线 / UI）只需要知道
    「这个文件读不了，原因是 X」，不需要分辨底层是哪个库出的问题。
    """


# ---------------------------------------------------------------- 标识与文件名


def make_doc_id(path: Path) -> str:
    """内容寻址的文档 ID：文件内容 sha256 的前 16 位十六进制。

    分块读取（1 MiB/次），避免把大 PDF 整个读进内存再算哈希。
    内容相同 → ID 相同（幂等的基础）；内容变化 → ID 变化。
    """
    target = Path(path)
    digest = hashlib.sha256()
    try:
        with target.open("rb") as fh:
            while True:
                block = fh.read(_READ_BLOCK)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise LoadError(f"{target.name}: 读取失败（{exc}）") from exc
    return digest.hexdigest()[:16]


def safe_filename(name: str) -> str:
    """把任意文件名清洗成可安全落盘/展示的形式。

    去掉目录部分（同时兼容 `/` 与 `\\`，不依赖当前平台）、
    删除 Windows 非法字符、限制长度不超过 120 字符。
    """
    base = str(name or "").replace("\\", "/").rsplit("/", 1)[-1]
    base = "".join(ch for ch in base if ch not in _ILLEGAL_FILENAME_CHARS).strip()
    if not base:
        return "untitled"
    if len(base) > _MAX_FILENAME_LEN:
        suffix = Path(base).suffix
        keep = _MAX_FILENAME_LEN - len(suffix)
        base = base[:keep] + suffix if 0 < keep < len(base) else base[:_MAX_FILENAME_LEN]
    return base


# ------------------------------------------------------------------ 各格式解析


def _load_text_file(path: Path) -> list[Page]:
    """TXT / MD：整篇读成单页。

    `errors="replace"` 是刻意的：GBK 编码的老文件不该因为几个坏字节
    就让整篇文档入库失败，宁可留下替换字符也要把正文捞出来。
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []
    return [Page(text=text, page_no=1)]


def _load_pdf(path: Path) -> list[Page]:
    """PDF：PyMuPDF 逐页抽取，天然保留真实页码（引用溯源的基础）。

    `try/finally` 确保文件句柄一定会释放；纯空白的页不产生 `Page`
    （否则会在下游切出没有意义的空 chunk）。
    """
    try:
        import pymupdf as fitz  # PyMuPDF ≥1.24 的正式模块名（`fitz` 已弃用并会告警）
    except ImportError:
        try:
            import fitz  # 更老的版本只有这个名字
        except ImportError as exc:  # pragma: no cover - 依赖缺失属于环境问题
            raise LoadError("解析 PDF 需要 PyMuPDF，请先执行 pip install pymupdf") from exc

    doc = None
    pages: list[Page] = []
    try:
        doc = fitz.open(str(path))
        for index in range(doc.page_count):
            text = doc.load_page(index).get_text("text")
            if text and text.strip():
                pages.append(Page(text=text, page_no=index + 1))
    finally:
        if doc is not None:
            doc.close()
    return pages


def _load_docx(path: Path) -> list[Page]:
    """DOCX：读全部段落，合并为第 1 页。

    DOCX 格式本身不含分页信息（分页是渲染时才决定的），
    所以诚实地标成 1 页，而不是伪造一堆页码去骗引用溯源。
    """
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover - 依赖缺失属于环境问题
        raise LoadError("解析 DOCX 需要 python-docx，请先执行 pip install python-docx") from exc

    document = docx.Document(str(path))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text and p.text.strip()]
    if not paragraphs:
        return []
    return [Page(text="\n".join(paragraphs), page_no=1)]


_LOADERS: dict[str, Callable[[Path], list[Page]]] = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".txt": _load_text_file,
    ".md": _load_text_file,
}


# -------------------------------------------------------------------- 元信息


def _extract_meta(pages: list[Page]) -> dict:
    """尽力从开头几页抽取标题 / 年份 / DOI，抽不到就留空字符串。

    绝不编造：宁可 `meta` 是空的，也不要给下游一个看起来像真的假标题。
    """
    head = "\n".join(page.text for page in pages[:_META_SCAN_PAGES])
    meta: dict = {"title": "", "year": "", "doi": ""}

    lines = [line.strip() for line in head.splitlines()]
    lines = [line for line in lines if line]
    if lines:
        # 学术论文首页最长的那一行通常就是标题。
        meta["title"] = max(lines, key=len)

    year = _YEAR_RE.search(head)
    if year:
        meta["year"] = year.group(0)

    doi = _DOI_RE.search(head)
    if doi:
        meta["doi"] = doi.group(0).rstrip(".,;)]")

    return meta


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# -------------------------------------------------------------------- 主入口


def load_document(path: Path) -> Document:
    """把一个文件加载成 `Document`。

    任何失败（后缀不支持、文件不存在、解析库报错、磁盘读失败）
    都统一抛 `LoadError`，错误信息里带上文件名与原始原因。
    空文件是合法输入：返回 `pages == []` 而不是异常，由调用方决定跳过策略。
    """
    target = Path(path)
    ext = target.suffix.lower()

    if ext not in SUPPORTED_EXTS:
        raise LoadError(f"{target.name}: 不支持的文件类型「{ext or '无后缀'}」，{_UNSUPPORTED_HINT}")
    if not target.is_file():
        raise LoadError(f"{target.name}: 文件不存在或不是普通文件")

    try:
        doc_id = make_doc_id(target)
        pages = _LOADERS[ext](target)
        # 元信息抽取也放进同一个 try：它的输入来自第三方库的解析结果，
        # 收口要彻底，不能让任何解析衍生的异常漏成堆栈。
        meta = _extract_meta(pages)
    except LoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - 刻意的异常收口，见模块 docstring
        raise LoadError(f"{target.name}: 解析失败（{type(exc).__name__}: {exc}）") from exc

    return Document(
        doc_id=doc_id,
        filename=safe_filename(target.name),
        filetype=ext.lstrip("."),
        pages=pages,
        meta=meta,
        ingested_at=_now_iso(),
    )


__all__ = ["SUPPORTED_EXTS", "LoadError", "load_document", "make_doc_id", "safe_filename"]
