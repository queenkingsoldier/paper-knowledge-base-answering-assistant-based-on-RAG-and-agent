"""批量入库管线：一批文件路径 → 向量库 + BM25 索引，并回报可展示的统计。

三件事必须在这里做对（都是 Review Focus #3 的「上传同一文件两次 / 上传损坏的 PDF」）：

1. **幂等**：`doc_id` 是内容寻址的，重复上传同一份内容得到同一个 ID，
   已入库的整篇跳过，绝不重复向量化（重复上传是用户的常规操作，不是异常）。
2. **单文件失败不中断整批**：一份损坏的 PDF 只能让它自己进 `IngestReport.errors`，
   其余文件必须照常入库——「批量上传 10 篇坏了 1 篇，另外 9 篇也没进去」
   是最让人恼火的失败模式。
3. **空内容不算失败**：空文件、只有页眉页脚的文件跳过即可，用户没做错什么。

失败重试的边界：本管线**不做**文件级重试。`load_document` 把所有解析错误
统一收口成 `LoadError`，其中「损坏的 PDF」「文件不存在」重试多少次都不会变好，
盲目重试只会让用户多等 3 秒才看到那条本来就确定的错误。真正需要重试的
外部 API 调用（embedding / rerank）由 `core/providers.py` 的 `_post_with_retry`
负责，已经是「最多 3 次、间隔 1s/2s/4s」的指数退避。

依赖方向：ingest → core（`retriever` 是鸭子类型，只用到它的 `.vectorstore`
与 `.bm25`，因此本模块不导入 retrieval，运行时依赖方向仍然单向）。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config import settings as default_settings
from core.logging_utils import get_logger
from core.schemas import Chunk
from ingest.chunkers import BaseChunker, get_chunker, strip_boilerplate
from ingest.loaders import load_document

if TYPE_CHECKING:  # pragma: no cover - 仅供类型标注，不在运行期导入
    from retrieval.hybrid import HybridRetriever

logger = get_logger(__name__)


@dataclass
class IngestReport:
    """一次批量入库的结果快照，供 UI 摘要与脚本统计直接消费。

    计数口径：`files_ok + files_skipped + files_failed == 本次传入的文件数`。
    「跳过」包含两种正常情况——已入库过（幂等）与没有可入库内容（空文件）。
    """

    files_ok: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    chunks_added: int = 0
    errors: list[dict] = field(default_factory=list)
    elapsed_ms: int = 0


class IngestPipeline:
    """把文件批量送进知识库：加载 → 去页眉页脚 → 分块 → 向量库 + BM25。"""

    def __init__(
        self,
        retriever: "HybridRetriever",
        chunker: BaseChunker | None = None,
        data_dir: Path | None = None,
    ) -> None:
        self.retriever = retriever
        # 缺省值一律来自 config，不在业务代码里硬编码（Global Constraints）。
        self.chunker: BaseChunker = chunker if chunker is not None else get_chunker()
        self.data_dir = Path(data_dir) if data_dir is not None else Path(default_settings.raw_dir)
        # 原始文档根目录（UI 上传落盘处）：确保存在，相对路径也以它为基准解释。
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # 用属性而不是在 __init__ 里缓存：`retriever.bm25` 可能被替换
    #（重建/清空索引），属性每次都读到当下的那个实例。
    @property
    def vectorstore(self) -> Any:
        return self.retriever.vectorstore

    @property
    def bm25(self) -> Any:
        return self.retriever.bm25

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def ingest_paths(
        self,
        paths: Sequence[Path],
        progress_cb: Callable[[str, float], None] | None = None,
    ) -> IngestReport:
        """入库一批文件，返回统计；任何单文件失败都不会中断整批。"""
        items = [Path(p) for p in (paths or [])]
        report = IngestReport()
        started = time.perf_counter()
        total = len(items)

        for index, raw_path in enumerate(items):
            path = self._resolve(raw_path)
            self._notify(progress_cb, path.name, (index / total) if total else 0.0)
            try:
                skipped, added = self._ingest_one(path)
            except Exception as exc:  # noqa: BLE001 - 单文件失败绝不中断整批（Review Focus #3）
                logger.warning("入库失败，已记录并继续处理其余文件 %s: %s", path, exc)
                report.files_failed += 1
                report.errors.append({"filename": path.name, "error": str(exc)})
                continue

            if skipped:
                report.files_skipped += 1
            else:
                report.files_ok += 1
            report.chunks_added += added

        # 只有真的写进了新 chunk 才落盘 BM25：纯跳过的一批不该产生写操作。
        if report.chunks_added > 0:
            self._save_bm25()

        # 结束哨兵：前端据此把进度条推到 100% 并收起转圈。
        self._notify(progress_cb, "", 1.0)
        report.elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "入库完成：成功 %d / 跳过 %d / 失败 %d，新增 %d 个 chunk，耗时 %d ms",
            report.files_ok, report.files_skipped, report.files_failed,
            report.chunks_added, report.elapsed_ms,
        )
        return report

    # ------------------------------------------------------------------
    # 单文件
    # ------------------------------------------------------------------
    def _ingest_one(self, path: Path) -> tuple[bool, int]:
        """处理单个文件，返回 `(是否跳过, 本次写入的 chunk 数)`。

        失败一律**抛异常**，由 `ingest_paths` 统一记入 `errors`——
        失败只有一个出口，不会出现「一半记 error、一半静默返回」的分裂。
        """
        # 解析失败（损坏 PDF / 文件不存在 / 后缀不支持）抛 LoadError。
        doc = load_document(path)
        # 先剔页眉页脚再分块：否则每页重复的页眉会变成一堆高相似度噪声 chunk。
        doc = replace(doc, pages=strip_boilerplate(doc.pages))
        chunks = self.chunker.split(doc)

        if self._already_indexed(doc.doc_id, chunks):
            logger.info("文档已入库，跳过重复向量化: %s（doc_id=%s）", doc.filename, doc.doc_id)
            return True, 0

        if not chunks:
            # 空文件 / 抽不出正文 / 整篇都是页眉页脚：跳过，且**不计失败**。
            logger.info("文件没有可入库内容，已跳过: %s", doc.filename)
            return True, 0

        # 先向量库后 BM25：向量库是权威副本。中途失败时两库不一致，
        # 下一次入库会因 `_already_indexed` 判否而重新处理，自动对齐（自愈）。
        written = self.vectorstore.upsert(chunks)
        self.bm25.add(chunks)
        logger.info("入库成功 %s：%d 个 chunk（doc_id=%s）", doc.filename, written, doc.doc_id)
        return False, written

    def _already_indexed(self, doc_id: str, chunks: list[Chunk]) -> bool:
        """`doc_id` 已在**两个**索引里才算已入库 → 整篇跳过，不重复向量化。

        要求两路同时命中：只命中向量库（例如 BM25 索引文件被删过）或只命中
        BM25（Chroma 目录被删过）时都走重新入库，把两库重新对齐。这比「只看
        向量库」多付一次 add 的代价，换来的是两个索引不会长期不一致——
        否则混合检索会静默退化成单路，且没有任何地方会报出这件事。
        """
        if not chunks:
            return False
        if not self.vectorstore.document_exists(doc_id):
            return False
        return any(chunk.doc_id == doc_id for chunk in self.bm25.chunks.values())

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _resolve(self, path: Path) -> Path:
        """绝对路径原样使用；相对路径优先按 `data_dir`（原始文档根目录）解释。"""
        if path.is_absolute() or path.exists():
            return path
        candidate = self.data_dir / path
        return candidate if candidate.exists() else path

    def _save_bm25(self) -> None:
        """落盘 BM25 索引；保存失败只告警，不把整批入库判为失败。

        内存里的索引依然可用，本次会话的检索不受影响；磁盘问题该记日志，
        但不该推翻「这些文件确实已经进了知识库」这个事实。
        """
        try:
            self.bm25.save()
        except Exception as exc:  # noqa: BLE001
            logger.warning("BM25 索引保存失败（内存索引仍然可用）: %s", exc)

    @staticmethod
    def _notify(
        progress_cb: Callable[[str, float], None] | None,
        name: str,
        fraction: float,
    ) -> None:
        """回调进度。UI 回调自己抛异常不该中断入库。"""
        if progress_cb is None:
            return
        try:
            progress_cb(name, fraction)
        except Exception as exc:  # noqa: BLE001
            logger.warning("进度回调抛出异常，已忽略: %s", exc)


__all__ = ["IngestPipeline", "IngestReport"]
