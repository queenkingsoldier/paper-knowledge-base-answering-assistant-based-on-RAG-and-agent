"""批量建索引 CLI。

用法：
    python scripts/build_index.py eval/demo_corpus          # 索引一个目录
    python scripts/build_index.py data/raw --dump-chunks    # 索引并导出 chunk 清单
    python scripts/build_index.py data/raw --strategy semantic --chunk-size 256
    python scripts/build_index.py data/raw --rebuild        # 清空后重建

`--dump-chunks` 会把每个 chunk 的 chunk_id / 文件名 / 页码 / 前 80 字写入
    eval/reports/chunks_<时间戳>.jsonl
该清单是编写评测集 gold_chunk_ids 的依据 —— 评测集的 gold 必须从真实产出的
chunk_id 里挑，不能凭想象填写。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}
REPORT_DIR = PROJECT_ROOT / "eval" / "reports"


def collect_paths(raw: list[str]) -> list[Path]:
    """把命令行参数展开成文件列表（目录会被递归扫描）。"""
    out: list[Path] = []
    for item in raw:
        p = Path(item)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_dir():
            out.extend(sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED))
        else:
            out.append(p)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="批量文档入库与索引构建")
    parser.add_argument("paths", nargs="*", default=["data/raw"],
                        help="文件或目录（默认 data/raw）")
    parser.add_argument("--strategy", default=None,
                        choices=["fixed", "recursive", "semantic"], help="分块策略")
    parser.add_argument("--chunk-size", type=int, default=None, help="分块大小（token 估算）")
    parser.add_argument("--rebuild", action="store_true", help="先清空索引再重建")
    parser.add_argument("--dump-chunks", action="store_true", help="导出 chunk 清单供评测集标注")
    args = parser.parse_args(argv)

    from config import settings
    from ingest.chunkers import get_chunker
    from ingest.pipeline import IngestPipeline
    from retrieval.hybrid import build_retriever

    paths = collect_paths(args.paths)
    if not paths:
        print(f"[FAIL] 未找到可索引的文件（支持 {'/'.join(sorted(SUPPORTED))}）")
        return 1

    retriever = build_retriever()

    if args.rebuild:
        # 必须同时清空向量库与关键词索引。只清向量库的话，换了分块参数后
        # 旧 chunk 中 chunk_id 未碰撞的条目会残留在 BM25 里，
        # 造成「检索到已不存在的片段」。
        removed = 0
        for doc in retriever.vectorstore.list_documents():
            removed += retriever.vectorstore.delete_document(doc["doc_id"])
        dropped = retriever.bm25.clear()
        retriever.bm25.save()
        print(f"[重建] 已清空向量库（{removed} 个 chunk）与 BM25 索引（{dropped} 个 chunk）")

    strategy = args.strategy or settings.chunk_strategy
    chunk_size = args.chunk_size or settings.chunk_size
    chunker = get_chunker(strategy, chunk_size=chunk_size, overlap=settings.chunk_overlap)

    print(f"[索引] 策略={strategy}  chunk_size={chunk_size}  文件数={len(paths)}")

    last = {"line": ""}

    def on_progress(name: str, frac: float) -> None:
        line = f"  [{frac * 100:5.1f}%] {name or '完成'}"
        print(" " * len(last["line"]) + "\r", end="")
        print(line, end="", flush=True)
        last["line"] = line

    pipeline = IngestPipeline(retriever=retriever, chunker=chunker, data_dir=settings.raw_dir)
    t0 = time.time()
    report = pipeline.ingest_paths(paths, progress_cb=on_progress)
    print()

    print("\n===== 入库结果 =====")
    print(f"  成功   : {report.files_ok}")
    print(f"  跳过   : {report.files_skipped}  (内容未变，已幂等跳过)")
    print(f"  失败   : {report.files_failed}")
    print(f"  新增块 : {report.chunks_added}")
    print(f"  耗时   : {report.elapsed_ms} ms")
    print(f"  库中块 : {retriever.vectorstore.count()}  (向量) / {retriever.bm25.count()}  (BM25)")

    if report.errors:
        print("\n  --- 失败明细 ---")
        for e in report.errors:
            print(f"    {e['filename']}: {e['error']}")

    if args.dump_chunks:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = REPORT_DIR / f"chunks_{stamp}.jsonl"
        docs = {d["doc_id"]: d["filename"] for d in retriever.vectorstore.list_documents()}
        n = 0
        with out.open("w", encoding="utf-8") as f:
            for doc_id, filename in docs.items():
                for row in retriever.vectorstore.iter_chunks(doc_id):
                    f.write(json.dumps({
                        "chunk_id": row["chunk_id"],
                        "filename": filename,
                        "page_no": row["page_no"],
                        "preview": row["text"][:80].replace("\n", " "),
                    }, ensure_ascii=False) + "\n")
                    n += 1
        print(f"\n  chunk 清单已导出（{n} 条）: {out}")
        print("  提示：评测集的 gold_chunk_ids 请从这个清单里挑选，不要凭想象填写。")

    return 0 if report.files_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
