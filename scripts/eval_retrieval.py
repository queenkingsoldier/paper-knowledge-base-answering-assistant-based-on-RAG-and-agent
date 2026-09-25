"""检索层评测：三档检索模式的 Hit@k / MRR / Recall 对比。

用法：
    python scripts/eval_retrieval.py --qa eval/qa_set_demo.jsonl \
        --modes vector hybrid hybrid_rerank --ks 1 3 5

评测集格式（JSONL，每行一条）：
    {"id": "q001", "type": "factual", "question": "……",
     "gold_answer": "……", "gold_chunk_ids": ["a1b2c3::3::2"], "gold_doc_ids": ["a1b2c3"],
     "difficulty": "easy"}

注意：`gold_chunk_ids` 必须经过人工核对。未经核对的自动标注会让评测失去意义。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# 无显示环境（服务器 / CI）下必须在导入 pyplot 之前指定后端，
# 否则 matplotlib 会尝试连接显示器并报错。
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORT_DIR = PROJECT_ROOT / "eval" / "reports"

# 中文字体：Windows 上优先用雅黑，缺失时退回默认（图表中文会显示为方框但不影响数据）。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 指标（纯函数，可独立单测）
# ---------------------------------------------------------------------------


def hit_at_k(ranked_ids: list[str], gold: set[str], k: int) -> int:
    """前 k 个结果中是否命中至少一个 gold。命中返回 1，否则 0。"""
    if not gold:
        return 0
    return 1 if gold & set(ranked_ids[:k]) else 0


def reciprocal_rank(ranked_ids: list[str], gold: set[str]) -> float:
    """第一个命中位置的倒数排名；未命中返回 0。"""
    if not gold:
        return 0.0
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked_ids: list[str], gold: set[str], k: int) -> float:
    """前 k 个结果覆盖了多少比例的 gold。"""
    if not gold:
        return 0.0
    return len(gold & set(ranked_ids[:k])) / len(gold)


# ---------------------------------------------------------------------------
# 评测集加载
# ---------------------------------------------------------------------------


def load_qa(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} 不是合法 JSON: {exc}") from exc
    return items


# ---------------------------------------------------------------------------
# 评测主流程
# ---------------------------------------------------------------------------


def evaluate(retriever, qa_pairs: list[dict], modes: list[str], ks: list[int],
             gold_level: str = "chunk") -> pd.DataFrame:
    """遍历 问题 × 模式，产出指标汇总表。

    `gold_level` 决定用什么作为「正确答案」：

    - `chunk`（默认）：用 `gold_chunk_ids`。粒度细，能反映「是否精确定位到那一段」。
      但它**与分块方式绑定**——换了 chunk_size 或分块策略，chunk_id 全变，同一份
      评测集就不再适用。
    - `doc`：用 `gold_doc_ids`，即「是否召回了正确文档里的任意片段」。
      它**与分块方式无关**，因此是横向比较不同 chunk_size / 分块策略时唯一
      公平的口径。做分块消融实验必须用这个。
    """
    if gold_level not in ("chunk", "doc"):
        raise ValueError(f"gold_level 只能是 chunk 或 doc，收到 {gold_level!r}")

    max_k = max(ks)
    rows: list[dict] = []

    for mode in modes:
        acc: dict[int, list[float]] = {k: [] for k in ks}
        recalls: dict[int, list[float]] = {k: [] for k in ks}
        rrs: list[float] = []
        degraded = 0

        for item in qa_pairs:
            if gold_level == "chunk":
                gold = set(item.get("gold_chunk_ids") or [])
            else:
                gold = set(item.get("gold_doc_ids") or [])
            if not gold:
                # 该问题本就不该检索知识库（纯算术 / 问时间），跳过不计分
                continue

            result = retriever.search(item["question"], mode=mode, top_k=max_k)
            if result.degraded:
                degraded += 1
            if gold_level == "chunk":
                ranked = [sc.chunk.chunk_id for sc in result.chunks]
            else:
                # 文档级：按文档去重后排名，取每个文档首次出现的位置
                seen: list[str] = []
                for sc in result.chunks:
                    if sc.chunk.doc_id not in seen:
                        seen.append(sc.chunk.doc_id)
                ranked = seen
            for k in ks:
                acc[k].append(hit_at_k(ranked, gold, k))
                recalls[k].append(recall_at_k(ranked, gold, k))
            rrs.append(reciprocal_rank(ranked, gold))

        n = len(rrs) or 1
        row = {"mode": mode, "n": len(rrs)}
        for k in ks:
            row[f"hit@{k}"] = round(sum(acc[k]) / n, 4)
        for k in ks:
            row[f"recall@{k}"] = round(sum(recalls[k]) / n, 4)
        row["mrr"] = round(sum(rrs) / n, 4)
        row["degraded_rate"] = round(degraded / n, 4)
        rows.append(row)

    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, ks: list[int], out_png: Path) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    metrics = [f"hit@{k}" for k in ks] + ["mrr"]
    df.set_index("mode")[metrics].plot(kind="bar", ax=axes[0], rot=0)
    axes[0].set_title("检索质量对比")
    axes[0].set_ylabel("分数")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", alpha=0.3)

    rec = [f"recall@{k}" for k in ks]
    df.set_index("mode")[rec].plot(kind="bar", ax=axes[1], rot=0)
    axes[1].set_title("Recall 对比")
    axes[1].set_ylabel("分数")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检索层三档对比评测")
    parser.add_argument("--qa", default="eval/qa_set_demo.jsonl", help="评测集 JSONL 路径")
    parser.add_argument("--modes", nargs="+", default=["vector", "hybrid", "hybrid_rerank"])
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--gold-level", choices=["chunk", "doc"], default="chunk",
                        help="chunk=用 gold_chunk_ids 精确匹配；"
                             "doc=用 gold_doc_ids 文档级匹配。"
                             "比较不同分块参数时必须用 doc，否则 chunk_id 变了不可比。")
    parser.add_argument("--out", default=None, help="CSV 输出路径")
    args = parser.parse_args(argv)

    qa_path = Path(args.qa)
    if not qa_path.is_absolute():
        qa_path = PROJECT_ROOT / qa_path
    if not qa_path.exists():
        print(f"[FAIL] 评测集不存在: {qa_path}")
        return 1

    qa_pairs = load_qa(qa_path)
    if not qa_pairs:
        print(f"[FAIL] 评测集为空: {qa_path}")
        return 1

    from retrieval.hybrid import build_retriever

    retriever = build_retriever()
    if retriever.vectorstore.count() == 0 and retriever.bm25.count() == 0:
        print("[FAIL] 知识库为空，请先运行 python scripts/build_index.py <目录>")
        return 1

    df = evaluate(retriever, qa_pairs, args.modes, args.ks, gold_level=args.gold_level)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = Path(args.out) if args.out else REPORT_DIR / f"retrieval_{stamp}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_comparison(df, args.ks, REPORT_DIR / f"retrieval_compare_{stamp}.png")

    print(f"\n评测集: {qa_path}  (共 {len(qa_pairs)} 条)")
    print(f"评分口径: {'chunk 级（gold_chunk_ids）' if args.gold_level == 'chunk' else 'doc 级（gold_doc_ids，与分块无关）'}")
    print(df.to_string(index=False))
    print(f"\n已写出: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
