"""Agent 决策层评测：工具调用准确率、平均推理轮次、延迟与 token 消耗。

用法：
    python scripts/eval_agent.py --qa eval/qa_set_demo.jsonl
    python scripts/eval_agent.py --qa eval/qa_set.jsonl --limit 20 --out eval/reports/agent.csv

评测集里可选字段 `expected_tool`：声明该问题「期望 Agent 首选哪个工具」。
- 写了 `expected_tool` 的条目参与「工具调用准确率」统计；
- 没写的只参与轮次 / 延迟 / token 统计。
`expected_tool` 取 `"__none__"` 表示期望「不调用任何工具、直接作答」。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境必须在导入 pyplot 前指定后端

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORT_DIR = PROJECT_ROOT / "eval" / "reports"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

NO_TOOL = "__none__"


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


def first_tool(steps) -> str:
    """返回轨迹中第一个真正执行的工具名；全程没调工具则返回 NO_TOOL。"""
    for s in steps:
        if s.action and s.action != "Finish":
            return s.action
    return NO_TOOL


def evaluate_agent(agent, qa_pairs: list[dict]) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    for item in qa_pairs:
        q = item["question"]
        t0 = time.time()
        try:
            result = agent.run(q)
            error = ""
        except Exception as exc:  # noqa: BLE001 - 评测要如实记录崩溃而不是自己挂掉
            rows.append({
                "id": item.get("id", ""), "type": item.get("type", ""),
                "question": q[:40], "expected_tool": item.get("expected_tool", ""),
                "actual_tool": "<CRASH>", "tool_correct": 0, "steps": 0,
                "degraded": "exception", "latency_ms": int((time.time() - t0) * 1000),
                "prompt_tokens": 0, "completion_tokens": 0, "answer_len": 0,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        actual = first_tool(result.steps)
        expected = item.get("expected_tool")
        correct = "" if expected is None else int(actual == expected)
        rows.append({
            "id": item.get("id", ""),
            "type": item.get("type", ""),
            "question": q[:40],
            "expected_tool": expected if expected is not None else "",
            "actual_tool": actual,
            "tool_correct": correct,
            "steps": len(result.steps),
            "degraded": result.degraded or "",
            "latency_ms": result.latency_ms.get("total", int((time.time() - t0) * 1000)),
            "prompt_tokens": result.tokens.get("prompt", 0),
            "completion_tokens": result.tokens.get("completion", 0),
            "answer_len": len(result.answer),
            "error": error,
        })

    df = pd.DataFrame(rows)

    judged = df[df["tool_correct"] != ""] if not df.empty else df
    summary = {
        "n": len(df),
        "tool_accuracy": round(judged["tool_correct"].mean(), 4) if len(judged) else None,
        "tool_judged_n": len(judged),
        "avg_steps": round(df["steps"].mean(), 3) if len(df) else 0,
        "max_steps": int(df["steps"].max()) if len(df) else 0,
        "crash_rate": round((df["actual_tool"] == "<CRASH>").mean(), 4) if len(df) else 0,
        "degraded_rate": round((df["degraded"] != "").mean(), 4) if len(df) else 0,
        "avg_latency_ms": int(df["latency_ms"].mean()) if len(df) else 0,
        "p95_latency_ms": int(df["latency_ms"].quantile(0.95)) if len(df) > 1 else 0,
        "total_prompt_tokens": int(df["prompt_tokens"].sum()) if len(df) else 0,
        "total_completion_tokens": int(df["completion_tokens"].sum()) if len(df) else 0,
    }
    return df, summary


def plot_summary(df: pd.DataFrame, summary: dict, out_png: Path) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))

    tool_counts = df["actual_tool"].value_counts()
    tool_counts.plot(kind="bar", ax=axes[0], rot=30)
    axes[0].set_title("实际工具调用分布")
    axes[0].set_ylabel("次数")
    axes[0].grid(axis="y", alpha=0.3)

    df["steps"].value_counts().sort_index().plot(kind="bar", ax=axes[1], rot=0)
    axes[1].set_title(f"推理轮次分布（均值 {summary['avg_steps']}）")
    axes[1].set_xlabel("轮次")
    axes[1].set_ylabel("问题数")
    axes[1].grid(axis="y", alpha=0.3)

    if "type" in df.columns and df["type"].astype(bool).any():
        df.groupby("type")["latency_ms"].mean().plot(kind="bar", ax=axes[2], rot=20)
        axes[2].set_title("各题型平均延迟")
        axes[2].set_ylabel("毫秒")
        axes[2].grid(axis="y", alpha=0.3)
    else:
        df["latency_ms"].plot(kind="hist", bins=min(20, max(3, len(df))), ax=axes[2])
        axes[2].set_title("延迟分布")
        axes[2].set_xlabel("毫秒")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent 决策层评测")
    parser.add_argument("--qa", default="eval/qa_set_demo.jsonl", help="评测集 JSONL 路径")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0 表示全部）")
    parser.add_argument("--out", default=None, help="CSV 输出路径")
    args = parser.parse_args(argv)

    qa_path = Path(args.qa)
    if not qa_path.is_absolute():
        qa_path = PROJECT_ROOT / qa_path
    if not qa_path.exists():
        print(f"[FAIL] 评测集不存在: {qa_path}")
        return 1

    qa_pairs = load_qa(qa_path)
    if args.limit:
        qa_pairs = qa_pairs[: args.limit]
    if not qa_pairs:
        print(f"[FAIL] 评测集为空: {qa_path}")
        return 1

    from agent import build_agent
    from retrieval.hybrid import build_retriever

    retriever = build_retriever()
    if retriever.vectorstore.count() == 0 and retriever.bm25.count() == 0:
        print("[FAIL] 知识库为空，请先运行 python scripts/build_index.py <目录>")
        return 1

    agent = build_agent(retriever=retriever)
    df, summary = evaluate_agent(agent, qa_pairs)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = Path(args.out) if args.out else REPORT_DIR / f"agent_{stamp}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_summary(df, summary, REPORT_DIR / f"agent_summary_{stamp}.png")

    print(f"\n评测集: {qa_path}  (共 {len(qa_pairs)} 条)")
    print("-" * 60)
    print(f"  工具调用准确率 : {summary['tool_accuracy']}  "
          f"（{summary['tool_judged_n']} 条标注了 expected_tool）")
    print(f"  平均推理轮次   : {summary['avg_steps']}  (最大 {summary['max_steps']})")
    print(f"  崩溃率         : {summary['crash_rate']}")
    print(f"  降级率         : {summary['degraded_rate']}")
    print(f"  平均延迟       : {summary['avg_latency_ms']} ms  (P95 {summary['p95_latency_ms']} ms)")
    print(f"  Token 消耗     : prompt {summary['total_prompt_tokens']} / "
          f"completion {summary['total_completion_tokens']}")
    print(f"\n已写出: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
