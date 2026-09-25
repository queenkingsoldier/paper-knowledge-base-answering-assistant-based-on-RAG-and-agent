"""校验评测集的 gold_chunk_ids 是否真实正确。

用法：
    python scripts/validate_qa_set.py eval/qa_set_demo.jsonl

为什么需要这个脚本：评测集里的 `gold_chunk_ids` 如果写错，整份检索评测就失去意义。
本脚本对每条样本断言两件事：

1. 每个 `gold_chunk_id` 都真实存在于当前索引中（防止凭想象填 ID）；
2. 每个 gold chunk 的**正文里确实含有** `verify_key` 这个关键片段
   （防止把答案标到不含答案的 chunk 上）。

`verify_key` 是「该问题的答案一定出现在哪个片段里」的字面证据，
既用于自动校验，也让 review 的人一眼看懂 gold 的依据。它可以是单个字符串，
也可以是列表——列表用于答案横跨多篇文档的问题，此时**每个**关键事实都必须
能在 gold 集合中的某一个 chunk 里找到。

注意：`gold_chunk_ids` 为空表示「该问题本就不该检索知识库」（如纯算术、问时间），
这类样本必须有 `expected_tool` 说明应走哪个工具。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验评测集 gold 标注")
    parser.add_argument("qa", nargs="?", default="eval/qa_set_demo.jsonl")
    args = parser.parse_args(argv)

    qa_path = Path(args.qa)
    if not qa_path.is_absolute():
        qa_path = PROJECT_ROOT / qa_path
    if not qa_path.exists():
        print(f"[FAIL] 评测集不存在: {qa_path}")
        return 1

    from retrieval.hybrid import build_retriever

    retriever = build_retriever()
    if retriever.vectorstore.count() == 0:
        print("[FAIL] 知识库为空，请先运行 python scripts/build_index.py eval/demo_corpus")
        return 1

    # 建立 chunk_id -> 正文 的索引
    texts: dict[str, str] = {}
    for d in retriever.vectorstore.list_documents():
        for row in retriever.vectorstore.iter_chunks(d["doc_id"]):
            texts[row["chunk_id"]] = row["text"]

    items: list[dict] = []
    with qa_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[FAIL] {qa_path.name}:{lineno} 不是合法 JSON: {exc}")
                return 1

    errors: list[str] = []
    checked = 0

    for item in items:
        qid = item.get("id", "?")
        gold = item.get("gold_chunk_ids") or []
        key = item.get("verify_key", "")

        if not gold:
            # 不检索知识库的样本：必须说明期望工具，否则标注不完整
            if not item.get("expected_tool"):
                errors.append(f"{qid}: gold_chunk_ids 为空但未声明 expected_tool")
            continue

        if not key:
            errors.append(f"{qid}: 有 gold_chunk_ids 却没有 verify_key，无法校验")
            continue

        # 每个 gold chunk 都必须真实存在
        for cid in gold:
            checked += 1
            if cid not in texts:
                errors.append(f"{qid}: gold chunk 不存在于索引中 → {cid}")

        # verify_key 可以是单个字符串，也可以是列表。
        # 列表用于「答案横跨多篇文档」的问题：每个关键事实都必须能在
        # gold 集合里的**某一个** chunk 中找到。
        # 注意不能用「每个 gold chunk 都要含 key」——对比类/归纳类问题的
        # 答案本来就分散在不同文档的不同 chunk 里，那样会误报。
        keys = [key] if isinstance(key, str) else [str(k) for k in key]
        for k in keys:
            hit = any(k in texts.get(cid, "") for cid in gold)
            if not hit:
                errors.append(f"{qid}: 没有任何 gold chunk 的正文含 verify_key {k!r}")

        # gold 里的 doc_id 必须与 chunk_id 前缀一致
        for did in item.get("gold_doc_ids") or []:
            if not any(cid.startswith(did + "::") for cid in gold):
                errors.append(f"{qid}: gold_doc_ids 含 {did}，但没有任何 gold chunk 属于它")

    print(f"评测集: {qa_path.name}")
    print(f"  样本数        : {len(items)}")
    print(f"  校验的 gold 数: {checked}")
    by_type: dict[str, int] = {}
    for it in items:
        by_type[it.get("type", "?")] = by_type.get(it.get("type", "?"), 0) + 1
    print("  题型分布      : " + "、".join(f"{k} {v} 条" for k, v in sorted(by_type.items())))

    if errors:
        print(f"\n[FAIL] 发现 {len(errors)} 处标注问题：")
        for e in errors:
            print(f"   - {e}")
        return 1

    print("\n[OK] 全部 gold 标注校验通过（chunk 存在且正文含 verify_key）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
