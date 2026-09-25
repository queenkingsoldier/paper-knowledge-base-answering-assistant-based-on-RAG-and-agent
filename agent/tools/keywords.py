"""关键词抽取工具（纯函数，无外部依赖）。

中文没有空格分词，这里用 2-4 字滑窗近似：一个反复出现的词组，它的子串也一定反复
出现，因此按频次排序就能把「扩散模型」这类真关键词顶到前面。英文直接取原词。
纯数字等无法分词的内容不会让函数崩，只是拿不到关键词。
"""

from __future__ import annotations

import re
from collections import Counter

from agent.tools.base import register_tool

# 连续中文块（每个块内独立滑窗，避免跨标点造出假词）
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
# 英文/数字原词：保留 3.14 这种小数整体，连字符按分隔处理
_ASCII_WORD = re.compile(r"[A-Za-z0-9]+(?:\.[0-9]+)*")

_MIN_GRAM = 2
_MAX_GRAM = 4


def extract_keywords(text: str, top_n: int = 8) -> list[str]:
    """从文本中抽取 top_n 个关键词（去重，按频次降序）。

    Args:
        text: 任意中英文文本或问题。
        top_n: 最多返回多少个关键词；<= 0 时返回空列表。

    Returns:
        关键词列表；空文本返回 `[]`。同频次时较长的词优先。
    """
    if not text or not text.strip():
        return []

    counter: Counter[str] = Counter()
    for run in _CJK_RUN.findall(text):
        for size in range(_MIN_GRAM, _MAX_GRAM + 1):
            for start in range(len(run) - size + 1):
                counter[run[start : start + size]] += 1
    for word in _ASCII_WORD.findall(text):
        counter[word] += 1

    # most_common 给出「频次降序」，再稳定地按长度二次排序：
    # 同频次时长词更具体（「扩散模型」优于它的子串「扩散」），仍相同则保持首次出现顺序。
    ranked = sorted(counter.most_common(), key=lambda kv: (-kv[1], -len(kv[0])))
    limit = max(0, int(top_n))
    return [term for term, _ in ranked[:limit]]


@register_tool(
    name="extract_keywords",
    description="从问题或文档片段中抽取核心关键词，可用于扩展检索词",
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "待抽取关键词的文本"},
            "top_n": {"type": "integer", "description": "返回的关键词个数，默认 8"},
        },
        "required": ["text"],
    },
)
def extract_keywords_tool(text: str, top_n: int = 8) -> str:
    """工具入口：把关键词列表拼成一行可读文本。"""
    keywords = extract_keywords(text, top_n=top_n)
    if not keywords:
        return "未抽取到关键词。"
    return "关键词：" + "、".join(keywords)
