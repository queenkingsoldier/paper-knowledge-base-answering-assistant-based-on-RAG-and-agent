"""轻量工具路由：不花 LLM 调用就能判定的问题，直接派给工具。

**为什么值得单独一层**：`"现在几点"`、`"3.14 * 2.56 等于多少"` 这类问题走一遍
ReAct 循环要 1~3 次 LLM 调用、等价于几百毫秒到数秒的延迟与实打实的 token 成本，
而答案是确定的。router 把它们直接短路到工具。

**设计取舍**：router 刻意只做「高精度、低召回」的判定——
把该检索的科研问题误判成算术，代价是用户拿到一个无关的计算结果；
把算术误判成普通问题，代价只是多一次 Agent 循环。
两者不对称，所以规则一律保守：拿不准就 `kind="agent"`。

**Review Focus #5**：`"3.14"` 只是数字，不是算式。因此算术判定要求
「整串只含表达式字符」+「至少一个运算符」+「至少两个数字」三条同时成立，
缺一条就落回 agent。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.logging_utils import get_logger

logger = get_logger(__name__)

# 表达式允许出现的字符（严格整串匹配用）。
_EXPR_CHARS = r"[-+*/().\d\s]"
_STRICT_EXPR_RE = re.compile(rf"^{_EXPR_CHARS}+$")
# 夹带中文时，从问题里抽出数字/运算符构成的片段。
_EXPR_FRAGMENT_RE = re.compile(rf"{_EXPR_CHARS}+")
_OPERATOR_RE = re.compile(r"[+\-*/]")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# 时间类问题的关键词与长度上限：长句里出现「时间」多半是在问论文的时间线，
# 不是在问现在几点，所以只有短问句才短路到 current_time。
_TIME_KEYWORDS = ("现在", "今天", "当前", "几点", "日期", "时间", "current time", "today")
_TIME_MAX_LEN = 30

_MIN_FRAGMENT_LEN = 3  # "1+1" 恰好 3 个字符，是能被接受的最短算式

# 中文口语里的运算符。用户写「3.14 乘以 2.56」而不是「3.14 * 2.56」是很自然的，
# 构建书 §5.3.1 举的例子恰好就是「3.14乘以2.56」，所以必须支持。
# 顺序很重要：长词必须排在短词前面，否则「乘以」会先被「乘」拆掉。
_CN_OPERATORS: tuple[tuple[str, str], ...] = (
    ("乘以", "*"), ("乘上", "*"), ("乘", "*"),
    ("除以", "/"), ("除", "/"),
    ("加上", "+"), ("加", "+"),
    ("减去", "-"), ("减", "-"),
)


def _normalize_cn_operators(text: str) -> str:
    """把中文运算符词替换成符号，并在两侧补空格以免粘连数字。

    「加速度」这类词里的「加」也会被替换，但它不会造成误判：算式判定还要求
    片段里至少有**两个数字**，`"+ 速度"` 只有一个数字，会被拒绝。
    """
    out = text
    for word, symbol in _CN_OPERATORS:
        out = out.replace(word, f" {symbol} ")
    return out


@dataclass
class RouteDecision:
    """一次路由判定的结果。

    kind 取值 `direct_tool | agent`；`direct_tool` 时 `tool` 必有值、`args` 可直接
    交给 `ToolExecutor.run_one`。`reason` 永远非空——前端要把它显示给用户看，
    这既是可解释性，也是排查误路由时唯一的线索。
    """

    kind: str
    tool: str | None = None
    args: dict = field(default_factory=dict)
    reason: str = ""


def _looks_like_expression(text: str) -> bool:
    """整串是否是一个「真算式」：只有数字、运算符与括号，且运算符和数字都够。"""
    if not _STRICT_EXPR_RE.match(text):
        return False
    if not _OPERATOR_RE.search(text):
        return False
    return len(_NUMBER_RE.findall(text)) >= 2


def _extract_expression(question: str) -> str | None:
    """从夹带中文的问题里抽出最长的数字表达式片段。

    中文口语里的运算符会先被归一化成符号，因此
    `"3.14 乘以 2.56 等于多少"`、`"3.14 * 2.56 等于多少"` 都能抽出 `"3.14 * 2.56"`。

    必须同时满足「有运算符」且「至少两个数字」才算算式——这条约束让
    `"3.14"`（纯数字）、`"论文1和论文2的区别"`（两个数字但无运算符）都不会被误判。
    """
    normalized = _normalize_cn_operators(question)
    fragments = [f.strip() for f in _EXPR_FRAGMENT_RE.findall(normalized)]
    candidates = [f for f in fragments if len(f) >= _MIN_FRAGMENT_LEN]
    if not candidates:
        return None
    longest = max(candidates, key=len)
    if not _OPERATOR_RE.search(longest):
        return None
    if len(_NUMBER_RE.findall(longest)) < 2:
        return None
    return longest


def _is_time_question(question: str) -> bool:
    lowered = question.lower()
    if len(question) >= _TIME_MAX_LEN:
        return False
    return any(k in lowered for k in _TIME_KEYWORDS)


def classify(question: str) -> RouteDecision:
    """判定一个问题该直接调工具还是交给 Agent。**永不抛异常。**

    Args:
        question: 用户原始问题，允许为空串、含全角标点、含换行。

    Returns:
        永远带非空 `reason` 的 `RouteDecision`。
    """
    try:
        text = (question or "").strip()
        if not text:
            return RouteDecision("agent", None, {}, "问题为空，交给 Agent 兜底处理")

        # 1) 算术：整串就是算式，或夹带中文但能抽出足够长的算式片段。
        if _looks_like_expression(text):
            return RouteDecision(
                "direct_tool",
                "calculator",
                {"expression": text},
                f"整句是一个纯数学表达式「{text}」，可直接计算，无需检索",
            )
        expression = _extract_expression(text)
        if expression:
            return RouteDecision(
                "direct_tool",
                "calculator",
                {"expression": expression},
                f"句中含算式「{expression}」，可直接计算，无需检索",
            )

        # 2) 时间：短问句 + 时间关键词。
        if _is_time_question(text):
            return RouteDecision(
                "direct_tool",
                "current_time",
                {},
                "这是关于当前日期时间的问题，current_time 工具即可直接回答",
            )

        # 3) 其余一律交给 Agent（需要检索、对比或多步推理）。
        return RouteDecision("agent", None, {}, "需要检索知识库并多步推理，交给 Agent")
    except Exception as e:  # noqa: BLE001 - 路由判定绝不能成为新的故障点
        logger.warning("路由判定异常，已降级为 agent：%s", e)
        return RouteDecision("agent", None, {}, f"路由判定异常（{e}），保守交给 Agent")
