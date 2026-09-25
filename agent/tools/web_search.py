"""联网搜索工具：DuckDuckGo HTML 端点（无需 API key）。

刻意不引入 BeautifulSoup 之类的解析依赖——需求只要标题和链接，正则足够，
少一个依赖就少一个装不上的理由。任何失败（网络、超时、解析）都返回可读字符串
而不是抛异常：Observation 要能交还给 Agent，让它自己决定换工具还是直接回答。

注意 `httpx.get` 通过模块级 `import httpx` 调用（而不是 `from httpx import get`），
测试才能 monkeypatch 掉它。
"""

from __future__ import annotations

import html
import re
import urllib.parse

import httpx

from agent.tools.base import register_tool
from core.logging_utils import get_logger

logger = get_logger(__name__)

_ENDPOINT = "https://html.duckduckgo.com/html/"
_TIMEOUT_S = 10.0
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
}

# <a class="result__a" href="...">标题</a>；class 和 href 之间可能夹着 rel 等属性。
_RESULT_RE = re.compile(
    r'<a\b([^>]*class="[^"]*result__a[^"]*"[^>]*)>(.*?)</a>', re.S | re.I
)
_HREF_RE = re.compile(r'href="([^"]+)"', re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_FAILURE_HINT = "请基于知识库内容回答，或告知用户当前无法联网。"


def _clean_title(raw: str) -> str:
    """去掉高亮用的 <b> 等标签并还原 HTML 实体，得到纯文本标题。"""
    return html.unescape(_TAG_RE.sub("", raw)).strip()


def _real_url(href: str) -> str:
    """DDG 结果常是 `//duckduckgo.com/l/?uddg=<编码后的真实地址>` 形式的跳转链接。"""
    if "uddg=" not in href:
        return href
    query = urllib.parse.urlparse(href).query
    target = urllib.parse.parse_qs(query).get("uddg", [""])[0]
    return target or href


def web_search(query: str, max_results: int = 3) -> str:
    """联网搜索并返回编号结果列表。

    Args:
        query: 搜索关键词；空白字符串视为错误，直接返回提示。
        max_results: 最多返回多少条结果。

    Returns:
        多行字符串 `[1] 标题\n    https://...`；失败时返回可读的错误说明。
    """
    keyword = (query or "").strip()
    if not keyword:
        return "错误：搜索关键词为空。"
    try:
        limit = max(0, int(max_results))
        response = httpx.get(
            _ENDPOINT, params={"q": keyword}, timeout=_TIMEOUT_S, headers=_HEADERS
        )
        status = getattr(response, "status_code", 200)
        if status >= 400:
            return f"联网搜索失败：HTTP {status}。{_FAILURE_HINT}"

        lines: list[str] = []
        for attrs, raw_title in _RESULT_RE.findall(response.text):
            if len(lines) >= limit:
                break
            match = _HREF_RE.search(attrs)
            if not match:
                continue
            url = _real_url(match.group(1))
            lines.append(f"[{len(lines) + 1}] {_clean_title(raw_title)}\n    {url}")
        if not lines:
            return f"联网搜索没有返回结果。{_FAILURE_HINT}"
        return "\n".join(lines)
    except Exception as e:  # 网络/超时/解析异常一律降级为可读字符串，不打断 Agent 循环
        logger.debug("联网搜索失败：%s", e)
        return f"联网搜索失败：{e}。{_FAILURE_HINT}"


@register_tool(
    name="web_search",
    description="联网搜索最新资料（DuckDuckGo）；知识库内容不足时使用，失败会返回可读提示",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "max_results": {"type": "integer", "description": "返回条数，默认 3"},
        },
        "required": ["query"],
    },
)
def web_search_tool(query: str, max_results: int = 3) -> str:
    """工具入口：注册表里叫 `web_search`，内部直接转调 `web_search()`。"""
    return web_search(query, max_results=max_results)
