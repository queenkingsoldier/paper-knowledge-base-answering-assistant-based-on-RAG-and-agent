"""前端布局与上传区状态的真实性测试。

用 Streamlit 官方的无头测试框架 `AppTest` 真正执行 `streamlit_app.py`，
然后遍历它构建出的元素树做结构性断言。这比「字符串里有没有 st.container」
强得多——它能真正区分「容器套对了」和「容器套错了层级」。

两个被测试的行为：
1. 三栏各自独立滚动，且 `st.chat_input` 必须在滚动容器**外面**（否则输入框
   会跟着对话一起被滚走）。
2. 入库成功后上传区复位（靠更换控件 key 实现）。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 无头运行 Streamlit 需要把临时目录放在工作区内
_TMP = PROJECT_ROOT / ".tmp" / "pytest_ui"
_TMP.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(_TMP)
os.environ["TMPDIR"] = os.environ["TEMP"] = os.environ["TMP"] = str(_TMP)


@pytest.fixture(scope="module")
def app_factory():
    """返回一个「跑一次前端脚本」的工厂，避免每个用例都重新启动一次运行时会。"""
    from streamlit.testing.v1 import AppTest

    def _run():
        at = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py"), default_timeout=180)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        return at

    return _run


def _children(node):
    return list((getattr(node, "children", None) or {}).values())


def _is_panel(node) -> bool:
    """判断某个 Block 是不是「设了固定高度的滚动容器」。

    `st.container(height=...)` 在协议里是一个 VERTICAL 的 flex_container，
    带 `height_config { use_stretch: true }`；而普通的 `st.container()`
    是 `height_config { use_content: true }`，据此可以精确区分。
    """
    if type(node).__name__ != "Block":
        return False
    proto = getattr(node, "proto", None)
    hc = getattr(proto, "height_config", None) if proto is not None else None
    return bool(hc is not None and getattr(hc, "use_stretch", False))


def _find_panels(node, acc=None):
    """深度优先收集所有滚动容器块。"""
    acc = acc if acc is not None else []
    for child in _children(node):
        if _is_panel(child):
            acc.append(child)
        _find_panels(child, acc)
    return acc


def _descendant_types(node, acc=None) -> set[str]:
    """某个节点下所有后代的 element 类型集合。"""
    acc = acc if acc is not None else set()
    for child in _children(node):
        t = getattr(child, "type", None)
        if t:
            acc.add(t)
        _descendant_types(child, acc)
    return acc


def _top_columns(root):
    """取页面顶层的三栏。

    注意不能直接拿 `root.children` 里的 Column：三栏被包在一个 HORIZONTAL 的
    flex_container（列行）里，而且文档列表每行还有自己的两列小 `st.columns`。
    所以这里找的是「直接子块中恰好含 3 个 Column 的那一个」。
    """
    for child in _children(root):
        cols = [c for c in _children(child) if type(c).__name__ == "Column"]
        if len(cols) == 3:
            return cols
    return []


# ---------------------------------------------------------------------------
# 布局
# ---------------------------------------------------------------------------


def test_panel_height_is_a_valid_streamlit_height():
    """PANEL_HEIGHT 必须是 Streamlit 接受的 int 或 "stretch"/"content"。

    写错成随便一个字符串时，Streamlit 会在运行时抛错；这条断言让它在导入期
    就被抓住，而不是等用户打开页面才发现。
    """
    import streamlit_app

    h = streamlit_app.PANEL_HEIGHT
    assert isinstance(h, int) or h in ("stretch", "content"), f"非法的 PANEL_HEIGHT: {h!r}"


def test_layout_has_exactly_three_scroll_panels(app_factory):
    """左、中、右三栏各有一个独立滚动区——这正是「左右栏不被落在页面上方」的实现。"""
    at = app_factory()
    panels = _find_panels(at.main)
    assert len(panels) == 3, f"期望 3 个滚动容器，实际 {len(panels)} 个"


def test_each_column_owns_exactly_one_scroll_panel(app_factory):
    """每个栏各自拥有一个滚动区，而不是三个都挤在某一栏里。"""
    at = app_factory()
    cols = _top_columns(at.main)
    assert len(cols) == 3, f"期望 3 栏，实际 {len(cols)} 栏"
    for i, col in enumerate(cols):
        own = _find_panels(col)
        assert len(own) == 1, f"第 {i + 1} 栏应恰好有 1 个滚动容器，实际 {len(own)} 个"


def test_chat_input_is_outside_the_scroll_panel(app_factory):
    """`st.chat_input` 必须是滚动容器的**兄弟**而非后代。

    一旦它被放进滚动区，输入框就会跟着对话内容一起滚走——这是本次改动最容易
    被改回去的地方，所以用结构性断言钉住。
    """
    at = app_factory()
    center = _top_columns(at.main)[1]
    panels = _find_panels(center)
    assert len(panels) == 1
    assert "chat_input" not in _descendant_types(panels[0]), (
        "chat_input 被放进了滚动容器内部，输入框会跟着对话滚走"
    )
    # 同时确认它确实在（在容器外面）
    assert "chat_input" in {getattr(c, "type", None) for c in _children(center)}


def test_message_area_comes_before_chat_input(app_factory):
    """中栏的渲染顺序：滚动消息区在前，输入框在后。"""
    at = app_factory()
    center = _top_columns(at.main)[1]
    order = [getattr(c, "type", None) for c in _children(center)]
    panel_idx = next(i for i, c in enumerate(_children(center)) if _is_panel(c))
    input_idx = order.index("chat_input")
    assert panel_idx < input_idx, f"消息区应在输入框之前，实际顺序 {order}"


# ---------------------------------------------------------------------------
# 上传区复位
# ---------------------------------------------------------------------------


def _fake_flash() -> dict:
    return {"ok": 2, "skipped": 0, "failed": 0, "chunks": 7, "ms": 123, "errors": []}


def test_uploader_starts_empty_with_first_generation_key(app_factory):
    at = app_factory()
    up = at.get("file_uploader")[0]
    assert up.key == "uploader_0"
    assert up.value == [], "全新会话的上传区不应挂任何文件"


def test_uploader_key_changes_when_generation_is_bumped(app_factory):
    """入库成功后 `_reset_uploader()` 自增代次，上传控件因而换成一个新的空控件。"""
    at = app_factory()
    assert at.get("file_uploader")[0].key == "uploader_0"

    at.session_state["uploader_seq"] = 1  # 等价于入库成功后 _reset_uploader() 的效果
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.get("file_uploader")[0].key == "uploader_1"

    at.run()  # 代次保持，不会自己回退
    assert at.get("file_uploader")[0].key == "uploader_1"


def test_uploader_key_and_reset_logic(monkeypatch):
    """直接单测两个辅助函数的逻辑（用普通 dict 顶替 session_state，不需要运行时）。"""
    import streamlit_app as app

    fake_state: dict = {}
    monkeypatch.setattr(app.st, "session_state", fake_state)

    assert app._uploader_key() == "uploader_0"
    app._reset_uploader()
    assert app._uploader_key() == "uploader_1"
    app._reset_uploader()
    assert app._uploader_key() == "uploader_2"
    assert fake_state["uploader_seq"] == 2


# ---------------------------------------------------------------------------
# 入库结果反馈
# ---------------------------------------------------------------------------


def test_ingest_result_is_shown_after_rerun(app_factory):
    """入库结果必须在重跑**之后**显示。

    原来代码是 `st.success(...)` 紧接 `st.rerun()`，消息会被立刻冲掉，
    用户其实从来没看到过入库结果。现在结果暂存到 session_state，重跑后显示。
    """
    at = app_factory()
    at.session_state["ingest_flash"] = _fake_flash()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    shown = [s.value for s in at.success]
    assert any("入库完成" in v for v in shown), f"未显示入库结果：{shown}"
    assert any("新增 7 块" in v for v in shown)


def test_ingest_result_is_shown_only_once(app_factory):
    """结果只提示一次，不会每轮重跑都重复弹出。"""
    at = app_factory()
    at.session_state["ingest_flash"] = _fake_flash()
    at.run()
    assert any("入库完成" in s.value for s in at.success)

    at.run()  # 第二轮：flash 已被 pop 掉
    assert not any("入库完成" in s.value for s in at.success)


def test_ingest_errors_are_surfaced(app_factory):
    at = app_factory()
    at.session_state["ingest_flash"] = {
        "ok": 1, "skipped": 0, "failed": 1, "chunks": 3, "ms": 50,
        "errors": [{"filename": "bad.pdf", "error": "不是合法 PDF"}],
    }
    at.run()
    assert any("bad.pdf" in e.value for e in at.error), "失败文件未在界面上报出"
