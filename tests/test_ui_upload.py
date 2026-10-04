"""上传区行为测试。

用 Streamlit 官方的无头测试框架 `AppTest` 真正执行 `streamlit_app.py`，
而不是只检查源码里有没有某段字符串。

覆盖两个行为：
1. 入库成功后上传区复位（靠更换控件 key 实现，`file_uploader` 没有官方清空接口）。
2. 入库结果能在重跑之后显示出来，且只显示一次。

注意：三栏独立滚动的改动已按要求回滚，因此这里不再有布局断言。
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
    """返回一个「跑一次前端脚本」的工厂，避免每个用例都重新启动一次运行时。"""
    from streamlit.testing.v1 import AppTest

    def _run():
        at = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py"), default_timeout=180)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        return at

    return _run


def _fake_flash() -> dict:
    return {"ok": 2, "skipped": 0, "failed": 0, "chunks": 7, "ms": 123, "errors": []}


# ---------------------------------------------------------------------------
# 上传区复位
# ---------------------------------------------------------------------------


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
