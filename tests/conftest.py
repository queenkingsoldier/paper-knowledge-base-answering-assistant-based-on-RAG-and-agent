"""测试全局夹具。

三件事：

1. 在任何项目模块被导入前把 Provider 切成 mock，
   这样整个测试套件不依赖真实 API key，也不产生任何外部调用与费用。

2. 把临时目录重定向到工作区内的 .tmp/pytest。
   本机的文件沙箱只允许写工作区，pytest 的 tmp_path 默认落在系统
   临时目录会被拒绝。

3. 把 os.mkdir 的 POSIX mode 中立化。
   Windows 上 UCRT 会把 mode 写进目录 DACL，而 pytest 的 tmp_path
   机制固定用 mode=0o700 建目录（_pytest/tmpdir.py）。0o700 生成的
   DACL 不含本会话沙箱的受限令牌组，于是紧接着的 os.scandir 直接
   报 WinError 5「拒绝访问」。实测对照：Path.mkdir() 可枚举，
   Path.mkdir(mode=0o700) 则 PermissionError。
   把 mode 统一成 0o777 即可绕开，对测试语义没有任何影响。
"""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("EMBEDDING_PROVIDER", "mock")
os.environ.setdefault("RERANK_PROVIDER", "mock")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PYTEST_TMP = _PROJECT_ROOT / ".tmp" / "pytest"
_PYTEST_TMP.mkdir(parents=True, exist_ok=True)

for _var in ("TMPDIR", "TEMP", "TMP"):
    os.environ[_var] = str(_PYTEST_TMP)
# 直接改写 tempfile 的模块级缓存，这是让 tmp_path 生效的关键一步。
tempfile.tempdir = str(_PYTEST_TMP)

_ORIG_MKDIR = os.mkdir


def _mode_neutral_mkdir(path, mode=0o777, *, dir_fd=None):
    """丢弃 mode 参数，始终以 0o777 建目录（见模块 docstring 第 3 条）。"""
    if dir_fd is not None:
        return _ORIG_MKDIR(path, 0o777, dir_fd=dir_fd)
    return _ORIG_MKDIR(path, 0o777)


os.mkdir = _mode_neutral_mkdir
