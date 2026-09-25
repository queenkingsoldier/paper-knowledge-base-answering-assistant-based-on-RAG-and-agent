"""测试全局夹具。

在任何项目模块被导入前把 Provider 切成 mock，
这样整个测试套件不依赖真实 API key，也不产生任何外部调用与费用。
"""

import os

os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("EMBEDDING_PROVIDER", "mock")
os.environ.setdefault("RERANK_PROVIDER", "mock")
