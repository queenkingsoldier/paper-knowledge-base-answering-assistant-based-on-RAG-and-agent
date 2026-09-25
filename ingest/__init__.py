"""文档摄入层：把磁盘上的文件变成可检索的 Chunk。

依赖方向：`ingest` 只依赖 `core`（schemas / logging_utils）与 `config`，
不反向依赖 retrieval / generation / ui。

- `loaders.py`  ：多格式解析（PDF/DOCX/TXT/MD）→ `Document`
- `chunkers.py` ：三种分块策略 → `list[Chunk]`
"""
