"""检索层：向量库、BM25 关键词索引、RRF 融合与重排序。

依赖方向：retrieval → core（schemas / providers / logging_utils）。
本层不 import 任何上层模块，也不直接发起 HTTP 调用（一切外部调用交给 provider）。
"""
