"""生成层：Prompt 模板、语义缓存与降级策略。

子模块按需显式导入（`generation.prompts` / `generation.cache` / `generation.fallback`
/ `generation.rag`），本文件刻意不做任何 eager re-export：

1. 避免 `import generation.cache` 这类轻量引用被迫连带加载 LLM 相关依赖；
2. 让子模块可以独立演进、独立测试，不会因为某个兄弟模块出错而整体导入失败。
"""
