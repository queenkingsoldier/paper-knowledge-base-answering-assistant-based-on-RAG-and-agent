"""全局配置中心。

所有可调参数集中在此，业务代码禁止硬编码默认值。
配置来源优先级：构造参数 > 环境变量 > .env 文件 > 此处默认值。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    # env_file 用绝对路径：相对路径会相对于「当前工作目录」解析，
    # 从别的目录启动 streamlit / 脚本时就会静默读不到 .env。
    model_config = SettingsConfigDict(env_file=str(PROJECT_ROOT / ".env"), extra="ignore")

    # ---- Provider 开关（mock 用于无 key 全链路验证）----
    llm_provider: str = "deepseek"  # deepseek | mock
    embedding_provider: str = "siliconflow"  # siliconflow | mock
    rerank_provider: str = "siliconflow"  # siliconflow | mock

    # ---- DeepSeek（生成）----
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # ---- SiliconFlow（向量化 + 重排序）----
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    embed_batch_size: int = 32

    # ---- 检索参数 ----
    chunk_strategy: str = "recursive"  # fixed | recursive | semantic
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieve_top_k: int = 20
    rerank_top_n: int = 5
    rrf_k: int = 60
    min_relevance: float = 0.2

    # ---- 生成参数 ----
    temperature: float = 0.3
    top_p: float = 0.95
    max_context_tokens: int = 3000
    max_output_tokens: int = 1500

    # ---- Agent 参数 ----
    max_iterations: int = 6
    tool_timeout_s: float = 30.0
    parallel_max_workers: int = 4
    enable_router: bool = True
    router_use_llm: bool = False

    # ---- 记忆参数 ----
    memory_max_tokens: int = 2000
    summarize_after_turns: int = 8

    # ---- 缓存 ----
    semantic_cache_enabled: bool = True
    cache_similarity_threshold: float = 0.95

    # ---- 存储路径 ----
    storage_dir: str = "storage"
    data_dir: str = "data"

    @model_validator(mode="after")
    def _check_keys_present(self) -> "Settings":
        """在配置构造期就报出缺失的 key。

        刻意不做「延迟到第一次请求才失败」——那样用户会先看到一堆
        无关的正常输出，然后在某个深埋的调用点撞上 401，排查成本高得多。
        """
        missing: list[str] = []
        if self.llm_provider != "mock" and not self.deepseek_api_key:
            missing.append("DEEPSEEK_API_KEY")
        if self.embedding_provider != "mock" and not self.siliconflow_api_key:
            missing.append("SILICONFLOW_API_KEY")
        if self.rerank_provider != "mock" and not self.siliconflow_api_key:
            missing.append("SILICONFLOW_API_KEY")
        if missing:
            names = ", ".join(sorted(set(missing)))
            raise ValueError(
                f"缺少必填配置: {names}。"
                f"请在 .env 中填写真实密钥（可参考 .env.example），"
                f"或把 LLM_PROVIDER / EMBEDDING_PROVIDER / RERANK_PROVIDER 设为 mock 以离线运行。"
            )
        return self

    # ---- 派生路径 ----
    @property
    def storage_path(self) -> Path:
        return PROJECT_ROOT / self.storage_dir

    @property
    def data_path(self) -> Path:
        return PROJECT_ROOT / self.data_dir

    @property
    def chroma_dir(self) -> Path:
        return self.storage_path / "chroma"

    @property
    def bm25_dir(self) -> Path:
        return self.storage_path / "bm25"

    @property
    def cache_dir(self) -> Path:
        return self.storage_path / "cache"

    @property
    def embed_cache_dir(self) -> Path:
        return self.cache_dir / "emb"

    @property
    def log_dir(self) -> Path:
        return self.storage_path / "logs"

    @property
    def session_dir(self) -> Path:
        return self.storage_path / "sessions"

    @property
    def raw_dir(self) -> Path:
        return self.data_path / "raw"


settings = Settings()
