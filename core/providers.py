"""外部服务 Provider 抽象层。

设计意图：把「调用 DeepSeek / SiliconFlow」这件事收敛到唯一一处。
上层代码只认抽象接口，因此：

1. 换供应商只需改本文件，不动任何业务逻辑；
2. 无 key 时切到 mock，全链路（含 Agent 与评测）仍可真实跑通并验证；
3. 磁盘缓存、分批、重试、脱敏等工程细节集中实现，不散落各处。

依赖约束：本模块只依赖 config 与 core.schemas，不得依赖任何上层模块。
第三方 SDK（openai）一律延迟导入，使 mock 模式无需其存在。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

import httpx
import numpy as np

from config import settings
from core.logging_utils import get_logger, mask_secret

logger = get_logger(__name__)

RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 1.0  # 秒，退避序列 1s / 2s / 4s

# ---------------------------------------------------------------------------
# 抽象接口
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """生成模型接口。"""

    last_usage: dict  # {"prompt_tokens": int, "completion_tokens": int}

    @abstractmethod
    def complete(self, messages: list[dict], **kwargs: Any) -> str:
        """一次性生成，返回完整文本。"""

    @abstractmethod
    def stream(self, messages: list[dict], **kwargs: Any) -> Iterator[str]:
        """流式生成，逐段 yield 文本增量。"""


class EmbeddingProvider(ABC):
    """向量化接口。"""

    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量向量化。返回顺序必须与输入严格一致。"""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class RerankProvider(ABC):
    """重排序接口。"""

    @abstractmethod
    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        """返回 [(原始下标, 相关性分数), ...]，按分数降序。"""


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _retry_sleep(attempt: int) -> None:
    time.sleep(RETRY_BACKOFF_BASE * (2**attempt))


def _should_retry(exc: Exception) -> bool:
    """只对「重试可能成功」的错误重试。

    401/403（密钥错）重试一百次也还是错的，立即失败才能让用户看到真问题。
    """
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (408, 409, 429) or exc.response.status_code >= 500
    return False


def _post_with_retry(
    url: str, *, headers: dict, payload: dict, timeout: float = 60.0
) -> dict:
    """带指数退避的 POST，返回解析后的 JSON。"""
    last_exc: Exception | None = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - 需要按类型决定是否重试
            last_exc = exc
            if attempt < RETRY_ATTEMPTS - 1 and _should_retry(exc):
                logger.warning("请求 %s 失败（第 %d 次），准备重试: %s", url, attempt + 1, exc)
                _retry_sleep(attempt)
                continue
            raise
    raise RuntimeError(f"请求重试耗尽: {last_exc}")  # pragma: no cover


# ---------------------------------------------------------------------------
# 真实 Provider：DeepSeek
# ---------------------------------------------------------------------------


class DeepSeekLLM(LLMProvider):
    """DeepSeek 生成模型（OpenAI 兼容接口）。"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None) -> None:
        from openai import OpenAI  # 延迟导入：mock 模式无需安装 openai

        self.api_key = api_key or settings.deepseek_api_key
        self.base_url = base_url or settings.deepseek_base_url
        self.model = model or settings.deepseek_model
        self.last_usage: dict = {}
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=60.0,
            max_retries=2,
        )
        logger.info("DeepSeekLLM 就绪: model=%s key=%s", self.model, mask_secret(self.api_key))

    def _params(self, messages: list[dict], **kwargs: Any) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", settings.temperature),
            "top_p": kwargs.get("top_p", settings.top_p),
            "max_tokens": kwargs.get("max_tokens", settings.max_output_tokens),
        }

    def complete(self, messages: list[dict], **kwargs: Any) -> str:
        params = self._params(messages, **kwargs)
        if kwargs.get("stop"):
            params["stop"] = kwargs["stop"]
        resp = self._client.chat.completions.create(**params)
        usage = getattr(resp, "usage", None)
        self.last_usage = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }
        return resp.choices[0].message.content or ""

    def stream(self, messages: list[dict], **kwargs: Any) -> Iterator[str]:
        params = self._params(messages, **kwargs)
        # include_usage 必须显式开启：流式模式下默认不返回 usage，
        # 而成本监控与评测都依赖它。
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        for chunk in self._client.chat.completions.create(**params):
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                self.last_usage = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                }
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                yield piece


# ---------------------------------------------------------------------------
# 真实 Provider：SiliconFlow
# ---------------------------------------------------------------------------


class SiliconFlowEmbedding(EmbeddingProvider):
    """SiliconFlow 向量化，带分批、磁盘缓存与重试。"""

    dim = 1024  # BAAI/bge-m3

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, batch_size: int | None = None) -> None:
        self.api_key = api_key or settings.siliconflow_api_key
        self.base_url = (base_url or settings.siliconflow_base_url).rstrip("/")
        self.model = model or settings.embedding_model
        self.batch_size = batch_size or settings.embed_batch_size
        self.cache_dir = settings.embed_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, list[float]] = {}
        logger.info("SiliconFlowEmbedding 就绪: model=%s key=%s",
                    self.model, mask_secret(self.api_key))

    # ---- 缓存 ----
    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str):
        return self.cache_dir / f"{key}.npy"

    def _cache_get(self, key: str) -> list[float] | None:
        if key in self._mem:
            return self._mem[key]
        path = self._cache_path(key)
        if path.exists():
            try:
                vec = np.load(path).tolist()
                self._mem[key] = vec
                return vec
            except Exception:  # noqa: BLE001 - 缓存损坏不该让主流程失败
                logger.warning("向量缓存损坏，已忽略: %s", path)
        return None

    def _cache_put(self, key: str, vec: list[float]) -> None:
        self._mem[key] = vec
        try:
            np.save(self._cache_path(key), np.asarray(vec, dtype=np.float32))
        except Exception:  # noqa: BLE001
            logger.warning("向量缓存写入失败: %s", key)

    # ---- 主流程 ----
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float] | None] = [None] * len(texts)
        pending: list[tuple[int, str, str]] = []  # (下标, 文本, 缓存key)

        for i, text in enumerate(texts):
            key = self._key(text)
            hit = self._cache_get(key)
            if hit is not None:
                results[i] = hit
            else:
                pending.append((i, text, key))

        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            vectors = self._request_batch([t for _, t, _ in batch])
            for (i, _text, key), vec in zip(batch, vectors):
                self._cache_put(key, vec)
                results[i] = vec

        missing = [i for i, v in enumerate(results) if v is None]
        if missing:  # pragma: no cover - 理论上不可达
            raise RuntimeError(f"部分文本未能向量化: {missing}")
        return [v for v in results if v is not None]

    def _request_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts, "encoding_format": "float"}
        data = _post_with_retry(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
        )
        # 不能假设响应顺序与输入一致，必须按 index 回填。
        items = sorted(data["data"], key=lambda d: d.get("index", 0))
        return [item["embedding"] for item in items]


class SiliconFlowRerank(RerankProvider):
    """SiliconFlow 重排序（交叉编码器）。"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None) -> None:
        self.api_key = api_key or settings.siliconflow_api_key
        self.base_url = (base_url or settings.siliconflow_base_url).rstrip("/")
        self.model = model or settings.rerank_model
        logger.info("SiliconFlowRerank 就绪: model=%s key=%s",
                    self.model, mask_secret(self.api_key))

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        if not documents:
            return []
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": max(1, min(top_n, len(documents))),
            # 关掉文档回传以压缩响应体：文本我们本地已有，靠 index 回查。
            "return_documents": False,
        }
        data = _post_with_retry(
            f"{self.base_url}/rerank",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
        )
        return [(int(r["index"]), float(r["relevance_score"])) for r in data["results"]]


# ---------------------------------------------------------------------------
# Mock Provider：无 key 也能验证全链路
# ---------------------------------------------------------------------------

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_WORD_RE = re.compile(r"[a-z0-9]+")


def _mock_tokens(text: str) -> list[str]:
    """轻量切词：英文小写单词 + 中文 bigram。仅用于 mock 打分。"""
    lowered = (text or "").lower()
    tokens = [w for w in _WORD_RE.findall(lowered) if len(w) >= 2]
    for seg in _CJK_RE.findall(lowered):
        if len(seg) == 1:
            tokens.append(seg)
        else:
            tokens.extend(seg[i : i + 2] for i in range(len(seg) - 1))
    return tokens


class MockEmbedding(EmbeddingProvider):
    """确定性伪向量。

    由文本的 sha256 播种，保证：
    - 相同文本永远得到相同向量（缓存与检索行为可复现）；
    - 不同文本得到不同向量（检索才有区分度）；
    - 输出已 L2 归一化，故余弦相似度 == 点积。
    """

    dim = 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256((text or "").encode("utf-8")).digest()
        raw = (digest * (self.dim // len(digest) + 1))[: self.dim]
        values = [(b - 127.5) / 127.5 for b in raw]
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]


class MockRerank(RerankProvider):
    """按查询与文档的词元重合度打分的伪重排序。"""

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        if not documents:
            return []
        q_tokens = set(_mock_tokens(query))
        scored: list[tuple[int, float]] = []
        for i, doc in enumerate(documents):
            d_tokens = _mock_tokens(doc)
            if not d_tokens:
                scored.append((i, 0.0))
                continue
            overlap = sum(1 for t in d_tokens if t in q_tokens)
            # 用重合数除以文档词元数的平方根做长度归一，避免长文档天然占优。
            scored.append((i, overlap / math.sqrt(len(d_tokens))))
        # 分数降序；同分时按下标升序，保证结果确定。
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored[: max(1, min(top_n, len(documents)))]


_MOCK_ANSWER = (
    "根据知识库中的资料，该问题涉及的核心内容已在检索到的文档片段中给出。"
    "（当前运行在 mock 模式，此文本用于验证链路连通性，不代表真实模型输出。）"
)


class MockLLM(LLMProvider):
    """固定话术的伪 LLM，同时能产生合法的 ReAct 输出。

    行为：在 ReAct 场景下先发一次 `kb_search`，看到 Observation 后再 `Finish`。
    这样 mock 模式跑出来的也是**真实的两步轨迹 + 引用**，而不是一步即答的空壳——
    否则「全链路可验证」就只是句空话。
    """

    # _force_conclude 的指令措辞（agent/react.py）。命中它就不再输出 Action 协议，
    # 只给答案正文，避免把「Action: Finish」当最终答案返回。
    _FORCE_CONCLUDE_MARKER = "不要再调用任何工具"

    def __init__(self) -> None:
        self.last_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0}
        self._answer = _MOCK_ANSWER

    @staticmethod
    def _joined(messages: list[dict]) -> str:
        return " ".join(str(m.get("content", "")) for m in messages)

    @classmethod
    def _wants_react(cls, messages: list[dict]) -> bool:
        joined = cls._joined(messages)
        return "Action:" in joined or "可用工具" in joined or "Action Input" in joined

    @staticmethod
    def _has_observation(messages: list[dict]) -> bool:
        """工具结果是以 user 角色、以 'Observation:' 开头的消息（见 agent/react.py）。"""
        return any(
            str(m.get("role")) == "user"
            and str(m.get("content", "")).startswith("Observation:")
            for m in messages
        )

    @staticmethod
    def _extract_question(messages: list[dict]) -> str:
        """取出原始用户问题（跳过 Observation 与强制收敛指令）。"""
        for m in messages:
            if str(m.get("role")) != "user":
                continue
            content = str(m.get("content", ""))
            if content.startswith("Observation:") or "不要再调用任何工具" in content:
                continue
            return content.strip()
        return "论文"

    def complete(self, messages: list[dict], **kwargs: Any) -> str:
        prompt_tokens = sum(max(1, len(str(m.get("content", ""))) // 3) for m in messages)
        joined = self._joined(messages)

        if self._FORCE_CONCLUDE_MARKER in joined:
            text = self._answer
        elif self._wants_react(messages):
            if self._has_observation(messages):
                body = json.dumps({"answer": self._answer}, ensure_ascii=False)
                text = (
                    "Thought: 已从知识库拿到观察结果，信息足够，直接给出最终答案。\n"
                    "Action: Finish\n"
                    f"Action Input: {body}"
                )
            elif "kb_search" in joined:
                # 第一次进入循环：先检索一次，让轨迹里出现真实的 Action/Observation
                body = json.dumps(
                    {"query": self._extract_question(messages), "top_k": 5},
                    ensure_ascii=False,
                )
                text = (
                    "Thought: 需要先查知识库里与问题相关的内容。\n"
                    "Action: kb_search\n"
                    f"Action Input: {body}"
                )
            else:
                # 没有 kb_search 工具时不要瞎调，直接收尾
                body = json.dumps({"answer": self._answer}, ensure_ascii=False)
                text = (
                    "Thought: 没有可用的检索工具，直接作答。\n"
                    "Action: Finish\n"
                    f"Action Input: {body}"
                )
        else:
            text = self._answer

        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": max(1, len(text) // 3),
        }
        return text

    def stream(self, messages: list[dict], **kwargs: Any) -> Iterator[str]:
        text = self.complete(messages, **kwargs)
        # 按 8 字符切段，模拟流式增量，便于前端打字机效果与引用缓冲逻辑被真实触发。
        for i in range(0, len(text), 8):
            yield text[i : i + 8]
        self.last_usage = {
            "prompt_tokens": sum(max(1, len(str(m.get("content", ""))) // 3) for m in messages),
            "completion_tokens": max(1, len(text) // 3),
        }


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_llm() -> LLMProvider:
    if settings.llm_provider == "mock":
        return MockLLM()
    if settings.llm_provider == "deepseek":
        return DeepSeekLLM()
    raise ValueError(
        f"未知的 LLM_PROVIDER: {settings.llm_provider}（可选 deepseek | mock）"
    )


@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingProvider:
    if settings.embedding_provider == "mock":
        return MockEmbedding()
    if settings.embedding_provider == "siliconflow":
        return SiliconFlowEmbedding()
    raise ValueError(
        f"未知的 EMBEDDING_PROVIDER: {settings.embedding_provider}（可选 siliconflow | mock）"
    )


@lru_cache(maxsize=1)
def get_reranker() -> RerankProvider:
    if settings.rerank_provider == "mock":
        return MockRerank()
    if settings.rerank_provider == "siliconflow":
        return SiliconFlowRerank()
    raise ValueError(
        f"未知的 RERANK_PROVIDER: {settings.rerank_provider}（可选 siliconflow | mock）"
    )


def reset_providers() -> None:
    """清空单例缓存。测试与切换配置时使用。"""
    get_llm.cache_clear()
    get_embedder.cache_clear()
    get_reranker.cache_clear()
