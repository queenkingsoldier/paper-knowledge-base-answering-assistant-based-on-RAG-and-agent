"""Chroma 向量库封装。

设计要点：

1. **坐标全量入 metadata**：正文放 Chroma 的 `documents`，来源坐标
   （`doc_id` / `filename` / `page_no` / `seq`）放 metadata。`page_no`
   必须是 int——引用溯源要把它直接印在答案里，字符串会让 `第 "7" 页`
   这种脏数据一路漏到前端。
2. **cosine 空间**：`hnsw:space=cosine`，与 provider 输出的归一化向量
   点积等价；Chroma 返回的是**距离**，故 `score = 1 - distance`。
3. **空集合短路**：集合为空时先 `count() == 0` 返回 `[]`。部分 Chroma
   版本在空集合上 `query` 会抛异常，首次启动就问问题是最常见的入口。
4. **增量与幂等**：`upsert` 以 `chunk_id` 为主键，重复入库同 ID 只覆盖不新增；
   文档级删除走 `where={"doc_id": ...}`。

依赖方向：retrieval → core，本模块不直接发 HTTP。
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb

from core.logging_utils import get_logger
from core.providers import EmbeddingProvider
from core.schemas import Chunk, ScoredChunk, estimate_tokens

logger = get_logger(__name__)

# Chroma 单次写入条数上限（实测 1.5.9 Rust 绑定为 5461），留足余量分批。
_UPSERT_BATCH = 512

# Chroma 1.5.x 的集合名约束：3-512 字符、字符集 [a-zA-Z0-9._-]、首尾必须是字母或数字。
_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$")
_INVALID_NAME_CHAR_RE = re.compile(r"[^a-zA-Z0-9._-]")


def _physical_collection_name(name: str) -> str:
    """把集合名调整为 Chroma 1.5.x 可接受的物理名。

    Chroma 1.5.x 收紧了对集合名的校验（长度 3-512）。计划书按 0.5.x 编写，
    其测试与示例使用 `"t"` 这类短名，在新版本会直接抛
    `InvalidArgumentError: Expected a name containing 3-512 characters`。
    为了让这种历史写法继续可用，非法名统一改写为
    `f"{清洗后的名字}-{sha1(name)[:8]}"`：确定性（同一入参永远得到同一集合）、
    可读、且带上哈希后缀以避免与合法名撞车。
    """
    if _NAME_RE.match(name):
        return name
    tail = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    cleaned = _INVALID_NAME_CHAR_RE.sub("", name).strip("._-")
    base = cleaned if cleaned and cleaned[0].isalnum() else "col"
    return f"{base}-{tail}"


def _utc_now() -> str:
    """ISO-8601 UTC 时间戳，用于记录文档入库时间。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class VectorStore:
    """Chroma 持久化向量库的薄封装。"""

    def __init__(self, persist_dir: Path, collection: str, embedder: EmbeddingProvider) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.collection_name = _physical_collection_name(collection)
        if self.collection_name != collection:
            logger.warning(
                "集合名 %r 不符合 Chroma 1.5.x 命名规则，已改写为 %r", collection, self.collection_name
            )
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        # cosine 空间：与归一化向量的点积等价，语义检索的标准选择。
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    @staticmethod
    def _metadata(chunk: Chunk, ingested_at: str) -> dict[str, Any]:
        return {
            "doc_id": chunk.doc_id,
            "filename": chunk.filename,
            "page_no": int(chunk.page_no),  # 必须为 int：引用溯源依赖它
            "seq": int(chunk.seq),
            "text": chunk.text,
            "token_count": int(chunk.token_count),
            "ingested_at": ingested_at,
        }

    def upsert(self, chunks: list[Chunk]) -> int:
        """幂等写入，返回实际写入的 chunk 数（同 ID 覆盖也算写入）。"""
        if not chunks:
            # Chroma 对空 upsert 会抛 ValueError，空批次直接短路。
            return 0
        # 同一批内可能出现重复 chunk_id（overlap 段落），按主键去重后写。
        unique: dict[str, Chunk] = {c.chunk_id: c for c in chunks}
        items = list(unique.values())
        vectors = self.embedder.embed([c.text for c in items])
        if len(vectors) != len(items):
            raise RuntimeError(f"向量化返回数量 {len(vectors)} 与 chunk 数 {len(items)} 不一致")

        ingested_at = _utc_now()
        written = 0
        for start in range(0, len(items), _UPSERT_BATCH):
            batch = items[start : start + _UPSERT_BATCH]
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=vectors[start : start + _UPSERT_BATCH],
                documents=[c.text for c in batch],
                metadatas=[self._metadata(c, ingested_at) for c in batch],
            )
            written += len(batch)
        logger.info("向量库写入 %d 个 chunk（集合 %s）", written, self.collection_name)
        return written

    # ------------------------------------------------------------------
    # 删除与查询
    # ------------------------------------------------------------------
    def document_exists(self, doc_id: str) -> bool:
        got = self._collection.get(where={"doc_id": doc_id}, limit=1)
        return len(got["ids"]) > 0

    def delete_document(self, doc_id: str) -> int:
        """删除整篇文档的所有 chunk，返回删除条数。"""
        # 先数后删：Chroma 的 delete 不返回条数，靠 get 拿到 ids 长度。
        ids = self._collection.get(where={"doc_id": doc_id})["ids"]
        if not ids:
            return 0
        self._collection.delete(where={"doc_id": doc_id})
        logger.info("向量库删除文档 %s 的 %d 个 chunk", doc_id, len(ids))
        return len(ids)

    @staticmethod
    def _to_chunk(chunk_id: str, meta: dict, document: str | None) -> Chunk:
        text = str(meta.get("text") or document or "")
        return Chunk(
            chunk_id=chunk_id,
            doc_id=str(meta.get("doc_id", "")),
            filename=str(meta.get("filename", "")),
            page_no=int(meta.get("page_no", 0)),
            seq=int(meta.get("seq", 0)),
            text=text,
            token_count=int(meta.get("token_count") or estimate_tokens(text)),
        )

    def query(self, query: str, top_k: int) -> list[ScoredChunk]:
        """向量检索，返回按相似度降序、rank 从 1 编号的结果。"""
        if top_k <= 0:
            return []
        # 空集合上 Chroma 的 query 行为不稳定（旧版本直接抛异常），先短路。
        if self.count() == 0:
            return []

        embedding = self.embedder.embed_one(query)
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = res["ids"][0]
        documents = res["documents"][0] if res["documents"] else []
        metadatas = res["metadatas"][0] if res["metadatas"] else []
        distances = res["distances"][0] if res["distances"] else []

        out: list[ScoredChunk] = []
        for i, chunk_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            document = documents[i] if i < len(documents) else None
            distance = float(distances[i]) if i < len(distances) else 0.0
            out.append(
                ScoredChunk(
                    chunk=self._to_chunk(chunk_id, meta, document),
                    score=1.0 - distance,  # cosine 距离 → 相似度
                    source="vector",
                    rank=i + 1,
                )
            )
        return out

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    def count(self) -> int:
        return int(self._collection.count())

    def list_documents(self) -> list[dict]:
        """按 doc_id 聚合出文档清单，供前端文档面板展示。"""
        metadatas = self._collection.get(include=["metadatas"])["metadatas"] or []
        agg: dict[str, dict] = {}
        for meta in metadatas:
            if not meta:
                continue
            doc_id = str(meta.get("doc_id", ""))
            if not doc_id:
                continue
            row = agg.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "filename": str(meta.get("filename", "")),
                    "chunk_count": 0,
                    "ingested_at": "",
                },
            )
            row["chunk_count"] += 1
            stamp = str(meta.get("ingested_at", ""))
            # 取最早的时间戳作为「该文档入库时间」。
            if stamp and (not row["ingested_at"] or stamp < row["ingested_at"]):
                row["ingested_at"] = stamp
        return sorted(agg.values(), key=lambda d: d["doc_id"])

    def health(self) -> tuple[bool, str]:
        try:
            self._collection.count()
            return True, "ok"
        except Exception as exc:  # noqa: BLE001 - 健康检查不该自己抛异常
            logger.warning("向量库健康检查失败: %s", exc)
            return False, str(exc)
