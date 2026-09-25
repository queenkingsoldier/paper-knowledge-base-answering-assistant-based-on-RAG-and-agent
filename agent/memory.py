"""多轮会话记忆：会话隔离 + 滑动窗口 + 摘要压缩。

三层机制（对应技术规格书 §5.3.4）：

1. **会话隔离**：`dict[session_id, list[Turn]]`，前端每新建一次会话生成一个新 UUID，
   各会话互不干扰——A 会话提问不该污染 B 会话的上下文。
2. **滑动窗口**：从最新往旧累加 `estimate_tokens`，超 `max_tokens` 即停。
   **以整轮（user + assistant）为最小裁剪单位**：只保留半个轮次会得到
   「助手回答了但问题被截掉」的残缺历史，模型据此续写必然跑偏。
3. **摘要压缩**：轮次数超过 `summarize_after_turns` 后，把最早的若干整轮交给 LLM
   压成一段阶段性摘要，用摘要 + 近期原文替代全部历史。摘要结果按
   `(会话, 被压缩的轮次数)` 缓存——同一个状态反复 `build_context` 不该反复花钱。

**持久化**：`store_dir=None` 时 `save/load` 静默 no-op（内存模式，测试与演示都够用）；
给出目录时落盘为 `{session_id}.json`，前端可据此列出/切换/删除历史会话。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from core.logging_utils import get_logger
from core.schemas import estimate_tokens

logger = get_logger(__name__)

SUMMARY_MAX_CHARS = 200

_SUMMARY_SYSTEM_PROMPT = (
    "你是对话压缩器。把用户与助手的多轮对话压缩成一段不超过 "
    f"{SUMMARY_MAX_CHARS} 字的阶段性摘要，保留关键事实、结论、论文名与页码引用；"
    "不要添加原对话中没有的信息，不要输出任何解释性前后缀。"
)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class Turn:
    """一轮对话消息。`ts` 缺省为当前时间，便于直接 `Turn(role=..., content=...)`。"""

    role: str  # user | assistant
    content: str
    ts: str = field(default_factory=_now_iso)
    cited_chunk_ids: list[str] = field(default_factory=list)


class ConversationMemory:
    """按会话管理与裁剪对话历史。"""

    def __init__(
        self,
        max_tokens: int = 2000,
        summarize_after_turns: int = 8,
        llm=None,
        store_dir: Path | str | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.summarize_after_turns = summarize_after_turns
        self.llm = llm  # 为 None 时只做滑窗，不做摘要（离线/测试模式）
        self.store_dir = Path(store_dir) if store_dir is not None else None
        self._sessions: dict[str, list[Turn]] = {}
        # session_id -> (被压缩的 turn 数, 摘要文本)
        self._summaries: dict[str, tuple[int, str]] = {}

    # ------------------------------------------------------------------
    # 会话管理
    # ------------------------------------------------------------------
    def new_session(self, session_id: str | None = None) -> str:
        """创建一个新会话并返回其 ID（默认 uuid4，保证全局唯一）。"""
        sid = session_id or uuid.uuid4().hex
        self._sessions.setdefault(sid, [])
        return sid

    def get(self, session_id: str) -> list[Turn]:
        """返回某会话的全部轮次；未知会话返回空列表（不抛异常）。"""
        return self._sessions.get(session_id, [])

    def append(self, session_id: str, turn: Turn) -> None:
        """追加一轮。会话不存在时自动创建，避免调用方先 new_session 的样板代码。"""
        self._sessions.setdefault(session_id, []).append(turn)

    def clear(self, session_id: str) -> None:
        """清空一个会话（内存 + 磁盘），用于前端的「清空对话」按钮。"""
        self._sessions.pop(session_id, None)
        self._summaries.pop(session_id, None)
        path = self._path(session_id)
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError as e:  # pragma: no cover - 磁盘异常不该影响交互
                logger.warning("删除会话文件失败 %s: %s", path, e)

    def list_sessions(self) -> list[dict]:
        """列出已知会话（内存中的 + 磁盘上的），供前端历史列表使用。"""
        infos: dict[str, dict] = {}
        for sid, turns in self._sessions.items():
            infos[sid] = {
                "session_id": sid,
                "turns": len(turns) // 2,
                "updated_at": turns[-1].ts if turns else "",
            }
        if self.store_dir is not None and self.store_dir.exists():
            for path in sorted(self.store_dir.glob("*.json")):
                sid = path.stem
                info = infos.setdefault(sid, {"session_id": sid, "turns": 0, "updated_at": ""})
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as e:
                    logger.warning("读取会话文件失败 %s: %s", path, e)
                    continue
                info["turns"] = len(payload.get("turns") or []) // 2
                info["updated_at"] = payload.get("updated_at") or info["updated_at"]
        return sorted(infos.values(), key=lambda d: (d["session_id"],))

    # ------------------------------------------------------------------
    # 上下文构造
    # ------------------------------------------------------------------
    def build_context(self, session_id: str) -> tuple[list[dict], str | None]:
        """返回 `(近期原文消息, 早期摘要 | None)`。

        - 轮次数未超 `summarize_after_turns`（或未注入 llm）→ 摘要为 None；
        - 超出时把最早的若干整轮压成摘要，只把最近 `summarize_after_turns` 轮的
          原文送进滑窗，避免上下文随对话无限增长；
        - 滑窗按整轮裁剪，返回列表以完整的 assistant 消息收尾。
        """
        # 先快照：摘要可能耗时，期间不该读到半更新的列表。
        turns = list(self._sessions.get(session_id, []))
        if not turns:
            return [], None

        pairs = len(turns) // 2
        summary: str | None = None
        recent = turns
        if self.llm is not None and pairs > self.summarize_after_turns:
            old_count = (pairs - self.summarize_after_turns) * 2  # 只压缩整轮
            old, trimmed = turns[:old_count], turns[old_count:]
            summary = self._summarize_cached(session_id, old)
            if summary:
                recent = trimmed
            # 摘要失败（summary 为 None）时保持 recent = 全部轮次：
            # 丢历史比丢摘要严重得多，滑窗仍会把上下文压回预算内。
        return self._window(recent), summary

    def _window(self, turns: list[Turn]) -> list[dict]:
        """从最新往旧按整轮累加 token，超预算即停。"""
        kept: list[Turn] = []
        used = 0
        i = len(turns) - 1
        while i >= 0:
            if i - 1 >= 0 and turns[i].role == "assistant" and turns[i - 1].role == "user":
                unit = [turns[i - 1], turns[i]]
                i -= 2
            else:  # 落单的消息（例如刚 append 的 user）自成一组
                unit = [turns[i]]
                i -= 1
            cost = sum(estimate_tokens(t.content) for t in unit)
            if kept and used + cost > self.max_tokens:
                break
            kept = unit + kept
            used += cost
        return [{"role": t.role, "content": t.content} for t in kept]

    # ------------------------------------------------------------------
    # 摘要
    # ------------------------------------------------------------------
    def _summarize_cached(self, session_id: str, old: list[Turn]) -> str | None:
        """摘要缓存：被压缩的轮次数没变就直接复用，不再调 LLM。"""
        cached = self._summaries.get(session_id)
        if cached is not None and cached[0] == len(old):
            return cached[1]
        text = self._summarize(old)
        if not text:
            return None
        self._summaries[session_id] = (len(old), text)
        return text

    def _summarize(self, turns: list[Turn]) -> str:
        """调用 LLM 压缩历史；失败只记日志并返回空串（记忆降级不该中断问答）。"""
        transcript = "\n".join(f"{t.role}: {t.content}" for t in turns)
        messages = [
            {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": f"请压缩以下对话：\n{transcript}"},
        ]
        try:
            return (self.llm.complete(messages) or "").strip()
        except Exception as e:  # noqa: BLE001 - 摘要失败则退回只用滑窗
            logger.warning("会话摘要生成失败（%s），本次跳过摘要", e)
            return ""

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def _path(self, session_id: str) -> Path | None:
        if self.store_dir is None:
            return None
        return self.store_dir / f"{session_id}.json"

    def save(self, session_id: str) -> None:
        """落盘一个会话；`store_dir=None`（内存模式）时静默 no-op。"""
        path = self._path(session_id)
        if path is None:
            return
        payload = {
            "session_id": session_id,
            "updated_at": _now_iso(),
            "turns": [asdict(t) for t in self._sessions.get(session_id, [])],
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as e:
            logger.warning("会话落盘失败 %s: %s", path, e)

    def load(self, session_id: str) -> bool:
        """从磁盘载入一个会话，成功返回 True；无此会话（或内存模式）返回 False。"""
        path = self._path(session_id)
        if path is None or not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            turns = [
                Turn(
                    role=str(item.get("role", "user")),
                    content=str(item.get("content", "")),
                    ts=str(item.get("ts", "")),
                    cited_chunk_ids=list(item.get("cited_chunk_ids") or []),
                )
                for item in payload.get("turns") or []
            ]
        except (OSError, ValueError, AttributeError) as e:
            logger.warning("会话载入失败 %s: %s", path, e)
            return False
        self._sessions[session_id] = turns
        self._summaries.pop(session_id, None)  # 摘要随内容重新计算
        return True
