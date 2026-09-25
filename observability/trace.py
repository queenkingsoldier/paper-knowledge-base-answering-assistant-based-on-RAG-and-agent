"""Agent 轨迹记录（JSONL 追加落盘 + 回读）。

为什么是 JSONL：一次问答写一行，天然可追加、可按行容错
（某一行损坏只丢那一条），也不会像整体 JSON 那样「写一半就全废」。
文件位置由调用方给：生产用 `settings.log_dir`，测试用 `tmp_path`。

**核心承诺：`record()` 绝不抛异常。**
观测是旁路，轨迹写不进去最多丢一条日志，
绝不能让用户的一次正常提问因为磁盘满、编码错或字段不可序列化而失败。
因此这里有双保险：

1. `json.dumps(payload, ensure_ascii=False, default=str)`——
   `default=str` 让任何不可序列化对象（例如 `object()`、numpy 标量）
   降级成字符串，而不是抛 `TypeError`；
2. 整个函数体包在 `try/except Exception` 里并 `logger.warning`。
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.logging_utils import get_logger
from core.schemas import AgentResult

from observability.metrics import compute

logger = get_logger(__name__)

TRACE_FILENAME = "trace.jsonl"

# 一条轨迹里会被写下的 step 字段（与 AgentStep 的字段一一对应）。
_STEP_FIELDS = (
    "index",
    "thought",
    "action",
    "action_input",
    "observation",
    "latency_ms",
    "tokens",
    "error",
)

# 引用字段（与 Citation 一一对应）。
_CITATION_FIELDS = ("index", "filename", "page_no", "chunk_id", "snippet")


class TraceRecorder:
    """把每次问答追加写入 `<log_dir>/trace.jsonl`。"""

    def __init__(self, log_dir: Path | str) -> None:
        self.log_dir = Path(log_dir)
        self.path = self.log_dir / TRACE_FILENAME

    # ------------------------------------------------------------------
    # 写
    # ------------------------------------------------------------------
    def record(self, result: AgentResult, session_id: str) -> None:
        """追加一条轨迹。**任何失败都只记 warning，绝不向上抛异常。**

        轨迹里同时带上 `metrics`（由 `compute` 生成），
        这样前端指标看板只要 `read_recent()` 就能直接 `aggregate()`，
        不必再回读 `result` 重算一遍。
        """
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            payload = self._payload(result, session_id)
            line = json.dumps(payload, ensure_ascii=False, default=str)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception as exc:  # noqa: BLE001 - 观测失败绝不能影响主流程
            logger.warning("轨迹写入失败（已忽略，不影响本次问答）: %s", exc)

    @staticmethod
    def _metrics(result: Any) -> dict:
        """算指标；指标算不出来也不能拖垮整条轨迹的写入。"""
        try:
            return compute(result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("指标计算失败（该条轨迹只保留原始字段）: %s", exc)
            return {}

    def _payload(self, result: Any, session_id: str) -> dict:
        """组装一条轨迹。字段全部走 `getattr`，容忍非 dataclass 的鸭子类型结果。"""
        steps = getattr(result, "steps", None) or []
        citations = getattr(result, "citations", None) or []
        return {
            "ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "session_id": session_id,
            "route": getattr(result, "route", None),
            "degraded": getattr(result, "degraded", None),
            "answer": getattr(result, "answer", ""),
            "metrics": self._metrics(result),
            "tokens": getattr(result, "tokens", None) or {},
            "latency_ms": getattr(result, "latency_ms", None) or {},
            "steps": [self._pick(step, _STEP_FIELDS) for step in steps],
            "citations": [self._pick(c, _CITATION_FIELDS) for c in citations],
        }

    @staticmethod
    def _pick(obj: Any, fields: tuple[str, ...]) -> dict:
        """按字段名抽取，缺失的字段直接跳过（不塞 None，保持 JSONL 干净）。"""
        if isinstance(obj, dict):
            return dict(obj)
        data = {name: getattr(obj, name) for name in fields if hasattr(obj, name)}
        return data or {"repr": repr(obj)}

    # ------------------------------------------------------------------
    # 读
    # ------------------------------------------------------------------
    def read_recent(self, n: int = 50) -> list[dict]:
        """返回**最新的 n 条，最新在前**；文件不存在返回 `[]`。

        用 `deque(maxlen=n)` 逐行扫描：既拿到「最后 n 条」，
        又不必把整个文件读进内存。单行 JSON 损坏只跳过该行。
        """
        try:
            limit = int(n)
        except (TypeError, ValueError):
            logger.warning("read_recent 收到非法 n=%r，按 0 处理", n)
            return []
        if limit <= 0:
            return []

        try:
            if not self.path.is_file():
                return []
            recent: deque[dict] = deque(maxlen=limit)
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except (json.JSONDecodeError, ValueError) as exc:
                        logger.warning("跳过损坏的轨迹行: %s", exc)
                        continue
                    if isinstance(item, dict):
                        recent.append(item)
            return list(recent)[::-1]
        except Exception as exc:  # noqa: BLE001 - 读不出来就当没有历史
            logger.warning("轨迹读取失败（返回空列表）: %s", exc)
            return []
