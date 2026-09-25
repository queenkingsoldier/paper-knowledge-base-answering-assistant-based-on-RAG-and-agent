"""日志与脱敏工具。"""

from __future__ import annotations

import logging
import sys


def mask_secret(value: str) -> str:
    """把 API key 脱敏成 sk-***abcd 形式，用于日志输出。

    承诺：任何日志里都不允许出现完整密钥。
    """
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:3]}***{value[-4:]}"


_CONFIGURED: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """返回配置好的 logger，重复调用不会叠加 handler。"""
    logger = logging.getLogger(name)
    if name not in _CONFIGURED:
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
            )
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _CONFIGURED.add(name)
    return logger
