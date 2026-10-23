"""Единый логгер. Уровень берётся из ALFA_LOG_LEVEL при каждом получении логгера."""

from __future__ import annotations

import logging
import os

_HANDLER_FLAG = "_alfa_handler"


def get_logger(name: str = "alfa") -> logging.Logger:
    logger_name = name if name == "alfa" or name.startswith("alfa.") else f"alfa.{name}"
    logger = logging.getLogger(logger_name)
    level_name = os.environ.get("ALFA_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    if not getattr(logger, _HANDLER_FLAG, False):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False
        setattr(logger, _HANDLER_FLAG, True)
    return logger
