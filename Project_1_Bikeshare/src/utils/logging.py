from __future__ import annotations

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a  root logger."""
    fmt = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
