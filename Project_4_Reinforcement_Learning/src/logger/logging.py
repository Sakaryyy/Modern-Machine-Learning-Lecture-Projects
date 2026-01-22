"""Logging utilities for the reinforcement learning project."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = [
    "LoggingConfig",
    "LoggingManager",
    "get_logger",
]


@dataclass(slots=True)
class LoggingConfig:
    """Dataclass describing how logging should be configured.

    Parameters
    ----------
    level:
        Numerical logging level compatible with :mod:`logging`.
    fmt:
        Format string describing how each log line should look.
    datefmt:
        Format string describing how timestamps should be rendered.
    log_to_file:
        If ``True`` the logger also writes to a file next to stdout.
    log_directory:
        Optional directory where log files should be saved when ``log_to_file``
        is enabled.
    filename:
        Name of the log file that will be created within ``log_directory``.
    """

    level: int = logging.INFO
    fmt: str = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    log_to_file: bool = False
    log_directory: Optional[Path] = None
    filename: str = "reinforcement_learning.log"


class LoggingManager:
    """Configure the Python logging framework for the project."""

    def __init__(self, config: LoggingConfig) -> None:
        self._config = config

    def configure(self) -> None:
        """Apply the configuration to the global logging setup."""

        handlers = [logging.StreamHandler(sys.stdout)]

        if self._config.log_to_file and self._config.log_directory is not None:
            log_file = self._resolve_log_file()
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))

        logging.basicConfig(
            level=self._config.level,
            format=self._config.fmt,
            datefmt=self._config.datefmt,
            handlers=handlers,
            force=True,
        )

    def _resolve_log_file(self) -> Path:
        """Return the file location where logs should be stored.

        Returns
        -------
        pathlib.Path
            Absolute path to the log file. The directory is created by
            :meth:`configure` before the handler is instantiated.
        """

        directory = self._config.log_directory or Path.cwd()
        if not directory.is_absolute():
            directory = directory.resolve()
        return directory / self._config.filename


def get_logger(name: str) -> logging.Logger:
    """Retrieve a configured logger instance.

    Parameters
    ----------
    name:
        Name of the logger which typically corresponds to ``__name__`` of the
        module requesting logging.

    Returns
    -------
    logging.Logger
        Configured logger instance. The global configuration needs to be
        performed via :class:`LoggingManager` before calling this helper.
    """

    return logging.getLogger(name)
