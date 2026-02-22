from contextlib import contextmanager
from contextvars import ContextVar
import json
import pathlib
import threading
import time
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from abc import ABC
from typing import Any, Dict
import wandb

# Thread-safe context for log metadata (uses contextvars, safe for threads and async)
_log_context: ContextVar[Dict[str, str | int | float]] = ContextVar('_log_context', default={})

@contextmanager
def inject_logs(**data: str | int | float):
    """
    Context manager to inject metadata into loggers via context variables.
    Thread-safe and async-safe. Supports nesting — inner contexts overlay outer ones.

    Args:
        **data: Key-value pairs to inject as metadata. Keys are uppercased.
    """
    current = _log_context.get()
    merged = {**current, **{k.upper(): v for k, v in data.items()}}
    token = _log_context.set(merged)
    try:
        yield
    finally:
        _log_context.reset(token)

def get_log_context() -> Dict[str, str | int | float]:
    """Get the current log context metadata (thread-safe)."""
    return _log_context.get()

global_logger = None
def set_global_logger(logger: "Logger") -> None:
    """
    Set the global logger.

    Args:
        logger: The logger to set as global.
    """
    global global_logger
    global_logger = logger

def get_global_logger() -> "Logger":
    """
    Get the global logger.

    Returns:
        The current global logger.
    """
    global global_logger
    if global_logger is None:
        global_logger = NoLogger()
    return global_logger

class Logger(BaseModel, ABC):
    metadata: Dict[str, str | int | float] = Field(default_factory=dict, description="Metadata for the logger.")

    def _log(self, data: Dict[str, Any]):
        pass

    def log(self, data: Dict[str, Any]):
        """
        Log a dictionary of metrics with the current path prefix.
        Automatically includes a timestamp and any context injected via inject_logs.

        Args:
            data: Metrics to log.
        """
        ctx_metadata = get_log_context()
        self._log({
            "timestamp": time.time(),
            **self.metadata,
            **ctx_metadata,
            **data,
        })

    def get_sublogger(self, **metadata: str | int | float) -> "Logger":
        """
        Create a sub-logger with additional metadata.

        Args:
            **metadata: Additional metadata to add to the sub-logger.

        Returns:
            A new Logger instance with the combined metadata.
        """
        combined_metadata = {**self.metadata, **metadata}
        return self.model_copy(update={"metadata": combined_metadata})

class WandbLogger(Logger):
    model_config = ConfigDict(populate_by_name=True)

    project_name: str = Field(..., description="The W&B project name.", alias="project")
    name: str
    config_items: Dict[str, Any] = Field(default_factory=dict, description="The W&B config items.", alias="config")
    
    _run: wandb.Run | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """
        Called by Pydantic *after* the model has been initialized and validated.
        This is the right place for side effects like wandb.init().
        """
        self._run = wandb.init(
            project=self.project_name,
            config=self.config_items,
            name=self.name,
        )

    def _log(self, data: Dict[str, Any]) -> None:
        if self._run is None:
            raise RuntimeError("Cannot log: W&B run is not active.")
        self._run.log(dict(data))

    def stop(self) -> None:
        """Finish the W&B run if it is still active."""
        if self._run is not None:
            self._run.finish()
            self._run = None

class PrintLogger(Logger):
    def _log(self, data: Dict[str, Any]):
        print(data)

class JsonLogger(Logger):
    file_path: pathlib.Path
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @model_validator(mode="after")
    def _validate_jsonl_extension(self) -> "JsonLogger":
        if self.file_path.suffix != ".jsonl":
            raise ValueError("file_path must have .jsonl extension")
        return self

    def _log(self, data: Dict[str, Any]):
        line = json.dumps(data, default=str, ensure_ascii=False) + "\n"
        with self._lock:
            with self.file_path.open("a", encoding="utf-8") as f:
                f.write(line)

class NoLogger(Logger):
    def _log(self, data: Dict[str, Any]):
        pass