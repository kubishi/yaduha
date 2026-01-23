from contextlib import contextmanager
from re import A
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from abc import abstractmethod, ABC
from typing import Any, Dict, List, Mapping, Optional, Generic, TypeVar, cast
from dotenv import load_dotenv

import wandb
import os

load_dotenv()

# context manager to add metadata via environment variables
@contextmanager
def inject_logs(**data: str | int | float):
    """
    Context manager to inject metadata into loggers via environment variables.

    Args:
        **data: Key-value pairs to inject as metadata.
    """
    old_env = {}
    try:
        for key, value in data.items():
            env_key = f"LOGGER_METADATA_{key.upper()}"
            old_env[env_key] = os.environ.get(env_key)
            os.environ[env_key] = str(value)
        yield
    finally:
        for key in data.keys():
            env_key = f"LOGGER_METADATA_{key.upper()}"
            if old_env[env_key] is None:
                del os.environ[env_key]
            else:
                os.environ[env_key] = old_env[env_key]

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

    @abstractmethod
    def _log(self, data: Dict[str, Any]):
        pass

    def log(self, data: Dict[str, Any]):
        """
        Log a dictionary of metrics with the current path prefix.

        Args:
            data: Metrics to log.
        """
        env_metadata = {}
        for key, value in os.environ.items():
            if key.startswith("LOGGER_METADATA_"):
                env_key = key[len("LOGGER_METADATA_") :]
                env_metadata[env_key] = value
        self._log({**self.metadata, **env_metadata, **data})

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

    def _log(self, data: Mapping[str, Any], **kwargs: Any) -> None:
        """
        Log a dictionary of metrics to this W&B run.

        Args:
            data: Metrics to log.
            step: Optional global step (epoch, iteration, etc.).
            **kwargs: Forwarded to Run.log (e.g., commit=False).
        """
        if self._run is None:
            raise RuntimeError("Cannot log: W&B run is not active.")

        self._run.log(dict(data), **kwargs)

    def stop(self) -> None:
        """Finish the W&B run if it is still active."""
        if self._run is not None:
            self._run.finish()
            self._run = None

class PrintLogger(Logger):
    def _log(self, data: Dict[str, Any]):
        print(data)

class NoLogger(Logger):
    def _log(self, data: Dict[str, Any]):
        pass