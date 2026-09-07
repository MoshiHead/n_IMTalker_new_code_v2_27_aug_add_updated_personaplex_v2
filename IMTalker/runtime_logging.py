"""runtime_logging.py — shared, thread/async-safe rotating log files for the
live PersonaPlex + IMTalker pipeline.

Two fixed-path, rotating, millisecond-timestamped log files, independent of
(and in addition to) the per-session files conversation_logger.py and
latency_logger.py already write:

  logs/system_runtime.log -- models/adapters loaded (name, source, path,
      device, dtype, quantization), STT/VAD/router/compressor/PersonaPlex/
      IMTalker init, avatar/video streaming init, GPU/CUDA/VRAM info, server
      startup, and errors/warnings/timeouts/fallbacks. Written directly by
      the model-loading code (liveTry.py, liveTry_cached.py,
      imtalker_personaplex_try_vad2_8998.py).

  logs/conversation.log -- the full per-turn conversation flow (USER ->
      SEARCH DECISION -> SEARCH QUERY -> SEARCH RESULTS -> SUMMARY ->
      CONTEXT INJECTED -> PERSONAPLEX RESPONSE -> AVATAR/STREAMING), with a
      conversation_id on every line. This file is populated by attaching a
      handler to conversation_logger.py's existing logger (see
      `attach_conversation_file`) and by mirroring latency_logger.py's
      per-stage lines (see latency_logger.py's `mirror_logger` param) --
      nothing that already computes this data is duplicated here.

Location: `$LOGS_DIR` if set, else `$SPEECH2AVATAR_ROOT/logs`, else
`<project root>/logs` (IMTalker/'s parent directory). Works unmodified on a
RunPod pod or any other host -- no local-machine assumptions.

Non-blocking: every logger here funnels through a stdlib
`logging.handlers.QueueHandler`, so a call on the hot GPU thread only
enqueues a LogRecord (an in-memory, lock-protected list append) and returns
immediately. A single background `QueueListener` thread per file does the
actual (rotating) disk I/O. This is the same pattern the standard library
recommends for logging from latency-sensitive code.

Secrets: every record passes through `_RedactSecretsFilter`, which scrubs
common API-key/token/password shapes before anything is written to disk or
console. Callers should still never deliberately log a raw secret -- this
filter is a safety net, not a license to log credentials.
"""
from __future__ import annotations

import atexit
import logging
import logging.handlers
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_SYSTEM_LOGGER: logging.Logger | None = None
_LISTENERS: list[logging.handlers.QueueListener] = []

_MAX_BYTES = int(os.environ.get("IMTALKER_LOG_MAX_BYTES", str(20 * 1024 * 1024)))  # 20 MB/file
_BACKUP_COUNT = int(os.environ.get("IMTALKER_LOG_BACKUP_COUNT", "5"))
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def logs_dir() -> Path:
    """Resolve the logs directory: $LOGS_DIR, else $SPEECH2AVATAR_ROOT/logs,
    else <project root>/logs (IMTalker/'s parent). No local-PC assumptions --
    all three resolve correctly on a RunPod pod."""
    override = os.environ.get("LOGS_DIR", "").strip()
    if override:
        return Path(override)
    root = os.environ.get("SPEECH2AVATAR_ROOT", "").strip()
    if root:
        return Path(root) / "logs"
    return Path(__file__).resolve().parent.parent / "logs"


# -- Secret redaction --------------------------------------------------------
# Belt-and-suspenders: callers should never log a raw secret, but every
# record is scrubbed anyway before it reaches any handler (console or file).
_SECRET_RES = [
    re.compile(r"hf_[A-Za-z0-9]{10,}"),
    re.compile(r"tvly-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password|passwd|authorization)\b\s*[=:]\s*[\"']?[^\s\"',}]{6,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{10,}"),
]


class _RedactSecretsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        redacted = msg
        for pat in _SECRET_RES:
            redacted = pat.sub("***REDACTED***", redacted)
        if redacted != msg:
            record.msg = redacted
            record.args = ()
        return True


class _DefaultFieldFilter(logging.Filter):
    """Fills in any LogRecord fields the active Formatter references but a
    given call site forgot to supply via `extra=`, so a missing field can
    never crash logging (which would otherwise silently kill the caller's
    thread if it weren't inside the queue listener)."""

    def __init__(self, defaults: dict[str, Any]) -> None:
        super().__init__()
        self._defaults = dict(defaults)

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in self._defaults.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


def _install_queued_rotating_file(
    logger: logging.Logger, file_path: Path, fmt: str, extra_defaults: dict[str, Any] | None = None,
) -> None:
    """Attach `file_path` (rotating, capped at _MAX_BYTES * _BACKUP_COUNT
    total) to `logger` via a QueueHandler -> background QueueListener thread,
    so the calling thread never blocks on disk I/O."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        str(file_path), maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8", delay=True,
    )
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=_DATEFMT))
    file_handler.addFilter(_RedactSecretsFilter())
    if extra_defaults:
        file_handler.addFilter(_DefaultFieldFilter(extra_defaults))

    log_queue: "queue.Queue" = queue.Queue(-1)  # unbounded: never block the producer
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(queue_handler)

    listener = logging.handlers.QueueListener(log_queue, file_handler, respect_handler_level=True)
    listener.start()
    _LISTENERS.append(listener)
    atexit.register(listener.stop)


# -- System log (logs/system_runtime.log) ------------------------------------

def get_system_logger() -> logging.Logger:
    """Singleton logger for model/adapter loading, device/GPU info, server
    startup, and runtime errors/warnings. Safe to call from any thread/module;
    idempotent."""
    global _SYSTEM_LOGGER
    with _LOCK:
        if _SYSTEM_LOGGER is not None:
            return _SYSTEM_LOGGER
        logger = logging.getLogger("imtalker.system")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            fmt = "[%(asctime)s.%(msecs)03d] [SYSTEM] %(message)s"
            console = logging.StreamHandler(sys.stdout)
            console.setFormatter(logging.Formatter(fmt, datefmt=_DATEFMT))
            console.addFilter(_RedactSecretsFilter())
            logger.addHandler(console)
            try:
                _install_queued_rotating_file(logger, logs_dir() / "system_runtime.log", fmt)
            except Exception as e:  # logging must never break startup
                print(f"[runtime_logging] system_runtime.log disabled, could not open: {e!r}", flush=True)
        _SYSTEM_LOGGER = logger
        return logger


def log_event(logger: logging.Logger, component: str, event: str, level: int = logging.INFO, **fields: Any) -> None:
    """One structured line: '[component] event key=val key=val ...'."""
    try:
        parts = " ".join(f"{k}={v!r}" for k, v in fields.items() if v is not None)
        msg = f"[{component}] {event}" + (f" {parts}" if parts else "")
        logger.log(level, msg)
    except Exception:
        pass


class Timer:
    """Context manager that logs '<event> START', then '<event> DONE
    duration_ms=...' or '<event> FAILED duration_ms=... error=...'. Use to
    wrap any model/adapter load so start/end/duration/failure are captured
    without hand-writing three log lines at every call site.

        with Timer(get_system_logger(), "PersonaPlex", "load_mimi_lm", repo=repo):
            self.mimi = ...
            self.lm = ...
    """

    def __init__(self, logger: logging.Logger, component: str, event: str, **fields: Any) -> None:
        self.logger = logger
        self.component = component
        self.event = event
        self.fields = fields
        self._t0: float | None = None

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        log_event(self.logger, self.component, f"{self.event} START", **self.fields)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed_ms = round((time.perf_counter() - (self._t0 or time.perf_counter())) * 1000.0, 1)
        if exc_type is None:
            log_event(self.logger, self.component, f"{self.event} DONE", duration_ms=elapsed_ms, **self.fields)
        else:
            log_event(
                self.logger, self.component, f"{self.event} FAILED", level=logging.ERROR,
                duration_ms=elapsed_ms, error=repr(exc), **self.fields,
            )
        return False  # never suppress the exception


# -- Conversation log (logs/conversation.log) --------------------------------

def attach_conversation_file(logger: logging.Logger) -> None:
    """Attach the fixed-path logs/conversation.log rotating handler to an
    existing `logging.Logger` (conversation_logger.py's own logger). Every
    line that logger already writes (turn HEARD/DECIDE/SEARCH/GROUND/SAID,
    web_search/compressor/ref_injected events, and -- via
    latency_logger.py's `mirror_logger` -- every per-stage timing line) then
    also lands here, unified, with a conversation_id on every line. Idempotent
    per logger instance."""
    if getattr(logger, "_imtalker_conversation_file_attached", False):
        return
    fmt = "[%(asctime)s.%(msecs)03d] [conversation_id=%(conversation_id)s] %(message)s"
    try:
        _install_queued_rotating_file(
            logger, logs_dir() / "conversation.log", fmt, extra_defaults={"conversation_id": "-"},
        )
        logger._imtalker_conversation_file_attached = True  # type: ignore[attr-defined]
    except Exception as e:
        print(f"[runtime_logging] conversation.log disabled, could not open: {e!r}", flush=True)
