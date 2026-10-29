"""Logging and LangSmith wiring.

One id ties both together: trace_id is used as the LangSmith root run_id, so a
log line's trace_id can be pasted straight into LangSmith to find the trace.
Tracing itself is switched on by env (LANGSMITH_TRACING / LANGSMITH_API_KEY);
this module only decides names, tags and metadata.
"""

import contextvars
import json
import logging
import os
import uuid

from langchain_core.runnables import RunnableLambda

from .config import triage_model

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")
log = logging.getLogger("field_triage")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "event": record.getMessage(),
            # contextvar, not a parameter: every log line in the run gets the
            # id without threading it through each function
            "trace_id": trace_id_var.get(),
            **getattr(record, "fields", {}),
        }
        return json.dumps(entry, default=str)


def setup_logging() -> None:
    if log.handlers:  # idempotent: avoid duplicate lines when called twice
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    log.addHandler(handler)
    log.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    log.propagate = False


def run_config(trace_id: str, extra_tags: tuple[str, ...] = (), extra_metadata: dict | None = None) -> dict:
    """RunnableConfig for the root run. Tags make runs filterable in LangSmith;
    dataset version is a tag so experiments on different data are never compared by accident."""
    model = triage_model()
    return {
        "run_name": "field-triage",
        "run_id": uuid.UUID(trace_id),
        "tags": [f"model:{model}", f"dataset:{os.getenv('DATASET_VERSION', 'v1')}", *extra_tags],
        "metadata": {"trace_id": trace_id, "model": model, **(extra_metadata or {})},
    }


def tracing_enabled() -> bool:
    on = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    if on and not os.getenv("LANGSMITH_API_KEY"):
        # Tracing "on" without a key fails quietly inside the SDK; say so loudly instead
        log.warning("LANGSMITH_TRACING=true but LANGSMITH_API_KEY is unset; traces will not upload")
        return False
    return on


def traced_node(name: str, fn):
    """Wrap a node so it shows up in LangSmith as `name` with a node:<name> tag,
    and logs what it did, including any errors it recorded."""

    def wrapper(state):
        out = fn(state)
        errors = [e.message for e in out.get("errors", [])]
        log.info("node_done", extra={"fields": {"node": name, "errors": errors}})
        return out

    return RunnableLambda(wrapper, name=name).with_config(tags=[f"node:{name}"])
