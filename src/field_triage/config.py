"""Env-driven config. Read at call time (not import time) so tests can monkeypatch."""

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "claude-sonnet-5-5"


def triage_model() -> str:
    return os.getenv("TRIAGE_MODEL", DEFAULT_MODEL)


def default_llm():
    from langchain_anthropic import ChatAnthropic

    # temperature 0: same report should give the same answer run to run,
    # which is what makes eval comparisons meaningful
    return ChatAnthropic(model=triage_model(), temperature=0, max_tokens=1024)
