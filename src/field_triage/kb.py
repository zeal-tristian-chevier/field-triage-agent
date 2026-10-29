"""Retrieval over the synthetic manual snippets.

Chroma (vector) is the primary path. BM25 is the fallback because Chroma's
default embedder downloads a model on first use and can fail offline / in CI;
retrieval failing should degrade answer quality, not take the agent down.
"""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

KB_PATH = Path(__file__).resolve().parents[2] / "data" / "kb" / "snippets.jsonl"


def load_snippets(path: Path = KB_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Retriever:
    def __init__(self, snippets: list[dict] | None = None):
        self.snippets = snippets or load_snippets()
        self._bm25 = BM25Okapi([_tokens(s["title"] + " " + s["text"]) for s in self.snippets])
        self._collection = None  # built lazily so BM25-only use never imports chroma models

    def _chroma(self):
        if self._collection is None:
            import chromadb

            # upsert, not add: the ephemeral client is shared in-process, so a second
            # Retriever would hit duplicate ids and silently fall back to BM25.
            # Ephemeral: 18 docs rebuild in well under a second, so no on-disk
            # index to go stale or commit by accident.
            col = chromadb.EphemeralClient().get_or_create_collection("kb")
            col.upsert(
                ids=[s["id"] for s in self.snippets],
                documents=[s["title"] + ". " + s["text"] for s in self.snippets],
            )
            self._collection = col
        return self._collection

    def search(self, query: str, k: int = 3) -> tuple[str, list[dict]]:
        """Return (backend_used, snippets). Raises only if both backends fail."""
        by_id = {s["id"]: s for s in self.snippets}
        try:
            res = self._chroma().query(query_texts=[query], n_results=k)
            return "chroma", [by_id[i] for i in res["ids"][0]]
        except Exception:
            scores = self._bm25.get_scores(_tokens(query))
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            return "bm25", [self.snippets[i] for i in ranked]
