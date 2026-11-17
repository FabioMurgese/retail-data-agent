"""Golden Bucket (light): analysts' Question -> SQL -> Report trios, retrieved at question time.

Trios live in `golden/trios/*.yaml`, one file each: easy to review in a pull request, and a stand-in
for the production bucket (GCS + a vector index, see docs/HLD.md). They carry what an LLM cannot
know: business definitions (net revenue, churn, return rate), data quirks ('Deutschland' = Germany)
and how to break a "why" question into queries.

Retrieval: Gemini embeddings (cosine similarity), document vectors cached on disk by content hash.
If embeddings are unavailable (no key, quota, outage) the retriever falls back to BM25 over the
question and tags, so a failure here degrades relevance a little and never blocks a turn.

Learning loop (system level): a user's `/good` on an answer backed by queries writes a *candidate*
trio to `golden/candidates/`. Candidates are never retrieved. `retail-agent golden curate`
(`curation.py`) checks, generalises and judges them automatically: safe, novel, well-judged ones join
`golden/trios/` with `status: probation`; doubtful ones go to the human review queue; unsafe ones are
rejected. Humans own the rules (glossary, thresholds) and the exceptions, not every example.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field

from retail_agent.policy import ConfigFileError, load_yaml_model

log = logging.getLogger(__name__)


class Trio(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    question: str
    tags: tuple[str, ...] = ()
    author: str
    created: date
    status: Literal["approved", "probation"] = "approved"  # probation = approved automatically, on trial
    source: str | None = None  # provenance of a curated trio (candidate, trace, judge score)
    metrics: tuple[str, ...] = ()  # glossary metrics it relies on: re-check these trios when one changes
    sql: str = Field(min_length=1)
    report: str = Field(min_length=1)

    @property
    def search_text(self) -> str:
        return f"{self.question}\nTopics: {', '.join(self.tags)}"


def load_trios(directory: Path) -> list[Trio]:
    trios = [load_yaml_model(Trio, path) for path in sorted(directory.glob("*.yaml"))]
    if duplicates := [i for i, n in Counter(t.id for t in trios).items() if n > 1]:
        raise ConfigFileError(f"{directory}: duplicate trio ids {duplicates}")
    return trios


class _LiteralDumper(yaml.SafeDumper):
    """Multi-line strings as `|` blocks, so SQL and reports stay readable in review."""


_LiteralDumper.add_representer(
    str,
    lambda dumper, text: dumper.represent_scalar(
        "tag:yaml.org,2002:str", text, style="|" if "\n" in text else None
    ),
)


def dump_yaml(data: dict) -> str:
    return yaml.dump(data, Dumper=_LiteralDumper, sort_keys=False, allow_unicode=True, width=110)


def write_candidate(
    directory: Path,
    *,
    trace_id: str,
    user_id: str,
    question: str,
    answer: str,
    queries: Sequence[str],
    today: date,
    warnings: Sequence[str] = (),
) -> Path:
    """Propose a trio from a well-rated answer: the last query becomes `sql`, the others are kept as
    `other_queries`. `warnings` (e.g. figures the grounding check could not verify) are kept for
    curation, which then sends the candidate to human review instead of approving it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    candidate = {
        "id": f"candidate-{trace_id}",
        "question": question,
        "author": f"{user_id} (via /good)",
        "created": today,
        "trace_id": trace_id,
        "warnings": list(warnings),
        "sql": queries[-1].strip() + "\n",
        "other_queries": [q.strip() + "\n" for q in queries[:-1]],
        "report": answer.strip() + "\n",
    }
    header = (
        f"# Candidate trio proposed by {user_id} on {today} (trace {trace_id}).\n"
        "# Not used by the assistant. `retail-agent golden curate` checks, generalises and judges it:\n"
        "# approved (on probation) into golden/trios/, sent to golden/review/, or rejected.\n"
    )
    path = directory / f"{today}-{trace_id}.yaml"
    path.write_text(header + dump_yaml(candidate), encoding="utf-8")
    return path


# --- Embeddings ------------------------------------------------------------------------------------


class Embedder(Protocol):
    model: str

    def embed(self, texts: Sequence[str], *, task: Literal["document", "query"]) -> list[list[float]]: ...


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str, dimensions: int = 768) -> None:
        from google import genai  # slow import, only when embeddings are used

        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str], *, task: Literal["document", "query"]) -> list[list[float]]:
        from google.genai import types

        config = types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT" if task == "document" else "RETRIEVAL_QUERY",
            output_dimensionality=self.dimensions,
        )
        response = self._client.models.embed_content(model=self.model, contents=list(texts), config=config)
        return [list(e.values) for e in response.embeddings]


def _normalise(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


# --- BM25 fallback ---------------------------------------------------------------------------------

_STOPWORDS = frozenset(
    (
        "a an and are as at be by did do does for from how in is it of on or our the to was we what "
        "which who why with this that last"
    ).split()
)


def _stem(token: str) -> str:
    """Tiny suffix stripper, enough for "spending" ~ "spend" and "users" ~ "user"."""
    for suffix, min_length in (("ing", 6), ("ed", 5), ("es", 5), ("s", 4)):
        if token.endswith(suffix) and len(token) >= min_length and not token.endswith("ss"):
            return token[: -len(suffix)]
    return token


def _tokens(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


class _BM25:
    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = [Counter(_tokens(d)) for d in documents]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_length = sum(self.lengths) / max(len(self.docs), 1)
        frequency = Counter(term for doc in self.docs for term in doc)
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in frequency.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> list[float]:
        terms = _tokens(query)
        result = []
        for doc, length in zip(self.docs, self.lengths, strict=True):
            score = 0.0
            for term in terms:
                if tf := doc.get(term):
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_length)
                    score += self.idf[term] * tf * (self.k1 + 1) / norm
            result.append(score)
        return result


# --- Retriever -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Retrieval:
    trios: list[tuple[Trio, float]]  # best first, with score
    method: Literal["embedding", "bm25"]
    error: str | None = None  # why embeddings were not used, if they were expected


class GoldenRetriever:
    def __init__(self, trios: Sequence[Trio], *, embedder: Embedder | None, cache_path: Path | None) -> None:
        self.trios = list(trios)
        self.embedder = embedder
        self.cache_path = cache_path
        self._bm25 = _BM25([t.search_text for t in self.trios])
        self._doc_vectors: list[list[float]] | None = None

    def search(self, question: str, k: int = 3) -> Retrieval:
        if not self.trios:
            return Retrieval([], "bm25")
        error = None
        if self.embedder is not None:
            try:
                return Retrieval(self._by_embedding(question, k), "embedding")
            except Exception as exc:  # quota, outage, bad key: degrade to BM25, never block the turn
                error = f"{type(exc).__name__}: {exc}"[:300]
                log.warning("golden retrieval: embeddings unavailable, using BM25 (%s)", error)
        scored = sorted(zip(self.trios, self._bm25.scores(question), strict=True), key=lambda p: -p[1])
        return Retrieval([(t, s) for t, s in scored[:k] if s > 0], "bm25", error)

    def nearest(self, text: str, k: int = 3) -> list[tuple[Trio, float]]:
        """Trios most similar to another trio's search text (document vs documents), for duplicate
        detection. Embeddings only: BM25 scores are not comparable across texts. Raises if unavailable."""
        if self.embedder is None:
            raise RuntimeError("no embedding model configured")
        if not self.trios:
            return []
        documents = self._document_vectors()
        vector = _normalise(self.embedder.embed([text], task="document")[0])
        scored = [
            (t, sum(a * b for a, b in zip(vector, v, strict=True)))
            for t, v in zip(self.trios, documents, strict=True)
        ]
        return sorted(scored, key=lambda p: -p[1])[:k]

    def _by_embedding(self, question: str, k: int) -> list[tuple[Trio, float]]:
        assert self.embedder is not None
        documents = self._document_vectors()
        query = _normalise(self.embedder.embed([question], task="query")[0])
        scored = [
            (t, sum(a * b for a, b in zip(query, v, strict=True)))
            for t, v in zip(self.trios, documents, strict=True)
        ]
        return sorted(scored, key=lambda p: -p[1])[:k]

    def _document_vectors(self) -> list[list[float]]:
        if self._doc_vectors is not None:
            return self._doc_vectors
        assert self.embedder is not None
        cache = self._read_cache()
        keys = [self._cache_key(t.search_text) for t in self.trios]
        missing = [(key, t.search_text) for key, t in zip(keys, self.trios, strict=True) if key not in cache]
        if missing:
            vectors = self.embedder.embed([text for _, text in missing], task="document")
            cache.update({key: _normalise(v) for (key, _), v in zip(missing, vectors, strict=True)})
            self._write_cache(cache)
        self._doc_vectors = [cache[key] for key in keys]
        return self._doc_vectors

    def _cache_key(self, text: str) -> str:
        assert self.embedder is not None
        return hashlib.sha256(f"{self.embedder.model}\n{text}".encode()).hexdigest()

    def _read_cache(self) -> dict[str, list[float]]:
        if self.cache_path is None or not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}  # a corrupt cache is only a cost, never an error

    def _write_cache(self, cache: dict[str, list[float]]) -> None:
        if self.cache_path is None:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(cache), encoding="utf-8")
        except OSError as exc:
            log.warning("golden retrieval: cannot write embedding cache %s: %s", self.cache_path, exc)
