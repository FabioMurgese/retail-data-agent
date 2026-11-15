"""Golden Bucket: the trios are valid for the guard, and retrieval works with and without embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retail_agent.golden import GoldenRetriever, Trio, load_trios
from retail_agent.policy import ConfigFileError, load_data_policy, load_user_directory
from retail_agent.safety.sql_guard import SqlGuard

ROOT = Path(__file__).resolve().parents[1]
TRIOS = load_trios(ROOT / "golden" / "trios")
POLICY = load_data_policy(ROOT / "config")
USERS = load_user_directory(ROOT / "config")


@pytest.mark.parametrize("trio", TRIOS, ids=lambda t: t.id)
def test_every_trio_passes_the_guard_for_every_user(trio):
    for profile in USERS.users.values():
        SqlGuard(POLICY, profile.scope, max_rows=1000).check(trio.sql)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Why are users in Ohio spending less than users in New York?", "state-spend-comparison"),
        ("What happened to churn in October?", "churn-rate-spike"),
        ("Build me a Q2 report with action items", "quarterly-report"),
        ("Which categories get returned the most?", "return-rate-by-category"),
        ("Who are our best customers?", "top-customers-by-spend"),
    ],
)
def test_bm25_finds_the_relevant_trio(question, expected):
    result = GoldenRetriever(TRIOS, embedder=None, cache_path=None).search(question, k=3)
    assert result.method == "bm25"
    assert result.trios[0][0].id == expected


class FakeEmbedder:
    model = "fake-embedding"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, int]] = []

    def embed(self, texts, *, task):
        self.calls.append((task, len(texts)))
        if self.fail:
            raise RuntimeError("429 quota exceeded")
        # one dimension per keyword: enough to make similarity meaningful
        return [[float(w in t.lower()) for w in ("churn", "report", "return", "customer")] for t in texts]


def test_embedding_retrieval_caches_document_vectors(tmp_path):
    cache = tmp_path / "vectors.json"
    embedder = FakeEmbedder()
    result = GoldenRetriever(TRIOS, embedder=embedder, cache_path=cache).search("churn last month", k=1)

    assert result.method == "embedding" and result.trios[0][0].id == "churn-rate-spike"
    assert embedder.calls == [("document", len(TRIOS)), ("query", 1)]
    assert len(json.loads(cache.read_text())) == len(TRIOS)

    second = FakeEmbedder()  # a new session reuses the cache: only the query is embedded
    GoldenRetriever(TRIOS, embedder=second, cache_path=cache).search("returns", k=1)
    assert second.calls == [("query", 1)]


def test_embedding_failure_falls_back_to_bm25():
    result = GoldenRetriever(TRIOS, embedder=FakeEmbedder(fail=True), cache_path=None).search("churn spike")
    assert result.method == "bm25" and "quota exceeded" in result.error
    assert result.trios[0][0].id == "churn-rate-spike"


def test_duplicate_trio_ids_are_rejected(tmp_path):
    source = (ROOT / "golden" / "trios" / "churn-rate-spike.yaml").read_text()
    (tmp_path / "a.yaml").write_text(source)
    (tmp_path / "b.yaml").write_text(source)
    with pytest.raises(ConfigFileError, match="duplicate trio ids"):
        load_trios(tmp_path)


def test_trio_search_text_includes_tags():
    assert "Topics: churn" in next(t for t in TRIOS if t.id == "churn-rate-spike").search_text
    assert isinstance(TRIOS[0], Trio)
