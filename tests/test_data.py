"""Tests for the synthetic task family (pure Python, no model needed)."""
import pytest

from sequential_adapt.data import (NONCE_WORDS, WIDE_LABEL_SPACE, make_tasks,
                                   nonce_pool)

LABELS = (" red", " blue", " green", " yellow", " purple", " orange")


def test_default_tasks_unchanged():
    """overlap=0 inside the original pool must reproduce the historical
    task sets exactly (older artifacts stay comparable)."""
    tasks = make_tasks(3, 4, LABELS)
    words = [f.word for t in tasks for f in t.facts]
    assert words == NONCE_WORDS[:12]
    for t_idx, task in enumerate(tasks):
        for i, fact in enumerate(task.facts):
            assert fact.label == LABELS[(i + t_idx) % len(LABELS)]


def test_nonce_pool_extends_deterministically():
    pool = nonce_pool(100)
    assert pool[:18] == NONCE_WORDS
    assert len(pool) == 100 and len(set(pool)) == 100
    assert all(w.isalpha() and w.islower() for w in pool)
    # silent-e real words (babe, bake, ...) must be unreachable
    assert not any(w.endswith("e") for w in pool[18:])
    assert nonce_pool(100) == pool  # deterministic


def test_large_task_family_no_longer_capped():
    tasks = make_tasks(6, 10, WIDE_LABEL_SPACE)
    words = [f.word for t in tasks for f in t.facts]
    assert len(words) == 60 and len(set(words)) == 60


def test_overlap_words_conflict_across_domains():
    tasks = make_tasks(3, 4, LABELS, overlap_words=2)
    # Shared words appear in every domain...
    for i in range(2):
        shared = {t.facts[i].word for t in tasks}
        assert len(shared) == 1
        # ...with a different label per domain (the conflict).
        labels = {t.facts[i].label for t in tasks}
        assert len(labels) == len(tasks)
    # Non-overlapping facts stay disjoint across domains.
    rest = [f.word for t in tasks for f in t.facts[2:]]
    assert len(rest) == len(set(rest))


def test_overlap_validation():
    with pytest.raises(ValueError):
        make_tasks(3, 4, LABELS, overlap_words=5)
    with pytest.raises(ValueError):
        make_tasks(7, 4, LABELS, overlap_words=1)  # 7 domains, 6 labels
