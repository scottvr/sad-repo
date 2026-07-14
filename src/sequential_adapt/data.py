"""Synthetic fact-mapping task family.

Each task is a "domain" holding a few nonce-word -> color facts, e.g.
domain A: blicket -> red.  The model is adapted to answer prompts like
"In domain A, the word blicket maps to the color" with " red".

Answers are restricted to single-token color words so accuracy is a
restricted argmax over the label space, which stays meaningful even for a
weak base model.
"""

import itertools
import string
from dataclasses import dataclass

NONCE_WORDS = [
    "blicket", "dax", "wug", "fep", "toma", "gazzer",
    "kiki", "bouba", "zorp", "mib", "lorp", "quen",
    "vimp", "snod", "trill", "yark", "pilk", "chiv",
]

# CVCV generator alphabet for extending the pool past the hand-picked 18.
# The final vowel excludes 'e' so the silent-e class of real words (babe,
# bade, bake, ...) can never be generated.
_NONCE_CONSONANTS = "bdfgjklmnprstvz"
_NONCE_VOWELS = "aeiou"
_NONCE_FINAL_VOWELS = "aiou"
# Best-effort blocklist for the remaining real CVCV words reachable by the
# generator ("no prior association" is the point of nonce facts).
_REAL_CVCV = {
    "baba", "bozo", "dada", "dado", "demo", "dino", "dodo", "dojo", "fifo",
    "fila", "gaga", "gala", "gogo", "guru", "jojo", "judo", "juju", "juno",
    "kaka", "kilo", "koko", "kudu", "lava", "lilo", "lima", "limo", "logo",
    "mama", "memo", "menu", "mesa", "milo", "mojo", "nana", "nova", "papa",
    "peso", "pogo", "polo", "saga", "sago", "silo", "soda", "sofa", "solo",
    "sumo", "taro", "toga", "tofu", "tuba", "tuna", "tutu", "vaso", "veto",
    "vino", "visa", "zulu",
}

DOMAIN_NAMES = list(string.ascii_uppercase)

# Wider single-token answer set for pressure runs (default label_space stays
# the original 6 for reproducibility). check_single_token_labels() verifies
# tokenization at runtime.
WIDE_LABEL_SPACE = (" red", " blue", " green", " yellow", " purple",
                    " orange", " black", " white", " brown", " pink",
                    " gray", " gold")


def nonce_pool(n: int) -> list:
    """Deterministic nonce-word pool of size n.

    The first 18 entries are the original hand-picked words, so any run
    that fit inside the old pool reproduces exactly. Beyond that, CVCV
    strings extend the pool (~5600 available)."""
    pool = list(NONCE_WORDS)
    if n <= len(pool):
        return pool[:n]
    seen = set(pool)
    for c1, v1, c2, v2 in itertools.product(_NONCE_CONSONANTS, _NONCE_VOWELS,
                                            _NONCE_CONSONANTS,
                                            _NONCE_FINAL_VOWELS):
        word = c1 + v1 + c2 + v2
        if word in seen or word in _REAL_CVCV:
            continue
        pool.append(word)
        seen.add(word)
        if len(pool) >= n:
            return pool
    raise ValueError(f"nonce pool exhausted at {len(pool)} < {n}")

# Prompt templates. {d}=domain, {w}=word. The answer is a leading-space color token.
TEMPLATES = [
    "In domain {d}, the word {w} maps to the color",
    "Domain {d} rule: {w} is assigned the color",
    "Within domain {d}, the color associated with {w} is",
]

# Context prompts describing a domain (used for controller context embeddings).
CONTEXT_TEMPLATES = [
    "In domain {d},",
    "Domain {d} rule:",
    "Within domain {d},",
    "Considering domain {d} conventions,",
    "We are now operating in domain {d}.",
    "Switch to domain {d}.",
]

NEUTRAL_PROBES = [
    "The capital of France is",
    "Once upon a time, there was a",
    "The weather today is",
    "Two plus two equals",
    "The quick brown fox jumps over the",
    "My favorite food is",
    "Water is made of hydrogen and",
    "The sun rises in the",
]


@dataclass
class Fact:
    domain: str
    word: str
    label: str  # e.g. " red" (leading space, single token)


@dataclass
class Task:
    name: str            # e.g. "task_A"
    domain: str          # e.g. "A"
    facts: list          # list[Fact]


def make_tasks(n_tasks: int, facts_per_task: int, label_space,
               overlap_words: int = 0) -> list:
    """Nonce-word -> label facts per domain; labels cycle so tasks differ.

    overlap_words: the first `overlap_words` facts of EVERY domain reuse the
    same shared nonce words. Because the label index is domain-shifted, the
    same word maps to a DIFFERENT label in each domain — deliberately
    conflicting facts that a single composed state cannot satisfy without
    genuine context conditioning. Requires n_tasks <= len(label_space) so
    the conflicting labels are actually distinct. 0 = fully disjoint
    (original behavior, byte-identical task sets).
    """
    if overlap_words > facts_per_task:
        raise ValueError("overlap_words cannot exceed facts_per_task")
    if overlap_words and n_tasks > len(label_space):
        raise ValueError("overlap conflicts need n_tasks <= len(label_space)")
    unique_per_task = facts_per_task - overlap_words
    pool = nonce_pool(overlap_words + n_tasks * unique_per_task)
    shared = pool[:overlap_words]
    unique = pool[overlap_words:]
    tasks = []
    for t in range(n_tasks):
        domain = DOMAIN_NAMES[t]
        facts = []
        for i in range(facts_per_task):
            if i < overlap_words:
                word = shared[i]
            else:
                word = unique[t * unique_per_task + (i - overlap_words)]
            # Offset labels per domain so domains use different label patterns
            # (and so shared words get conflicting labels across domains).
            label = label_space[(i + t) % len(label_space)]
            facts.append(Fact(domain=domain, word=word, label=label))
        tasks.append(Task(name=f"task_{domain}", domain=domain, facts=facts))
    return tasks


def fact_prompt(fact: Fact, template_idx: int) -> str:
    return TEMPLATES[template_idx].format(d=fact.domain, w=fact.word)


def task_prompts(task: Task, template_idx: int):
    """(prompt, answer) pairs for one template."""
    return [(fact_prompt(f, template_idx), f.label) for f in task.facts]


def context_prompts(task: Task):
    return [t.format(d=task.domain) for t in CONTEXT_TEMPLATES]


def check_single_token_labels(tokenizer, label_space):
    """Every label must encode to exactly one token; returns label -> token id."""
    out = {}
    for label in label_space:
        ids = tokenizer.encode(label)
        if len(ids) != 1:
            raise ValueError(f"Label {label!r} is not a single token: {ids}")
        out[label] = ids[0]
    return out
