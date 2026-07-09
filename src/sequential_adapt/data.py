"""Synthetic fact-mapping task family.

Each task is a "domain" holding a few nonce-word -> color facts, e.g.
domain A: blicket -> red.  The model is adapted to answer prompts like
"In domain A, the word blicket maps to the color" with " red".

Answers are restricted to single-token color words so accuracy is a
restricted argmax over the label space, which stays meaningful even for a
weak base model.
"""

from dataclasses import dataclass

NONCE_WORDS = [
    "blicket", "dax", "wug", "fep", "toma", "gazzer",
    "kiki", "bouba", "zorp", "mib", "lorp", "quen",
    "vimp", "snod", "trill", "yark", "pilk", "chiv",
]

DOMAIN_NAMES = ["A", "B", "C", "D", "E", "F"]

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


def make_tasks(n_tasks: int, facts_per_task: int, label_space) -> list:
    """Disjoint nonce words per domain; labels cycle so tasks differ."""
    needed = n_tasks * facts_per_task
    if needed > len(NONCE_WORDS):
        raise ValueError(f"Need {needed} nonce words, have {len(NONCE_WORDS)}")
    tasks = []
    for t in range(n_tasks):
        domain = DOMAIN_NAMES[t]
        facts = []
        for i in range(facts_per_task):
            word = NONCE_WORDS[t * facts_per_task + i]
            # Offset labels per domain so domains use different label patterns.
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
