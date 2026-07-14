"""Experiment configuration. One dataclass, no framework."""

from dataclasses import dataclass, field, asdict

SITE_GROUPS = {
    "both": ("attn.c_attn", "mlp.c_fc"),
    "attn": ("attn.c_attn",),
    "mlp": ("mlp.c_fc",),
}


def resolve_site_suffixes(sites: str = "both", layers=None) -> tuple:
    """Map a --sites/--layers CLI spec to Config.site_suffixes.

    Total coefficient dims = n_sites * n_components, so restricting sites
    or layers varies dims independently of k. `layers` is None (all
    layers, generic suffixes) or a spec like "0-2" / "0,3,5", producing
    layer-qualified suffixes ("h.0.attn.c_attn", ...) that each match
    exactly one module in GPT-2-family models.
    """
    if sites not in SITE_GROUPS:
        raise ValueError(f"sites must be one of {sorted(SITE_GROUPS)}, "
                         f"got {sites!r}")
    base = SITE_GROUPS[sites]
    if layers is None:
        return base
    idxs = set()
    for part in str(layers).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo, hi = int(lo), int(hi)
            if hi < lo:
                raise ValueError(f"bad layer range {part!r}")
            idxs.update(range(lo, hi + 1))
        elif part:
            idxs.add(int(part))
    if not idxs or min(idxs) < 0:
        raise ValueError(f"bad layer spec {layers!r}")
    return tuple(f"h.{i}.{suf}" for i in sorted(idxs) for suf in base)


@dataclass
class Config:
    # Model
    model_name: str = "distilgpt2"
    device: str = "auto"  # "auto" -> cuda if available, else cpu
    seed: int = 0

    # Adapter geometry
    # Module-name suffixes (within transformer blocks) that receive adapters.
    site_suffixes: tuple = ("attn.c_attn", "mlp.c_fc")
    rank: int = 4              # rank of each basis component
    n_components: int = 8      # frozen random low-rank components per site
    lora_rank: int = 4         # rank for the independent trainable LoRA baseline
    lora_alpha: float = 8.0

    # Task family
    n_tasks: int = 3
    facts_per_task: int = 4
    overlap_words: int = 0     # shared nonce words with conflicting labels across domains (adversarial; 0 = disjoint)
    # Templates 0..1 are used for training. Task accuracy/retention is
    # measured on template `eval_template` (a TRAINED phrasing: the question
    # is whether learned behavior survives later adaptation). Template
    # `heldout_template` is never trained on and feeds the paraphrase
    # coherence probe (generalization).
    train_templates: tuple = (0, 1)
    eval_template: int = 0
    heldout_template: int = 2

    # Fitting
    steps: int = 200
    lr: float = 5e-2           # for coefficients over random bases
    lora_lr: float = 5e-3      # for trainable LoRA matrices
    l2: float = 1e-4           # L2 on coefficients
    ortho_penalty: float = 0.1  # cosine^2 penalty vs earlier tasks' coefficients (controller method)
    hard_ortho: bool = False   # project new coeffs onto the orthogonal complement of earlier tasks' directions (supersedes ortho_penalty)
    anchor_weight: float = 0.0  # KL(base||adapted) on neutral probes during fitting (0 = off)
    replay_weight: float = 0.0  # CE on earlier tasks' examples, evaluated in the composed state, during later fitting (0 = off)
    replay_fraction: float = 1.0  # fraction of each earlier task's examples used for replay (deterministic subsample; 1.0 = full rehearsal)
    train_gates: bool = True   # Model C: per-site gates for the controller method

    # Controller (Model B)
    controller_hidden: int = 64
    controller_steps: int = 500
    controller_lr: float = 1e-2

    # Eval
    label_space: tuple = (" red", " blue", " green", " yellow", " purple", " orange")

    extras: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self):
        d = asdict(self)
        d["site_suffixes"] = list(self.site_suffixes)
        d["train_templates"] = list(self.train_templates)
        d["label_space"] = list(self.label_space)
        return d


def smoke_config(**overrides) -> Config:
    """Reduced-step settings for smoke test and unit tests."""
    defaults = dict(steps=80, controller_steps=300)
    defaults.update(overrides)
    return Config(**defaults)
