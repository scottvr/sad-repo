"""Low-rank adapters over a frozen base model.

Two adapter kinds:

1. `LoRAAdapter` — a standard trainable low-rank delta (B @ A), one per task.
   Used for the independent-adapter and naive-stacking baselines.

2. `SharedBasisAdapter` — a FROZEN random bank of K rank-r components per
   site. A task is only a coefficient vector c (K numbers per site); the
   weight delta is sum_k c_k * (B_k @ A_k). Deltas are linear in c, so task
   composition is exact vector addition and reversal is exact negation in
   parameter space. Any residual behavioral interference is therefore a
   property of composition through the frozen network, not of sloppy
   parameter bookkeeping. This is the substrate for Model B (coefficient
   controller) and Model C (gating).

Both wrap GPT-2-style Conv1D (weight [d_in, d_out]) or nn.Linear modules by
replacing them with an `AdapterSite` that adds delta(x) to the base output.
"""

import torch
import torch.nn as nn


def _module_dims(module):
    """(d_in, d_out) for nn.Linear or transformers Conv1D (duck-typed)."""
    if isinstance(module, nn.Linear):
        return module.in_features, module.out_features
    w = getattr(module, "weight", None)
    if w is not None and hasattr(module, "nf"):  # transformers Conv1D
        return w.shape[0], w.shape[1]
    raise TypeError(f"Cannot adapt module of type {type(module)}")


class LoRAAdapter(nn.Module):
    """Trainable delta(x) = scale * (x @ A^T) @ B^T. B starts at zero."""

    def __init__(self, d_in, d_out, rank, alpha, generator=None):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, rank))
        with torch.no_grad():
            self.A.normal_(0.0, 0.02, generator=generator)
        self.scale = alpha / rank

    def forward(self, x):
        return (x @ self.A.T) @ self.B.T * self.scale


class SharedBasisAdapter(nn.Module):
    """Frozen random components; delta(x) = sum_k c_k * (x @ A_k^T) @ B_k^T.

    Coefficients are NOT parameters of this module. `coeff_terms` is a list of
    (sign, task_entry, site_idx) tuples set externally by the AdapterBank;
    task_entry is the bank's {"coeffs": [n_sites, K], "gate_logits": ...}
    dict. The effective coefficient vector is recomputed lazily on every
    forward so that optimizer updates to coefficients/gate logits are always
    reflected and no autograd graph is reused across steps. Empty list =>
    adapter is a no-op.
    """

    def __init__(self, d_in, d_out, rank, n_components, generator=None):
        super().__init__()
        A = torch.empty(n_components, rank, d_in)
        B = torch.empty(n_components, d_out, rank)
        with torch.no_grad():
            A.normal_(0.0, d_in ** -0.5, generator=generator)
            B.normal_(0.0, rank ** -0.5, generator=generator)
        self.register_buffer("A_bases", A)
        self.register_buffer("B_bases", B)
        self.n_components = n_components
        self.coeff_terms = []  # list[(sign, task_entry, site_idx)]

    def effective_coeff(self):
        if not self.coeff_terms:
            return None
        total = None
        for sign, entry, i in self.coeff_terms:
            c = entry["coeffs"][i]
            if entry.get("gate_logits") is not None:
                c = torch.sigmoid(entry["gate_logits"][i]) * c
            term = sign * c
            total = term if total is None else total + term
        return total

    def forward(self, x):
        c = self.effective_coeff()
        if c is None:
            return torch.zeros(x.shape[:-1] + (self.B_bases.shape[1],),
                               dtype=x.dtype, device=x.device)
        # xa: [..., K, r]
        xa = torch.einsum("...i,kri->...kr", x, self.A_bases)
        return torch.einsum("...kr,kor,k->...o", xa, self.B_bases, c)


class AdapterSite(nn.Module):
    """Replaces a base linear/Conv1D module; adds enabled adapters' deltas."""

    def __init__(self, base_module):
        super().__init__()
        self.base = base_module
        self.d_in, self.d_out = _module_dims(base_module)
        self.adapters = nn.ModuleDict()
        self.enabled = {}

    def add_adapter(self, name, adapter, enabled=True):
        self.adapters[name] = adapter
        self.enabled[name] = enabled

    def set_enabled(self, name, flag):
        if name not in self.adapters:
            raise KeyError(name)
        self.enabled[name] = flag

    def forward(self, x):
        y = self.base(x)
        for name, adapter in self.adapters.items():
            if self.enabled.get(name, False):
                y = y + adapter(x)
        return y


def attach_adapter_sites(model, site_suffixes):
    """Replace matching submodules with AdapterSite wrappers.

    Returns an ordered dict name -> AdapterSite. Idempotent-hostile: call once.
    """
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, AdapterSite):
            raise RuntimeError("attach_adapter_sites called twice on this model")
        if any(name.endswith(suf) for suf in site_suffixes):
            targets.append(name)
    sites = {}
    for name in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        site = AdapterSite(getattr(parent, child))
        setattr(parent, child, site)
        sites[name] = site
    if not sites:
        raise ValueError(f"No modules matched suffixes {site_suffixes}")
    return sites


class AdapterBank:
    """Shared-basis coefficient store across all sites.

    A task is: coeffs [n_sites, K] (+ optional gate logits [n_sites]).
    `apply([(task, sign), ...])` composes tasks onto the model; coefficients
    referenced live, so optimizing a task's tensors updates behavior directly.
    """

    def __init__(self, sites: dict, rank: int, n_components: int, seed: int = 0,
                 device: str = "cpu"):
        self.site_names = list(sites.keys())
        self.sites = sites
        self.n_components = n_components
        self.device = device
        gen = torch.Generator().manual_seed(seed)  # CPU gen: same bases on any device
        for name in self.site_names:
            site = sites[name]
            shared = SharedBasisAdapter(site.d_in, site.d_out, rank,
                                        n_components, generator=gen)
            site.add_adapter("shared", shared.to(device), enabled=True)
        self.tasks = {}      # name -> {"coeffs": [n_sites, K], "gate_logits": [n_sites] | None}
        self.applied = []    # list[(task_name, sign)]

    @property
    def n_sites(self):
        return len(self.site_names)

    def new_task(self, name, train_gates=False, requires_grad=True):
        coeffs = torch.zeros(self.n_sites, self.n_components,
                             device=self.device, requires_grad=requires_grad)
        gate_logits = None
        if train_gates:
            # init ~= gate 0.98 so gated and ungated tasks start comparable
            gate_logits = torch.full((self.n_sites,), 4.0, device=self.device,
                                     requires_grad=requires_grad)
        self.tasks[name] = {"coeffs": coeffs, "gate_logits": gate_logits}
        return self.tasks[name]

    def set_task(self, name, coeffs, gate_logits=None):
        coeffs = coeffs.detach().clone()
        if coeffs.shape != (self.n_sites, self.n_components):
            raise ValueError(f"coeffs shape {tuple(coeffs.shape)} != "
                             f"{(self.n_sites, self.n_components)}")
        gl = None if gate_logits is None else gate_logits.detach().clone()
        self.tasks[name] = {"coeffs": coeffs, "gate_logits": gl}

    def trainable_params(self, name):
        t = self.tasks[name]
        params = [t["coeffs"]]
        if t["gate_logits"] is not None:
            params.append(t["gate_logits"])
        return params

    def gates(self, name):
        t = self.tasks[name]
        if t["gate_logits"] is None:
            return torch.ones(self.n_sites)
        return torch.sigmoid(t["gate_logits"]).cpu()

    def flat_coeffs(self, name):
        """Effective (gated) coefficients flattened to [n_sites * K]."""
        t = self.tasks[name]
        c = t["coeffs"]
        if t["gate_logits"] is not None:
            c = torch.sigmoid(t["gate_logits"]).unsqueeze(-1) * c
        return c.reshape(-1).detach().clone()

    def apply(self, applied):
        """applied: list of (task_name, sign). Rebinds live coefficient refs;
        gates/coefficients are read lazily at forward time."""
        self.applied = list(applied)
        for i, name in enumerate(self.site_names):
            terms = [(float(sign), self.tasks[task_name], i)
                     for task_name, sign in applied]
            self.sites[name].adapters["shared"].coeff_terms = terms

    def apply_flat(self, flat_coeffs):
        """Apply a raw flat coefficient vector (e.g. a controller prediction)."""
        c = flat_coeffs.detach().to(self.device).reshape(self.n_sites,
                                                         self.n_components)
        entry = {"coeffs": c, "gate_logits": None}
        self.applied = [("<flat>", 1.0)]
        for i, name in enumerate(self.site_names):
            self.sites[name].adapters["shared"].coeff_terms = [(1.0, entry, i)]

    def clear(self):
        self.apply([])
