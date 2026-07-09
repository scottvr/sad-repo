import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")))

from sequential_adapt.config import smoke_config  # noqa: E402
from sequential_adapt.experiments import ExperimentContext  # noqa: E402

TINY = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="session")
def cfg():
    # Very short fits: tests check mechanics, not learning quality.
    return smoke_config(model_name=TINY, steps=15, controller_steps=50,
                        n_tasks=2, facts_per_task=2)


@pytest.fixture(scope="session")
def ctx(cfg):
    return ExperimentContext(cfg, log=lambda *_: None)
