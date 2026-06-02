import random
import numpy as np
from copy import deepcopy

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency during early project setup
    torch = None

def set_seed(seed: int):
    random.seed(seed)

    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_experiment_seeds(config: dict) -> list[int]:
    seeds = config.get("project", {}).get("random_seeds", [])
    if not seeds:
        raise ValueError("config.project.random_seeds must contain at least one seed")
    return [int(seed) for seed in seeds]


def get_primary_seed(config: dict) -> int:
    return get_experiment_seeds(config)[0]


def clone_config_with_seed(config: dict, seed: int) -> dict:
    cloned = deepcopy(config)
    cloned.setdefault("project", {})
    cloned["project"]["random_seeds"] = [int(seed)]
    return cloned
