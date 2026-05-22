import random
import numpy as np

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
