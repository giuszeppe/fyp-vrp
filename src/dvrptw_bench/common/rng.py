"""Random seed utilities."""

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and Torch if available."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
