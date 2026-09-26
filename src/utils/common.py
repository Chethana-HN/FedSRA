from pathlib import Path
import random

import numpy as np
import torch
import yaml


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)