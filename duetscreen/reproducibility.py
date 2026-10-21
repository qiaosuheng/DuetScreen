"""Random-state helpers for independent classifier training runs.

Data sampling owns a separate RNG in data_preprocessing.py. Importing this
module does not seed anything or construct a model.
"""

import random

import numpy as np
import torch


def seed_training(seed: int) -> None:
    """Initialize training RNGs once, before loaders and model construction.

    ``torch.manual_seed`` seeds CPU and all CUDA devices, including layers
    initialized on CPU before being moved to a GPU. Do not call this at every
    epoch or batch: the random streams must advance during a training run.
    Seeding alone does not make nondeterministic CUDA kernels deterministic.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_data_worker(worker_id: int) -> None:
    """Seed Python/NumPy from the worker seed assigned by PyTorch.

    The DataLoader's own generator controls this seed independently of model
    initialization. With persistent workers, this runs only at worker startup.
    """
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
