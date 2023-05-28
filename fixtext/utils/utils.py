import random
import numpy as np
import torch as th

def set_seed(seed: int):
    """
    Set seed so that the experiments are replicable.

    Args:
        seed (int): The seed to be set.
    """

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False