import os
from logging import getLogger, INFO, StreamHandler, Formatter, Logger
from random import seed
from typing import Tuple

import numpy as np
import sys
from torch import manual_seed, cuda
from torch.backends import cudnn

Seq = Tuple[str, ...]

SEED = 42


def global_config(seed_: int = SEED, device: str = '0'):
    setup_logging()
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    set_seed(seed_)
    sys.setrecursionlimit(10000)


def setup_logging() -> Logger:
    formatter = Formatter('[%(asctime)s][%(levelname)s][%(name)s.%(funcName)s:%(lineno)3d] > %(message)s')
    formatter.datefmt = '%Y-%m-%d %H:%M:%S'
    handler = StreamHandler(stream=sys.stdout)
    handler.setLevel(INFO)
    handler.setFormatter(formatter)
    logger = getLogger()
    logger.setLevel(INFO)
    logger.handlers.clear()
    logger.addHandler(handler)

    return logger


def set_seed(val: int = SEED):
    seed(val)
    np.random.seed(val)
    manual_seed(val)
    cuda.manual_seed_all(val)
    cudnn.deterministic = True
    cudnn.benchmark = False


def require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)
