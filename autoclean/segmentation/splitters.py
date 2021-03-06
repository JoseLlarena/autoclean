from functools import lru_cache
from typing import Tuple, Callable as Fn, Iterable

from autoclean import Seq, require
from autoclean.segmentation import CACHE


@lru_cache(maxsize=CACHE)
def string_splits(seq: Seq, is_valid: Fn[[str], bool] = lambda split: True) -> Iterable[Tuple[Seq, Seq]]:
    """
    Returns all possible splits of the given sequence that are allowed according to the given predicate

    :param seq: the sequence to generate splits for
    :param is_valid: whether the tail split is a valid sequence
    :return: a collection of splits, in the form of a head and a joined-up tail of tokens
    """
    require(len(seq) > 0, 'sequence should have at least 1 element')

    return tuple((head := seq[:idx], tail := (tail_split,))
                 for idx in reversed(range(len(seq)))
                 if is_valid(tail_split := ''.join(seq[idx:])))
