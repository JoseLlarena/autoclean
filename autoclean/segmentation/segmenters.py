from functools import lru_cache
from typing import Callable as Fn

from autoclean import Seq
from autoclean.segmentation import CACHE


@lru_cache(maxsize=CACHE)
def viterbi(seq: Seq, cost_of: Fn, splits_of: Fn) -> Seq:
    """
    Segments the given sequence using the Viterbi algorithm, parameterised by a splitting function to generate the
    splits, and a cost function to score those splits. Based on Peter Norvig's chapter 14 of "Beautiful Data" by
    Seagaran and Hammerbacher.

    It uses recursion heavily and so it can only handle reasonably long sentences. It may require increasing the
    default Python recursion depth limit.

    The algorithm proceeds by splitting the input sequence into a head of starting words and a tail of joined up end
    words. This is done recursively for each head until the only head left is the empty sequence,
    at which point, the recursion ends and then the cost of each segmentation of a given head is evaluated and the
    one with the lowest cost is returned, the rest discarded and not recursed into. The segmentations are created
    by joining the segmentation of a head with its tail, for each head-tail split.

    :param seq: the sequence to segment
    :param cost_of: the cost function to score segmentations
    :param splits_of: the function to generate segmentations
    :return: a segmented sequence
    """

    if len(seq) < 2:
        return seq

    segmentations = (viterbi(head, cost_of, splits_of) + tail for head, tail in splits_of(seq))

    return min(segmentations, key=cost_of, default=seq)
