from functools import lru_cache
from logging import getLogger

from pypey import pype, Fn, px, TOTAL

from autoclean import Seq
from autoclean.models import ngram_cost_fn_from, rnn_cost_fn_from
from autoclean.segmentation import CACHE
from autoclean.segmentation.segmenters import viterbi
from autoclean.segmentation.splitters import string_splits

BOS = '·'
EOS = '⎵'

LOG = getLogger(__package__)


def segment(in_path: str, out_path: str, corpus_path: str, lm: str, smoothed: bool):
    """
    Segments sequences in given input path and writes the resulting segmentation to the given output path.

    :param in_path: path to file with sequences to be segmented
    :param out_path:  path to file to write segmentations to
    :param corpus_path: path to file with data to train the probabilistic model with
    :param lm: the type of language model: rnn, unigram, bigram, trigram, fourgram, fivegram or sixgram
    :param smoothed: whether the ngram language model should be smoothed, ignored for an rnn language model
    :return: nothing, but writes to file and console
    """
    LOG.info(f'Reading in corpus from [{corpus_path}] ...')
    corpus = (pype
              .file(corpus_path)
              .map(str.strip)
              .map(str.split)
              .map(tuple)
              .eager())

    if lm == 'rnn':
        LOG.info(f'Estimating rnn language model from [{corpus_path}] ...')
        cost_fn = px(rnn_cost_of, lm=rnn_cost_fn_from(corpus))
    else:
        order = int(lm)
        LOG.info(f'Estimating [{order + 1}]-ngram language model from [{corpus_path}] ...')
        cost_fn = px(ngram_cost_of, lm=ngram_cost_fn_from(corpus, order, smoothed), order=order)

    LOG.info(f'Segmenting and saving to [{out_path}] ...')
    all_words = corpus.flat().to(set) | {BOS, EOS}
    splitting_fn = px(string_splits, is_valid=all_words.__contains__)

    LOG.info(f'Reading in file to segment from [{in_path}] ...')
    (pype
     .file(in_path)
     .map(str.strip)
     .map(lambda sent: BOS + ''.join(sent.split()) + EOS)
     .map(tuple)
     .map(lambda sent: viterbi(sent, cost_fn, splitting_fn))
     .map(lambda seg: ' '.join(seg)[1:-1].strip())
     .to_file(out_path))

    LOG.info('Done.')


def evaluate(in_path: str, gold_path: str):
    """
    Evaluates the per-sentence accuracy of a segmentation

    :param in_path: path to file with proposed segmented sequences
    :param gold_path: path to file with true segmented sequences, must be the same size as the input path
    :return: nothing, but writes to console
    """
    LOG.info(f'Reading segmented text from [{in_path}] ...')
    LOG.info(f'Reading gold standard text from [{gold_path}] ...')

    item_to_count_freq = (pype
                          .file(in_path)
                          .zip(pype.file(gold_path).map(str.strip))
                          .map(lambda seg, gold: seg == gold)
                          .freqs()
                          .map(lambda item, count, freq: (item, (count, freq)))
                          .to(dict))

    if True not in item_to_count_freq:
        item_to_count_freq[True] = 0, 0.

    LOG.info(f'Accuracy: [{item_to_count_freq[True][-1]:.2%}], '
             f'[{item_to_count_freq[True][0]:6,d}] correct out of [{item_to_count_freq[TOTAL][0]:6,d}] sequences')

    LOG.info('Done.')


@lru_cache(maxsize=CACHE)
def rnn_cost_of(sequence: Seq, lm: Fn) -> float:
    return lm(sequence) / len(sequence)


def ngram_cost_of(sequence: Seq, lm: Fn, order: int) -> float:
    return unnormed_ngram_cost_of(sequence, lm, order=order) / len(sequence)


@lru_cache(maxsize=CACHE)
def unnormed_ngram_cost_of(sequence: Seq, lm: Fn, order: int) -> float:
    if len(sequence) == 1:
        return lm(sequence)

    return unnormed_ngram_cost_of(sequence[:-1], lm, order) + lm(sequence)
