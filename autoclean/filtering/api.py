from datetime import datetime
from logging import getLogger
from os.path import join, sep, basename
from typing import Sequence, Tuple, TypeVar, Union, Iterable

from pypey import pype, px, Pype

from autoclean.models import estimate_ngram_lm_from, NgramLM, UnigramLM

Text = TypeVar('Text')
Cost = Tuple[Text, Union[int, float]]
Ranking = Tuple[Text, int]

LOG = getLogger(__package__)


def filter_out(in_path: str, out_path: str, threshold: float):
    """
    Select clean sequences in the given input file and writes them, sorted in increasing cost, to the given output file.

    :param in_path: path to file with sequences to be filtered
    :param out_path: path to file to write selected sequences to
    :param threshold: the maximum value of the cost metric for a sequences to be selected
    :return: nothing, but writes to disk
    """

    high_lm = estimate_ngram_lm_from(in_path, order=5, smoothed=True)
    low_lm = estimate_ngram_lm_from(in_path, order=0, smoothed=False)

    LOG.info(f'Reading from [{in_path}] and writing to [{out_path}] ...')

    (pype
     .file(in_path)
     .map(str.strip)
     .select(bool)
     .map(str.split)
     .zip_with(px(slor_cost, high_lm=high_lm, low_lm=low_lm))
     .reject(lambda sentence, cost: cost > threshold)
     .sort(lambda sentence, cost: (cost, sentence))
     .map(lambda sentence, cost: ' '.join(sentence))
     .to_file(out_path))

    LOG.info('Done.')


def evaluate(in_domain_path: str, out_domain_path: str):
    """
    Evaluates the goodness of the filtering with precision, recall and fallout. It creates a mixed dataset with
    the in-domain and out-of-domain datasets, and then sorts it with the scoring function. Finally it computes the
    metrics for cut-offs at 100% to 10% of the dataset.

    :param in_domain_path: path to in-domain corpus file
    :param out_domain_path: path to out-of-domain corpus file
    :return: nothing, but writes to console
    """

    mixed_path = join(sep, 'tmp', f'{basename(in_domain_path)}.'
                                  f'{basename(out_domain_path)}.'
                                  f'{datetime.now().isoformat(timespec="seconds")}.'
                                  f'{abs(hash(in_domain_path + out_domain_path))}.txt')

    LOG.info(f'Reading from [{in_domain_path}] and to [{out_domain_path}] ...')

    (pype
     .file(in_domain_path)
     .cat(pype.file(out_domain_path))
     .map(str.strip)
     .select(bool)
     .to_file(mixed_path))

    high_lm = estimate_ngram_lm_from(mixed_path, order=5, smoothed=True)
    low_lm = estimate_ngram_lm_from(mixed_path, order=0, smoothed=False)

    LOG.info('Calculating 10%-cutoff rankings metrics ...')

    rankings = (pype
                .file(mixed_path)
                .map(str.split)
                .map(tuple)
                .zip_with(px(slor_cost, high_lm=high_lm, low_lm=low_lm))
                .eager()
                .to(rank_of)
                .eager())

    out_domain = (pype
                  .file(out_domain_path)
                  .map(str.strip)
                  .select(bool)
                  .map(str.split)
                  .map(tuple)
                  .uniq()
                  .eager())

    print()
    for cutoff in range(10, 0, -1):
        cutoff /= 10
        precision, recall, fallout = rank_metrics(rankings, out_domain, cutoff)

        print(f'@{cutoff:4.0%} '
              f'precision [{precision:8.4f}%] '
              f'recall [{recall:8.4f}%] '
              f'fallout [{fallout:8.4f}%]\n')

    LOG.info('Done.')


def slor_cost(sequence: Sequence[str], high_lm: NgramLM, low_lm: UnigramLM) -> float:
    """
    Computes the cost of the sequence using a combination of the syntactic log-odds ratio and the average bigram
    cross-entropy.

    :param sequence: sequence to compute cost for
    :param high_lm: bigram or higher language model
    :param low_lm: unigram language model
    :return: cost of the given sequence according with the high and low language models
    """
    joint = high_lm.xent_of(sequence, bos=False, eos=False, normed=True)
    margs = low_lm.xent_of(sequence, bos=False, eos=False, normed=True)
    pmi = margs - joint

    bigrams = tuple(zip(sequence, sequence[1:])) or sequence

    bigram_score = sum(high_lm.xent_of(LR, bos=False, eos=False, normed=False) for LR in bigrams) / len(bigrams)

    return pmi + bigram_score


def rank_of(word_cost: Pype[Cost], sort: bool = True) -> Pype[Ranking]:
    """
    Returns ranked items, possibly sorted by ascending rank, then by item.The item cost needs to be non-negative
    No ordering is assumed.

    :param word_cost: a stream of pairs of item and cost
    :param sort: True if ranks should be sorted, by ascending rank then by item, False otherwise
    :return: A stream of pairs of item and rank, possibly ordered by rank then item
    """
    cost_to_rank = word_cost.pick(-1).uniq().sort().enum(start=1, swap=True).to(dict)

    rankings = word_cost.map(lambda word, cost: (word, cost_to_rank[cost]))

    return rankings.sort(lambda word, rank: (rank, word)) if sort else rankings


def rank_metrics(rankings: Pype[Ranking], out_of_domain: Iterable[Text], cutoff: float = 1.) \
        -> Tuple[float, float, float]:
    """
    Computes precision, recall and fallout for the given ranking.
    
    Precision is defined as the fraction of in-domain sequences in the top N*cutoff true rankings.

    Recall is defined as the fraction of all in-domain sequences in the top N*cutoff true rankings. It gives a measure
    of how many of the in-domain sequences have been preserved.

    Fallout is defined as the fraction of all out-of-domain sequences in the top N*cutoff true rankings. It gives
    a measure of how good the cleaning is

    Note the metrics are computed for the top N*cutoff true rankings and not not just the N*cutoff rankings because when
    where there are many out-of-domain sequences, metrics could never be 100%/0%, as as sufficiently high cut-off will
    invariably contain some out-of-domain sequences.

    :param rankings: collection of sequences and their rankings, assumes rankings are sorted by rank, ascending
    :param out_of_domain: collection of sequences that are out of domain
    :param cutoff: fraction of the top rankings the metrics should be computed for
    :return: a triple of precision, recall and fallout
    """

    out_of_domain = set(out_of_domain)

    N = rankings.select(lambda word, rank: word not in out_of_domain).size()
    top_N = int(round(N * cutoff)) or 1

    in_domain_N = rankings.take(top_N).select(lambda word, rank: word not in out_of_domain).size()

    return 100 * in_domain_N / top_N, 100 * in_domain_N / N, 100 * (top_N - in_domain_N) / len(out_of_domain)
