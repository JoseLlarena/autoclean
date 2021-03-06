from pypey import pype

from autoclean.filtering.api import rank_of, rank_metrics


def test_it_computes_rank_of_scoring_no_ties():
    costs = pype([('a', 50), ('b', 55), ('c', 60), ('d', 65)])
    ranking = pype([('a', 1), ('b', 2), ('c', 3), ('d', 4)])

    assert rank_of(costs).to(list) == ranking.to(list)


def test_it_computes_rank_of_scoring_only_ties():
    costs = pype([('a', 50), ('b', 50), ('c', 50), ('d', 50)])
    ranking = pype([('a', 1), ('b', 1), ('c', 1), ('d', 1)])

    assert rank_of(costs).to(list) == ranking.to(list)


def test_it_computes_rank_of_scoring_with_ties():
    costs = pype([('a', 50), ('b', 55), ('c', 60), ('d', 60)])
    ranking = pype([('a', 1), ('b', 2), ('c', 3), ('d', 3)])

    assert rank_of(costs).to(list) == ranking.to(list)


def test_it_computes_perfect_rank_precision():
    ranks = pype([('a', 1), ('b', 2), ('c', 3), ('d', 3)])
    out_of_domain = {'d'}

    assert rank_metrics(ranks, out_of_domain)[0] == 100.


def test_it_computes_rank_precision():
    ranks = pype([('a', 1), ('d', 2), ('b', 3), ('c', 3)])
    out_of_domain = {'d'}

    assert rank_metrics(ranks, out_of_domain)[0] == 100*2 / 3


def test_it_computes_perfect_rank_fallout():
    ranks = pype([('a', 1), ('b', 2), ('c', 3), ('d', 3)])
    out_of_domain = {'d'}

    assert rank_metrics(ranks, out_of_domain)[-1] == 0.


def test_it_computes_rank_fallout():
    ranks = pype([('a', 1), ('d', 2), ('b', 3), ('c', 3)])
    out_of_domain = {'d', 'e'}

    assert rank_metrics(ranks, out_of_domain)[-1] == 50.
