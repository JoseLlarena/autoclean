from autoclean.segmentation.segmenters import viterbi
from autoclean.segmentation.splitters import string_splits


def test_splitting():
    output = list(string_splits(tuple('funday')))

    assert output == [(('f', 'u', 'n', 'd', 'a'), ('y',)),
                      (('f', 'u', 'n', 'd'), ('ay',)),
                      (('f', 'u', 'n'), ('day',)),
                      (('f', 'u'), ('nday',)),
                      (('f',), ('unday',)),
                      ((), ('funday',))]


def test_segmentation():
    def cost_fn(chunk):
        return 0 if chunk in {('fun',), ('fun', 'day'), ('day',)} else 1

    assert tuple(viterbi(tuple('funday'), cost_fn, string_splits)) == ('fun', 'day')
