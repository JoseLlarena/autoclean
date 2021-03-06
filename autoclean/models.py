from __future__ import annotations

from datetime import datetime
from importlib.resources import path
from logging import getLogger
from os.path import join, basename, sep
from shlex import split
from subprocess import run, DEVNULL, STDOUT
from typing import Dict, Union, Iterator, Iterable, Tuple
from typing import Sequence

from kenlm import Model as KenLM, Config, LoadMethod
from math import log
from pypey import Fn, pype, Pype
from torch import Tensor, tensor, no_grad, device as Device
from torch import cuda
from torch.nn import GRU, Linear, Embedding, CrossEntropyLoss, LogSoftmax, Module, Identity
from torch.optim import Optimizer
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, IterableDataset

from autoclean import Seq

BOS = '·'
EOS = '⎵'
PAD_IDX = 0
MIN_LOGPROB = 100  # matches kenlm/srilm mininum
LN_10 = log(10)

LOG = getLogger(__package__)


class UnigramLM(dict):
    """
    Encapsulates a unigram language model, with a fixed OOV log-probability
    """

    def __init__(self, items: Iterable, min_logprob: float):
        super().__init__(items)
        self.min_logprob = min_logprob

    def xent_of(self, seq: Sequence[str], bos: bool = True, eos: bool = True, normed: bool = True, joint: bool = True):
        """
        Returns the cross-entropy of the given sequence

        :param seq: sequence to evaluate the cross-entropy of
        :param bos: whether the sequence is a prefix of a larger sequence
        :param eos: whether the sequence is a suffix of a larger sequence
        :param normed: true if the cross-entropy should be normalised by the length of the sequence
        :param joint: True for the joint cross-entropy, False for the negative log condition probability.
        :return: the possibly normalised cross-entropy or the negative conditional log prob of the sequence

        """
        if isinstance(seq, str):
            seq = seq,

        if len(seq) == 1 and seq[0] == BOS:
            return 0

        if joint:
            return (sum(map(self.__getitem__, seq)) + (self[EOS] if eos else 0)) / \
                   ((len(seq) + int(eos)) if normed else 1)

        return -self[seq[-1]]

    def __missing__(self, key: str) -> float:
        return self.min_logprob

    @property
    def order(self) -> int:
        return 0

    def __hash__(self):
        return hash(tuple(sorted(self.items())))


class NgramLM:
    """
    Encapsulates a bigram or higher language model, with a fixed OOV log-probability
    """

    def __init__(self, kenlm: KenLM, file_path: str):
        super().__init__()
        self._kenlm = kenlm
        self._file_path = file_path

    @property
    def order(self) -> int:
        return self._kenlm.order - 1

    def xent_of(self, seq: Sequence[str], bos: bool = True, eos: bool = True, normed: bool = True, joint: bool = True):
        """
       Returns the cross-entropy of the given sequence

       :param seq: sequence to evaluate the cross-entropy of
       :param bos: whether the sequence is a prefix of a larger sequence
       :param eos: whether the sequence is a suffix of a larger sequence
       :param normed: true if the cross-entropy should be normalised by the length of the sequence
       :param joint: True for the joint cross-entropy, False for the negative log conditional probability
       :return: the possibly normalised cross-entropy or the negative conditional log prob of the sequence
       """
        if len(seq) == 1 and seq[0] == BOS:
            return 0

        if joint:
            return -self._kenlm.score(' '.join(seq), bos=bos, eos=eos) / ((len(seq) + int(eos)) if normed else 1)

        return -tuple(self._kenlm.full_scores(' '.join(seq), bos=bos, eos=eos))[-1][0]

    def __hash__(self) -> int:
        return hash(self._file_path)


def ngram_cost_fn_from(corpus: Pype[Seq], order: int = 1, smoothed: bool = False) -> Fn:
    """
    Returns a function giving the cost of a sequence, based on an ngram language model.

    :param corpus: the corpus to train the language model used in the cost function
    :param order: the order of the ngram language model
    :param smoothed: whether the language model should be smoothed
    :return: a function computing the cost of a sequence
    """
    corpus_path = join(sep, 'tmp', f'corpus.'
                                   f'{datetime.now().isoformat(timespec="seconds")}.'
                                   f'{order + 1}.txt')

    corpus.map(' '.join).to_file(corpus_path)

    lm = estimate_ngram_lm_from(corpus_path, order, smoothed)

    def xent_of(words: Seq, joint: bool = True) -> float:
        if len(words) == 1 and words[0] == BOS:
            return 0

        trimmed_words = words[1:] if words[0] == BOS else words
        trimmed_words = trimmed_words[:-1] if trimmed_words[-1] == EOS else trimmed_words

        return lm.xent_of(trimmed_words, bos=words[0] == BOS, eos=words[-1] == EOS, normed=False, joint=joint)

    return xent_of


def estimate_ngram_lm_from(corpus_file_path: str, order: int, smoothed: bool = True) -> Union[UnigramLM, NgramLM]:
    """
    Estimates an n-gram language model from the given path to a text corpus

    :param corpus_file_path: path to training file
    :param order: order of the model, currently supporting only 0-5
    :param smoothed: whether the language model should smooth probabilities
    :return: a language model
    """
    return ngram_lm_from(_estimate(corpus_file_path, order, smoothed))


def ngram_lm_from(model_file_path: str) -> Union[UnigramLM, NgramLM]:
    """
    Loads a language model from an arpa or binary file, as created by KenLM or SRILM

    :param model_file_path: path to arpa or binary model
    :return: An n-gram language model
    """
    if model_file_path.endswith('.arpa'):
        return UnigramLM(pype
                         .file(model_file_path)
                         .drop_while(lambda line: line.strip() != '\\1-grams:')
                         .reject(lambda line: line.strip() == '\\1-grams:')
                         .take_while(lambda line: len(line.strip()) > 0)
                         .map(lambda line: line.split('\t'))
                         .reject(lambda bits: len(bits) != 2)
                         .map(lambda prob, gram: (gram, -float(prob))), MIN_LOGPROB)

    c = Config()
    c.load_method = LoadMethod.POPULATE_OR_READ  # this load method consumes more memory but makes prob lookup faster
    return NgramLM(KenLM(model_file_path, c), model_file_path)


def _estimate(corpus_file_path: str, order: int, smoothed: bool) -> str:
    LOG.info(f'estimating [{order + 1}]-gram from [{corpus_file_path}] ...')
    estimation = _kenlm_estimate if smoothed else _srilm_estimate
    arpa_path = estimation(corpus_file_path, order)

    LOG.info(f'binarising [{order + 1}]-gram from [{arpa_path}] ...')
    return arpa_path if order == 0 else _kenlm_binarise(arpa_path, arpa_path.replace('arpa', 'bin'))


def _srilm_estimate(corpus_file_path: str, order: int) -> str:
    arpa_output_path = join(sep, 'tmp',
                            f'{basename(corpus_file_path)}.'
                            f'{datetime.now().isoformat(timespec="seconds")}.'
                            f'{abs(hash(corpus_file_path))}.'
                            f'{order + 1}.arpa')

    with path('autoclean', 'ngram-count') as p:
        command = f'{p} ' \
                  f'-text {corpus_file_path} ' \
                  f'-order {order + 1} ' \
                  f'-lm {arpa_output_path} ' \
                  f'-addsmooth 0 ' \
                  f'-gt3min 1 -gt4min 1 -gt5min 1 -gt6min 1 -gt7min 1'

    run(split(command), check=True, stdout=DEVNULL, stderr=DEVNULL)

    return arpa_output_path


def _kenlm_estimate(corpus_file_path: str, order: int) -> str:
    arpa_output_path = join(sep, 'tmp',
                            f'{basename(corpus_file_path)}.'
                            f'{datetime.now().isoformat(timespec="seconds")}.'
                            f'{abs(hash(corpus_file_path))}.'
                            f'{order + 1}.arpa')

    with path('autoclean', 'lmplz') as p:
        command = f'{p} ' \
                  f'--discount_fallback ' \
                  f'--skip_symbols ' \
                  f'-o {order + 1} ' \
                  f'--text {corpus_file_path} ' \
                  f'--arpa {arpa_output_path}'

        run(split(command), check=True, stdout=DEVNULL, stderr=STDOUT)

        return arpa_output_path


def _kenlm_binarise(arpa_path: str, model_path: str) -> str:
    with path('autoclean', 'build_binary') as p:
        command = f'{p} {arpa_path} {model_path}'
        run(split(command), check=True, stdout=DEVNULL, stderr=STDOUT)

        return model_path


class SelfSupDataset(IterableDataset):
    """
    Encapsulates a self-supervised dataset with a single vocabulary for input and target
    """

    def __init__(self, data: Pype[Tuple[Tensor, Tensor]], token_to_id: Dict[str, int], pad: str):
        self._data = data
        self.token_to_id = token_to_id
        self.id_to_token = {idx: token for token, idx in token_to_id.items()}
        self.pad = pad

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        return self._data.it()

    def shuffle(self) -> SelfSupDataset:
        return SelfSupDataset(self._data.shuffle(), self.token_to_id, self.pad)

    def to_token(self, key: Union[Tensor, int]) -> str:
        return self.id_to_token[int(key.item()) if isinstance(key, Tensor) else key]

    def to_id(self, token: str) -> int:
        return self.token_to_id[token]

    @property
    def n(self) -> int:
        return self._data.size()

    @property
    def vocab_n(self) -> int:
        return len(self.id_to_token)


class NeuralLM(Module):
    """
    Encapsulates a neural language model, based on a GRU
    """

    def __init__(self, embedding: Module, gru: Module, proj: Module, out: Module = Identity()):
        super().__init__()
        self.embedding = embedding
        self.gru = gru
        self.proj = proj
        self.out = out
        self.history = None

    def device(self) -> Device:
        return next(self.parameters()).device

    def forward(self, x: Tensor) -> Tensor:
        h, self.history = self.gru(self.embedding(x), self.history)

        self.history.detach_()

        return self.out(self.proj(h))


def rnn_cost_fn_from(corpus: Pype[Seq],
                     device: Device = Device(cuda.current_device() if cuda.is_available() else 'cpu')) -> Fn:
    """
     Returns a function giving the cost of a sequence, based on a neural language model.

    :param corpus: the corpus to train the language model used in the cost function
    :param device: the device to run training and inference of the neural language model
    :return: a function computing the cost of a sequence
    """
    dataset = dataset_from(corpus, max_len=corpus.map(len).to(max))
    vocab_n = dataset.vocab_n
    embed_dim = 64
    hid_dim = 128
    bs = 64
    epochs = 6
    layers = 1

    lm = NeuralLM(Embedding(vocab_n, embed_dim),
                  GRU(embed_dim, hid_dim, layers, batch_first=True),
                  Linear(hid_dim, vocab_n)).to(device)

    lm = train(lm, loaders_from(dataset, bs, epochs), CrossEntropyLoss(), Adam(lm.parameters(), lr=.01)).eval()
    lm.history = None
    lm.out = LogSoftmax(dim=-1)

    @no_grad()
    def xent(words: Seq, reset: bool = True) -> float:
        if reset:
            lm.history = None
        try:
            [dataset.to_id(word) for word in words]
        except KeyError:
            return MIN_LOGPROB

        context = tensor([dataset.to_id(word) for word in words]).unsqueeze(0).to(lm.device())
        logsoftmaxes = lm(context)

        return sum(-logsoftmaxes[0, t, dataset.to_id(word)] for t, word in enumerate(words[:-1])) / LN_10

    return xent


def dataset_from(sequences: Pype[Tuple[str, ...]], max_len: int, pad: str = '∘') -> SelfSupDataset:
    """
    Creates a self-supervised dataset from the given sequences

    :param sequences: the sequences to build a dataset from
    :param max_len: the maximum allowed length of a sequence, longer sequences will be truncated an shorter ones padded
    :param pad: symbol to pad sequences shorter than the maximum length
    :return: a self-supervised dataset
    """
    token_to_id = sequences.flat().uniq().sort().enum(start=PAD_IDX + 1, swap=True).to(dict)
    token_to_id[pad] = PAD_IDX

    data = (sequences
            .map(lambda sent: sent[:max_len] + (pad,) * (max_len + 1 - len(sent[:max_len])))
            .map(lambda sent: tensor(list(map(token_to_id.__getitem__, sent))))
            .map(lambda sent: (_in := sent[:-1], target := sent[1:]))
            .eager())

    return SelfSupDataset(data, token_to_id, pad)


def loaders_from(data: SelfSupDataset, bs: int, epochs: int, shuffle: bool = True) -> Iterable[DataLoader]:
    """
    Returns a collections of `epoch` data loaders

    :param data: a self-supervised dataset
    :param bs: max batch size
    :param epochs: number of epochs == number of data loaders
    :param shuffle: whether the dataset should be shuffled in each epoch/data loader
    :return: a collection of data loaders
    """
    for _ in range(epochs):
        yield DataLoader(data.shuffle() if shuffle else data, shuffle=False, batch_size=bs, drop_last=True)


def train(model: Module,
          loaders: Iterable[Iterable[Tuple[Tensor, Tensor]]],
          loss_fn: Fn,
          optimiser: Optimizer) -> Module:
    """
    Trains the given model with the data in the given data loaders

    :param model: a model to train
    :param loaders: a collection of loaders, one per epoch
    :param loss_fn: a loss function
    :param optimiser: an optimiser
    :return: the trained model
    """
    model.train()

    for epoch, loader in enumerate(loaders):
        accum_loss, n = 0., 0

        for idx, (xs, ys) in enumerate(loader):
            model.zero_grad()

            y_hats = model(xs.to(model.device()))

            loss = loss_fn(y_hats.transpose(-2, -1), ys.to(model.device()))
            loss.backward()

            accum_loss += loss.item()
            n += xs.size()[0]

            optimiser.step()

        LOG.info(f'epoch [{epoch:3,d}] loss: [{accum_loss / n:8.4f}]')

    return model
