from click import argument, option, Choice, group

from autoclean import global_config
from autoclean.segmentation.api import segment, evaluate


@group()
def cli():
    pass


@cli.command(help='segments text in given in_path writing to the given out_path')
@argument('in_path', type=str)
@argument('out_path', type=str)
@option('--corpus_path', '-c', type=str, help='path to corpus to train the given language model with')
@option('--lm',
        type=Choice(['rnn', '0', '1', '2', '3', '4', '5'], case_sensitive=False),
        help='type of lm to use to score segmentations')
@option('--smoothed', '-s', default=False, is_flag=True, help='True if the ngram language models should be smoothed')
def seg(in_path: str, out_path: str, corpus_path: str, lm: str, smoothed: bool):
    segment(in_path, out_path, corpus_path or in_path, lm, smoothed)


@cli.command(help='evaluates the quality of the segmentations in given in_path with the ground truths in gold_path')
@argument('in_path', type=str)
@argument('gold_path', type=str)
def eval(in_path: str, gold_path: str):
    evaluate(in_path, gold_path)


if __name__ == '__main__':
    global_config()
    cli()
