from click import argument, group

from autoclean import global_config
from autoclean.filtering.api import filter_out, evaluate


@group()
def cli():
    pass


@cli.command(help='path to text to segment type of lm to use to score segmentations')
@argument('in_path', type=str)
@argument('out_path', type=str)
@argument('threshold', type=float)
def filter(in_path: str, out_path: str, threshold: float):
    filter_out(in_path, out_path, threshold)


@cli.command(help='path to text to segment type of lm to use to score segmentations')
@argument('in_domain_path', type=str)
@argument('out_domain_path', type=str)
def eval(in_domain_path: str, out_domain_path: str):
    evaluate(in_domain_path, out_domain_path)


if __name__ == '__main__':
    global_config()
    cli()
