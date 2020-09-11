import argparse
import dgl
import numpy as np
import torch

from data_loader.data_util import _iterate_datasets
from torch_model.models import main, parse_args
from torch_model.util_dgl import set_logger

logger = set_logger(log_file=True)
parser = argparse.ArgumentParser(description='GTC model evaluation.')
parser.add_argument("--times", type=int, default=5)
parser.add_argument("--start", type=int, default=0, help="Datset start index.")
parser.add_argument("--end", type=int, default=14,
                    help="Datset end index (exclusive).")
parser.add_argument("--gid", type=int, default=0)
parser.add_argument("--display", dest="display", action="store_true")

args = parser.parse_args()
logger.info(args)


def iterate_times(method):
    def worker(*ar, **kw):
        results = []
        for _ in range(args.times):
            logger.info("Times {} begins.".format(_))
            res = method(*ar, **kw)
            results.append(res)
        return results
    return worker


@iterate_times
def evaluate():
    # fname = _iterate_datasets()
    fname = ["ia-workplace-contacts", "ia-contact", "fb-forum", "soc-sign-bitcoinotc", "ia-escorts-dynamic",
             "ia-retweet-pol", "ia-movielens-user2tags-10m", "soc-wiki-elec", "ia-slashdot-reply-dir", "ia-frwikinews-user-edits"]
    fname = fname[args.start: args.end]
    cmds = []
    if args.display:
        default = " --dataset {} --bidirected --gpu --gid {} --display --opt SGD "
    else:
        default = " --dataset {} --bidirected --gpu --gid {} --no-display --opt SGD "
    # cmds.append(default + "-te concat")
    cmds.append(default + " -pc -nc --margin 0.1")
    cmds.append(default + " -pc -nc --margin 0.2")
    cmds.append(default + " -pc -nc --margin 0.4")
    cmds.append(default + " -pc -nc --margin 0.8")
    # cmds.append(default + "-te outer")
    # cmds.append(default + "-pc")
    # cmds.append(default + "-nc")
    # cmds.append(default + "-pc -nc -te concat")
    # cmds.append(default + "-pc -nc")
    # cmds.append(default + "-pc -nc -te outer")
    for data in fname:
        for cmd in cmds:
            margs = parse_args().parse_args(cmd.format(data, args.gid).split())
            main(margs, logger)


if __name__ == "__main__":
    evaluate()
