import argparse
import dgl
import numpy as np
import torch
from itertools import product

from data_loader.data_util import _iterate_datasets
from torch_model.node_model import main, parse_args
from torch_model.util_dgl import set_logger

logger = set_logger(log_file=True)
parser = argparse.ArgumentParser(description='GTC model evaluation.')
parser.add_argument("--times", type=int, default=1)
parser.add_argument("-d", "--dataset", type=str, default="JODIE-wikipedia",
                    choices=["JODIE-wikipedia", "JODIE-mooc", "JODIE-reddit"])
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
    data = args.dataset
    cmds = []
    if args.display:
        default = " --dataset {} --display --lr {} -bs {} --sampling {} {} "
    else:
        default = " --dataset {} --no-display --lr {} -bs {} --sampling {} {} "

    if data == "JODIE-wikipedia":
        bss = [256, 128, 64]
        cmds.append(default.format(data, 3e-4, 128, "balance", ""))
    else:
        bss = [1024, 512, 256]
        cmds.append(default.format(data, 1e-4, 1024, "balance", ""))
        cmds.append(default.format(data, 1e-4, 1024, "balance", " --neg-ratio 10"))
        cmds.append(default.format(data, 1e-4, 512, "balance", ""))
        cmds.append(default.format(data, 1e-4, 512, "balance", " --neg-ratio 10"))
    # for lr, bs in product([1e-4, 3e-4, 1e-3], bss):
        # cmds.append(default.format(data, lr, bs, "normal", ""))
        # cmds.append(default.format(data, lr, bs, "resample", ""))
        # cmds.append(default.format(data, lr, bs, "resample", "-pw"))
        # cmds.append(default.format(data, lr, bs, "balance", ""))
        # cmds.append(default.format(data, lr, bs, "balance", "--neg-ratio 10"))
        # cmds.append(default.format(data, lr, bs, "balance", "--neg-ratio 20"))
        # cmds.append(default.format(data, lr, bs, "balance", "--neg-ratio 40"))

    for cmd in cmds:
        cmd = cmd + f" --gpu --gid {args.gid}"
        margs = parse_args().parse_args(cmd.split())
        main(margs, logger)


if __name__ == "__main__":
    evaluate()
