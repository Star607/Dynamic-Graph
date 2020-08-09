import numpy as np
import os
import pandas as pd


def _load_data(dataset="JODIE-reddit", mode="format_data", root_dir="/nfs/zty/Graph/"):
    edges = pd.read_csv("{}/{}/{}.edges".format(root_dir, mode, dataset))
    nodes = pd.read_csv("{}/{}/{}.nodes".format(root_dir, mode, dataset))
    return edges, nodes


def _iterate_datasets(dataset="all", mode="format_data", root_dir="/nfs/zty/Graph/"):
    fname = [f for f in os.listdir(os.path.join(root_dir, mode)) if f.endswith(".edges")]
    fpath = [os.path.join(root_dir, mode, f) for f in fname]
    lines = [len(open(f, "r").readlines()) for f in fpath]
    # sort the dataset by data size
    forder = [f[:-6] for l, f in sorted(zip(lines, fname))]
    return forder


def load_data(dataset="ia-contact", mode="format", root_dir="/nfs/zty/Graph/"):
    """We split dataset into two files: dataset.edges, and dataset.nodes.

    """
    # Load edges and nodes dataframes from the following directories.
    # format_data/train_data/valid_data/test_data
    # label_train_data/label_valid_data/label_test_data
    mode = os.path.join(root_dir, f"{mode}_data")
    fname = _iterate_datasets(mode=mode)
    if dataset == "all":
        return [_load_data(dataset=name, mode=mode) for name in fname]
    elif isinstance(dataset, str):
        return _load_data(dataset=dataset, mode=mode)
    elif isinstance(dataset, int):
        return _load_data(dataset=fname[dataset], mode=mode)
    elif isinstance(dataset, list):
        return [_load_data(dataset=fname[i], mode=mode) for i in dataset]
    else:
        raise NotImplementedError
