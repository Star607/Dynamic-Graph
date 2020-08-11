import numpy as np
import os
import pandas as pd
from random import shuffle


def _load_data(dataset="JODIE-reddit", mode="format_data", root_dir="/nfs/zty/Graph/"):
    edges = pd.read_csv("{}/{}/{}.edges".format(root_dir, mode, dataset))
    nodes = pd.read_csv("{}/{}/{}.nodes".format(root_dir, mode, dataset))
    return edges, nodes


def _iterate_datasets(dataset="all", mode="test_data", root_dir="/nfs/zty/Graph/"):
    if dataset != "all":
        if isinstance(dataset, str):
            return [dataset]
        elif isinstance(dataset, list) and isinstance(dataset[0], str):
            return dataset
    fname = [f for f in os.listdir(os.path.join(root_dir, mode)) if f.endswith(".edges")]
    fpath = [os.path.join(root_dir, mode, f) for f in fname]
    lines = [len(open(f, "r").readlines()) for f in fpath]
    # sort the dataset by data size
    forder = [f[:-6] for l, f in sorted(zip(lines, fname))]
    if dataset != "all":
        if isinstance(dataset, int):
            return forder[dataset]
        elif isinstance(dataset, list) and isinstance(dataset[0], int):
            return [forder[i] for i in dataset]
        else:
            raise NotImplementedError
    return forder


def load_data(dataset="ia-contact", mode="format", root_dir="/nfs/zty/Graph/"):
    """We split dataset into two files: dataset.edges, and dataset.nodes.

    """
    # Load edges and nodes dataframes from the following directories.
    # Return: a list of (edges, nodes) tuple of required datasets.
    # format_data/train_data/valid_data/test_data
    # label_train_data/label_valid_data/label_test_data
    mode = "{}_data".format(mode)
    if dataset == "all":
        fname = _iterate_datasets(mode=mode)
        return [_load_data(dataset=name, mode=mode) for name in fname]
    elif isinstance(dataset, str):
        return [_load_data(dataset=dataset, mode=mode)]
    elif isinstance(dataset, int):
        fname = _iterate_datasets(mode=mode)
        return [_load_data(dataset=fname[dataset], mode=mode)]
    elif isinstance(dataset, list) and isinstance(dataset[0], str):
        return [_load_data(dataset=name, mode=mode) for name in dataset]
    elif isinstance(dataset, list) and isinstance(dataset[0], int):
        fname = _iterate_datasets(mode=mode)
        return [_load_data(dataset=fname[i], mode=mode) for i in dataset]
    else:
        raise NotImplementedError


def load_split_edges(dataset="all", root_dir="/nfs/zty/Graph"):
    train_tuples = load_data(dataset=dataset, mode="train", root_dir=root_dir)
    train_edges = [edges for edges, nodes in train_tuples]
    nodes = [nodes for edges, nodes in train_tuples]
    valid_edges = [edges for edges, _ in load_data(
        dataset=dataset, mode="valid", root_dir=root_dir)]
    test_edges = [edges for edges, _ in load_data(
        dataset=dataset, mode="test", root_dir=root_dir)]
    return list(zip(train_edges, valid_edges, test_edges)), nodes


def load_label_edges(dataset="ia-contact", root_dir="/nfs/zty/Graph"):
    train_tuples = load_data(dataset=dataset, mode="label_train", root_dir=root_dir)
    train_edges = [edges for edges, nodes in train_tuples]
    nodes = [nodes for edges, nodes in train_tuples]
    valid_edges = [edges for edges, _ in load_data(
        dataset=dataset, mode="label_valid", root_dir=root_dir)]
    test_edges = [edges for edges, _ in load_data(
        dataset=dataset, mode="label_test", root_dir=root_dir)]
    return list(zip(train_edges, valid_edges, test_edges)), nodes
