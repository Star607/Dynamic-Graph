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

def load_graph(dataset=None):
    """Concat the temporal edges, transform into nstep time slots, and return edges, pivot_time.
    """
    edges, nodes = load_split_edges(dataset=dataset)
    train_edges, val_edges, test_edges = edges[0]
    val_time = val_edges["timestamp"].min()
    test_time = test_edges["timestamp"].min()

    edges = pd.concat([train_edges, val_edges, test_edges])
    nodes = nodes[0]
    # padding node is 0, so add 1 here.
    id2idx = {row.node_id: row.id_map + 1 for row in nodes.itertuples()}
    edges["from_node_id"] = edges["from_node_id"].map(id2idx)
    edges["to_node_id"] = edges["to_node_id"].map(id2idx)
    return edges, len(nodes), val_time, test_time


def load_label_data(dataset=None):
    edges, nodes = load_label_edges(dataset=dataset)
    train_edges, val_edges, test_edges = edges[0]
    pivot_time = train_edges["timestamp"].max()

    nodes = nodes[0]
    # padding node is 0, so add 1 here.
    id2idx = {row.node_id: row.id_map + 1 for row in nodes.itertuples()}
    ans = []
    for df in [train_edges, val_edges, test_edges]:
        df["from_node_id"] = df["from_node_id"].map(id2idx)
        df["to_node_id"] = df["to_node_id"].map(id2idx)
        df = df[["from_node_id", "to_node_id", "timestamp", "label"]]
        df.columns = ["u", "i", "ts", "label"]
        ans.append(df)
    return ans[0], ans[1], ans[2]
