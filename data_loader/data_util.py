import numpy as np
import os
import pandas as pd


def load_data(data_dir="./graph_data/", dataset="ia-contact"):
    """We split dataset into two files: dataset.edges, and dataset.nodes.

    """
    cwd = os.getcwd()  # default is '/nfs/zty/Graph/Dynamic-Graph'
    data_dir = os.path.join(cwd, data_dir)
    if isinstance(dataset, str):
        pass
    elif isinstance(dataset, int):
        pass
    pass


def iterate_datasets(dataset="all", project_dir="/nfs/zty/Graph/"):
    fname = os.listdir(os.path.join(project_dir, "train_data"))
    fpath = [os.path.join(project_dir, "train_data/{}".format(f))
             for f in fname]
    lines = [len(open(f, "r").readlines()) for f in fpath]
    # sort the dataset by train data size
    forder = [f for l, f in sorted(zip(lines, fname))]
    fpath = [os.path.join(project_dir, "train_data/{}".format(f))
             for f in forder]
    if dataset != "all":
        forder = [name for name, file in zip(
            forder, fpath) if name[:-4] == dataset]
        fpath = [file for name, file in zip(
            forder, fpath) if name[:-4] == dataset]
    return forder, fpath
