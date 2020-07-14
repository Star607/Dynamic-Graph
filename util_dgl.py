import argparse
import os
import logging
import time

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch
from torch import nn
import dgl
import scipy.sparse as ssp


def construct_dglgraph(edges, nodes):
    ''' Edges should be a pandas DataFrame, and its columns should be like from_node_id, to_node_id, timestamp, state_label, features_separated_by_comma. Here `state_label` varies in edge classification tasks.

    Nodes should be a pandas DataFrame, and its columns should be like node_id, id_map, role, label, features_separated_by_comma.
    '''
    efeature = None
    nfeature = None
    pass


def main(args):
    pass


def nodeflow_test():
    pass


def fullgraph_test():
    pass


def subgraph_test():
    pass


def test():
    src = np.array([0, 0, 1, 2, 2])
    dst = np.array([1, 1, 2, 0, 0])
    etime = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    g = dgl.DGLGraph((u, v))
    g.ndata["x"] = torch.rand((g.number_of_nodes(), 5))
    g.edata["timestamp"] = etime
    return g


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # main(args)
    test()
