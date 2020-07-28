import argparse
from datetime import datetime
import itertools
import logging
import os
import sys
import time

import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch.conv import SAGEConv
from dgl.utils import expand_as_pair, check_eq_shape
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as ssp
import torch
import torch as th
import torch.nn as nn
from torch.nn import functional as F
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data_loader.minibatch import load_data
from model.utils import get_free_gpu
from torch_model.util_dgl import set_logger, construct_dglgraph, padding_node, TSAGEConv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", dest="gpu", action="store_true",
                        help="Whether use GPU.")
    return parser.parse_args(["--gpu"])


class TGraphSAGE(nn.Module):
    def __init__(self, g, in_feats, n_hidden, out_feats, n_layers, activation, dropout, aggregator_type="mean"):
        super(TGraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g

        self.layers.append(TSAGEConv(in_feats, n_hidden, aggregator_type,
                                     feat_drop=dropout, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(TSAGEConv(n_hidden, n_hidden, aggregator_type,
                                         feat_drop=dropout, activation=activation))
        # for node classification task, n_classes mean the node logits
        # for link prediction task, n_classes mean the output dimension
        # for link classification task, n_classes mean the link logits
        self.layers.append(TSAGEConv(n_hidden, out_feats, aggregator_type,
                                     feat_drop=dropout, activation=None))

    def forward(self, features):
        """In the 1st layer, we use the node features/embeddings as the features for each edge.
        In the next layers, we store the edge features in the edges, named `src_feat{current_layer}` and `dst_feat{current_layer}`.
        """
        g = self.g.local_var()
        g.ndata["nfeat"] = features
        for i, layer in enumerate(self.layers):
            cl = i + 1
            src_feat, dst_feat = layer(g, current_layer=cl)
            g.edata[f"src_feat{cl}"] = src_feat
            g.edata[f"dst_feat{cl}"] = dst_feat
        return src_feat, dst_feat


def evaluate(model, features):
    model.eval()
    with torch.no_grad():
        src_feat, dst_feat = model(features)
        logits = torch.cat((src_feat, dst_feat))
        _, indices = torch.max(logits, dim=1)
        return indices.cpu().numpy()


def eval_linkprediction(model, features, pairs):
    model.eval()
    with torch.no_grad():
        src_feat, dst_feat = model(features)
        src_pair, dst_pair = src_feat[pairs[0]], dst_feat[pairs[1]]
        logits = torch.sigmoid((src_pair * dst_pair).sum(dim=1))
        return logits.cpu().numpy()


def main():
    args = parse_args()
    logger = set_logger()
    logger.info(args)
    edges, nodes = load_data(dataset=args.dataset)
    if args.gpu:
        device = torch.device("cuda:{}".format(get_free_gpu()))
    else:
        device = torch.device("cpu")

    edges, nodes = padding_node(edges, nodes)
    delta = edges["timestamp"].shift(-1) - edges["timestamp"]
    assert np.all(delta[:len(delta) - 1] >= 0)  # pandas loc[low:high] includes high
    g = construct_dglgraph(edges, nodes, device)
    import copy
    nfeat_copy = copy.deepcopy(g.ndata["nfeat"])
    logger.info("Begin Conv on Device %s", device)

    in_feats = g.ndata["nfeat"].shape[-1]
    out_feats = 128
    n_edges = g.number_of_edges()
    model = TGraphSAGE(g, in_feats, args.n_hidden, out_feats,
                       args.n_layers, F.relu, args.dropout, args.agg_type)
    model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    if g.ndata["nfeat"].requires_grad:
        logger.info(
            "Optimization includes randomly initialized dim-{} node embeddings.".format(g.ndata["nfeat"].shape[-1]))
        optimizer = torch.optim.Adam(itertools.chain(
            [g.ndata["nfeat"]], model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        logger.info("Optimization only includes convolution parameters.")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # for epoch in range(args.epochs):
    epoch_bar = trange(args.epochs, disable=(not args.display))
    for epoch in epoch_bar:
        model.train()
        src_feats, dst_feats = model(g.ndata["nfeat"])
        neg_feats = dst_feats[torch.randint(high=n_edges, size=(n_edges,))]
        logits = torch.cat([src_feats * dst_feats, src_feats * neg_feats]
                           ).sum(dim=1)  # 2E training samples
        labels = torch.cat((torch.ones(n_edges), torch.zeros(n_edges))).to(device)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert not torch.all(torch.eq(nfeat_copy, g.ndata["nfeat"]))

        src_pair = np.hstack((np.arange(n_edges), np.arange(n_edges)))
        dst_pair = np.hstack((np.arange(n_edges), np.random.randint(0, n_edges, n_edges)))
        logits = eval_linkprediction(model, g.ndata["nfeat"], [src_pair, dst_pair])
        labels = np.concatenate((np.ones(n_edges), np.zeros(n_edges)))
        acc = accuracy_score(labels, logits >= 0.5)
        f1 = f1_score(labels, logits >= 0.5)
        auc = roc_auc_score(labels, logits)
        epoch_bar.update()
        epoch_bar.set_postfix(loss=loss.item(), acc=acc, f1=f1, auc=auc)


def parse_args():
    parser = argparse.ArgumentParser(description='Temporal GraphSAGE')
    parser.add_argument("--dataset", type=str, default="ia-contact")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="dropout probability")
    parser.add_argument("--gpu", dest="gpu", action="store_true",
                        help="Whether use GPU.")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false",
                        help="Whether use GPU.")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=128,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=0,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--agg-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument("--display", dest="display", action="store_true")
    parser.add_argument("--no-display", dest="display", action="store_false"
                        )
    return parser.parse_args()


if __name__ == "__main__":
    main()
