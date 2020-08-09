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
from model.utils import get_free_gpu, timeit
from torch_model.util_dgl import set_logger, construct_dglgraph, padding_node, TSAGEConv
# A cpp extension computing upper_bound along the last dimension of an non-decreasing matrix. It saves huge memory use.
import upper_bound_cpp

# Change the order so that it is the one used by "nvidia-smi" and not the
# one used by all other programs ("FASTEST_FIRST")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", dest="gpu", action="store_true",
                        help="Whether use GPU.")
    return parser.parse_args(["--gpu"])


class TGraphSAGE(nn.Module):
    def __init__(self, g, in_feats, n_hidden, out_feats, n_layers, activation, dropout, aggregator_type="mean", task="linkpred"):
        super(TGraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g

        self.layers.append(TSAGEConv(in_feats, n_hidden, aggregator_type))
        for i in range(n_layers - 1):
            self.layers.append(TSAGEConv(n_hidden, n_hidden, aggregator_type))
        # for node classification task, n_classes mean the node logits
        # for link prediction task, n_classes mean the output dimension
        # for link classification task, n_classes mean the link logits
        # if task == "nodeclass":
        #     self.layers.append(TSAGEConv(n_hidden, out_feats, aggregator_type,
        #                                  feat_drop=dropout, activation=None))
        # elif task == "linkpred":
        #     self.layers.append()
        self.layers.append(TSAGEConv(n_hidden, out_feats, aggregator_type))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, features):
        """In the 1st layer, we use the node features/embeddings as the features for each edge.
        In the next layers, we store the edge features in the edges, named `src_feat{current_layer}` and `dst_feat{current_layer}`.
        """
        g = self.g.local_var()
        g.ndata["nfeat"] = features
        for i, layer in enumerate(self.layers):
            # print(f"layer {i}")
            cl = i + 1
            src_feat, dst_feat = layer(g, current_layer=cl)
            g.edata[f"src_feat{cl}"] = self.activation(self.dropout(src_feat))
            g.edata[f"dst_feat{cl}"] = self.activation(self.dropout(dst_feat))
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


def prepare_dataset(edges, nodes):
    """Use batch_iterator to maintain train, valid, and test data.
    """
    pass


@timeit
def _prepare_deg_indices(g):
    """Compute on CPU devices."""
    def _group_func_wrapper(groupby):
        def _compute_deg_indices(edges):
            buc, deg, dim = edges.src["nfeat"].shape
            t = edges.data["timestamp"].view(buc, deg, 1)
            # It doesn't change the behavior but saves to the 1/deg memory use.
            indices = upper_bound_cpp.forward(t.squeeze(-1)).add_(-1)
            # indices = (t.permute(0, 2, 1) <= t).sum(dim=-1).add_(-1)
            # assert torch.all(torch.eq(another_indices, indices))
            return {f"{groupby}_deg_indices": indices}
        return _compute_deg_indices
    g = g.local_var()
    g.edata["timestamp"] = g.edata["timestamp"].cpu()

    src_deg_indices = _group_func_wrapper(groupby="src")
    dst_deg_indices = _group_func_wrapper(groupby="dst")
    g.group_apply_edges(group_by="src", func=src_deg_indices)
    g.group_apply_edges(group_by="dst", func=dst_deg_indices)
    return {"src_deg_indices": g.edata["src_deg_indices"],
            "dst_deg_indices": g.edata["dst_deg_indices"]}


def main():
    args = parse_args()
    logger = set_logger()
    logger.info(args)
    edges, nodes = load_data(dataset=args.dataset)
    edges = edges.sort_values("timestamp").reset_index(drop=True)
    id2idx = {row["node_id"]:  row["id_map"]
              for index, row in nodes.iterrows()}
    edges["from_node_id"] = edges["from_node_id"].map(id2idx)
    edges["to_node_id"] = edges["to_node_id"].map(id2idx)
    # edges, nodes = padding_node(edges, nodes)
    delta = edges["timestamp"].shift(-1) - edges["timestamp"]
    assert np.all(delta[:len(delta) - 1] >= 0)  # pandas loc[low:high] includes high

    if args.gpu:
        if args.gid >= 0:
            device = torch.device("cuda:{}".format(args.gid))
        else:
            device = torch.device("cuda:{}".format(get_free_gpu()))
    else:
        device = torch.device("cpu")
    g = construct_dglgraph(edges, nodes, device, bidirected=args.bidirected)
    deg_indices = _prepare_deg_indices(g)
    for k, v in deg_indices.items():
        g.edata[k] = v.to(g.ndata["nfeat"].device).unsqueeze(-1).detach()

    # import copy
    # nfeat_copy = copy.deepcopy(g.ndata["nfeat"])
    logger.info("Begin Conv on Device %s, GPU Memory %d GB", device,
                torch.cuda.get_device_properties(device).total_memory // 2**30)

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
        params = itertools.chain([g.ndata["nfeat"]], model.parameters())
    else:
        logger.info("Optimization only includes convolution parameters.")
        params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

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
        # clip gradients by value: https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
        for p in params:
            p.register_hook(lambda grad: torch.clamp(grad, -args.clip, args.clip))
        optimizer.step()
        # assert not torch.all(torch.eq(nfeat_copy, g.ndata["nfeat"]))

        src_pair = np.hstack((np.arange(n_edges), np.arange(n_edges)))
        dst_pair = np.hstack((np.arange(n_edges), np.random.randint(0, n_edges, n_edges)))
        logits = eval_linkprediction(model, g.ndata["nfeat"], [src_pair, dst_pair])
        labels = np.concatenate((np.ones(n_edges), np.zeros(n_edges)))
        acc = accuracy_score(labels, logits >= 0.5)
        f1 = f1_score(labels, logits >= 0.5)
        auc = roc_auc_score(labels, logits)
        if epoch % 100 == 0:
            logger.info("epoch:%d acc: %.4f, auc: %.4f, f1: %.4f", epoch, acc, auc, f1)
        epoch_bar.update()
        epoch_bar.set_postfix(loss=loss.item(), acc=acc, f1=f1, auc=auc)


def parse_args():
    parser = argparse.ArgumentParser(description='Temporal GraphSAGE')
    parser.add_argument("--dataset", type=str, default="ia-contact")
    parser.add_argument("--bidirected", dest="bidirected", action="store_true",
                        help="For non-bipartite graphs, set this as True.")
    parser.add_argument("--no-bidirected", dest="bidirected", action="store_false",
                        help="For bipartite graphs, set this as False.")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="dropout probability")
    parser.add_argument("--gpu", dest="gpu", action="store_true",
                        help="Whether use GPU.")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false",
                        help="Whether use GPU.")
    parser.add_argument("--gid", type=int, default=-1,
                        help="Specify GPU id.")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-hidden", type=int, default=128,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=0,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight for L2 loss")
    parser.add_argument("--clip", type=float, default=5.0, help="Clip gradients by value.")
    parser.add_argument("--agg-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument("--display", dest="display", action="store_true")
    parser.add_argument("--no-display", dest="display", action="store_false"
                        )
    return parser.parse_args()


if __name__ == "__main__":
    main()
