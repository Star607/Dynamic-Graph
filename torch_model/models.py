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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data_loader.data_util import load_data, load_split_edges, load_label_edges
from model.utils import get_free_gpu, timeit
from torch_model.util_dgl import set_logger, construct_dglgraph, padding_node, TSAGEConv
from torch_model.layers import TemporalLinkLayer
# A cpp extension computing upper_bound along the last dimension of an non-decreasing matrix. It saves huge memory use.
import upper_bound_cpp

# Change the order so that it is the one used by "nvidia-smi" and not the
# one used by all other programs ("FASTEST_FIRST")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class TGraphSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers, activation, dropout, aggregator_type="mean"):
        super(TGraphSAGE, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(TSAGEConv(in_feats, n_hidden, aggregator_type))
        for i in range(n_layers - 2):
            self.layers.append(TSAGEConv(n_hidden, n_hidden, aggregator_type))
        self.layers.append(TSAGEConv(n_hidden, out_feats, aggregator_type))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, g, features):
        """In the 1st layer, we use the node features/embeddings as the features for each edge.
        In the next layers, we store the edge features in the edges, named `src_feat{current_layer}` and `dst_feat{current_layer}`.
        """
        g = g.local_var()
        g.ndata["nfeat"] = features
        for i, layer in enumerate(self.layers):
            # print(f"layer {i}")
            cl = i + 1
            src_feat, dst_feat = layer(g, current_layer=cl)
            g.edata[f"src_feat{cl}"] = self.activation(self.dropout(src_feat))
            g.edata[f"dst_feat{cl}"] = self.activation(self.dropout(dst_feat))
        return src_feat, dst_feat


class TemporalLinkTrainer(nn.Module):
    def __init__(self, in_feats, args):
        super(TemporalLinkTrainer, self).__init__()
        self.conv = TGraphSAGE(in_feats, args.n_hidden, in_feats,
                               args.n_layers, F.relu, args.dropout, args.agg_type)
        self.pred = TemporalLinkLayer(in_feats, 1, time_encoding=args.time_encoding)

    def forward(self, g, features, label_edges, bidirected=False):
        g = g.local_var()
        src_feat, dst_feat = self.conv(g, features)
        g.edata["src_feat"] = src_feat
        g.edata["dst_feat"] = dst_feat
        logits = self.pred(g, label_edges, bidirected=bidirected)
        return logits.squeeze()

    def infer(self, g, features, edges, bidirected=False):
        pass


def prepare_dataset(dataset):
    edges, nodes = load_split_edges(dataset=dataset)
    edges = pd.concat(edges[0]).reset_index(drop=True)
    nodes = nodes[0]
    labels, _ = load_label_edges(dataset=dataset)
    train_labels, val_labels, test_labels = labels[0]
    id2idx = {row.node_id: row.id_map for row in nodes.itertuples()}

    def _f(edges):
        edges["from_node_id"] = edges["from_node_id"].map(id2idx)
        edges["to_node_id"] = edges["to_node_id"].map(id2idx)
        return edges
    edges, train_labels,  val_labels, test_labels = [
        _f(e) for e in [edges, train_labels,  val_labels, test_labels]]
    tmax, tmin = edges["timestamp"].max(), edges["timestamp"].min()
    # scaler = MinMaxScaler().fit(edges["timestamp"])
    def scaler(s): return (s - tmin) / (tmax - tmin)
    edges["timestamp"] = scaler(edges["timestamp"])
    val_labels["timestamp"] = scaler(val_labels["timestamp"])
    test_labels["timestamp"] = scaler(test_labels["timestamp"])
    return nodes, edges, train_labels, val_labels, test_labels


@timeit
def _prepare_deg_indices(g):
    """Compute on CPU devices."""
    def _group_func_wrapper(groupby):
        def _compute_deg_indices(edges):
            buc, deg, dim = edges.src["nfeat"].shape
            t = edges.data["timestamp"].view(buc, deg, 1)
            # It doesn't change the behavior but saves to the 1/deg memory use.
            indices = upper_bound_cpp.upper_bound(t.squeeze(-1)).add_(-1)
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


@timeit
def _par_deg_indices(g):
    """Compute on CPU devices."""
    def _group_func_wrapper(groupby):
        def _compute_deg_indices(edges):
            buc, deg, dim = edges.src["nfeat"].shape
            t = edges.data["timestamp"].view(buc, deg, 1)
            # It doesn't change the behavior but saves to the 1/deg memory use.
            indices = upper_bound_cpp.upper_bound_par(t.squeeze(-1)).add_(-1)
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


@timeit
def _deg_indices_full(g):
    u, v, eids = g.out_edges(g.nodes(), 'all')
    etime = g.edata["timestamp"][eids].cpu()
    degs = g.out_degrees()
    src_deg_indices = upper_bound_cpp.upper_bound_full(u, etime, degs)
    # eids is a permutation of torch.arange(g.number_of_edges())
    # we can reverse the permutation by torch.argsort: eids[torch.argsort(eids)] == torch.arange(g.numer_of_edges())
    src_deg_indices = src_deg_indices[torch.argsort(eids)]
    u, v, eids = g.in_edges(g.nodes(), 'all')
    etime = g.edata["timestamp"][eids].cpu()
    degs = g.out_degrees()
    dst_deg_indices = upper_bound_cpp.upper_bound_full(v, etime, degs)
    dst_deg_indices = dst_deg_indices[torch.argsort(eids)]
    return {"src_deg_indices": src_deg_indices.add_(-1),
            "dst_deg_indices": dst_deg_indices.add(-1)}


@timeit
def _par_deg_indices_full(g):
    u, v, eids = g.out_edges(g.nodes(), 'all')
    etime = g.edata["timestamp"][eids].cpu()
    degs = g.out_degrees()
    src_deg_indices = upper_bound_cpp.upper_bound_full_par(u, etime, degs)
    # eids is a permutation of torch.arange(g.number_of_edges())
    # we can reverse the permutation by torch.argsort: eids[torch.argsort(eids)] == torch.arange(g.numer_of_edges())
    src_deg_indices = src_deg_indices[torch.argsort(eids)]
    u, v, eids = g.in_edges(g.nodes(), 'all')
    etime = g.edata["timestamp"][eids].cpu()
    degs = g.out_degrees()
    dst_deg_indices = upper_bound_cpp.upper_bound_full_par(v, etime, degs)
    dst_deg_indices = dst_deg_indices[torch.argsort(eids)]
    return {"src_deg_indices": src_deg_indices.add_(-1),
            "dst_deg_indices": dst_deg_indices.add(-1)}


def _df2np(edges):
    return edges.iloc[:, :3].to_numpy().transpose()


def train(args):
    pass


def main():
    args = parse_args()
    logger = set_logger()
    logger.info(args)
    nodes, edges, train_labels, val_labels, test_labels = prepare_dataset(args.dataset)
    delta = edges["timestamp"].shift(-1) - edges["timestamp"]
    # pandas loc[low:high] includes high
    assert np.all(delta[:len(delta) - 1] >= 0)

    if args.gpu:
        if args.gid >= 0:
            device = torch.device("cuda:{}".format(args.gid))
        else:
            device = torch.device("cuda:{}".format(get_free_gpu()))
    else:
        device = torch.device("cpu")
    g = construct_dglgraph(edges, nodes, device, bidirected=args.bidirected)
    for _ in range(3):
        deg_indices = _prepare_deg_indices(g)
        par_deg_indices = _par_deg_indices(g)
        full_deg_indices = _deg_indices_full(g)
        par_full_deg_indices = _par_deg_indices_full(g)
        for key in deg_indices.keys():
            assert torch.equal(deg_indices[key], par_deg_indices[key]), key
            assert torch.equal(deg_indices[key], full_deg_indices[key]), key
            assert torch.equal(deg_indices[key], par_full_deg_indices[key]), key

    exit(0)

    for k, v in deg_indices.items():
        g.edata[k] = v.to(device).unsqueeze(-1).detach()

    logger.info("Begin Conv on Device %s, GPU Memory %d GB", device,
                torch.cuda.get_device_properties(device).total_memory // 2**30)

    in_feats = g.ndata["nfeat"].shape[-1]
    out_feats = 128
    n_edges = g.number_of_edges()
    model = TemporalLinkTrainer(in_feats, args)
    model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    if g.ndata["nfeat"].requires_grad:
        logger.info(
            "Optimization includes randomly initialized dim-{} node embeddings.".format(g.ndata["nfeat"].shape[-1]))
        params = itertools.chain([g.ndata["nfeat"]], model.parameters())
    else:
        logger.info("Optimization only includes convolution parameters.")
        params = model.parameters()
    optimizer = torch.optim.Adam(
        params, lr=args.lr, weight_decay=args.weight_decay)

    # for epoch in range(args.epochs):
    epoch_bar = trange(args.epochs, disable=(not args.display))
    for epoch in epoch_bar:
        model.train()
        logits = model(g, g.ndata["nfeat"], _df2np(train_labels))
        labels = torch.tensor(train_labels["label"]).float().to(device)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        # clip gradients by value: https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
        for p in params:
            p.register_hook(lambda grad: torch.clamp(
                grad, -args.clip, args.clip))
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(g, g.ndata["nfeat"], _df2np(val_labels))
            logits = torch.sigmoid(logits).cpu().numpy()
            labels = val_labels["label"]
            acc = accuracy_score(labels, logits >= 0.5)
            f1 = f1_score(labels, logits >= 0.5)
            auc = roc_auc_score(labels, logits)

        if epoch % 100 == 0:
            logger.info("epoch:%d acc: %.4f, auc: %.4f, f1: %.4f",
                        epoch, acc, auc, f1)
        epoch_bar.update()
        epoch_bar.set_postfix(loss=loss.item(), acc=acc, f1=f1, auc=auc)


def parse_args():
    parser = argparse.ArgumentParser(description='Temporal GraphSAGE')
    parser.add_argument("--dataset", type=str, default="ia-contact")
    parser.add_argument("--no-bidirected", dest="bidirected", action="store_false",
                        help="For bipartite graphs, set this as False.")
    parser.add_argument("--bidirected", dest="bidirected", action="store_true",
                        help="For non-bipartite graphs, set this as True.")
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
    parser.add_argument("--time-encoding", "-te", type=str, default="cosine",
                        help="Time encoding function.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-hidden", type=int, default=128,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight for L2 loss")
    parser.add_argument("--clip", type=float, default=5.0,
                        help="Clip gradients by value.")
    parser.add_argument("--agg-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument("--display", dest="display", action="store_true")
    parser.add_argument("--no-display", dest="display", action="store_false"
                        )
    return parser.parse_args()


if __name__ == "__main__":
    main()
