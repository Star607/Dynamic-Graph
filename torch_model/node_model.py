from .models import TGraphSAGE, write_result
import argparse
from datetime import datetime
import itertools
import logging
import os
import sys
import time

import dgl  # import dgl after torch will cause `GLIBCXX_3.4.22` not found.
from dgl.nn.pytorch.conv import SAGEConv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data_loader.data_util import load_data, load_split_edges, load_label_edges
from model.utils import get_free_gpu, timeit, EarlyStopMonitor
from torch_model.util_dgl import set_logger, construct_dglgraph
from torch_model.layers import TemporalNodeLayer, TSAGEConv
from torch_model.eid_precomputation import _prepare_deg_indices, _par_deg_indices, _deg_indices_full, _par_deg_indices_full, _latest_edge, LatestNodeInteractionFinder
# A cpp extension computing upper_bound along the last dimension of an non-decreasing matrix. It saves huge memory use.
import upper_bound_cpp

# Change the order so that it is the one used by "nvidia-smi" and not the
# one used by all other programs ("FASTEST_FIRST")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class TemporalNodeTrainer(nn.Module):
    def __init__(self, g, in_feats, n_hidden, out_feats, args):
        super(TemporalNodeTrainer, self).__init__()
        self.logger = logging.getLogger()
        self.nfeat = g.ndata["nfeat"]
        self.efeat = g.edata["efeat"]
        self.conv = TGraphSAGE(in_feats, n_hidden, out_feats,
                               args.n_layers, F.relu, args.dropout,
                               args.agg_type, args.time_encoding)
        self.pred = TemporalNodeLayer(
            out_feats, 1, time_encoding=args.time_encoding)
        self.loss_fn = nn.BCEWithLogitsLoss()
        if args.norm:
            self.norm = nn.LayerNorm(out_feats)

    def forward(self, g, node_ids, ts, labels, bidirected=False):
        logits = self.infer(g, node_ids, ts, bidirected=bidirected)
        labels = torch.tensor(labels).to(self.nfeat).unsqueeze(-1)
        loss = self.loss_fn(logits, labels)
        return loss
    
    def infer(self, g, node_ids, ts, bidirected=False):
        g = g.local_var()
        node_ids, ts = torch.tensor(node_ids), torch.tensor(ts).to(self.nfeat)
        g.ndata["deg"] = (g.in_degrees() +
                          g.out_degrees()).to(g.ndata["nfeat"])
        if bidirected:
            g.ndata["deg"] /= 2
        src_feat, dst_feat = self.conv(g)
        if hasattr(self, "norm"):
            src_feat, dst_feat = self.norm(src_feat), self.norm(dst_feat)
        g.edata["src_feat"] = src_feat
        g.edata["dst_feat"] = dst_feat

        node_eids = _latest_edge(g, node_ids, ts, mode="out")
        logits = self.pred(g, node_eids, ts)
        return logits


def prepare_node_dataset(dataset):
    edges, nodes = load_split_edges(dataset=dataset)
    train_labels, val_labels, test_labels = edges[0]
    edges = pd.concat(edges[0]).reset_index(drop=True)
    nodes = nodes[0]
    id2idx = {row.node_id: row.id_map for row in nodes.itertuples()}

    def _f(edges):
        edges["from_node_id"] = edges["from_node_id"].map(id2idx)
        edges["to_node_id"] = edges["to_node_id"].map(id2idx)
        return edges
    edges, train_labels, val_labels, test_labels = [
        _f(e) for e in [edges, train_labels, val_labels, test_labels]]

    tmax, tmin = edges["timestamp"].max(), edges["timestamp"].min()
    def scaler(s): return (s - tmin) / (tmax - tmin)
    edges["timestamp"] = scaler(edges["timestamp"])
    train_labels["timestamp"] = scaler(train_labels["timestamp"])
    val_labels["timestamp"] = scaler(val_labels["timestamp"])
    test_labels["timestamp"] = scaler(test_labels["timestamp"])
    return nodes, edges, train_labels, val_labels, test_labels


def eval_nodeclass(model, g, val_data, batch_size=None):
    if batch_size is None:
        batch_size = val_data.shape[0]
    model.eval()
    val_data = val_data.iloc[:batch_size]
    with torch.no_grad():
        node_ids = val_data["from_node_id"].to_numpy()
        ts = val_data["timestamp"].to_numpy()
        labels = val_data["state_label"].to_numpy()
        logits = model.infer(g, node_ids, ts).sigmoid().cpu().numpy()
        acc = accuracy_score(labels, logits >= 0.5)
        f1 = f1_score(labels, logits >= 0.5)
        auc = roc_auc_score(labels, logits)
    return acc, f1, auc


def main(args, logger):
    logger.info(args)

    # Set device utility.
    if args.gpu:
        if args.gid >= 0:
            device = torch.device("cuda:{}".format(args.gid))
        else:
            device = torch.device("cuda:{}".format(get_free_gpu()))
        logger.info("Begin Conv on Device %s, GPU Memory %d GB", device,
                    torch.cuda.get_device_properties(device).total_memory // 2**30)
    else:
        device = torch.device("cpu")

    # Load nodes, edges, and labeled dataset for training, validation and test.
    nodes, edges, train_data, val_data, test_data = prepare_node_dataset(
        args.dataset)
    delta = edges["timestamp"].shift(-1) - edges["timestamp"]
    # Pandas loc[low:high] includes high, so we use slice operations here instead.
    assert np.all(delta[:len(delta) - 1] >= 0)

    # Set DGLGraph, node_features, edge_features, and edge timestamps.
    g = construct_dglgraph(edges, nodes, device, bidirected=args.bidirected)
    if not args.trainable:
        g.ndata["nfeat"] = torch.zeros_like(g.ndata["nfeat"])
    deg_indices = _par_deg_indices_full(g)
    for k, v in deg_indices.items():
        g.edata[k] = v.to(device).unsqueeze(-1).detach()

    # Set model configuration.
    # Input features: node_featurs + edge_features + time_encoding
    in_feats = (g.ndata["nfeat"].shape[-1] + g.edata["efeat"].shape[-1])
    model = TemporalNodeTrainer(
        g, in_feats, args.n_hidden, args.n_hidden, args)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(
            grad, -args.clip, args.clip))

    train_ids = np.arange(len(train_data))
    batch_size = args.batch_size
    num_batch = np.int(np.ceil(len(train_data) / batch_size))
    epoch_bar = trange(args.epochs, disable=(not args.display))
    early_stopper = EarlyStopMonitor(max_round=5)
    for epoch in epoch_bar:
        np.random.shuffle(train_ids)
        batch_bar = trange(num_batch, disable=(not args.display))
        for idx in batch_bar:
            model.eval()
            batch_ids = train_ids[idx * batch_size: (idx + 1) * batch_size]
            node_ids = train_data.loc[batch_ids, "from_node_id"].to_numpy()
            ts = train_data.loc[batch_ids, "timestamp"].to_numpy()
            labels = train_data.loc[batch_ids, "state_label"].to_numpy()
            loss = model(g, node_ids, ts, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc, f1, auc = eval_nodeclass(
                model, g, val_data, batch_size=args.batch_size)
            batch_bar.set_postfix(loss=loss.item(), acc=acc, f1=f1, auc=auc)
        acc, f1, auc = eval_nodeclass(model, g, val_data)
        epoch_bar.update()
        epoch_bar.set_postfix(loss=loss.item(), acc=acc, f1=f1, auc=auc)

        lr = "%.4f" % args.lr
        trainable = "train" if hasattr(model, "nfeat") else "no-train"
        norm = "norm" if hasattr(model, "norm") else "no-norm"
        def ckpt_path(
            epoch): return f'./ckpt/{args.dataset}-{args.agg_type}-{trainable}-{norm}-{lr}-{epoch}-{args.hostname}-{device.type}-{device.index}.pth'
        if early_stopper.early_stop_check(auc):
            logger.info(
                f"No improvement over {early_stopper.max_round} epochs.")
            logger.info(
                f'Loading the best model at epoch {early_stopper.best_epoch}')
            model.load_state_dict(torch.load(
                ckpt_path(early_stopper.best_epoch)))
            logger.info(
                f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            break
        else:
            torch.save(model.state_dict(), ckpt_path(epoch))
    model.eval()
    _, _, val_auc = eval_nodeclass(model, g, val_data)
    acc, f1, auc = eval_nodeclass(model, g, test_data)
    params = {"best_epoch": early_stopper.best_epoch,
              "bidirected": args.bidirected, "trainable": trainable,
              "norm": norm, "lr": lr,
              "n_layers": args.n_layers,
              "time_encoding": args.time_encoding, "dropout": args.dropout,
              "weight_decay": args.weight_decay}
    write_result(val_auc, (acc, f1, auc), args.dataset,
                 params, postfix="NC-GTC")


def parse_args():
    import socket
    parser = argparse.ArgumentParser(description='Temporal GraphSAGE')
    parser.add_argument("-d", "--dataset", type=str, default="JODIE-wikipedia",
                        choices=["JODIE-wikipedia", "JODIE-mooc", "JODIE-reddit"])
    parser.add_argument("--bidirected", dest="bidirected", action="store_true",
                        help="For non-bipartite graphs, set this as True.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--log-file", action="store_true")
    parser.add_argument("--gpu", dest="gpu", action="store_true",
                        help="Whether use GPU.")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false",
                        help="Whether use GPU.")
    hostname = socket.gethostname()
    parser.add_argument("--hostname", action="store_const",
                        const=hostname, default=hostname)
    parser.add_argument("--gid", type=int, default=-1,
                        help="Specify GPU id.")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--trainable", "-train",
                        dest="trainable", action="store_true")
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--epochs", type=int, default=50,
                        help="number of training epochs")
    parser.add_argument("--time-encoding", "-te", type=str, default="cosine",
                        help="Time encoding function.", choices=["concat", "cosine", "outer"])
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
                        help="Aggregator type: mean/gcn/pool/lstm/cosine")
    parser.add_argument("--display", dest="display", action="store_true")
    parser.add_argument("--no-display", dest="display", action="store_false"
                        )
    return parser


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    logger = set_logger()
    main(args, logger)
