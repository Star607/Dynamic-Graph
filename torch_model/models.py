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
import scipy.sparse as ssp
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data_loader.data_util import load_data, load_split_edges, load_label_edges
from model.utils import get_free_gpu, timeit, EarlyStopMonitor
from torch_model.util_dgl import set_logger, construct_dglgraph
from torch_model.layers import TemporalLinkLayer, TSAGEConv
from torch_model.eid_precomputation import _prepare_deg_indices, _par_deg_indices, _deg_indices_full, _par_deg_indices_full, _latest_edge, LatestNodeInteractionFinder
# A cpp extension computing upper_bound along the last dimension of an non-decreasing matrix. It saves huge memory use.
import upper_bound_cpp

# Change the order so that it is the one used by "nvidia-smi" and not the
# one used by all other programs ("FASTEST_FIRST")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class TGraphSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers, activation,
                 dropout, agg_type="mean", time_encoding="cosine"):
        super(TGraphSAGE, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(TSAGEConv(in_feats, n_hidden,
                                     agg_type, time_encoding=time_encoding))
        for i in range(n_layers - 2):
            self.layers.append(TSAGEConv(n_hidden, n_hidden,
                                         agg_type, time_encoding=time_encoding))
        if n_layers >= 2:
            self.layers.append(TSAGEConv(n_hidden, out_feats,
                                         agg_type, time_encoding=time_encoding))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, g):
        """In the 1st layer, we use the node features/embeddings as the features
        for each edge. In the next layers, we store the edge features in the edges,
        named `src_feat{current_layer}` and `dst_feat{current_layer}`.
        """
        g = g.local_var()

        def combine_feats(edges):
            return {"src_feat0": torch.cat(
                [edges.src["nfeat"], edges.data["efeat"]], dim=1),
                "dst_feat0": torch.cat(
                [edges.dst["nfeat"], edges.data["efeat"]], dim=1)}
        g.apply_edges(func=combine_feats)

        for i, layer in enumerate(self.layers):
            cl = i + 1
            src_feat, dst_feat = layer(g, current_layer=cl)
            g.edata[f"src_feat{cl}"] = self.activation(self.dropout(src_feat))
            g.edata[f"dst_feat{cl}"] = self.activation(self.dropout(dst_feat))
        src_feat, dst_feat = g.edata[f"src_feat{cl}"], g.edata[f"dst_feat{cl}"]
        return src_feat, dst_feat


class NegativeSampler(object):
    def __init__(self, g, k):
        # self.weights = g.in_degrees().float() ** 0.75
        self.n_nodes = g.number_of_nodes()
        self.k = k

    def __call__(self, g, eids):
        src, _ = g.find_edges(eids)
        n = len(src)
        # dst = self.weights.multinomial(self.k * n, replacement=True)
        dst = torch.randint(self.n_nodes, (self.k * n,))
        src = src.repeat_interleave(self.k * n)
        return src, dst


class TemporalLinkTrainer(nn.Module):
    def __init__(self, g, in_feats, n_hidden, out_feats, args,
                 remain_history=True, pos_contra=False, neg_contra=False):
        super(TemporalLinkTrainer, self).__init__()
        self.logger = logging.getLogger()
        if args.trainable and g.ndata["nfeat"].requires_grad:
            self.nfeat = g.ndata["nfeat"]
            self.logger.info(
                "Optimization includes randomly initialized dim-{} node embeddings.".format(self.nfeat.shape[-1]))
        else:
            self.logger.info(
                "Optimization only includes convolution parameters.")

        self.conv = TGraphSAGE(in_feats, n_hidden, out_feats,
                               args.n_layers, F.relu, args.dropout,
                               args.agg_type, args.time_encoding)
        self.pred = TemporalLinkLayer(
            out_feats, 1, time_encoding=args.time_encoding)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.n_neg = args.n_neg
        self.n_hist = args.n_hist
        self.remain_history = remain_history
        self.margin = nn.MarginRankingLoss(args.margin)
        self.pos_contra = pos_contra
        self.neg_contra = neg_contra
        self.neg_sampler = NegativeSampler(g, self.n_neg)
        if args.norm:
            self.norm = nn.LayerNorm(out_feats)

    def forward(self, g, batch_eids, bidirected=False):
        g = g.local_var()
        g.ndata["deg"] = (g.in_degrees() +
                          g.out_degrees()).to(g.ndata["nfeat"])
        if bidirected:
            g.ndata["deg"] /= 2
        src_feat, dst_feat = self.conv(g)
        if hasattr(self, "norm"):
            src_feat, dst_feat = self.norm(src_feat), self.norm(dst_feat)
        g.edata["src_feat"] = src_feat
        g.edata["dst_feat"] = dst_feat

        src, dst = g.find_edges(batch_eids)
        t = g.edata["timestamp"][batch_eids]
        src_eids = LatestNodeInteractionFinder(g, src, t, mode="out")
        dst_eids = LatestNodeInteractionFinder(g, dst, t, mode="in")
        _, neg_dst = self.neg_sampler(g, batch_eids)
        neg_eids = LatestNodeInteractionFinder(g, neg_dst, t, mode="in")
        # self.pred(g, batch_eids, bidirected=bidirected)
        pos_logits = self.pred(g, src_eids, dst_eids, t).squeeze()
        neg_logits = self.pred(g, src_eids.repeat(
            self.n_neg), neg_eids, t.repeat(self.n_neg)).squeeze()
        loss = self.loss_fn(pos_logits, torch.ones_like(pos_logits))
        loss += self.loss_fn(neg_logits, torch.zeros_like(neg_logits))
        loss += self.contrastive(g, t, src_eids, pos_logits, neg_logits)
        return loss

    def contrastive(self, g, t, src_eids, pos_logits, neg_logits):
        loss = 0.
        if not (self.pos_contra or self.neg_contra):
            return loss
        assert self.n_neg == self.n_hist, "We only implement for the equal \
            number history samples and negative samples."
        upper_ = (g.edata["src_deg_indices"][src_eids] + 1).view(-1, 1)
        cands = (torch.rand(len(src_eids), self.n_hist)
                 * upper_).long().view(-1)
        if self.remain_history:
            contra_eids = cands
        else:
            nodes = g.find_edges(cands)[1]
            contra_eids = LatestNodeInteractionFinder(g, nodes, t, mode="in")
        contra_logits = self.pred(g, src_eids, contra_eids, t).squeeze()
        if self.pos_contra:
            loss += self.margin(pos_logits.repeat(self.n_hist),
                                contra_logits, torch.ones_like(contra_logits))
        if self.neg_contra:
            loss += self.margin(contra_logits, neg_logits,
                                torch.ones_like(contra_logits))
        return loss

    def infer(self, g, edges, bidirected=False):
        self.eval()
        g = g.local_var()
        g.ndata["deg"] = (g.in_degrees() +
                          g.out_degrees()).to(g.ndata["nfeat"])
        if bidirected:
            g.ndata["deg"] /= 2
        src_feat, dst_feat = self.conv(g)
        g.edata["src_feat"] = src_feat
        g.edata["dst_feat"] = dst_feat

        u, v, t = edges[0], edges[1], torch.tensor(
            edges[2]).float().to(g.ndata["deg"].device)
        src_eids = LatestNodeInteractionFinder(g, u, t, mode="out")
        dst_eids = LatestNodeInteractionFinder(g, v, t, mode="in")
        logits = self.pred(g, src_eids, dst_eids, t)
        return logits


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
    edges, train_labels, val_labels, test_labels = [
        _f(e) for e in [edges, train_labels, val_labels, test_labels]]
    tmax, tmin = edges["timestamp"].max(), edges["timestamp"].min()
    def scaler(s): return (s - tmin) / (tmax - tmin)
    edges["timestamp"] = scaler(edges["timestamp"])
    val_labels["timestamp"] = scaler(val_labels["timestamp"])
    test_labels["timestamp"] = scaler(test_labels["timestamp"])
    return nodes, edges, train_labels, val_labels, test_labels


def _check_upper_bound_extension(g):
    for _ in range(3):
        deg_indices = _prepare_deg_indices(g)
        par_deg_indices = _par_deg_indices(g)
        full_deg_indices = _deg_indices_full(g)
        par_full_deg_indices = _par_deg_indices_full(g)
        for key in deg_indices.keys():
            assert torch.equal(deg_indices[key], par_deg_indices[key]), key
            assert torch.equal(deg_indices[key], full_deg_indices[key]), key
            assert torch.equal(
                deg_indices[key], par_full_deg_indices[key]), key


def _check_latest_edge_extension(g, edges):
    u = torch.tensor(edges["from_node_id"])
    v = torch.tensor(edges["to_node_id"])
    t = torch.tensor(edges["timestamp"])
    for _ in range(3):
        pyeids = LatestNodeInteractionFinder(g, u, t, mode="in")
        cppeids = _latest_edge(g, u, t, mode="in")
        assert torch.equal(pyeids, cppeids)
        pyeids = LatestNodeInteractionFinder(g, v, t, mode="out")
        cppeids = _latest_edge(g, v, t, mode="out")
        assert torch.equal(pyeids, cppeids)


def _df2np(edges):
    return edges.iloc[:, 0].to_numpy(), edges.iloc[:, 1].to_numpy(), edges.iloc[:, 2].to_numpy()


def eval_linkpred(model, g, df, batch_size=None):
    if batch_size is None:
        batch_size = df.shape[0]
    model.eval()
    with torch.no_grad():
        logits = model.infer(
            g, _df2np(df.iloc[:batch_size]))
        logits = torch.sigmoid(logits).cpu().numpy()
        labels = df["label"].iloc[:batch_size]
        acc = accuracy_score(labels, logits >= 0.5)
        f1 = f1_score(labels, logits >= 0.5)
        auc = roc_auc_score(labels, logits)
    return acc, f1, auc


def write_result(val_auc, metrics, dataset, params):
    res_path = "results/{}-GTC.csv".format(dataset)
    headers = ["method", "dataset", "valid_auc",
               "accuracy", "f1", "auc", "params"]
    acc, f1, auc = metrics
    if not os.path.exists(res_path):
        f = open(res_path, 'w')
        f.write(",".join(headers) + "\r\n")
        f.close()
        os.chmod(res_path, 0o777)
    with open(res_path, 'a') as f:
        result_str = "GTC,{},{:.4f},{:.4f},{:.4f},{:.4f}".format(
            dataset, val_auc, acc, f1, auc)
        logging.info(result_str)
        params_str = ",".join(["{}={}".format(k, v)
                               for k, v in params.items()])
        params_str = "\"{}\"".format(params_str)
        row = result_str + "," + params_str + "\r\n"
        f.write(row)


def main():
    # Set arg_parser, logger, and etc.
    args = parse_args()
    logger = set_logger()
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
    nodes, edges, train_labels, val_labels, test_labels = prepare_dataset(
        args.dataset)
    delta = edges["timestamp"].shift(-1) - edges["timestamp"]
    # Pandas loc[low:high] includes high, so we use slice operations here instead.
    assert np.all(delta[:len(delta) - 1] >= 0)

    # Set DGLGraph, node_features, edge_features, and edge timestamps.
    g = construct_dglgraph(edges, nodes, device, bidirected=args.bidirected)
    # For each entry in the adjacency list `u: (v, t)` of node `u`, compute
    # the related upper_bound with respect to `t`. So that we can use `cumsum`
    # or `cummax` to accelerate the computation speed. Otherwise, we have to
    # compute a mask matrix multiplication for each `(v, t)`, which costs even
    # 12GB memory for 58K interactions.
    deg_indices = _par_deg_indices_full(g)
    for k, v in deg_indices.items():
        g.edata[k] = v.to(device).unsqueeze(-1).detach()

    # Set model configuration.
    # Input features: node_featurs + edge_features + time_encoding
    in_feats = (g.ndata["nfeat"].shape[-1] + g.edata["efeat"].shape[-1])
    model = TemporalLinkTrainer(
        g, in_feats, args.n_hidden, args.n_hidden, args)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # clip gradients by value: https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(
            grad, -args.clip, args.clip))

    # Only use positive edges, so we have to multiply eids with 2.
    train_eids = np.arange(train_labels.shape[0] // 2)
    num_batch = np.int(np.ceil(len(train_eids) / args.batch_size))
    epoch_bar = trange(args.epochs, disable=(not args.display))
    early_stopper = EarlyStopMonitor(max_round=10)
    for epoch in epoch_bar:
        np.random.shuffle(train_eids)
        batch_bar = trange(num_batch, disable=(not args.display))
        for idx in batch_bar:
            model.train()
            batch_eids = train_eids[idx * args.batch_size:
                                    (idx + 1) * args.batch_size]
            mul = 2 if args.bidirected else 1
            loss = model(g, batch_eids * mul)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc, f1, auc = eval_linkpred(
                model, g, val_labels, batch_size=args.batch_size)
            batch_bar.set_postfix(loss=loss.item(), acc=acc, f1=f1, auc=auc)

        acc, f1, auc = eval_linkpred(model, g, val_labels)
        # if epoch % 100 == 0:
        #     logger.info("epoch:%d acc: %.4f, auc: %.4f, f1: %.4f",
        #                 epoch, acc, auc, f1)
        epoch_bar.update()
        epoch_bar.set_postfix(loss=loss.item(), acc=acc, f1=f1, auc=auc)

        lr = "%.4f" % args.lr
        trainable = "train" if hasattr(model, "nfeat") else "no-train"
        norm = "norm" if hasattr(model, "norm") else "no-norm"
        contra = "contra" if (
            model.pos_contra or model.neg_contra) else "no-contra"
        def ckpt_path(
            epoch): return f'./ckpt/{args.dataset}-{args.agg_type}-{trainable}-{norm}-{contra}-{lr}-{epoch}.pth'
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
    _, _, val_auc = eval_linkpred(model, g, val_labels)
    acc, f1, auc = eval_linkpred(model, g, test_labels)
    params = {"best_epoch": early_stopper.best_epoch,
              "bidirected": args.bidirected, "trainable": trainable,
              "norm": norm, "contra": contra, "lr": lr,
              "n_hist": args.n_hist,
              "n_neg": args.n_neg, "n_layers": args.n_layers,
              "time_encoding": args.time_encoding, "dropout": args.dropout,
              "weight_decay": args.weight_decay}
    write_result(val_auc, (acc, f1, auc), args.dataset, params)


def parse_args():
    parser = argparse.ArgumentParser(description='Temporal GraphSAGE')
    parser.add_argument("--dataset", type=str, default="ia-contact")
    parser.add_argument("--no-bidirected", dest="bidirected", action="store_false",
                        help="For bipartite graphs, set this as False.")
    parser.add_argument("--bidirected", dest="bidirected", action="store_true",
                        help="For non-bipartite graphs, set this as True.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--log-file", action="store_true")
    parser.add_argument("--gpu", dest="gpu", action="store_true",
                        help="Whether use GPU.")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false",
                        help="Whether use GPU.")
    parser.add_argument("--gid", type=int, default=-1,
                        help="Specify GPU id.")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--trainable", action="store_true")
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--time-encoding", "-te", type=str, default="cosine",
                        help="Time encoding function.", choices=["concat", "cosine", "outer"])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-hidden", type=int, default=128,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of hidden gcn layers")
    parser.add_argument("--n-neg", type=int, default=1,
                        help="number of negative samples")
    parser.add_argument("--contrastive", action="store_true")
    parser.add_argument("--n-hist", type=int, default=1,
                        help="number of history samples")
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight for L2 loss")
    parser.add_argument("--clip", type=float, default=5.0,
                        help="Clip gradients by value.")
    parser.add_argument("--agg-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm/cosine")
    parser.add_argument("--display", dest="display", action="store_true")
    parser.add_argument("--no-display", dest="display", action="store_false"
                        )
    return parser.parse_args()


if __name__ == "__main__":
    main()
