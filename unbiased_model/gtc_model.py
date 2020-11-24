import argparse
from datetime import datetime
import itertools
import logging
import os
import random
import sys
import time

import dgl  # import dgl after torch will cause `GLIBCXX_3.4.22` not found.
import dgl.function as fn
from numba import jit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.rnn import LSTM
from torch.utils.checkpoint import checkpoint
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data_loader.data_util import load_data, load_split_edges, load_label_edges
from model.utils import get_free_gpu, timeit, EarlyStopMonitor
from torch_model.util_dgl import set_logger, compute_degrees, construct_adj, construct_dglgraph
from torch_model.layers import TimeEncodingLayer
from torch_model.models import set_random_seed, parse_args, prepare_dataset, write_result

# Change the order so that it is the one used by "nvidia-smi" and not the
# one used by all other programs ("FASTEST_FIRST")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class NegativeSampler(object):
    def __init__(self, dst_nodes: np.array, k: int) -> None:
        # self.weights = g.in_degrees().float() ** 0.75
        self.dst_nodes = dst_nodes
        self.n_nodes = self.dst_nodes.shape[0]
        self.k = k

    def __call__(self, n: int) -> np.array:
        dst = np.random.randint(self.n_nodes, size=(self.k * n, ))
        dst = self.dst_nodes[dst]
        return dst


@timeit
# @jit
def history_sampler(src_idx, cut_time, adj_ngh_l, adj_ts_l, k):
    out_ngh_ids = np.zeros_like(src_idx.repeat(k))
    out_ngh_ts = np.zeros_like(cut_time.repeat(k))

    for i, (idx, t) in enumerate(zip(src_idx, cut_time)):
        nghs = adj_ngh_l[idx]
        ts = adj_ts_l[idx]
        left = np.searchsorted(ts, t, side="left")
        left = max(1, left)
        hist = np.random.randint(left, size=k)
        out_ngh_ids[i * k:(i + 1) * k] = nghs[hist]
        out_ngh_ts[i * k:(i + 1) * k] = ts[hist]
    return out_ngh_ids, out_ngh_ts


@timeit
# @jit
def find_idx(src_idx, cut_time, new_nodes_l, adj_ts_l):
    out_node_ids = np.zeros_like(src_idx)
    # out_node_ts = np.zeros_like(cut_time)

    for i, (idx, t) in enumerate(zip(src_idx, cut_time)):
        nodes = new_nodes_l[idx]
        ts = adj_ts_l[idx]
        left = np.searchsorted(ts, t, side="left")
        left = max(0, left - 1)
        out_node_ids[i] = nodes[left]
        # out_node_ts[i] = ts[left]
    return out_node_ids


class GTCLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 agg_type,
                 edge_feats,
                 time_enc="cosine") -> None:
        super(GTCLayer, self).__init__()
        self._in_src_feats, self._in_dst_feats = in_feats, in_feats
        self._out_feats = out_feats
        self._agg_type = agg_type
        self.encode_time = TimeEncodingLayer(in_feats,
                                             in_feats,
                                             time_encoding=time_enc)
        self.fc_edge = nn.Linear(edge_feats, in_feats)
        if agg_type == "pool":
            self.fc_pool = nn.Linear(in_feats, in_feats)
        if agg_type == "lstm":
            self.lstm = LSTM(in_feats, in_feats, batch_first=True)
        if agg_type != "gcn":
            self.fc_self = nn.Linear(in_feats, out_feats)
        self.fc_neigh = nn.Linear(in_feats, out_feats)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if self._agg_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._agg_type == 'lstm':
            self.lstm.reset_parameters()
        if self._agg_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _lstm_reducer(self, edge_feat):
        """LSTM processing for temporal edges.
        """
        # (seq_len, batch_size, dim) <= (bucket_size, deg, dim)
        edge_feat = edge_feat.permute(1, 0, 2)
        batch_size = edge_feat.shape[1]
        h = (edge_feat.new_zeros((1, batch_size, self._in_src_feats)),
             edge_feat.new_zeros((1, batch_size, self._in_src_feats)))
        rst, (h_, c_) = self.lstm(edge_feat, h)
        return rst

    def forward(self, agg_graph: dgl.DGLGraph, prop_graph: dgl.DGLGraph,
                traversal_order, new_node_ids) -> torch.Tensor:
        tg = agg_graph.local_var()
        pg = prop_graph.local_var()

        nfeat = tg.ndata["nfeat"]
        # h_self = nfeat
        h_self = self.encode_time(nfeat, tg.ndata["timestamp"])
        tg.ndata["nfeat"] = h_self
        tg.edata["efeat"] = self.fc_edge(tg.edata["efeat"])
        # efeat = tg.edata["efeat"]
        # tg.apply_edges(lambda edges: {
        #     "efeat":
        #     torch.cat((edges.src["nfeat"], edges.data["efeat"]), dim=1)
        # })
        # tg.edata["efeat"] = self.encode_time(tg.edata["efeat"], tg.edata["timestamp"])
        degs = tg.ndata["degree"]

        # agg_graph aggregation
        if self._agg_type == "pool":
            tg.edata["efeat"] = F.relu(self.fc_pool(tg.edata["efeat"]))
            tg.update_all(fn.u_add_e("nfeat", "efeat", "m"), fn.max("m", "neigh"))
            h_neigh = tg.ndata["neigh"]
        elif self._agg_type in ["mean", "gcn", "lstm"]:
            tg.update_all(fn.u_add_e("nfeat", "efeat", "m"), fn.sum("m", "neigh"))
            h_neigh = tg.ndata["neigh"]
        else:
            raise KeyError("Aggregator type {} not recognized.".format(
                self._agg_type))

        pg.ndata["neigh"] = h_neigh
        # prop_graph propagation
        if False:
            if self._agg_type == "mean":
                pg.prop_nodes(traversal_order,
                            message_func=fn.copy_src("neigh", "tmp"),
                            reduce_func=fn.sum("tmp", "acc"))
                h_neigh = h_neigh + pg.ndata["acc"]
                h_neigh = h_neigh / degs.unsqueeze(-1)
            elif self._agg_type == "gcn":
                pg.prop_nodes(traversal_order,
                            message_func=fn.copy_src("neigh", "tmp"),
                            reduce_func=fn.sum("tmp", "acc"))
                h_neigh = h_neigh + pg.ndata["acc"]
                h_neigh = (h_self + h_neigh) / (degs.unsqueeze(-1) + 1)
            elif self._agg_type == "pool":
                pg.prop_nodes(traversal_order,
                            message_func=fn.copy_src("neigh", "tmp"),
                            reduce_func=fn.max("tmp", "acc"))
                h_neigh = torch.max(h_neigh, pg.ndata["acc"])
            elif self._agg_type == "lstm":
                h_neighs = [
                    self._lstm_reducer(h_neigh[ids]) for ids in new_node_ids
                ]
                h_neighs = torch.cat(h_neighs, dim=0)
                ridx = torch.arange(h_neighs.shape[0])
                ridx[np.concatenate(new_node_ids)] = torch.arange(
                    h_neighs.shape[0])
                h_neigh = h_neighs[ridx]
        else:
            if self._agg_type == "mean":
                h_neighs = [torch.cumsum(h_neigh[ids], dim=0) for ids in new_node_ids]
                h_neighs = torch.cat(h_neighs, dim=0)
                ridx = torch.arange(h_neighs.shape[0])
                ridx[np.concatenate(new_node_ids)] = torch.arange(
                    h_neighs.shape[0])
                h_neigh = h_neighs[ridx]
                h_neigh = h_neigh / degs.unsqueeze(-1)
            elif self._agg_type == "gcn":
                h_neighs = [torch.cumsum(h_neigh[ids], dim=0) for ids in new_node_ids]
                h_neighs = torch.cat(h_neighs, dim=0)
                ridx = torch.arange(h_neighs.shape[0])
                ridx[np.concatenate(new_node_ids)] = torch.arange(
                    h_neighs.shape[0])
                h_neigh = h_neighs[ridx]
                h_neigh = (h_self + h_neigh) / (degs.unsqueeze(-1) + 1)
            elif self._agg_type == "pool":
                h_neighs = [torch.cummax(h_neigh[ids], dim=0) for ids in new_node_ids]
                h_neighs = torch.cat(h_neighs, dim=0)
                ridx = torch.arange(h_neighs.shape[0])
                ridx[np.concatenate(new_node_ids)] = torch.arange(
                    h_neighs.shape[0])
                h_neigh = h_neighs[ridx]
            elif self._agg_type == "lstm":
                h_neighs = [
                    self._lstm_reducer(h_neigh[ids]) for ids in new_node_ids
                ]
                h_neighs = torch.cat(h_neighs, dim=0)
                ridx = torch.arange(h_neighs.shape[0])
                ridx[np.concatenate(new_node_ids)] = torch.arange(
                    h_neighs.shape[0])
                h_neigh = h_neighs[ridx]

        if self._agg_type == "gcn":
            rst = self.fc_neigh(h_neigh)
        else:
            rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)
        return rst


class GTCEncoder(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 out_feats,
                 edge_feats,
                 n_layers,
                 activation,
                 dropout,
                 agg_type="mean",
                 time_enc="cosine") -> None:
        super(GTCEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            GTCLayer(in_feats, n_hidden, agg_type, edge_feats, time_enc=time_enc))
        for i in range(n_layers - 2):
            self.layers.append(
                GTCLayer(n_hidden, n_hidden, agg_type, edge_feats, time_enc=time_enc))
        if n_layers >= 2:
            self.layers.append(
                GTCLayer(n_hidden, out_feats, agg_type, edge_feats, time_enc=time_enc))
        self.dropout = nn.Dropout(dropout)
        self.act = activation

    def forward(self, agg_graph: dgl.DGLGraph, prop_graph: dgl.DGLGraph,
                new_node_ids: list) -> torch.Tensor:
        tg = agg_graph.local_var()
        pg = prop_graph.local_var()
        torder = dgl.topological_nodes_generator(prop_graph)
        torder = tuple([t.to(pg.device) for t in torder])

        feats = []
        for i, layer in enumerate(self.layers):
            feat = layer(tg, pg, torder, new_node_ids)
            # print("Layer %d" % i)
            # feat = self.act(self.dropout(feat))
            # tg.ndata["nfeat"] = pg.ndata["nfeat"] = feat
            feats.append(self.act(self.dropout(feat)))
            tg.ndata["nfeat"] = pg.ndata["nfeat"] = feats[-1]
        # return feat
        return feats[-1]


class GTCLinkLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 dropout,
                 time_enc,
                 n_classes=1,
                 concat=True) -> None:
        super(GTCLinkLayer, self).__init__()
        self.concat = concat
        self.encode_time = TimeEncodingLayer(in_feats,
                                             in_feats,
                                             time_encoding=time_enc)
        mul = 2 if concat else 1
        self.fc = nn.Linear(in_feats * mul, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_embeds, node_ts, src_idx, dst_idx, t):
        featu, tu = node_embeds[src_idx], node_ts[src_idx]
        featv, tv = node_embeds[dst_idx], node_ts[dst_idx]
        embed_u = self.encode_time(featu, t - tu)
        embed_v = self.encode_time(featv, t - tv)

        if self.concat:
            x = torch.cat([embed_u, embed_v], dim=1)
        else:
            x = embed_u + embed_v
        logits = self.fc(self.dropout(x))
        return logits.squeeze()


class GTCTrainer(nn.Module):
    def __init__(self, nfeat, in_feats, edge_feats, n_hidden, out_feats, args) -> None:
        super(GTCTrainer, self).__init__()
        self.logger = logging.getLogger()
        self.logger.info("GTC initialization.")
        self.nfeat = nfeat
        self.conv = GTCEncoder(in_feats, n_hidden, out_feats, edge_feats,
                                args.n_layers,
                               F.relu, args.dropout, args.agg_type,
                               args.time_encoding)
        self.pred = GTCLinkLayer(out_feats, args.dropout, args.time_encoding)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.n_neg = args.n_neg
        self.n_hist = args.n_hist
        self.no_ce = args.no_ce
        self.pos_contra = args.pos_contra
        self.neg_contra = args.neg_contra
        self.margin = nn.MarginRankingLoss(args.margin)
        self.remain_history = args.remain_history
        self.lam = args.lam
        if args.norm:
            self.norm = nn.LayerNorm(out_feats)
    
    def forward(self,
                agg_graph: dgl.DGLGraph,
                prop_graph: dgl.DGLGraph,
                new_node_ids: list,
                new_node_tss: list,
                batch_samples: tuple,
                neg_dst: np.array,
                contra_samples=None) -> torch.Tensor:
        agg_graph = agg_graph.local_var()
        prop_graph = prop_graph.local_var()

        nfeat = self.conv(agg_graph, prop_graph, new_node_ids)
        node_ts = agg_graph.ndata["timestamp"]
        if hasattr(self, "norm"):
            nfeat = self.norm(nfeat)
        src, dst, t = batch_samples

        src_idx = find_idx(src, t, new_node_ids, new_node_tss)
        dst_idx = find_idx(dst, t, new_node_ids, new_node_tss)
        neg_idx = find_idx(neg_dst, t.repeat(self.n_neg), new_node_ids,
                           new_node_tss)

        t_th = torch.tensor(t).to(nfeat)
        pos_logits = self.pred(nfeat, node_ts, src_idx, dst_idx, t_th)
        neg_logits = self.pred(nfeat, node_ts, src_idx.repeat(self.n_neg),
                               neg_idx, t_th.repeat(self.n_neg))
        if self.no_ce:
            loss = 0.0
        else:
            loss = self.loss_fn(pos_logits, torch.ones_like(pos_logits))
            loss += self.loss_fn(neg_logits, torch.zeros_like(neg_logits))

        if contra_samples is not None:
            hist, hist_t = contra_samples
            if self.remain_history:
                hist_idx = find_idx(hist, hist_t, new_node_ids, new_node_tss)
            else:
                hist_idx = find_idx(hist, t.repeat(self.n_hist), new_node_ids,
                                    new_node_tss)
            cont_logits = self.pred(nfeat, node_ts,
                                    src_idx.repeat(self.n_hist), hist_idx,
                                    t_th.repeat(self.n_hist))
            pos_logits = pos_logits.sigmoid()
            neg_logits = neg_logits.sigmoid()
            cont_logits = cont_logits.sigmoid()

            cont_loss = 0.
            if self.pos_contra:
                cont_loss += self.margin(pos_logits.repeat(self.n_hist),
                                         cont_logits,
                                         torch.ones_like(cont_logits))
            if self.neg_contra:
                times = self.n_neg // self.n_hist
                cont_loss += self.margin(cont_logits.repeat(times), neg_logits,
                                         torch.ones_like(neg_logits))
            loss += self.lam * cont_loss
        return loss

    def infer(self, agg_graph: dgl.DGLGraph, prop_graph: dgl.DGLGraph,
              new_node_ids: list, new_node_tss: list,
              batch_samples: tuple) -> torch.Tensor:
        agg_graph = agg_graph.local_var()
        prop_graph = prop_graph.local_var()

        nfeat = self.conv(agg_graph, prop_graph, new_node_ids)
        node_ts = agg_graph.ndata["timestamp"]
        if hasattr(self, "norm"):
            nfeat = self.norm(nfeat)
        src, dst, t = batch_samples
        src_idx = find_idx(src, t, new_node_ids, new_node_tss)
        dst_idx = find_idx(dst, t, new_node_ids, new_node_tss)

        t_th = torch.tensor(t).to(nfeat)
        logits = self.pred(nfeat, node_ts, src_idx, dst_idx, t_th)
        return logits


class GTCUtility(object):
    @staticmethod
    def split_adj(adj_list: list, ts_list: list):
        """Get the adjacency lists, and return the constructed temporal graph and propagation graph.

        Params:
        ----------
            adj_list : List[np.array(dtype=np.int64)]
                stores the neighbors chronologically
            ts_list : List[np.array(dtype=np.float)]
                stores the timestamps of interactions

        Returns:
        ----------
            new_node_ids : List[dict(tuple, int)]
                maps the (node_id, timestamp) to new_node_id
            old_id_map : List[tuple(int, float)]
                maps the new_node_id to (node_id, timestamp)
            agg_graph : dgl.Graph
                nodes are relabeled with their occurred time points, and edges link the nodes interacted at the same time point
            prop_graph : dgl.Graph
                nodes share the same definition as agg_graph, and edges link the nodes referring to the same node in the origin graph chronologically
        """
        remap_id = 0
        new_node_ids = []
        new_node_tss = []
        new_id_map = []
        old_id_map = []
        for i, neighbors in enumerate(adj_list):
            label_ts = np.sort(np.unique(ts_list[i]))
            new_node_ids.append(remap_id + np.arange(len(label_ts)))
            new_node_tss.append(label_ts)
            new_id_map.append({(i, ts): idx
                               for ts, idx in zip(label_ts, new_node_ids[-1])})
            old_id_map.extend([(i, ts) for ts in label_ts
                               ])  # map new_id to (old_id, timestamp)
            remap_id += len(label_ts)  # set next start_id

        # link nodes referring to the same node chronologically
        u = np.concatenate([node_ids[:-1] for node_ids in new_node_ids])
        v = np.concatenate([node_ids[1:] for node_ids in new_node_ids])
        prop_graph = dgl.graph((u, v), num_nodes=remap_id)

        u = []
        v = []
        for i, (neighbors, ts) in enumerate(zip(adj_list, ts_list)):
            u.append([new_id_map[i][(i, t)] for t in ts])
            v.append(
                [new_id_map[nid][(nid, t)] for nid, t in zip(neighbors, ts)])
        u = np.concatenate(u)
        v = np.concatenate(v)
        # themporal graph remains the same order as ts_list
        agg_graph = dgl.graph((u, v), num_nodes=remap_id)
        return new_node_ids, new_node_tss, old_id_map, agg_graph, prop_graph

    @staticmethod
    def split_edges(edges: pd.DataFrame):
        """Get the edge list, and return the constructed temporal graph and propagation graph.

        Params
        ----------
            edges: pandas.DataFrame
                columns=['from_node_id', 'to_node_id', 'timestamp']

        Returns:
        ----------
            new_node_ids : List[dict(tuple, int)]
                maps the (node_id, timestamp) to new_node_id
            old_id_map : List[tuple(int, float)]
                maps the new_node_id to (node_id, timestamp)
            agg_graph : dgl.Graph
                nodes are relabeled with their occurred time points, and edges link the nodes interacted at the same time point
            prop_graph : dgl.Graph
                nodes share the same definition as agg_graph, and edges link the nodes referring to the same node in the origin graph chronologically
        """
        delta = edges["timestamp"].shift(-1) - edges["timestamp"]
        # Pandas loc[low:high] includes high, so we use slice operations here instead.
        assert np.all(delta[:len(delta) - 1] >= 0)

        num_nodes = len(
            set(edges["from_node_id"]).union(set(edges["to_node_id"])))
        remap_id = 0
        new_node_ids = [list() for _ in range(num_nodes)]
        new_id_map = [dict() for _ in range(num_nodes)]
        old_id_map = []

        u = []
        v = []
        for row in edges.itertuples():
            fid, tid, ts = row.from_node_id, row.to_node_id, row.timestamp
            for idx, nlist in zip([u, v], [fid, tid]):
                if (idx, ts) in new_id_map[idx]:
                    nlist.append(new_id_map[idx][(idx, ts)])
                else:
                    new_id_map[idx][(idx, ts)] = remap_id
                    nlist.append(remap_id)
                    remap_id += 1

        # link nodes referring to the same node chronologically
        u = np.concatenate([node_ids[:-1] for node_ids in new_node_ids])
        v = np.concatenate([node_ids[1:] for node_ids in new_node_ids])
        prop_graph = dgl.graph((u, v), num_nodes=remap_id)

        # temporal graph remains the same order as edges
        agg_graph = dgl.graph((u, v), num_nodes=remap_id)
        return new_node_ids, old_id_map, agg_graph, prop_graph

    @staticmethod
    def split_dglgraph(graph: dgl.DGLGraph):
        ts = graph.edata["timestamp"]
        assert ((ts[1:] - ts[:-1] >= 0).all())
        u, v = graph.edges()
        u, v, ts = u.cpu().numpy(), v.cpu().numpy(), ts.cpu().numpy()
        edges = pd.DataFrame(
            [u, v, ts], columns=["from_node_id", "to_node_id", "timestamp"])

        return GTCUtility.split_edges(edges)


@torch.no_grad()
def eval_linkpred(model: GTCTrainer, val_labels, agg_graph, prop_graph,
                  new_node_ids, adj_ts_l):
    model.eval()
    src = val_labels["from_node_id"].to_numpy()
    dst = val_labels["to_node_id"].to_numpy()
    t = val_labels["timestamp"].to_numpy().astype('float32')
    logits = model.infer(agg_graph, prop_graph, new_node_ids, adj_ts_l,
                         (src, dst, t))
    logits = torch.sigmoid(logits).cpu().numpy()
    labels = val_labels["label"]
    acc = accuracy_score(labels, logits >= 0.5)
    f1 = f1_score(labels, logits >= 0.5)
    auc = roc_auc_score(labels, logits)
    return acc, f1, auc


def main(args, logger):
    set_random_seed()
    logger.info("Set random seeds.")
    logger.info(args)

    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)

    # Set device utility.
    if args.gpu:
        if args.gid >= 0:
            device = torch.device("cuda:{}".format(args.gid))
        else:
            device = torch.device("cuda:{}".format(get_free_gpu()))
        logger.info(
            "Begin Conv on Device %s, GPU Memory %d GB", device,
            torch.cuda.get_device_properties(device).total_memory // 2**30)
    else:
        device = torch.device("cpu")
        logger.info("Begin COnv on Device CPU.")

    # Load nodes, edges, and labeled dataset for training, validation and test.
    nodes, edges, train_labels, val_labels, test_labels = prepare_dataset(
        args.dataset)
    delta = edges["timestamp"].shift(-1) - edges["timestamp"]
    # Pandas loc[low:high] includes high, so we use slice operations here instead.
    assert np.all(delta[:len(delta) - 1] >= 0)

    # Set DGLGraph, node_features, edge_features, and edge timestamps.
    g = construct_dglgraph(edges, nodes, device, node_dim=args.n_hidden)
    nfeat = g.ndata["nfeat"]
    efeat = g.edata["efeat"]
    tfeat = g.edata["timestamp"]
    if nfeat.requires_grad and not args.trainable:
        g.ndata["nfeat"] = torch.zeros_like(g.ndata["nfeat"])

    # Prepare the agg_graph, prop_graph and their required features.
    src = edges["from_node_id"].to_numpy()
    dst = edges["to_node_id"].to_numpy()
    t = edges["timestamp"].to_numpy().astype('float32')
    adj_eid_l, adj_ngh_l, adj_ts_l = construct_adj(src, dst, t, len(nodes))
    new_node_ids, new_node_tss, old_id_map, agg_graph, prop_graph = GTCUtility.split_adj(
        adj_ngh_l, adj_ts_l)
    old_ids = [m[0] for m in old_id_map]
    old_ids = torch.tensor(old_ids, device=device)
    old_tss = [m[1] for m in old_id_map]
    agg_graph = agg_graph.to(device)
    prop_graph = prop_graph.to(device)
    agg_graph.ndata["timestamp"] = torch.tensor(old_tss).to(nfeat)

    old_eids = torch.tensor(np.concatenate(adj_eid_l)).to(device)
    agg_graph.edata["efeat"] = efeat[old_eids].detach()
    agg_graph.edata["timestamp"] = tfeat[old_eids].detach()
    degs = compute_degrees(new_node_ids, len(old_id_map))
    agg_graph.ndata["degree"] = torch.tensor(degs).to(efeat)
    prop_graph.ndata["degree"] = agg_graph.ndata["degree"]

    # Set model configuration.
    in_feats = g.ndata["nfeat"].shape[-1]
    edge_feats = g.edata["efeat"].shape[-1]
    model = GTCTrainer(nfeat, in_feats, edge_feats, args.n_hidden, args.n_hidden, args)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    if nfeat.requires_grad:
        logger.info("Trainable node embeddings.")
    else:
        logger.info("Freeze node embeddings.")
    neg_sampler = NegativeSampler(np.unique(dst), args.n_neg)
    num_batch = np.int(np.ceil(len(train_labels) / args.batch_size))
    epoch_bar = trange(args.epochs, disable=(not args.display))
    early_stopper = EarlyStopMonitor(max_round=5)

    lr = "%.4f" % args.lr
    trainable = "train" if args.trainable else "no-train"
    norm = "norm" if hasattr(model, "norm") else "no-norm"
    pos = "pos" if model.pos_contra else "no-pos"
    neg = "neg" if model.neg_contra else "no-neg"
    lam = '%.1f' % args.lam
    margin = '%.1f' % (args.margin)

    for epoch in epoch_bar:
        model.train()
        optimizer.zero_grad()
        batch_samples = (src, dst, t)
        neg_dst = neg_sampler(len(src))
        if args.pos_contra or args.neg_contra:
            contra_samples = history_sampler(src, t, adj_ngh_l, adj_ts_l,
                                             args.n_hist)
        else:
            contra_samples = None
        start = time.time()
        # These features are not leaf-tensors.
        agg_graph.ndata["nfeat"] = nfeat[old_ids]

        loss = model(agg_graph, prop_graph, new_node_ids, new_node_tss,
                     batch_samples, neg_dst, contra_samples)
        # loss = checkpoint(model, agg_graph, prop_graph, new_node_ids, new_node_tss,
                    #  batch_samples, neg_dst, contra_samples)
        loss.backward()
        optimizer.step()
        print('One epoch {:.2f}s'.format(time.time() - start))

        acc, f1, auc = eval_linkpred(model, val_labels, agg_graph, prop_graph,
                                     new_node_ids, new_node_tss)
        epoch_bar.update()
        # epoch_bar.set_postfix(loss=loss.item(), acc=acc, f1=f1, auc=auc)

        def ckpt_path(epoch):
            return f'./ckpt/{args.dataset}-{args.agg_type}-{trainable}-{norm}-{pos}-{neg}-{lr}-{epoch}-{args.hostname}-{device.type}-{device.index}.pth'

        if early_stopper.early_stop_check(auc):
            logger.info(
                f"No improvement over {early_stopper.max_round} epochs.")
            logger.info(
                f'Loading the best model at epoch {early_stopper.best_epoch}')
            model.load_state_dict(
                torch.load(ckpt_path(early_stopper.best_epoch)))
            logger.info(
                f'Loaded the best model at epoch {early_stopper.best_epoch} for inference'
            )
            break
        else:
            torch.save(model.state_dict(), ckpt_path(epoch))
    model.eval()
    # _, _, val_auc = eval_linkpred(model, g, val_labels)
    _, _, val_auc = eval_linkpred(model, val_labels, agg_graph, prop_graph,
                                  new_node_ids, new_node_tss)
    # acc, f1, auc = eval_linkpred(model, g, test_labels)
    acc, f1, auc = eval_linkpred(model, test_labels, agg_graph, prop_graph,
                                 new_node_ids, new_node_tss)
    params = {
        "best_epoch": early_stopper.best_epoch,
        "trainable": args.trainable,
        "lr": "%.4f" % (args.lr),
        "agg_type": args.agg_type,
        "no-ce": args.no_ce,
        "norm": norm,
        "pos_contra": args.pos_contra,
        "neg_contra": args.neg_contra,
        "n_hist": args.n_hist,
        "n_neg": args.n_neg,
        "n_layers": args.n_layers,
        "time_encoding": args.time_encoding,
        "lambda": args.lam,
        "margin": args.margin
    }
    write_result(val_auc, (acc, f1, auc), args.dataset, params)
    MODEL_SAVE_PATH = f'./saved_models/{args.dataset}-{args.agg_type}-{lr}-{lam}-{margin}.pth'
    model = model.cpu()
    torch.save(model.state_dict(), MODEL_SAVE_PATH)


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    logger = set_logger()
    main(args, logger)
