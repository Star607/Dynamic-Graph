import argparse
from datetime import datetime
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", dest="gpu", action="store_true",
                        help="Whether use GPU.")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false",
                        help="Whether use GPU.")
    return parser.parse_args(["--gpu"])


args = parse_args()


def set_logger():
    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # fh = logging.FileHandler('log/dgl-{}.log'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    # fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    # fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = set_logger()
logger.info(args)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print("%r  %2.2f s" % (method.__name__, te - ts))
        return result
    return timed


class TSAGEConv(SAGEConv):
    r"""Temporally GraphSAGE layer means aggregation only performing over the valid temporal neighbors. And each edge get distinct embeddings for source node and destionation node. Finally, we return the edge emebddings for all nodes at different timestamps, whose space cost is O(2*E*dim).

    All params remain the same as ``SAGEConv`` in ``dgl.nn.pytorch.conv.sagecong.py``.

    Parameters
    ----------
    in_feats : int, or pair of ints
    out_feats : int
    feat_drop : float
    aggregator_type : str
        Aggregator type to use (``mean``, ``gcn``, ``pool``, ``lstm``).
    bias : bool
    norm : callable activation function/layer or None, optional
    activation : callable activation function/layer or None, optional
    """

    def __init__(self, **kwargs):
        super(TSAGEConv, self).__init__(**kwargs)

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

    def group_func_wrapper(self, groupby, src_feat, dst_feat):
        """Set the group by function. The `onehop_conv` performs different aggregations over all valid temporal neighbors. The final embeddings are stored in the edges, named `src_feat` or `dst_feat`.
        """
        def onehop_conv(edges):
            if src_feat.startswith("src_feat"):
                h_self = edges.data[src_feat]
                h_neighs = edges.data[dst_feat]
            else:
                h_self = edges.src[src_feat]
                h_neighs = edges.dst[dst_feat]
            if groupby not in ["src", "dst"]:
                raise NotImplementedError
            deg_self, deg_neighs = edges.src["deg"], edges.dst["deg"]
            if groupby == "dst":
                h_self, h_neighs = h_neighs, h_self
                deg_self, deg_neighs = deg_neighs, deg_self
            assert h_self.shape == h_neighs.shape
            _, deg, dim = h_self.shape
            # assert the timestamp is increasing
            orders = torch.argsort(edges.data["timestamp"], dim=1)
            assert torch.all(torch.eq(torch.arange(deg).to(orders), orders))
            mask = torch.tril(torch.ones(deg, deg)).to(h_neighs)
            # sum over all valid neighbors: (bucket_size, deg, dim)
            # mask_feat = torch.matmul(mask, h_neighs)
            if self._aggre_type == "mean":
                mask_feat = torch.matmul(mask, h_neighs)
                mask_feat = mask_feat / mask.sum(dim=1, keepdim=True)
            elif self._aggre_type == "gcn":
                mask_feat = torch.matmul(mask, h_neighs)
                mask_feat = mask_feat + h_self
                norm_cof = (deg_neighs + 1.0).sqrt() * (deg_self + 1.0).sqrt()
                mask_feat = mask_feat / norm_cof.unsqueeze(-1)
            elif self._aggre_type == "pool":
                # If performing max_pooling, we compute maximum till i-th row using ``cummax()``.
                h_neighs = F.relu(self.fc_pool(h_neighs))
                mask_feat = h_neighs.cummax(dim=1).values
            elif self._aggre_type == 'lstm':
                raise NotImplementedError
            else:
                raise NotImplementedError

            if self._aggre_type == "gcn":
                rst = self.fc_neigh(mask_feat)
            else:
                rst = self.fc_self(h_self) + self.fc_neigh(mask_feat)

            if self.activation is not None:
                rst = self.activation(rst)
            if self.norm is not None:
                rst = self.norm(rst)
            return {f'{groupby}_feat': rst}
        return onehop_conv

    def forward(self, graph, current_layer=1):
        r"""We utilize ``dgl.DGLGraph.group_apply_edges`` to compute TGraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.

        current_layer : int, default as `1`.
            As we compute embeddings for each node at its each edge, the count of total embeddings are ``2 * E`` (E is the number of edges), comprise of source node embeddings and destination embeddings. 
            In the 1st layer, we accesses the previous layer embeddings via node features, whose shape is also ``O(E * dim)``.
            In the next layers, we accesses the previous layer embeddings via ``EdgeBatch.data["src_feat%d"%(current_layer-1)]``.

        src_feat_name : str
        dst_feat_name : str

        Returns
        ----------
        src_feat : Tensor
        dst_feat : Tensor
        """
        graph = graph.local_var()
        # compute for gcn normalization
        graph.ndata["deg"] = graph.in_degrees() + graph.out_degrees()
        if current_layer >= 2:
            src_feat = f'src_feat{current_layer - 1}'
            dst_feat = f'dst_feat{current_layer - 1}'
        else:
            src_feat = dst_feat = "nfeat"

        src_conv = self.group_func_wrapper(groupby="src", src_feat=src_feat, dst_feat=dst_feat)
        dst_conv = self.group_func_wrapper(groupby="dst", src_feat=src_feat, dst_feat=dst_feat)
        graph.group_apply_edges(group_by="src", func=src_conv)
        graph.group_apply_edges(group_by="dst", func=dst_conv)
        return graph.edata["src_feat"], graph.edata["dst_feat"]


class GTAConv(TSAGEConv):
    pass


class TGATConv(nn.Module):
    pass


class GTRConv(nn.Module):
    pass


def construct_dglgraph(edges, nodes, device, bidirected=False):
    ''' Edges should be a pandas DataFrame, and its columns should be columns comprise of  from_node_id, to_node_id, timestamp, state_label, features_separated_by_comma. Here `state_label` varies in edge classification tasks.

    Nodes should be a pandas DataFrame, and its columns should be columns comprise of node_id, id_map, role, label, features_separated_by_comma.

    By default, we use the single directional edges to store the bi-directional edge messages for memory reduction. If `bidirected` is set `True`, we add the inverse edges into the DGLGraph. In this case, we retain edges in the increasing temporal order.
    '''
    src = edges["from_node_id"]
    dst = edges["to_node_id"]
    etime = torch.tensor(edges["timestamp"], device=device)
    efeature = torch.tensor(edges.iloc[:, 4:].to_numpy(), device=device) if len(
        edges.columns) > 4 else torch.ones((len(edges), 1), device=device)

    def xavier_normal(
        *args): return nn.init.xavier_normal(torch.empty(*args, device=device))
    nfeature = torch.tensor(nodes.iloc[:, 4:].to_numpy(), device=device) if len(
        nodes.columns) > 4 else nn.Parameter(xavier_normal(len(nodes), 4))
    if bidirected:
        # In this way, we repeat the edge one by one, remaining the increasing temporal order.
        u = np.vstack((src, dst)).transpose().flatten()
        v = np.vstack((dst, src)).transpose().flatten()
        src, dst = u, v
        etime = etime.repeat_interleave(2)
        efeature = efeature.repeat_interleave(2, dim=0)
    # Adding edges in the time increasing order, so that group_apply_edges will process the neighbors temporally ascendingly.
    # We only add single directed edges, but treat them as undirected edges for representation. That is we store both source and destionation node representations at timestamp t on the same edge (s, d, t), assuming
    g = dgl.DGLGraph((src, dst))
    g.ndata["nfeat"] = nfeature  # .to(device)
    g.edata["timestamp"] = etime  # .to(device)
    g.edata["efeat"] = efeature  # .to(device)
    return g


def test_graph(name=None):
    """Load data using ``data_loader.minibatch.load_data()``. If name is not given, we return a sample graph with 10 nodes and 45 edges, which is a complete graph.
    """
    if name is None:
        nodes = pd.DataFrame(columns=["node_id", "id_map", "role", "label"])
        nodes["node_id"] = np.arange(10)
        nodes["id_map"] = np.arange(10)
        nodes["role"] = 0
        nodes["label"] = 0
        edges = pd.DataFrame(
            columns=["from_node_id", "to_node_id", "timestamp", "state_label"])
        edges["from_node_id"] = np.concatenate([
            [i for _ in range(9 - i)] for i in range(10)])
        edges["to_node_id"] = np.concatenate([
            [j for j in range(i + 1, 10)] for i in range(10)])
        edges["timestamp"] = np.arange(45, dtype=np.float)
        edges["state_label"] = 0
        dtypes = edges.dtypes
        dtypes[["from_node_id", "to_node_id"]] = int
        edges = edges.astype(dtypes)
        return edges, nodes
    else:
        edges, nodes = load_data(dataset=name)
        # add node 0
        nodes.loc[len(nodes)] = [0] * len(nodes)
        # add self-loop edge (0, 0)
        edges.loc[len(edges)] = [0] * len(edges.columns)
        return edges, nodes


def main():
    if args.gpu:
        device = torch.device("cuda:{}".format(get_free_gpu()))
    else:
        device = torch.device("cpu")

    edges, nodes = test_graph()
    g = construct_dglgraph(edges, nodes, device)

    logger.info(f"Begin Conv. Device {device}")
    dim = g.ndata["nfeat"].shape[-1]
    dims = [dim, 108, 4]
    # for l in range(1, 3):
    #     logger.info(f"Graph Conv Layer {l}.")
    #     model = TSAGEConv(in_feats=dims[l-1], out_feats=dims[l], aggregator_type="mean")
    #     model = model.to(device)
    #     src_feat, dst_feat = model(g, current_layer=l)
    #     g.edata[f"src_feat{l}"] = src_feat
    #     g.edata[f"dst_feat{l}"] = dst_feat
    model = TSAGEConv(in_feats=dims[0], out_feats=dims[1], aggregator_type="mean")
    model = model.to(device)
    import copy
    nfeat_copy = copy.deepcopy(g.ndata["nfeat"])
    loss_fn = nn.CosineEmbeddingLoss(margin=0.5)
    import itertools
    optimizer = torch.optim.Adam(itertools.chain([g.ndata["nfeat"]], model.parameters()), lr=0.01)
    # print(nfeat_copy)
    for i in range(10):
        logger.info("Epoch %3d", i)
        optimizer.zero_grad()
        src_feat, dst_feat = model(g, current_layer=1)
        labels = torch.ones((g.number_of_edges()), device=device)
        loss = loss_fn(src_feat, dst_feat, labels)
        loss.backward()
        optimizer.step()
        print("nfeat")
        print(g.ndata["nfeat"].storage().data_ptr())
        print("nfeat copy")
        print(nfeat_copy.storage().data_ptr())
        assert not torch.all(torch.eq(nfeat_copy, g.ndata["nfeat"]))
    print(src_feat.shape, dst_feat.shape)
    # z = src_feat.sum()
    # z.backward()
    print(g.ndata["nfeat"].grad)
    return src_feat, dst_feat


def nodeflow_test():
    pass


def fullgraph_test():
    pass


def subgraph_test():
    pass


if __name__ == "__main__":
    main()
