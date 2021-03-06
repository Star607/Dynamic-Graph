import dgl
import time
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
import torch
from collections import namedtuple

from data_loader.minibatch import TemporalEdgeBatchIterator


class SparseEdgeBatchIterator(TemporalEdgeBatchIterator):
    def __init__(self, edges, nodes, val_ratio, test_ratio, neg_sample_size, remove_unseen_nodes=True, **kwargs):
        self.eps = 1e-7
        # padding zero node and a self-loop edge
        edges, nodes = self._reindex_node(edges, nodes)
        self.id2idx = {row["node_id"]: row["id_map"] for i, row in nodes.iterrows()}
        self.graph = dgl.DGLGraph((edges["from_node_id"], edges["to_node_id"]))
        self.graph.edata["timestamp"] = edges["timestamp"]
        if remove_unseen_nodes:
            edges, train_nums, val_nums = self._remove_unseen_nodes(edges, val_ratio, test_ratio)
        else:
            train_nums = int(len(edges) * 1 - val_ratio - test_ratio)
            val_nums = int(len(edges) * val_ratio)
        # overwrite adjacency list by csr_matrix
        self.adj_ids, self.adj_tss, self.degrees = self.construct_adj(
            self.edges)

    def _reindex_node(self, edges, nodes):
        nodeset = set(edges["from_node_id"]).union(set(edges["to_node_id"]))
        assert len(set(nodes["node_id"])) == len(nodeset) and np.all(
            [n in nodeset for n in nodes])  # check node existence
        ids = set(nodes["id_map"])
        assert len(ids) == len(nodes)  # check non-repeated node ids
        if max(ids) - min(ids) + 1 > len(ids):  # ids not consecutive
            ids = sorted(ids)
            id_remap = {imap: idx for idx, imap in enumerate(ids)}
            nodes["id_map"] = nodes["id_map"].map(id_remap)
        ids = set(nodes["id_map"])
        assert max(ids) - min(ids) + 1 == len(ids)  # check id consecutive
        return edges, nodes

    def _remove_unseen_nodes(self, edges, val_ratio, test_ratio):
        pass

    def construct_adj(self, edges):
        r"""Construct truncated adjacency list and each entry is sorted in temporal order except padding zeros.

        Args:
        --------
        edges : temporal ascending interactions between ndoes with timestamps.

        Return:
        --------
        adj_ids : csr_matrix of adjacency list, each row stores the temporal neighbors of each node_id, with neighbours sorted by timestamp ascendingly.
        adj_tss : csr_matrix of adjacency list timestamps.
        """
        eid2aid = np.array(
            (len(edges), 2))  # change edge index to adjacency list index
        adj_ids_list = [[] for _ in range(len(self.id2idx))]
        adj_tss_list = [[] for _ in range(len(self.id2idx))]
        aid2eid = [[] for _ in range(len(self.id2idx))]
        # print(len(self.id2idx), len(self.id2idx))
        # Attention! df.iterrows will change dtype for each column!
        for row in edges.itertuples():
            from_id, to_id, ts = row.from_node_id, row.to_node_id, row.timestamp
            if from_id >= len(self.id2idx) or to_id >= len(self.id2idx):
                print(row)
            adj_ids_list[from_id].append(to_id)
            adj_tss_list[from_id].append(ts)
            adj_ids_list[to_id].append(from_id)
            adj_tss_list[to_id].append(ts)

            eid2aid[row.Index] = [
                len(adj_ids_list[from_id]) - 1, len(adj_ids_list[to_id]) - 1]
            aid2eid[from_id].append(row.Index)
            aid2eid[to_id].append(row.Index)

        adj_lens = np.array([len(l) for l in adj_ids_list])
        indptr = np.cumsum(adj_lens)
        indices = np.concatenate([np.arange(l) for l in adj_lens])
        adj_ids = csr_matrix(
            (np.concatenate(adj_ids_list), indices, indptr), type=np.int64)
        adj_tss = csr_matrix(
            (np.concatenate(adj_tss_list), indices, indptr), type=np.float64)

        self.eid2aid = pd.DataFrame(eid2aid, columns=["u", "v"])
        self.aid2eid = csr_matrix(np.concatenate(
            aid2eid), indices, indptr, type=np.int64)
        return adj_ids, adj_tss, adj_lens

    def aids2eids(self, aids):
        r"""
        Args:
        --------
        aids: (N, 2), each entry is a tuple of uid and the corresponding adjacency index of uid

        Return:
        --------
        eids: (N), the corresponding edge indices
        """
        uids = aids[:, 0]
        cols = aids[:, 1]
        return np.squeeze(np.array(self.aid2eid[uids, cols]))

    def eids2aids(self, eids):
        r"""In adjacency list, we store the undirected edge. However, the original edge has exact from_node_id and to_node_id. So we return the corresponding adjacency list indices with respect to (from_node_id, to_node_id) as a (N,2) dataframe, with columns of `u,v`.
        """
        return self.eid2aid.loc[eids]
