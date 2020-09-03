import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class TimeEncodingLayer(nn.Module):
    """Given `(E_u, t)`, output `f2(act(f1(E_u, Encode(t))))`.
    """

    def __init__(self, in_features, out_features, time_encoding="concat"):
        super(TimeEncodingLayer, self).__init__()
        self.time_encoding = time_encoding
        if time_encoding == "concat":
            self.fc1 = nn.Linear(in_features + 1, out_features)
        elif time_encoding == "cosine":
            self.basis_freq = nn.Parameter(
                0.1 * torch.linspace(0, 9, in_features))
            self.phase = nn.Parameter(torch.zeros(in_features))
            self.fc1 = nn.Linear(in_features * 2, out_features)
        elif time_encoding == "outer":
            hidden_dim = np.int(np.sqrt(in_features))
            self.basis_freq = nn.Parameter(
                0.1 * torch.linspace(0, 9, hidden_dim))
            self.phase = nn.Parameter(torch.zeros(hidden_dim))
            self.trans = nn.Linear(in_features, hidden_dim)
            self.fc1 = nn.Linear(np.square(hidden_dim), out_features)
            nn.init.xavier_normal_(self.trans.weight)
        else:
            raise NotImplementedError
        self.act = nn.ReLU()

        nn.init.xavier_normal_(self.fc1.weight)

    def forward(self, u, t):
        if self.time_encoding == "concat":
            x = self.fc1(torch.cat([u, t.view(-1, 1)], dim=1))
        elif self.time_encoding == "cosine":
            t = torch.cos(t.view(-1, 1) * self.basis_freq.view(1, -1) +
                          self.phase.view(1, -1))
            x = self.fc1(torch.cat([u, t], dim=-1))
        elif self.time_encoding == "outer":
            t = torch.cos(t.view(-1, 1) * self.basis_freq.view(1, -1) +
                          self.phase.view(1, -1))
            u = self.trans(u)
            assert(u.shape[0] == t.shape[0])
            batch_size = u.shape[0]
            x = torch.bmm(u.view(batch_size, -1, 1),
                          t.view(batch_size, 1, -1)).view(batch_size, -1)
            x = self.fc1(x)
        else:
            raise NotImplementedError

        return self.act(x)


class TemporalLinkLayer(nn.Module):
    """Given a list of `(u, v, t)` tuples, predicting the edge probability between `u` and `v` at time `t`. Firstly, we find the latest `E(u, t_u)` and `E(v, t_v)` before the time `t`. Then we compute `E(u, t)` and `E(v, t)` using an outer product temporal encoding layer for `E(u, t_u)` and `E(v, t_v)` respectively. Finally, we concatenate the embeddings and output probability logits via a two layer MLP like `TGAT`.
    """

    def __init__(self, in_features=128, out_features=1, concat=True, time_encoding="concat", dropout=0.2):
        super(TemporalLinkLayer, self).__init__()
        self.concat = concat
        self.time_encoding = time_encoding
        mul = 2 if concat else 1
        self.encode_time = TimeEncodingLayer(
            in_features, in_features, time_encoding=time_encoding)
        self.fc = nn.Linear(in_features * mul, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, src_eids, dst_eids, t):
        """For each `(u, v, t)`, we get embedding_u by
        `g.edata['src_feat'][src_eids]`, get embedding_v by
        `g.edata['dst_feat'][dst_eids]`.

        Finally, output `g(e_u, e_v, t)`.
        """
        featu = g.edata["src_feat"][src_eids]
        tu = g.edata["timestamp"][src_eids]
        featv = g.edata["dst_feat"][dst_eids]
        tv = g.edata["timestamp"][dst_eids]
        embed_u = self.encode_time(featu, t-tu)
        embed_v = self.encode_time(featv, t-tv)

        if self.concat:
            x = torch.cat([embed_u, embed_v], dim=1)
        else:
            x = embed_u + embed_v
        return self.fc(self.dropout(x))


class TemporalNodeLayer(nn.Module):

    def __init__(self, in_features=128, out_features=1, time_encoding="concat", dropout=0.2):
        super(TemporalNodeLayer, self).__init__()
        self.time_encoding = time_encoding
        self.encode_time = TimeEncodingLayer(
            in_features, in_features, time_encoding=time_encoding)
        self.fc = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, eids, t, mode="src"):
        x = g.edata[f"{mode}_feat"][eids]
        tx = g.edata["timestamp"][eids]
        x = self.encode_time(x, t - tx)
        return self.fc(self.dropout(x))


class TSAGEConv(nn.Module):
    r"""Temporally GraphSAGE layer means aggregation only performing over the valid temporal neighbors. And each edge get distinct embeddings for source node and destionation node. Finally, we return the edge emebddings for all nodes at different timestamps, whose space cost is O(2*E*dim).

    All params remain the same as ``SAGEConv`` in ``dgl.nn.pytorch.conv.sagecong.py``.

    Parameters
    ----------
    in_feats : int, or pair of ints
    out_feats : int
    feat_drop : float
    aggregator_type : str
        Aggregator type to use (``mean``, ``gcn``, ``pool``, ``lstm``).
    """

    def __init__(self, in_feats, out_feats, aggregator_type, time_encoding="cosine"):
        super(TSAGEConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = in_feats, in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.encode_time = TimeEncodingLayer(
            in_feats, in_feats, time_encoding=time_encoding)
        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats,
                                self._in_src_feats, batch_first=True)
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
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

    def group_func_wrapper(self, groupby, src_feat, dst_feat):
        """Set the group by function. The `onehop_conv` performs different aggregations over all valid temporal neighbors. The final embeddings are stored in the edges, named `src_feat` or `dst_feat`.

        Parameters:
        ------------
        deg_indices: Pre-computed index matrices for batch nodes. It is stored as a dictionary, with degree as keys, and the index matrix as values. It differs in source groupby and destination groupby modes.
        """

        if groupby not in ["src", "dst"]:
            raise NotImplementedError

        def onehop_conv(edges):
            # The first layer is a combination of ndoe features and edge features.
            h_self, h_neighs = edges.data[src_feat], edges.data[dst_feat]
            deg_self, deg_neighs = edges.src["deg"], edges.dst["deg"]
            if groupby == "dst":
                h_self, h_neighs = h_neighs, h_self
                deg_self, deg_neighs = deg_neighs, deg_self
            assert h_self.shape == h_neighs.shape

            # print("bucket shape", h_self.shape)
            buc, deg, dim = h_self.shape
            # Attention! There are edges with the same timestamp. The lower triangular assumption is not hold, and we comment the following codes.
            # assert the timestamp is increasing
            # orders = torch.argsort(edges.data["timestamp"], dim=1)
            # assert torch.all(torch.eq(torch.arange(deg).to(orders), orders))
            # mask = torch.tril(torch.ones(deg, deg)).to(h_neighs)
            # sum over all valid neighbors: (bucket_size, deg, dim)
            # mask_feat = torch.matmul(mask, h_neighs) / mask.sum(dim=-1, keepdim=True)
            # (B, Deg, 1) if the last dimension is 1
            ts = edges.data["timestamp"].view(buc, deg, 1)
            # The following mask matrix would crush out of CUDA memory. For the
            # 58k degree node, it consumes 12GB memory.
            # mask = (ts.permute(0, 2, 1) <= ts).float()  # (B, Deg, Deg)
            # We assume the batch mechanism keeps stable during training.
            # (bucket, deg, dim)
            indices = edges.data[f"{groupby}_deg_indices"].expand(-1, -1, dim)

            if self._aggre_type == "mean":
                # mask_feat = torch.bmm(mask, h_neighs)
                # mask_feat = mask_feat / mask.sum(dim=-1, keepdim=True)
                # mean_cof = torch.arange(deg).add_(1.0).unsqueeze_(-1)
                mean_cof = edges.data[f"{groupby}_deg_indices"].add(
                    1.0).view(buc, deg, 1)
                h_feat = h_neighs.cumsum(dim=1) / mean_cof
                mask_feat = h_feat.gather(dim=1, index=indices)
            elif self._aggre_type == "gcn":
                # mask_feat = torch.bmm(mask, h_neighs)
                h_feat = h_neighs.cumsum(dim=1)
                mask_feat = h_feat.gather(dim=1, index=indices)
                # norm_cof = deg_self.to(mask_feat) + 1
                norm_cof = edges.data[f"{groupby}_deg_indices"].add(
                    1.0).view(buc, deg)
                mask_feat = (mask_feat + h_self) / norm_cof.unsqueeze(-1)
            elif self._aggre_type == "pool":
                # mask_feat = torch.bmm(mask, h_neighs)
                # mask_feat = mask_feat / mask.sum(dim=-1, keepdim=True)
                # Since we get (upper_bound - 1) indices, we can use cummax() + gather() to perform max_pooling operation.
                h_neighs = F.relu(self.fc_pool(h_neighs))
                h_feat = h_neighs.cummax(dim=1).values
                mask_feat = h_feat.gather(dim=1, index=indices)
            elif self._aggre_type == 'lstm':
                raise NotImplementedError
            else:
                raise NotImplementedError

            if self._aggre_type == "gcn":
                rst = self.fc_neigh(mask_feat)
            else:
                rst = self.fc_self(h_self) + self.fc_neigh(mask_feat)

            return {f'{groupby}_feat': rst}
        return onehop_conv

    def forward(self, graph, current_layer=1):
        r"""We utilize ``dgl.DGLGraph.group_apply_edges`` to compute TGraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.

        current_layer : int, default as `1`.
            As we compute embeddings for each node at its each edge, the count of total embeddings are ``4 * E`` (E is the number of edges), comprise of source node embeddings and destination embeddings.
            In the 1st layer, we accesses the previous layer embeddings via node features, whose shape is also ``O(E * dim)``.
            In the next layers, we accesses the previous layer embeddings via ``EdgeBatch.data["src_feat%d"%(current_layer-1)]``.

        Returns
        ----------
        src_feat : Tensor
        dst_feat : Tensor
        """
        g = graph.local_var()

        src_name = f'src_feat{current_layer - 1}'
        dst_name = f'dst_feat{current_layer - 1}'
        # add dropout layer for node embeddings
        src_feat, dst_feat = g.edata[src_name], g.edata[dst_name]
        g.edata[src_name] = self.encode_time(src_feat, g.edata["timestamp"])
        g.edata[dst_name] = self.encode_time(dst_feat, g.edata["timestamp"])

        src_conv = self.group_func_wrapper(
            groupby="src",  src_feat=src_name, dst_feat=dst_name)
        dst_conv = self.group_func_wrapper(
            groupby="dst",  src_feat=src_name, dst_feat=dst_name)
        g.group_apply_edges(group_by="src", func=src_conv)
        g.group_apply_edges(group_by="dst", func=dst_conv)

        return g.edata["src_feat"], g.edata["dst_feat"]


class GTAConv(TSAGEConv):
    pass


class TGATConv(nn.Module):
    pass


class GTRConv(nn.Module):
    pass
