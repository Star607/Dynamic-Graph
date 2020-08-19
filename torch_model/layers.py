import argparse
import itertools
import logging
import os
import sys
import time
from datetime import datetime

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as ssp
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import trange

import upper_bound_cpp
from torch_model.util_dgl import timeit
from torch_model.eid_precomputation import LatestNodeInteractionFinder


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
            nn.init.xavier_normal_(self.trans)
        else:
            raise NotImplementedError
        self.act = nn.ReLU()

        nn.init.xavier_normal_(self.fc1.weight)

    def forward(self, u, t):
        if self.time_encoding == "concat":
            x = self.fc1(torch.cat([u, t], dim=1))
        elif self.time_encoding == "cosine":
            t = torch.cos(t * self.basis_freq.view(1, -1) +
                          self.phase.view(1, -1))
            x = self.fc1(torch.cat([u, t], dim=-1))
        elif self.time_encoding == "outer":
            t = torch.cos(t * self.basis_freq.view(1, -1) +
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
        tv = g.edata["dst_feat"][dst_eids]
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

    def forward(self, g, eids, mode="src"):
        x = g.edata[f"{mode}_feat"][eids]
        x = self.encode_time(x)
        return self.fc(self.dropout(x))
