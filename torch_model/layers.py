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


class TemporalLinkLayer(nn.Module):
    """Given a list of `(u, v, t)` tuples, predicting the edge probability between `u` and `v` at time `t`. Firstly, we find the latest `E(u, t_u)` and `E(v, t_v)` before the time `t`. Then we compute `E(u, t)` and `E(v, t)` using an outer product temporal encoding layer for `E(u, t_u)` and `E(v, t_v)` respectively. Finally, we concatenate the embeddings and output probability logits via a two layer MLP like `TGAT`.
    """

    def __init__(self, in_features=128, out_features=1, concat=True, time_encoding="concat"):
        super(TemporalLinkLayer, self).__init__()
        self.concat = concat
        self.time_encoding = time_encoding
        mul = 2 if concat else 1
        if time_encoding == "concat":
            self.fc1 = nn.Linear((in_features + 1) * mul, in_features)
        elif time_encoding == "cosine":
            self.basis_freq = nn.Parameter(0.1 * torch.linspace(0, 9, in_features))
            self.phase = nn.Parameter(torch.zeros(in_features))
            self.fc1 = nn.Linear((in_features * 2) * mul, in_features)
        elif time_encoding == "outer":
            hidden_dim = np.int(np.sqrt(in_features))
            self.basis_freq = nn.Parameter(0.1 * torch.linspace(0, 9, in_features))
            self.phase = nn.Parameter(torch.zeros(hidden_dim))
            self.trans = nn.Linear(in_features, hidden_dim)
            self.fc1 = nn.Linear(np.square(hidden_dim) * mul, in_features)
            nn.init.xavier_normal_(self.trans)
        else:
            raise NotImplementedError
        self.fc2 = nn.Linear(in_features, out_features)
        self.act = nn.ReLU()

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, g, edges, bidirected=False):
        """If bidirected, we can access the temporal node embeddings from either `src_feat` or `dst_feat`. Otherwise, a node will have two identical embeddings stored as both `src_feat` and `dst_feat`.
        """
        # Firstly, finding the corresponding eids of (u, t_u) and (v, t_v) for edges (u, v, t).
        u, v, t = edges[0], edges[1], edges[2]
        if bidirected:
            u_eids = LatestNodeInteractionFinder(g, u, t, mode="in")
            v_eids = LatestNodeInteractionFinder(g, v, t, mode="out")
        else:
            u_eids = LatestNodeInteractionFinder(g, u, t, mode="in")
            v_eids = LatestNodeInteractionFinder(g, v, t, mode="out")
        u_embeds = g.edges[u_eids].data["src_feat"]
        u_t = g.edges[u_eids].data["timestamp"]
        v_embeds = g.edges[v_eids].data["dst_feat"]
        v_t = g.edges[v_eids].data["timestamp"]
        if self.time_encoding == "concat":
            u_embeds = torch.cat([u_embeds, u_t.unsqueeze(-1)], dim=1)
            v_embeds = torch.cat([v_embeds, v_t.vnsqveeze(-1)], dim=1)
        elif self.time_encoding == "cosine":
            u_t = u_t.unsqueeze(-1) * self.basis_freq.view(1, -1) + self.phase.view(1, -1)
            u_embeds = torch.cat([u_embeds, torch.cos(u_t)], dim=1)
            v_t = v_t.unsqueeze(-1) * self.basis_freq.view(1, -1) + self.phase.view(1, -1)
            v_embeds = torch.cat([v_embeds, torch.cos(v_t)], dim=1)
        elif self.time_encoding == "outer":
            u_embeds, v_embeds = self.trans(u_embeds), self.trans(v_embeds)
            u_t = u_t.unsqueeze(-1) * self.basis_freq.view(1, -1) + self.phase.view(1, -1)
            u_embeds = torch.bmm(u_embeds.unsqueeze(2), torch.cos(
                u_t).unsqueeze(1)).view(u.shape[0], -1)
            v_t = v_t.unsqueeze(-1) * self.basis_freq.view(1, -1) + self.phase.view(1, -1)
            v_embeds = torch.bmm(v_embeds.unsqueeze(2), torch.cos(
                v_t).unsqueeze(1)).view(v.shape[0], -1)
        if self.concat:
            x = torch.cat([u_embeds, v_embeds], dim=1)
            h = self.act(self.fc1(x))
        else:
            x = u_embeds + v_embeds
            h = self.act(self.fc1(x))
        return self.fc2(h)


class TemporalNodeLayer(nn.Module):

    def __init__(self, in_features=128, out_features=1, time_encoding="concat"):
        super(TemporalNodeLayer, self).__init__()

    def forward(self, g, nodes, bidirected=False):
        pass
