import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tgat.module import MergeLayer, AttnModel, LSTMPool, MeanPool, \
    MultiHeadAttention, \
    TimeEncode, PosEncode, EmptyEncode, TGAN


class SoftmaxAttention(nn.Module):
    def __init__(self, feat_dim: int, samplers: int) -> None:
        super(SoftmaxAttention, self).__init__()
        self.query = torch.nn.Linear(feat_dim, 1, bias=False)
        self.linears = torch.nn.ModuleList(
            [nn.Linear(feat_dim, feat_dim) for _ in range(samplers)])

    def forward(self, embeds: list) -> torch.Tensor:
        k = len(embeds)
        x = [torch.tanh(self.linears[i](embeds[i])) for i in range(k)]
        x = [self.query(x[i]) for i in range(k)]  # (k, n, 1)
        weights = torch.softmax(torch.cat(x, dim=1), dim=1)  # (n, k)
        embeds = torch.cat([i.unsqueeze(dim=1) for i in embeds],
                           dim=1)  # (n, k, d)
        ans = weights.unsqueeze(-1) * embeds
        return torch.sum(ans, dim=1)  # (n, d)


class SamplingFusion(TGAN):
    def __init__(self, k_samplers: int, *args, **kwargs) -> None:
        super(SamplingFusion, self).__init__(*args, **kwargs)
        feat_dim = self.feat_dim
        num_layers = kwargs['num_layers']

        self.fusion_layer_list = torch.nn.ModuleList([
            SoftmaxAttention(feat_dim, k_samplers) for _ in range(num_layers)
        ])
        self.k_samplers = k_samplers

    def forward(self, src_idx_l, target_idx_l, num_neighbors=20):
        src_embed = self.tem_conv(src_idx_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, self.num_layers, num_neighbors)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        return score

    def contrast(self, src_idx_l, target_idx_l, background_idx_l,
                 num_neighbors):
        src_embed = self.tem_conv(src_idx_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, self.num_layers,
                                     num_neighbors)
        background_embed = self.tem_conv(background_idx_l, self.num_layers,
                                         num_neighbors)
        pos_score = self.affinity_score(src_embed,
                                        target_embed).squeeze(dim=-1)
        neg_score = self.affinity_score(src_embed,
                                        background_embed).squeeze(dim=-1)
        return pos_score.sigmoid(), neg_score.sigmoid()

    def tem_conv(self,
                 src_list,
                 curr_layers,
                 num_neighbors=20) -> torch.Tensor:
        """Here we precomputed the k-hop neighbors instead of computing during attention models.
        """
        assert (curr_layers >= 0)
        assert num_neighbors % self.k_samplers == 0
        src_idx_l, src_eids_l, cut_time_l = src_list

        device = self.n_feat_th.device
        batch_size = len(src_idx_l[0])
        src_nodes_th = src_idx_l[0]
        cut_times_th = cut_time_l[0].unsqueeze(dim=1)
        src_tembed = self.time_encoder(torch.zeros_like(cut_times_th))
        src_nfeat = self.node_raw_embed(src_nodes_th)

        if curr_layers == 0:
            return src_nfeat

        # get node features at previous layer
        src_conv_feat = self.tem_conv(
            (src_idx_l[:-1], src_eids_l[:-1], cut_time_l[:-1]),
            curr_layers - 1, num_neighbors)
        k = self.k_samplers
        # get neighbor node features at previous layer
        src_ngh_nodes_th = src_idx_l[1].view(batch_size, k,
                                             -1).permute(1, 0,
                                                         2)  # [k, batch, -1]
        src_ngh_eids_th = src_eids_l[1].view(batch_size, k,
                                             -1).permute(1, 0, 2)
        src_ngh_t_th = cut_time_l[1].view(batch_size, k, -1).permute(1, 0, 2)
        src_ngh_tdelta = cut_times_th.unsqueeze(dim=0) - src_ngh_t_th

        # next layer also perform sampling fusion
        div_neighbors = num_neighbors // self.k_samplers
        src_ngh_conv_feat = self.tem_conv(
            (src_idx_l[1:], src_eids_l[1:], cut_time_l[1:]), curr_layers - 1,
            num_neighbors)
        src_ngh_feat = src_ngh_conv_feat.view(batch_size, k, div_neighbors,
                                              -1).permute(1, 0, 2, 3)

        sampling_feats = []
        for i in range(k):
            i_nodes = src_ngh_nodes_th[i].squeeze(0)  # [batch, num_neighbors]
            i_eids = src_ngh_eids_th[i].squeeze(0)
            i_t = src_ngh_tdelta[i].squeeze(0)
            i_ngh_feat = src_ngh_feat[i].squeeze(
                0)  # [batch, num_neighbors, DIM]
            # get edge time features and node features
            i_ngh_tembed = self.time_encoder(i_t)
            i_ngh_efeat = self.edge_raw_embed(i_eids)

            # attention aggregation
            mask = i_nodes == 0
            attn_m = self.attn_model_list[curr_layers - 1]
            local, weight = attn_m(src_conv_feat, src_tembed, i_ngh_feat,
                                   i_ngh_tembed, i_ngh_efeat, mask)
            sampling_feats.append(local)

        # fuse feats under different sampling strategies
        fusion_layer = self.fusion_layer_list[curr_layers - 1]
        fusion_feats = fusion_layer(sampling_feats)
        return fusion_feats


class LGFusion(SamplingFusion):
    def __init__(self, *args, **kwargs) -> None:
        super(LGFusion, self).__init__(*args, **kwargs)
        feat_dim = self.feat_dim
        num_layers = kwargs['num_layers']
        attn_mode = kwargs['attn_mode']
        n_head = kwargs['n_heads']
        drop_out = kwargs['drop_out']

        self.global_fusion = MultiHeadAttention(1,
                                                feat_dim,
                                                feat_dim,
                                                feat_dim,
                                                dropout=drop_out)

    def forward(self,
                src_idx_l,
                target_idx_l,
                cut_time_l,
                num_neighbors=20,
                global_anchors=None):
        src_embed = self.lg_conv(src_idx_l, cut_time_l, self.num_layers,
                                 num_neighbors, global_anchors)
        target_embed = self.lg_conv(target_idx_l, cut_time_l, self.num_layers,
                                    num_neighbors, global_anchors)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        return score

    def contrast(self,
                 src_idx_l,
                 target_idx_l,
                 background_idx_l,
                 cut_time_l,
                 num_neighbors=20,
                 global_anchors=None):
        src_embed = self.lg_conv(src_idx_l, cut_time_l, self.num_layers,
                                 num_neighbors, global_anchors)
        target_embed = self.lg_conv(target_idx_l, cut_time_l, self.num_layers,
                                    num_neighbors, global_anchors)
        background_embed = self.lg_conv(background_idx_l, cut_time_l,
                                        self.num_layers, num_neighbors,
                                        global_anchors)
        pos_score = self.affinity_score(src_embed,
                                        target_embed).squeeze(dim=-1)
        neg_score = self.affinity_score(src_embed,
                                        background_embed).squeeze(dim=-1)
        return pos_score.sigmoid(), neg_score.sigmoid()

    def lg_conv(self,
                src_idx_l,
                cut_time_l,
                curr_layers,
                num_neighbors=20,
                global_anchors=None) -> torch.Tensor:
        batch_size = len(src_idx_l)
        num_anchors = len(global_anchors) // len(src_idx_l)
        src_embeds = self.tem_conv(src_idx_l, cut_time_l, curr_layers,
                                   num_neighbors)
        ext_cut_time = cut_time_l.repeat(num_anchors)
        global_embeds = self.tem_conv(global_anchors, ext_cut_time,
                                      curr_layers - 1, num_neighbors)

        mask = torch.zeros((batch_size, 1, num_anchors))
        output, attn = self.global_fusion(q=src_embeds,
                                          k=global_embeds,
                                          v=global_embeds,
                                          mask=mask)
        output = output.squeeze(1)
        attn = attn.squeeze(1)
        output = self.merge_layer(output, src_embeds)
        return output
