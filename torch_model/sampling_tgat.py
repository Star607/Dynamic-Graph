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
        x = [F.tanh(self.linears[i](embeds[i])) for i in range(k)]
        x = [self.query(x[i]) for i in range(k)]  # (k, n, 1)
        weights = torch.cat(x, dim=1)  # (n, k)
        ans = weights.unsqueeze(-1) * embeds
        return torch.sum(ans, dim=-1)  # (n, d)


class SamplingFusion(TGAN):
    def __init__(self, k_samplers: int, *args, **kwargs) -> None:
        super(SamplingFusion, self).__init__(*args, **kwargs)
        feat_dim = self.feat_dim
        num_layers = kwargs['num_layers']

        self.fusion_layer_list = torch.nn.ModuleList([
            SoftmaxAttention(feat_dim, k_samplers) for _ in range(num_layers)
        ])
        self.samplers = None

    def set_samplers(self, samplers: list) -> None:
        self.samplers = samplers

    def tem_conv(self, src_idx_l, cut_time_l, curr_layers,
                 num_neighbors) -> torch.Tensor:
        assert (curr_layers >= 0)
        device = self.n_feat_th.device
        batch_size = len(src_idx_l)
        src_nodes_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_times_th = torch.from_numpy(cut_time_l).float().to(device)
        src_tembed = self.time_encoder(torch.zeros_like(cut_times_th))
        src_nfeat = self.node_raw_embed(src_nodes_th)

        if curr_layers == 0:
            return src_nfeat

        # get node features at previous layer
        src_conv_feat = self.tem_conv(src_idx_l, cut_time_l, curr_layers - 1,
                                      num_neighbors)

        div_neighbors = num_neighbors // len(self.samplers)
        sampling_feats = []
        for i in range(len(self.samplers)):
            if i == len(self.samplers) - 1:
                cur_neighbors = num_neighbors - i * div_neighbors
            else:
                cur_neighbors = div_neighbors

            # get neighbor node features at previous layer
            src_ngh_nodes, src_ngh_eids, src_ngh_t = self.samplers[
                i].get_temporal_neighbor(src_idx_l, cut_time_l, cur_neighbors)

            src_ngh_nodes_th = torch.from_numpy(src_ngh_nodes).long().to(
                device)
            src_ngh_eids_th = torch.from_numpy(src_ngh_eids).long().to(device)
            src_ngh_tdelta = cut_time_l[:, np.newaxis] - src_ngh_t
            src_ngh_t_th = torch.from_numpy(src_ngh_tdelta).float().to(device)

            src_ngh_conv_feat = self.tem_conv(src_ngh_nodes.flatten(),
                                              src_ngh_t.flatten(),
                                              curr_layers - 1, num_neighbors)
            src_ngh_feat = src_ngh_conv_feat.view(batch_size, num_neighbors,
                                                  -1)

            # get edge time features and node features
            src_ngh_tembed = self.time_encoder(src_ngh_t_th)
            src_ngh_efeat = self.edge_raw_embed(src_ngh_eids_th)

            # attention aggregation
            mask = src_ngh_nodes_th == 0
            attn_m = self.attn_model_list[curr_layers - 1]
            local, weight = attn_m(src_conv_feat, src_tembed, src_ngh_feat,
                                   src_ngh_tembed, src_ngh_efeat, mask)
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

    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors,
                global_anchors):
        src_embed = self.lg_conv(src_idx_l, cut_time_l, self.num_layers,
                                 num_neighbors, global_anchors)
        target_embed = self.lg_conv(target_idx_l, cut_time_l, self.num_layers,
                                    num_neighbors, global_anchors)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        return score

    def contrast(self, src_idx_l, target_idx_l, background_idx_l, cut_time_l,
                 num_neighbors, global_anchors):
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

    def lg_conv(self, src_idx_l, cut_time_l, curr_layers, num_neighbors,
                global_anchors) -> torch.Tensor:
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
