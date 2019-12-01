import numpy as np
import pandas as pd
import tensorflow as tf
from collections import namedtuple

from model.layers import Layer
from model.models import GeneralizedModel
# SAGEInfo is a namedtuple that specifies the parameters
# of the recursive GraphSAGE layers
SAGEInfo = namedtuple("SAGEInfo",
                      ['layer_name',  # name of the layer (to get feature embedding etc.)
                       'num_samples',
                       ])


class AttentionAggregator(Layer):
    pass


class GraphTemporalAttention(GeneralizedModel):
    pass
# class GTA(Layer):
#     def __init__(self, adj, layer_infos, batch_size=512, edge_features=None, node_features=None):
#         self.adj = adj
#         self.dims = []
#         if node_features:
#             self.node_features = node_features
#         else:
#             self.node_features = tf.Variable(initializer(
#                 [self.n_nodes, self.emb_dim]), name='node_embedding')
#         if edge_features:
#             self.edge_features = edge_features
#         else:
#             self.edge_features = tf.eye(self.emb_dim)

#     def construct_placeholders(self, layer_infos):

#         self.placeholders = {
#             'batch1': tf.placeholder(tf.int32, shape=(None), name='batch1'),
#             'batch2': tf.placeholder(tf.int32, shape=(None), name='batch2'),
#             'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
#             'batch_size': tf.placeholder(tf.int32, name='batch_size')
#         }
#         self.support_sizes = self.sample()

#     def sample(self):
#         pass

#     def build(self):
#         pass

#     def loss(self):
#         pass
