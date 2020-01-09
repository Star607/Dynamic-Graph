import time
import numpy as np
import pandas as pd
import tensorflow as tf
# import tensorflow as tf
from tensorflow.layers import conv1d
from collections import namedtuple

from .aggregators import *
from .inits import glorot, zeros
from .layers import Layer, Dense
from .models import GeneralizedModel
from .prediction import BipartiteEdgePredLayer
from data_loader.minibatch import TemporalNeighborSampler

# SAGEInfo is a namedtuple that specifies the parameters
# of the recursive GraphSAGE layers
SAGEInfo = namedtuple("SAGEInfo",
                      ['layer_name',  # name of the layer (to get feature embedding etc.)
                       'neigh_sampler',  # callable neigh_sampler constructor
                       'num_samples',
                       'output_dim'  # the output (i.e., hidden) dimension
                       ])

flags = tf.app.flags
FLAGS = flags.FLAGS


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print("%r  %2.2f s" % (method.__name__, te - ts))
        return result
    return timed


class NGCFAggregator(Layer):
    def __init__(self, input_dim, output_dim, neigh_input_dim=None, dropout=0.,
                 act=tf.nn.leaky_relu, name=None, concat=False, **kwargs):
        super(NGCFAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + "_vars"):
            self.vars["self_weights"] = glorot(
                [input_dim, output_dim], name="self_weights")
            self.vars["neigh_weights"] = glorot(
                [neigh_input_dim, output_dim], name="neigh_weights")

        if self.logging:
            self._log_vars()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        pass


class AttentionAggregator(Layer):
    """Referring to Graph Attention Networks(Veličković, Petar, et al.), aggregate neighbors' features via attention mechanism. Given attention weight $a_i$, the node embedding can be computed by $W_0 * h + W_1 * \\sum a_i * h_i$.
        TODO: implement layers with mask
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None, mask_value=0., dropout=0.,
                 act=tf.nn.leaky_relu, name=None, concat=False, **kwargs):
        super(AttentionAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars["feat_weights"] = glorot(
                [input_dim, output_dim], name="neigh_weights")
            self.vars["attn_weights"] = glorot(
                [output_dim, 1], name="attn_weights")
            dim_mult = 2 if concat else 1
            self.vars["bias"] = zeros([dim_mult * output_dim], name="bias")

        if self.logging:
            self._log_vars()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        batch_size = tf.shape(self_vecs)[0]
        num_neighbors = tf.shape(neigh_vecs)[1]
        # num_neighbors = tf.cast(tf.shape(neigh_vecs)[0] / batch_size, tf.int32)
        neigh_vecs = tf.reshape(neigh_vecs, [-1, self.neigh_input_dim])

        # shape: (support_size * num_neighbors, output_dim)
        from_neighs = tf.matmul(neigh_vecs, self.vars["feat_weights"])
        h_self = tf.matmul(self_vecs, self.vars["feat_weights"])

        # shape: (support_size * num_neighbors, 1)
        neighs_logits = tf.matmul(from_neighs, self.vars["attn_weights"])
        neighs_logits = tf.reshape(
            neighs_logits, [batch_size, num_neighbors])
        # shape: (support_size, 1)
        self_logits = tf.matmul(h_self, self.vars["attn_weights"])
        logits = neighs_logits + self_logits
        # shape: (support_size, num_neighbors)
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))
        # print(coefs)
        # shape: (support_size * num_neighbors, 1)
        coefs = tf.reshape(tf.nn.dropout(coefs, rate=self.dropout), [-1, 1])

        # shape: (support_size, num_neighbors, output_dim)
        h_neighs = tf.reshape(tf.multiply(coefs, from_neighs), [
                              batch_size, num_neighbors, -1])
        h_neighs = tf.reduce_sum(h_neighs, axis=1)

        if not self.concat:
            output = tf.add_n([h_self, h_neighs])
        else:
            output = tf.concat([h_self, h_neighs], axis=1)
        output += self.vars["bias"]
        return self.act(output)


class GraphTemporalAttention(GeneralizedModel):
    def __init__(self, placeholders, features, adj_info, ts_info, degrees, layer_infos,
                 context_layer_infos, sampler, bipart=False, n_users=1, edge_features=None,
                 concat=False, aggregator_type="meanpool", embed_dim=128, **kwargs):
        super(GraphTemporalAttention, self).__init__(**kwargs)
        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        elif aggregator_type == "attention":
            self.aggregator_cls = AttentionAggregator
        else:
            raise Exception("Unknown aggregator: ", aggregator_type)

        self.batch_from = placeholders["batch_from"]
        self.batch_to = placeholders["batch_to"]
        self.timestamp = placeholders["timestamp"]
        self.context_from = placeholders["context_from"]
        self.context_to = placeholders["context_to"]
        self.context_timestamp = placeholders["context_timestamp"]
        self.batch_size = placeholders["batch_size"]

        self.adj_info = adj_info
        self.ts_info = ts_info
        zero_padding = tf.zeros([1, embed_dim], name="zero_padding")
        embeds = glorot([len(adj_info), embed_dim],
                        name="node_emebddings")
        self.embeds = tf.concat([zero_padding, embeds], axis=0)
        if features is None:
            self.features = self.embeds
        else:
            self.features = tf.Variable(
                tf.constant(features, dtype=tf.float32), trainable=False)
            self.features = tf.concat([self.embeds, self.features], axis=1)
        # print("embed shape: ", tf.shape(self.embeds))

        self.layer_infos = layer_infos
        self.ctx_layer_infos = context_layer_infos
        self.dims = [
            (0 if features is None else features.shape[1]) + embed_dim]
        self.dims.extend(
            [layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.ctx_dims = [self.dims[0]]
        self.ctx_dims.extend(
            [context_layer_infos[i].output_dim for i in range(len(context_layer_infos))])
        # Attention layer output dim
        assert(self.dims[-1] == self.ctx_dims[-1])
        self.dims.append(self.dims[-1])
        self.ctx_dims.append(self.ctx_dims[-1])

        self.placeholders = placeholders
        self.sampler = sampler
        self.bipart = bipart
        self.n_users = n_users
        self.degrees = degrees[n_users:]
        self.concat = concat
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)
        self.build()

    @timeit
    def sample(self, inputs, timestamp, layer_infos, batch_size):
        # for neg_samples and context_samples
        samples = [inputs]
        timemasks = [timestamp]
        support_size = 1
        support_sizes = [support_size]
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1
            num_samples = layer_infos[t].num_samples
            node, tmask = self.sampler(
                (samples[k], timemasks[k], batch_size, num_samples))
            samples.append(tf.reshape(node, [-1]))
            timemasks.append(tf.reshape(tmask, [-1]))
            batch_size *= num_samples
            support_size *= num_samples
            support_sizes.append(support_size)
        return samples, timemasks, support_sizes

    def aggregate(self, samples, dims, num_samples, support_sizes,
                  aggregators=None, batch_size=None, name=None, concat=False):
        if batch_size is None:
            batch_size = self.batch_size

        hidden = [tf.nn.embedding_lookup(
            self.features, nodes) for nodes in samples]

        if aggregators is None:
            aggregators = []
            # len(num_samples) = len(layer_infos) = len(dims) - 1 = len(support_sizes) - 1
            for layer in range(len(num_samples)):
                dim_mult = 2 if concat and (layer != 0) else 1
                # if layer == len(num_samples) - 1:
                act = tf.nn.leaky_relu
                # else:
                # def act(x): return x
                agg = self.aggregator_cls(
                    dim_mult*dims[layer], dims[layer+1], act=act, name=name, concat=concat)
                aggregators.append(agg)

        for layer in range(len(num_samples)):
            next_hidden = []
            agg = aggregators[layer]
            for hop in range(len(num_samples) - layer):
                # print("layer %d hop %d" % (layer, hop))
                dim_mult = 2 if concat and (layer != 0) else 1
                neigh_dims = [batch_size * support_sizes[hop],
                              num_samples[hop],
                              dim_mult * dims[layer]]
                h = agg((hidden[hop], tf.reshape(hidden[hop+1], neigh_dims)))
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0], aggregators

    def _build(self):
        samples_from, _, support_sizes = self.sample(
            self.batch_from, self.timestamp, self.layer_infos, FLAGS.batch_size)
        samples_to, _, _ = self.sample(
            self.batch_to, self.timestamp, self.layer_infos, FLAGS.batch_size)
        num_samples = [layer.num_samples for layer in self.layer_infos]
        num_samples = num_samples[::-1]
        self.embed_from, self.aggs = self.aggregate(
            samples_from, self.dims, num_samples, support_sizes, concat=self.concat)
        self.embed_to, _ = self.aggregate(
            samples_to, self.dims, num_samples, support_sizes, self.aggs, concat=self.concat)

        labels = tf.reshape(tf.cast(self.placeholders["batch_to"], dtype=tf.int64),
                            [self.batch_size, 1])
        self.batch_neg, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,
            num_true=1,
            num_sampled=FLAGS.neg_sample_size,
            unique=True,
            range_max=len(self.degrees),
            distortion=0.75,
            unigrams=self.degrees.tolist()))
        self.batch_neg = self.batch_neg + self.n_users
        self.neg_ts = tf.random.shuffle(self.timestamp)[:FLAGS.neg_sample_size]
        # self.neg_ts = self.repeat_vector(self.timestamp, FLAGS.neg_sample_size)
        # self.batch_neg = tf.tile(self.batch_neg, [self.batch_size])
        samples_neg, _, _ = self.sample(
            self.batch_neg, self.neg_ts, self.layer_infos, FLAGS.neg_sample_size)
        self.embed_neg, _ = self.aggregate(
            samples_neg, self.dims, num_samples, support_sizes, self.aggs, batch_size=FLAGS.neg_sample_size, concat=self.concat)

        # context_size = FLAGS.context_size * FLAGS.batch_size
        # ctx_samples_from, _, ctx_support_sizes = self.sample(
        #     self.context_from, self.context_timestamp, self.ctx_layer_infos, context_size)
        # ctx_samples_to, _, _ = self.sample(
        #     self.context_to, self.context_timestamp, self.ctx_layer_infos, context_size)
        # ctx_num_samples = [layer.num_samples for layer in self.ctx_layer_infos]
        # ctx_num_samples = ctx_num_samples[::-1]
        # self.ctx_embed_from, self.ctx_aggs = self.aggregate(
        #     ctx_samples_from, self.ctx_dims, ctx_num_samples, ctx_support_sizes, batch_size=context_size, concat=self.concat)
        # self.ctx_embed_to, self.ctx_aggs = self.aggregate(
        #     ctx_samples_to, self.ctx_dims, ctx_num_samples, ctx_support_sizes, self.ctx_aggs, batch_size=context_size, concat=self.concat)
        # self.context_embed = tf.reshape(tf.concat(
        #     [self.ctx_embed_from, self.ctx_embed_to], axis=1), [self.batch_size, FLAGS.context_size, -1])

        # self.aggs.extend(self.ctx_aggs)

        # dim_mult = 2 if self.concat else 1
        # att_agg = AttentionAggregator(
        #     dim_mult * self.dims[-1], dim_mult * self.dims[-1])
        # self.output_from = att_agg((self.embed_from, self.context_embed))
        # self.output_to = att_agg((self.embed_to, self.context_embed))
        # self.output_neg = att_agg((self.embed_neg, self.context_embed))
        # self.aggs.append(att_agg)

        self.output_from = self.embed_from
        self.output_to = self.embed_to
        self.output_neg = self.embed_neg

        dim_mult = 2 if self.concat else 1
        self.link_pred_layer = BipartiteEdgePredLayer(
            dim_mult * self.dims[-1], dim_mult * self.dims[-1], self.placeholders, act=tf.nn.sigmoid, bilinear_weights=False, neg_sample_weights=1/20, name="edge_predict")
        # self.output_from = tf.nn.l2_normalize(self.output_from, 1)
        # self.output_to = tf.nn.l2_normalize(self.output_to, 1)
        # self.output_neg = tf.nn.l2_normalize(self.output_neg, 1)

        self.sample_ops = [samples_from, samples_to,
                           samples_neg]
        self.count_nonzero_ops = []
        for samples in self.sample_ops:
            cnt = [tf.math.count_nonzero(s) for s in samples]
            self.count_nonzero_ops.append(cnt)

        self.embed_ops = [self.embed_from, self.embed_to,
                          self.embed_neg]
        # self.sample_ops = [samples_from, samples_to,
        #                    samples_neg, ctx_samples_from, ctx_samples_to]
        # self.embed_ops = [self.embed_from, self.embed_to,
        #                   self.embed_neg, self.ctx_embed_from, self.ctx_embed_to]

    def repeat_vector(self, vec, times):
        vec = tf.reshape(vec, [-1, 1])
        vec = tf.tile(vec, [1, times])
        return tf.reshape(vec, [-1])

    def repeat_matrix(self, mat, times):
        """Repeat matrix across axis 0"""
        dim = tf.shape(mat)[1]
        mat = tf.tile(mat, [1, times])
        return tf.reshape(mat, [-1, dim])

    def build(self):
        self._build()
        self._loss()
        self._accuracy()
        self.loss = self.loss / tf.cast(self.batch_size, tf.float32)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        self.pred_op = self._predict()

    def _bpr_loss(self):
        pass

    def _loss(self):
        for aggregator in self.aggs:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
            for name, var in aggregator.vars.items():
                tf.summary.histogram(name, var)
        self.loss += self.link_pred_layer.loss(
            self.output_from, self.output_to, self.output_neg)
        tf.summary.scalar('loss', self.loss)

    def _accuracy(self):
        # shape: [batch_size]
        aff = self.link_pred_layer.affinity(self.output_from, self.output_to)
        # shape : [batch_size x num_neg_samples]
        self.neg_aff = self.link_pred_layer.neg_cost(
            self.output_from, self.output_neg)
        self.neg_aff = tf.reshape(
            self.neg_aff, [self.batch_size, FLAGS.neg_sample_size])
        _aff = tf.expand_dims(aff, axis=1)
        self.aff_all = tf.concat(axis=1, values=[_aff, self.neg_aff])
        size = tf.shape(self.aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
        _, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        self.mrr = tf.reduce_mean(
            tf.div(1.0, tf.cast(self.ranks[:, 0] + 1, tf.float32)))
        self.print_op = [tf.print("affinity: ", self.aff_all[-1, :5]),
                         tf.print("probs: ", tf.nn.sigmoid(self.aff_all[-1, :5]))]
        tf.summary.scalar('mrr', self.mrr)

    def _predict(self):
        logits = self.link_pred_layer.affinity(
            self.output_from, self.output_to)
        return tf.nn.sigmoid(logits)
