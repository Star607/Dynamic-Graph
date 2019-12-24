import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from tensorflow.layers import conv1d
from collections import namedtuple

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


class AttentionAggregator(Layer):
    """Referring to Graph Attention Networks(Veličković, Petar, et al.), aggregate neighbors' features via attention mechanism. Given attention weight $a_i$, the node embedding can be computed by $W_0 * h + W_1 * \\sum a_i * h_i$.
        TODO: implement layers with mask
    """

    def __init__(self, input_dim, output_dim, mask_value=0., dropout=0.,
                 act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(AttentionAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.act = act
        self.concat = concat

        if name is not None:
            name = '/' + name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars["feat_weights"] = glorot(
                [input_dim, output_dim], name="neigh_weights")
            self.vars["attn_weights"] = glorot(
                [output_dim, 1], name="attn_weights")

            self.vars["bias"] = zeros([output_dim], name="bias")

        if self.logging:
            self._log_vars()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs, num_neighbors = inputs

        dims = tf.shape(self_vecs)
        support_size = dims[0]

        # shape: (batch_size * num_neighbors, output_dim)
        from_neighs = tf.matmul(neigh_vecs, self.vars["feat_weights"])
        h_self = tf.matmul(self_vecs, self.vars["feat_weights"])

        # shape: (batch_size * num_neighbors, 1)
        neighs_logits = tf.matmul(from_neighs, self.vars["attn_weights"])
        neighs_logits = tf.reshape(
            neighs_logits, [support_size, num_neighbors])
        # shape: (batch_size, 1)
        self_logits = tf.matmul(h_self, self.vars["attn_weights"])
        logits = neighs_logits + self_logits
        # shape: (batch_size, num_neighbors)
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))
        # print(coefs)
        # shape: (batch_size * num_neighbors, 1)
        coefs = tf.reshape(tf.nn.dropout(coefs, rate=self.dropout), [-1, 1])

        # shape: (batch_size, num_neighbors, output_dim)
        h_neighs = tf.reshape(tf.multiply(coefs, from_neighs), [
                              support_size, num_neighbors, -1])
        h_neighs = tf.reduce_sum(h_neighs, axis=1)

        if not self.concat:
            output = tf.add_n([h_self, h_neighs])
        else:
            output = tf.concat([h_self, h_neighs], axis=1)
        output += self.vars["bias"]
        return self.act(output)


class GraphTemporalAttention(GeneralizedModel):
    def __init__(self, placeholders, features, adj, degrees, layer_infos, context_layer_infos,
                 batch_size=512, learning_rate=1e-4,  concat=True, aggregator_type="attention",
                 embed_dim=128, **kwargs):
        super(GraphTemporalAttention, self).__init__(**kwargs)
        self.aggregator_cls = AttentionAggregator

        self.batch_from = placeholders["batch_from"]
        self.batch_to = placeholders["batch_to"]
        self.timestamp = placeholders["timestamp"]
        self.context_from = placeholders["context_from"]
        self.context_to = placeholders["context_to"]
        self.context_timestamp = placeholders["context_timestamp"]

        self.adj_info, self.ts_info = adj
        self.embeds = tf.get_variable(
            "node_embeddings", [len(self.adj_info), embed_dim])
        if features is None:
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(
                features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degress = degrees
        self.concat = concat

        self.dims = [
            (0 if features is None else features.shape[1]) + embed_dim]
        self.dims.extend(
            [layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = batch_size
        self.placeholders = placeholders
        self.layer_infos = layer_infos
        self.context_layer_infos = context_layer_infos
        self.context_dims = [
            (0 if features is None else features.shape[1]) + embed_dim]
        self.context_dims.extend(
            [layer_infos[i].output_dim for i in range(len(context_layer_infos))])

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate)
        self.build()

    def sample(self, inputs, timestamp, layer_infos, batch_size):
        samples = [inputs]
        timemasks = [timestamp]
        support_size = batch_size
        support_sizes = [support_size]
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1
            support_size *= layer_infos[t].num_samples
            sampler = layer_infos[t].neigh_sampler
            node, tsmask = sampler((samples[k], timemasks[k], support_size,
                                    layer_infos[t].num_samples))
            samples.append(tf.reshape(node, [-1]))
            timemasks.append(tf.reshape(tsmask, [-1]))
            support_sizes.append(support_size)
        return samples, support_sizes

    def aggregate(self, samples, input_features, dims, num_samples, support_sizes,
                  batch_size=None, aggregators=None, name=None, concat=False):
        if batch_size is None:
            batch_size = self.batch_size

        hidden = [tf.nn.embedding_lookup(
            input_features, node_samples) for node_samples in samples]
        if aggregators is None:
            aggregators = []
            for layer in range(len(num_samples)-1):
                dim_mult = 2 if concat and (layer != 0) else 1
                agg = self.aggregator_cls(
                    dim_mult*dims[layer], dims[layer+1], act=lambda x: x, name=name, concat=concat)
                aggregators.append(agg)

            dim_mult = 2 if concat else 1
            agg = self.aggregator_cls(
                dim_mult*dims[-2], dims[-1], name=name, concat=concat)
            aggregators.append(agg)

        for layer in range(len(num_samples)):
            next_hidden = []
            agg = aggregators[layer]
            for hop in range(len(num_samples)-layer):
                dim_mult = 2 if concat and (layer != 0) else 1
                neigh_dims = [batch_size * support_sizes[hop],
                              num_samples[len(num_samples)-hop-1],
                              dim_mult*dims[layer]]
                h = agg((hidden[hop], tf.reshape(hidden[hop+1], neigh_dims)))
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0], aggregators

    def _build(self):
        labels = tf.reshape(tf.cast(self.placeholders["batch_to"], dtype=tf.int64), [
                            self.batch_size, 1])
        self.neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,
            num_true=1,
            num_sampled=FLAGS.neg_sample_size,
            unique=False,
            range_max=len(self.degress),
            distortion=0.75,
            unigrams=self.degress.tolist()))

        from_samples, support_sizes = self.sample(
            self.batch_from, self.timestamp, self.layer_infos, FLAGS.batch_size)
        to_samples, support_sizes = self.sample(
            self.batch_to, self.timestamp, self.layer_infos, FLAGS.batch_size)
        num_samples = [
            layer_info.num_samples for layer_info in self.layer_infos]
        # self.hidden_from, self.aggs = self.aggregate(
        #     from_samples, [self.features], self.dims, num_samples, support_sizes, concat=self.concat)
        # self.hidden_to, _ = self.aggregate(
        #     to_samples, [self.features], self.dims, num_samples, support_sizes, aggregators=self.aggs, concat=self.concat)

        neg_timestamp = tf.tile(self.timestamp, [FLAGS.neg_sample_size])
        self.neg_timestamp = tf.reshape(tf.transpose(tf.reshape(
            neg_timestamp, shape=[-1, FLAGS.neg_sample_size])), shape=[-1])
        neg_samples = tf.tile(self.neg_samples, [self.batch_size])
        neg_samples, neg_support_sizes = self.sample(
            neg_samples, self.neg_timestamp, self.layer_infos, FLAGS.neg_sample_size * FLAGS.batch_size)
        # self.hidden_neg, _ = self.aggregate(neg_samples, [self.features], self.dims, num_samples,
                                            # neg_support_sizes, batch_size=FLAGS.neg_sample_size, aggregators=self.aggs, concat=self.concat)

        context_froms, support_sizes = self.sample(
            self.context_from, self.context_timestamp, self.context_layer_infos, FLAGS.context_size * FLAGS.batch_size)
        context_tos, _ = self.sample(
            self.context_to, self.context_timestamp, self.context_layer_infos)
        num_samples = [
            layer_info.num_samples for layer_info in self.context_layer_infos]
        # self.ctxhid_from, self.context_aggs = self.aggregate(
            # context_froms, [self.features], self.context_dims, num_samples, support_sizes, concat=self.concat)
        # self.ctxhid_to, _ = self.aggregate(
            # context_tos, [self.features], self.context_dims, num_samples, support_sizes, concat=self.concat)
        self.context_hidden = tf.concat(
            [self.ctxhid_from, self.ctxhid_to], axis=0)

        att_agg = AttentionAggregator(self.dims[0], self.dims[-1])
        self.output_from = att_agg((self.hidden_from, self.context_hidden))
        self.output_to = att_agg((self.hidden_to, self.context_hidden))
        self.output_neg = att_agg((self.hidden_neg, self.context_hidden))

        dim_mult = 2 if self.concat else 1
        self.link_pred_layer = BipartiteEdgePredLayer(
            dim_mult * self.dims[-1], dim_mult*self.dims[-1], self.placeholders, act=tf.nn.sigmoid, bilinear_weights=False, name="edge_predict")

        self.output_from = tf.nn.l2_normalize(self.output_from, 1)
        self.output_to = tf.nn.l2_normalize(self.output_to, 1)
        self.output_neg = tf.nn.l2_normalize(self.output_neg, 1)

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

    def _loss(self):
        for aggregator in self.aggs + self.context_aggs:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

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
        self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
        size = tf.shape(self.aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
        _, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        self.mrr = tf.reduce_mean(
            tf.div(1.0, tf.cast(self.ranks[:, -1] + 1, tf.float32)))
        tf.summary.scalar('mrr', self.mrr)


if __name__ == "__main__":
    # check AttentionAggregator work
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True  # pylint: disable=no-member
    config.allow_soft_placement = True
    tf.enable_eager_execution(config=config)

    self_vecs = tf.random.normal([10, 128])
    neigh_vecs = tf.random.normal([250, 128])
    agg = AttentionAggregator(128, 128)
    print(agg((self_vecs, neigh_vecs, 25)))
