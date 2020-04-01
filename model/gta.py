import time
import numpy as np
import pandas as pd
import tensorflow as tf
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


def repeat_vector(vec, times):
    # input: ([1, 2, 3], times=2)
    # output: [1, 1, 2, 2, 3, 3]
    vec = tf.reshape(vec, [-1, 1])
    vec = tf.tile(vec, [1, times])
    return tf.reshape(vec, [-1])


def repeat_matrix(mat, times):
    """Repeat matrix across axis 0"""
    dim = tf.shape(mat)[1]
    mat = tf.tile(mat, [1, times])
    return tf.reshape(mat, [-1, dim])


class NGCFAggregator(Layer):
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
        coefs = tf.reshape(tf.nn.dropout(
            coefs, keep_prob=1-self.dropout), [-1, 1])

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


class SinusoidalTimedeltaAggregator(Layer):
    pass


class GraphTemporalAttention(GeneralizedModel):
    def __init__(self, placeholders, features, adj_info, ts_info, degrees, layer_infos,
                 context_layer_infos, sampler, bipart=False, n_users=1, edge_features=None,
                 concat=False, aggregator_type="meanpool", embed_dim=128, **kwargs):
        super(GraphTemporalAttention, self).__init__(**kwargs)

        self.eps = 1e-7
        self.margin = 0.1

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
        self.batch_neg = placeholders["batch_neg"]
        self.timestamp = placeholders["timestamp"]
        self.context_from = placeholders["context_from"]
        self.context_to = placeholders["context_to"]
        self.context_timestamp = placeholders["context_timestamp"]
        self.batch_size = placeholders["batch_size"]
        self.context_size = placeholders["batch_size"] * FLAGS.context_size

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
        # Context Attention layer output dim
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
        # Also for neg_samples and context_samples
        samples = [inputs]
        timemasks = [timestamp]
        timefilters = [timestamp]
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1
            num_samples = layer_infos[t].num_samples
            # filter the adjacency list according to batch timestamp
            node, tmask = self.sampler(
                (samples[k], timefilters[k], batch_size, num_samples))
            samples.append(tf.reshape(node, [-1]))
            timemasks.append(tf.reshape(tmask, [-1]))
            new_filter = repeat_vector(timestamp, layer_infos[t].num_samples)
            timefilters.append(new_filter)
            batch_size *= num_samples
        return samples, timemasks

    def compute_support_sizes(self, layer_infos):
        support_size = 1
        support_sizes = [support_size]
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1
            support_size *= layer_infos[t].num_samples
            support_sizes.append(support_size)
        return support_sizes

    def aggregate(self, samples, dims, num_samples, support_sizes,
                  aggregators, batch_size, name=None, concat=False):
        hidden = [tf.nn.embedding_lookup(
            self.features, nodes) for nodes in samples]

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
        return hidden[0]

    def config_aggregators(self, layer_infos, dims, concat=False, name=None):
        aggregators = []
        # len(num_samples) = len(layer_infos) = len(dims) - 1 = len(support_sizes) - 1
        for layer in range(len(layer_infos)):
            dim_mult = 2 if concat and (layer != 0) else 1
            # if layer == len(num_samples) - 1:
            act = tf.nn.leaky_relu
            # else:
            # def act(x): return x
            agg = self.aggregator_cls(
                dim_mult*dims[layer], dims[layer+1], act=act, name=name, concat=concat)
            aggregators.append(agg)
        return aggregators

    def build(self):
        self._build()
        self.pos_scores = tf.reduce_sum(tf.multiply(
            self.output_from, self.output_to), axis=1)
        self.neg_scores = tf.reduce_sum(tf.multiply(
            self.output_from, self.output_neg), axis=1)
        self._loss()
        if FLAGS.loss == "xent":
            self._accuracy()
        self.loss = self.loss / tf.cast(self.batch_size, tf.float32)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        preds = [self._predict(self.pos_scores),
                 self._predict(self.neg_scores)]
        self.pred_op = tf.concat(preds, axis=0)

    def _build(self):
        support_sizes = self.compute_support_sizes(self.layer_infos)
        forward_batches = [self.batch_from, self.batch_to, self.batch_neg]
        samples_from, samples_to, samples_neg = [self.sample(
            batch, self.timestamp, self.layer_infos, FLAGS.batch_size)[0] for batch in forward_batches]

        forward_aggs = self.config_aggregators(self.layer_infos, self.dims)
        num_samples = [layer.num_samples for layer in self.layer_infos]
        num_samples = num_samples[::-1]
        forward_samples = [samples_from, samples_to, samples_neg]
        forward_embeds = [self.aggregate(
            samples, self.dims, num_samples, support_sizes, forward_aggs, self.batch_size, concat=self.concat) for samples in forward_samples]
        self.embed_from, self.embed_to, self.embed_neg = forward_embeds[
            0], forward_embeds[1], forward_embeds[2]

        if FLAGS.use_context:
            support_sizes = self.compute_support_sizes(self.ctx_layer_infos)
            contxt_batches = [self.context_from, self.context_to]
            ctx_samples_from, ctx_samples_to = [self.sample(
                batch, self.context_timestamp, self.ctx_layer_infos, FLAGS.batch_size * FLAGS.context_size)[0] for batch in contxt_batches]

            context_aggs = self.config_aggregators(
                self.ctx_layer_infos, self.ctx_dims)
            num_samples = [layer.num_samples for layer in self.ctx_layer_infos]
            num_samples = num_samples[::-1]
            context_samples = [ctx_samples_from, ctx_samples_to]
            context_embeds = [self.aggregate(
                samples, self.ctx_dims, num_samples, support_sizes, context_aggs, self.context_size, concat=self.concat) for samples in context_samples]
            # self.context_embed = tf.concat(context_embeds, axis=1)
            self.context_embed = tf.math.add_n(context_embeds)
            self.context_embed = tf.reshape(
                self.context_embed, [self.batch_size, FLAGS.context_size, -1])

            dim_mult = 2 if self.concat else 1
            att_agg = AttentionAggregator(
                dim_mult * self.dims[-1], dim_mult * self.dims[-1])
            self.output_from = att_agg((self.embed_from, self.context_embed))
            self.output_to = att_agg((self.embed_to, self.context_embed))
            self.output_neg = att_agg((self.embed_neg, self.context_embed))
        else:
            self.output_from = self.embed_from
            self.output_to = self.embed_to
            self.output_neg = self.embed_neg
        # self.output_from, self.output_to, self.output_neg = [
            # tf.nn.l2_normalize(output) for output in [output_from, output_to, output_neg]]

        if FLAGS.use_context:
            self.sample_ops = forward_samples + context_samples
            self.embed_ops = forward_embeds + context_embeds
            self.aggs = forward_aggs + context_aggs + [att_agg]
        else:
            self.sample_ops = forward_samples  # + context_samples
            self.embed_ops = forward_embeds  # + context_embeds
            self.aggs = forward_aggs  # + context_aggs + [att_agg]

    def _loss(self):
        self.loss = 0
        for agg in self.aggs:
            for var in agg.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        regularizer = tf.nn.l2_loss(
            self.output_from) + tf.nn.l2_loss(self.output_to) + tf.nn.l2_loss(self.output_neg)
        self.loss += FLAGS.weight_decay * regularizer

        pos_scores, neg_scores = self.pos_scores, self.neg_scores
        if FLAGS.loss == "xent":
            self.loss += self._xent_loss(pos_scores, neg_scores)
        elif FLAGS.loss == "hinge":
            self.loss += self._hinge_loss(pos_scores, neg_scores)
        elif FLAGS.loss == "bpr":
            self.loss += self._bpr_loss(pos_scores, neg_scores)
        else:
            raise NotImplementedError(
                "Loss {} not implemented yet.".format(FLAGS.loss))

        tf.summary.scalar("loss", self.loss)
        pass

    def _bpr_loss(self, pos_scores, neg_scores):
        mf_loss = tf.reduce_sum(tf.nn.softplus(-(pos_scores - neg_scores)))
        return mf_loss

    def _xent_loss(self, pos_scores, neg_scores):
        pos_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(pos_scores), logits=pos_scores)
        neg_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(neg_scores), logits=neg_scores)
        return tf.reduce_sum(pos_xent) + tf.reduce_sum(neg_xent)

    def _hinge_loss(self, pos_scores, neg_scores):
        diff = tf.nn.relu(neg_scores - pos_scores + self.margin, name="diff")
        return tf.reduce_sum(diff)

    def _accuracy(self):
        """Use only for training set calculation.
        """
        # if self.placeholders["batch_neg"] is None:
        #     raise Exception("placeholder of batch_neg is None.")
        if FLAGS.loss != "xent":
            raise Exception(
                "accuracy can be used only under cross entropy loss.")
        pos_probs, neg_probs = self._predict(
            self.pos_scores), self._predict(self.neg_scores)
        probs = tf.concat([pos_probs, neg_probs], axis=0)
        preds = tf.cast(probs >= 0.5, tf.int32)
        labels = tf.concat(
            [tf.ones_like(pos_probs), tf.zeros_like(neg_probs)], axis=0)
        labels = tf.cast(labels, tf.int32)
        self.acc, self.acc_update = tf.metrics.accuracy(labels, preds)
        self.auc, self.auc_update = tf.metrics.auc(labels, probs)
        tf.summary.scalar("acc", self.acc)
        tf.summary.scalar("auc", self.auc)

    def _predict(self, scores):
        if FLAGS.loss == "xent":
            return tf.reshape(tf.nn.sigmoid(scores), [-1])
        elif FLAGS.loss == "hinge" or FLAGS.loss == "bpr":
            return tf.reshape(scores, -1)
        else:
            raise NotImplementedError(
                "Loss {} not implemented yet.".format(FLAGS.loss))

def ModelTrainer(object):
    def __init__(self, flags=FLAGS):
        self.test = False
        pass

    def log_dir(self):
        log_dir = FLAGS.base_log_dir + "{}-{}".format(FLAGS.dataset, FLAGS.method)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            os.chmod(log_dir, 0o777)
        return log_dir

    def construct_placeholders(self):
        # Define placeholders: (None,) means 1-D tensor
        placeholders = {
            "batch_from": tf.placeholder(tf.int32, shape=(None,), name="batch_from"),
            "batch_to": tf.placeholder(tf.int32, shape=(None,), name="batch_to"),
            "batch_neg": tf.placeholder(tf.int32, shape=(None,), name="batch_neg"),
            "timestamp": tf.placeholder(tf.float64, shape=(None,), name="timestamp"),
            "batch_size": tf.placeholder(tf.int32, name="batch_size"),
            "context_from": tf.placeholder(tf.int32, shape=(None,), name="context_from"),
            "context_to": tf.placeholder(tf.int32, shape=(None,), name="context_to"),
            "context_timestamp": tf.placeholder(tf.float64, shape=(None,), name="timestamp"),
        }
        return placeholders

    def config_tensorflow(self):
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        # Initialize session
        sess = tf.Session(config=config)
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)
        return merged, sess, summary_writer

    def train(self):
        print("Loading training data...")
        edges, nodes = load_data(datadir="./ctdne_data/", dataset=FLAGS.dataset)
        print("Done loading training data...")

        placeholders = construct_placeholders()
        batch = TemporalEdgeBatchIterator(edges, nodes, placeholders,
                                        batch_size=FLAGS.batch_size, max_degree=FLAGS.max_degree, context_size=FLAGS.context_size)

        adj_info, ts_info = batch.adj_ids, batch.adj_tss
        if FLAGS.sampler == "mask":
            sampler = MaskNeighborSampler(adj_info, ts_info)
        elif FLAGS.sampler == "temporal":
            sampler = TemporalNeighborSampler(adj_info, ts_info)
        else:
            raise NotImplementedError("Sampler %s not supported." % FLAGS.sampler)

        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                    SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        ctx_layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        model = GraphTemporalAttention(
            placeholders, None, adj_info, ts_info, batch.degrees, layer_infos,
            ctx_layer_infos, sampler, bipart=batch.bipartite, n_users=batch.n_users)

        merged, sess, summary_writer = config_tensorflow()
        sess.run(tf.global_variables_initializer())
        batch_num = len(batch.train_idx) // FLAGS.batch_size
        
        for epoch in range(FLAGS.epochs):
            print("Epoch %04d" % (epoch + 1))
            batch.shuffle()
            tot_loss = tot_acc = val_loss = val_acc = 0
            sess.run(tf.local_variables_initializer())
            with trange(batch_num) as batch_bar:
                while not batch.end():
                    batch_bar.set_description("batch %04d" % batch_num)
                    feed_dict = batch.next_train_batch()
                    outs = sess.run(
                        [merged, model.opt_op, model.loss, model.acc, model.auc, model.acc_update, model.auc_update], feed_dict)
                    tot_loss += outs[2] * feed_dict[placeholders["batch_size"]]
                    tot_acc += outs[3] * feed_dict[placeholders["batch_size"]]
                    loss, acc = outs[2], outs[3]
                    if batch.batch_num % FLAGS.val_batches == 0:
                        val_loss, val_acc = valid_batch(sess, model, batch)
                    batch_bar.update()
                    batch_bar.set_postfix(
                        loss=loss, acc=acc, val_loss=val_loss, val_acc=val_acc)
                    summary_writer.add_summary(
                        outs[0], epoch * batch_num + batch.batch_num)
            tot_loss /= len(batch.train_idx)
            tot_acc /= len(batch.train_idx)
            # print("Full train_loss %.4f train_acc %.4f" % (tot_loss, tot_acc))
            # if epoch % FLAGS.val_epochs == 0:
            val_loss, acc, f1, auc = valid_full(sess, model, batch, placeholders)
            print("Full train_loss %.4f train_acc %.4f val_loss %.4f val_acc %.4f val_f1 %.4f val_auc %.4f" %(tot_loss, tot_acc, val_loss, acc, f1, auc))
            # epoch_bar.set_postfix(loss=tot_loss, acc=tot_acc,
                                # val_loss=val_loss, val_f1=f1, val_auc=auc)
        print("Optimized finishied!")
        loss, acc, f1, auc = valid_full(sess, model, batch, placeholders, mode="test")
        write_result(loss, acc, f1, auc)
        print("Dataset %s Test set statistics: loss %.4f acc %.4f f1 %.4f auc %.4f" % (FLAGS.dataset, loss, acc, f1, auc))

    def save_models(self, sess):
        pass
    
    def restore_models(self, ckpt_path):
        pass