from __future__ import print_function
from __future__ import division
import os
import time
from collections import namedtuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import trange

from data_loader.minibatch import TemporalEdgeBatchIterator
from data_loader.neigh_samplers import UniformNeighborSampler
from data_loader.data_util import load_split_edges, load_label_edges
from model.aggregators import *
from model.gta import SAGEInfo
from model.inits import glorot, zeros
from model.layers import Dense, Layer
from model.models import GeneralizedModel
from model.prediction import BipartiteEdgePredLayer
from main import FLAGS
from model.utils import EarlyStopMonitor, get_free_gpu


class SAGETrainer():
    def __init__(self, edges, nodes, val_ratio, test_ratio):
        self.placeholders = self.construct_placeholders()
        self.nodes = nodes
        self.edges = edges
        # config model
        self.preprocess(edges, nodes, val_ratio=val_ratio,
                        test_ratio=test_ratio)
        self.params = {
            "dropout": FLAGS.dropout,
            "weight_decay": FLAGS.weight_decay,
            "epochs": FLAGS.epochs,
            "max_degree": FLAGS.max_degree
        }

    def log_dir(self):
        log_dir = FLAGS.base_log_dir + \
            "{}-{}".format(FLAGS.dataset, FLAGS.method)
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
            "dropout": tf.placeholder_with_default(0., shape=(), name="dropout")
        }
        return placeholders

    def config_tensorflow(self):
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        # Initialize session
        sess = tf.Session(config=config)
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.log_dir(), sess.graph)
        return sess, merged, summary_writer

    def preprocess(self, edges, nodes, val_ratio, test_ratio):
        batch = TemporalEdgeBatchIterator(edges, nodes, self.placeholders,
                                          val_ratio, test_ratio,
                                          batch_size=FLAGS.batch_size, max_degree=FLAGS.max_degree, context_size=FLAGS.context_size)
        self.batch = batch
        adj_info, ts_info, _ = batch.construct_adj(
            batch.edges.iloc[batch.train_idx])
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        self.model = GraphSAGE(
            self.placeholders, None, adj_info, ts_info, batch.degrees, layer_infos, sampler)

        self.sess, self.merged, self.summary_writer = self.config_tensorflow()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def train(self, epoch=0):
        self.batch.shuffle()
        tot_loss = tot_acc = 0
        placeholders = self.placeholders
        batch = self.batch
        batch_num = len(self.batch.train_idx) // FLAGS.batch_size
        with trange(batch_num, disable=(not FLAGS.display)) as batch_bar:
            self.sess.run(tf.local_variables_initializer())
            while not batch.end():
                batch_bar.set_description("batch %04d" % batch_num)
                feed_dict = batch.next_train_batch()
                feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                outs = self.sess.run(
                    [self.merged, self.model.opt_op, self.model.loss, self.model.acc_update, self.model.auc_update], feed_dict=feed_dict)
                # tot_loss += outs[2] * feed_dict[placeholders["batch_size"]]
                # tot_acc += outs[3] * feed_dict[placeholders["batch_size"]]
                loss, acc, auc = outs[2], outs[3], outs[4]

                batch_bar.update()
                batch_bar.set_postfix(loss=loss, auc=auc)
                self.summary_writer.add_summary(
                    outs[0], epoch * batch_num + self.batch.batch_num)

    def valid(self):
        placeholders = self.placeholders
        batch = self.batch
        batch_num = len(self.batch.val_idx) // FLAGS.batch_size

        self.sess.run(tf.local_variables_initializer())
        for feed_dict in batch.valid_batch():
            outs = self.sess.run(
                [self.model.acc_update, self.model.auc_update], feed_dict=feed_dict)
            acc, auc = outs[0], outs[1]
        return auc

    def test(self, edges):
        batch_num = len(edges) // (FLAGS.batch_size * 2)
        preds = []
        for feed_dict in self.batch.test_batch(edges):
            # # print(feed_dict)
            # break
            outs = self.sess.run(
                [self.model.pred_op], feed_dict=feed_dict)
            # outs = self.sess.run([self.model.samples_from], feed_dict=feed_dict)
            # print("samples_from", outs[0])
            # print("loss: %.2f" % outs[1])
            preds.extend(outs[0])
        return np.array(preds)

    def save_models(self, epoch):
        save_path = "saved_models/{dataset}/{use_context}-{context_size}-{dropout:.2f}.ckpt".format(
            dataset=FLAGS.dataset, context_size=FLAGS.context_size, use_context=FLAGS.use_context, dropout=FLAGS.dropout)
        if not os.path.exists("saved_models/{}".format(FLAGS.dataset)):
            os.makedirs("saved_models/{}".format(FLAGS.dataset))
            os.chmod("saved_models/{}".format(FLAGS.dataset), 0o777)
        if not hasattr(self, "saver"):
            self.saver = tf.train.Saver(max_to_keep=5)
        self.saver.save(self.sess, save_path, global_step=epoch)
        # os.chmod(save_path, 0o777)

    def restore_models(self, epoch):
        load_path = "saved_models/{dataset}/{use_context}-{dropout:.2f}.ckpt".format(
            dataset=FLAGS.dataset, use_context=FLAGS.use_context, dropout=FLAGS.dropout)
        print("Load model from path {}".format(load_path))
        if not hasattr(self, "saver"):
            self.saver = tf.train.Saver(max_to_keep=5)
        self.saver.restore(self.sess, f"{load_path}-{epoch}")


class GraphSAGE(GeneralizedModel):
    def __init__(self, placeholders, features, adj_info, ts_info, degrees, layer_infos, sampler, concat=False, aggregator_type="meanpool", embed_dim=128, **kwargs):
        super(GraphSAGE, self).__init__(**kwargs)

        self.eps = 1e-7
        self.margin = 0.1
        self.aggregator_cls = MeanPoolingAggregator
        self.config_placeholders(placeholders)

        self.adj_info, self.ts_info = adj_info, ts_info
        self.config_embeds(adj_info, features, embed_dim)
        self.layer_infos = layer_infos
        self.dims = [
            (0 if features is None else features.shape[1]) + embed_dim]
        self.dims.extend(
            [layer_infos[i].output_dim for i in range(len(layer_infos))])

        self.concat = FLAGS.concat
        self.placeholders = placeholders
        self.sampler = sampler
        self.degrees = degrees
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)
        self.build()

    def config_placeholders(self, placeholders):
        self.batch_from = placeholders["batch_from"]
        self.batch_to = placeholders["batch_to"]
        self.batch_neg = placeholders["batch_neg"]
        self.batch_size = placeholders["batch_size"]

    def config_embeds(self, adj_info, features, embed_dim):
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

    def sample(self, inputs, layer_infos, batch_size):
        samples = [inputs]
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1
            num_samples = layer_infos[t].num_samples
            node = self.sampler((samples[k], num_samples))
            samples.append(tf.reshape(node, [-1]))
            batch_size *= num_samples
        return samples

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
                # print("layer %d hop %d" % (layer, hop), neigh_dims)
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
        self.pred_op = self._predict(self.pos_scores)

    def _build(self):
        support_sizes = self.compute_support_sizes(self.layer_infos)
        forward_batches = [self.batch_from, self.batch_to, self.batch_neg]
        samples_from, samples_to, samples_neg = [self.sample(
            batch, self.layer_infos, FLAGS.batch_size) for batch in forward_batches]

        forward_aggs = self.config_aggregators(self.layer_infos, self.dims)
        num_samples = [layer.num_samples for layer in self.layer_infos]
        num_samples = num_samples[::-1]
        forward_samples = [samples_from, samples_to, samples_neg]
        forward_embeds = [self.aggregate(
            samples, self.dims, num_samples, support_sizes, forward_aggs, self.batch_size, concat=self.concat) for samples in forward_samples]
        self.embed_from, self.embed_to, self.embed_neg = forward_embeds[
            0], forward_embeds[1], forward_embeds[2]

        self.output_from = self.embed_from
        self.output_to = self.embed_to
        self.output_neg = self.embed_neg

        self.sample_ops = forward_samples
        self.embed_ops = forward_embeds
        self.aggs = forward_aggs

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
        else:
            raise NotImplementedError(
                "Loss {} not implemented yet.".format(FLAGS.loss))
        tf.summary.scalar("loss", self.loss)

    def _xent_loss(self, pos_scores, neg_scores):
        pos_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(pos_scores), logits=pos_scores)
        neg_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(neg_scores), logits=neg_scores)
        return tf.reduce_sum(pos_xent) + tf.reduce_sum(neg_xent)

    def _accuracy(self):
        """Use only for training set calculation.
        """
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
        tf.summary.scalar("acc", self.acc_update)
        tf.summary.scalar("auc", self.auc_update)

    def _predict(self, scores):
        if FLAGS.loss == "xent":
            return tf.reshape(tf.nn.sigmoid(scores), [-1])
        else:
            raise NotImplementedError(
                "Loss {} not implemented yet.".format(FLAGS.loss))


def write_result(vallabel, valpreds, label, preds, params):
    val_auc = roc_auc_score(vallabel, valpreds)
    acc = accuracy_score(label, preds > 0.5)
    f1 = f1_score(label, preds > 0.5)
    auc = roc_auc_score(label, preds)
    res_path = "comp_results/{}-{}.csv".format(FLAGS.dataset, "GraphSAGE")
    headers = ["method", "dataset", "valid_auc", "accuracy", "f1", "auc", "params"]
    if not os.path.exists(res_path):
        f = open(res_path, 'w')
        f.write(",".join(headers) + "\r\n")
        f.close()
        os.chmod(res_path, 0o777)
    with open(res_path, 'a') as f:
        result_str = "{},{},{:.4f},{:.4f},{:.4f},{:.4f}".format(
            "GraphSAGE", FLAGS.dataset, val_auc, acc, f1, auc)
        params_str = ",".join(["{}={}".format(k, v)
                               for k, v in params.items()])
        params_str = "\"{}\"".format(params_str)
        row = result_str + "," + params_str + "\r\n"
        print("{}-GraphSAGE: acc {:.3f}, f1 {:.3f}, auc {:.3f}".format(FLAGS.dataset,
                                                                       acc, f1, auc))
        f.write(row)


def main_sage(argv=None):
    print("Loading training data {}.".format(FLAGS.dataset))
    edges, nodes = load_split_edges(dataset=FLAGS.dataset)
    edges, nodes = edges[0], nodes[0]
    label_edges, _ = load_label_edges(dataset=FLAGS.dataset)
    label_edges = label_edges[0]
    print("Done loading training data.")
    trainer = SAGETrainer(edges, nodes, val_ratio=0.05, test_ratio=0.25)
    early_stopper = EarlyStopMonitor()
    for epoch in range(FLAGS.epochs):
        trainer.train(epoch=epoch)
        val_auc = trainer.valid()
        # trainer.save_models(epoch=epoch)
        print(f"val_auc: {val_auc}")
        if early_stopper.early_stop_check(val_auc):
            print(f"No improvement over {early_stopper.max_round} epochs")
            trainer.params["epochs"] = epoch
            # trainer.restore(epoch=epoch-2)
            break
    _, valid_edges, test_edges = label_edges
    validy = trainer.test(valid_edges)
    testy = trainer.test(test_edges)
    write_result(valid_edges["label"], validy, test_edges["label"], testy, trainer.params)


if __name__ == "__main__":
    tf.app.run(main=main_sage)
