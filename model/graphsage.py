import os
import time
from collections import namedtuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tensorflow.layers import conv1d
from tqdm import trange

from data_loader.minibatch import (TemporalEdgeBatchIterator,
                                   TemporalNeighborSampler, load_data)
from data_loader.neigh_samplers import (MaskNeighborSampler,
                                        TemporalNeighborSampler)
from model.aggregators import *
from model.gta import GraphTemporalAttention, SAGEInfo
from model.inits import glorot, zeros
from model.layers import Dense, Layer
from model.models import GeneralizedModel
from model.prediction import BipartiteEdgePredLayer

flags = tf.app.flags
FLAGS = flags.FLAGS


class ModelTrainer():
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
            "use_context": FLAGS.use_context,
            "epochs": FLAGS.epochs,
            "context_size": FLAGS.context_size,
            "sampler": FLAGS.sampler,
            "dynamic_neighbor": FLAGS.dynamic_neighbor,
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
        # placeholders = construct_placeholders()
        # test_edges are both positive and negative samples
        self.batch = TemporalEdgeBatchIterator(edges, nodes, self.placeholders,
                                               val_ratio, test_ratio,
                                               batch_size=FLAGS.batch_size, max_degree=FLAGS.max_degree, context_size=FLAGS.context_size)

        adj_info, ts_info = self.batch.adj_ids, self.batch.adj_tss
        if FLAGS.sampler == "mask":
            sampler = MaskNeighborSampler(adj_info, ts_info)
        elif FLAGS.sampler == "temporal":
            sampler = TemporalNeighborSampler(adj_info, ts_info)
        else:
            raise NotImplementedError(
                "Sampler %s not supported." % FLAGS.sampler)

        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        ctx_layer_infos = [
            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        self.model = GraphTemporalAttention(
            self.placeholders, None, adj_info, ts_info, self.batch.degrees, layer_infos,
            ctx_layer_infos, sampler, bipart=self.batch.bipartite, n_users=self.batch.n_users)

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
        # when test_ratio is set, the context edges are also determined
        # assert timestamp is increasing
        ts_delta = edges["timestamp"].shift(-1) - edges["timestamp"]
        assert (np.all(ts_delta[:len(ts_delta) - 1] >= 0))
        if not FLAGS.dynamic_neighbor:
            edges["timestamp"] = min(edges["timestamp"])

        batch_num = len(edges) // (FLAGS.batch_size * 2)
        # print("test_idx %d test_edges %d" %
        #   (len(self.batch.test_idx), len(edges)))
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

    def save_models(self):
        save_path = "saved_models/{dataset}/{use_context}-{context_size}-{dropout:.2f}.ckpt".format(
            dataset=FLAGS.dataset, context_size=FLAGS.context_size, use_context=FLAGS.use_context, dropout=FLAGS.dropout)
        if not os.path.exists("saved_models/{}".format(FLAGS.dataset)):
            os.makedirs("saved_models/{}".format(FLAGS.dataset))
            os.chmod("saved_models/{}".format(FLAGS.dataset), 0o777)
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(self.sess, save_path)
        #os.chmod(save_path, 0o777)

    def restore_models(self):
        load_path = "saved_models/{dataset}/{use_context}-{dropout:.2f}.ckpt".format(
            dataset=FLAGS.dataset, use_context=FLAGS.use_context, dropout=FLAGS.dropout)
        print("Load model from path {}".format(load_path))
        saver = tf.train.Saver()
        saver.restore(self.sess, load_path)
