import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.layers import conv1d
from collections import namedtuple
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from model.aggregators import *
from model.inits import glorot, zeros
from model.layers import Layer, Dense
from model.models import GeneralizedModel
from model.prediction import BipartiteEdgePredLayer
from data_loader.minibatch import TemporalNeighborSampler
from data_loader.minibatch import load_data, TemporalEdgeBatchIterator
from data_loader.neigh_samplers import MaskNeighborSampler, TemporalNeighborSampler
from model.gta import GraphTemporalAttention, SAGEInfo

flags = tf.app.flags
FLAGS = flags.FLAGS


class ModelTrainer():
    def __init__(self, edges, nodes, train_edges, test_edges):
        self.placeholders = self.construct_placeholders()
        self.nodes = nodes
        self.edges = edges
        # config model
        self.preprocess(edges, nodes, train_edges, test_edges)
        self.params = {
            "dropout": FLAGS.dropout,
            "weight_decay": FLAGS.weight_decay,
            "use_context": FLAGS.use_context
        }


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

    def preprocess(self, edges, nodes, train_edges, test_edges):

        # placeholders = construct_placeholders()
        # test_edges are both positive and negative samples
        self.batch = TemporalEdgeBatchIterator(edges, nodes, self.placeholders, 
                                        len(train_edges), len(test_edges) // 2,
                                        batch_size=FLAGS.batch_size, max_degree=FLAGS.max_degree, context_size=FLAGS.context_size)

        adj_info, ts_info = self.batch.adj_ids, self.batch.adj_tss
        if FLAGS.sampler == "mask":
            sampler = MaskNeighborSampler(adj_info, ts_info)
        elif FLAGS.sampler == "temporal":
            sampler = TemporalNeighborSampler(adj_info, ts_info)
        else:
            raise NotImplementedError("Sampler %s not supported." % FLAGS.sampler)

        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                    SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        ctx_layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
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
        with trange(batch_num, disable=True) as batch_bar:
            while not batch.end():
                batch_bar.set_description("batch %04d" % batch_num)
                feed_dict = batch.next_train_batch()
                feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                outs = self.sess.run(
                    [self.merged, self.model.opt_op, self.model.loss, self.model.acc, self.model.auc], feed_dict=feed_dict)
                tot_loss += outs[2] * feed_dict[placeholders["batch_size"]]
                tot_acc += outs[3] * feed_dict[placeholders["batch_size"]]
                loss, acc = outs[2], outs[3]
               
                batch_bar.update()
                batch_bar.set_postfix(loss=loss, acc=acc)
                self.summary_writer.add_summary(
                    outs[0], epoch * batch_num + self.batch.batch_num)

    def test(self, edges):
        # when test_ratio is set, the context edges are also determined
        # assert timestamp is increasing
        ts_delta = edges["timestamp"].shift(-1) - edges["timestamp"]
        assert (np.all(ts_delta[:len(ts_delta) - 1] >= 0))

        id2idx = {row["node_id"]: row["id_map"]
                       for index, row in self.nodes.iterrows()}
        id2idx["padding_node"] = 0
        edges["from_node_id"] = edges["from_node_id"].map(id2idx)
        edges["to_node_id"] = edges["to_node_id"].map(id2idx)
        batch_num = len(edges) // (FLAGS.batch_size * 2)
        preds = []
        for feed_dict in self.batch.test_batch(edges):
            # # print(feed_dict)
            # break
            outs = self.sess.run([self.model.pred_op, self.model.loss], feed_dict=feed_dict)
            # outs = self.sess.run([self.model.samples_from], feed_dict=feed_dict)
            # print("samples_from", outs[0])
            # print("loss: %.2f" % outs[1])
            preds.extend(outs[0])
        return np.array(preds)

    def save_models(self):
        save_path = "saved_models/{dataset}/{use_context}-{dropout:.2f}.ckpt".format(dataset=FLAGS.dataset, use_context=FLAGS.use_context, dropout=FLAGS.dropout)
        if not os.path.exists("saved_models/{}".format(FLAGS.dataset)):
            os.makedirs("saved_models/{}".format(FLAGS.dataset))
            os.chmod("saved_models/{dataset}".format(dataset=FLAGS.dataset), 0o777)
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(self.sess, save_path)
        #os.chmod(save_path, 0o777)
    
    def restore_models(self):
        load_path = "saved_models/{dataset}/{use_context}-{dropout:.2f}.ckpt".format(dataset=FLAGS.dataset, use_context=FLAGS.use_context, dropout=FLAGS.dropout)
        print("Load model from path {}".format(load_path))
        saver = tf.train.Saver()
        saver.restore(self.sess, load_path)
