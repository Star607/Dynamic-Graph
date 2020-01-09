import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
import tensorflow as tf

from data_loader.minibatch import load_data, TemporalEdgeBatchIterator
from data_loader.neigh_samplers import MaskNeighborSampler, TemporalNeighborSampler
from model.gta import GraphTemporalAttention, SAGEInfo

flags = tf.app.flags
FLAGS = flags.FLAGS


class AdaptorGTA(object):
    """A wrapper for GTA model.
    """

    def __init__(self):
        self.edges, self.nodes = load_data(
            datadir="../graph_data", dataset=FLAGS.dataset)
        self.placeholders = self.construct_placeholders()
        self.batch = TemporalEdgeBatchIterator(
            self.edges, self.nodes, self.placeholders, batch_size=FLAGS.batch_size, max_degree=FLAGS.max_degree, context_size=FLAGS.context_size)

        self.sess = None
        pass

    def construct_placeholders(self):
        # Define placeholders
        placeholders = {
            "batch_from": tf.placeholder(tf.int32, shape=(None), name="batch_from"),
            "batch_to": tf.placeholder(tf.int32, shape=(None), name="batch_to"),
            "timestamp": tf.placeholder(tf.float64, shape=(None), name="timestamp"),
            "batch_size": tf.placeholder(tf.int32, name="batch_size"),
            "context_from": tf.placeholder(tf.int32, shape=(None), name="context_from"),
            "context_to": tf.placeholder(tf.int32, shape=(None), name="context_to"),
            "context_timestamp": tf.placeholder(tf.float64, shape=(None), name="timestamp"),
        }
        return placeholders

    def train(self):
        pass

    def validate(self):
        pass

    def test(self):
        pass
