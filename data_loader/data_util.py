import os
from collections import defaultdict, namedtuple
from datetime import datetime

import keras
import keras.backend as K
import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf
from numba import jit

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("max_degree", 100, "maximum node degree")
flags.DEFINE_integer("samples_1", 25, "number of samples in layer 1")
flags.DEFINE_integer("samples_2", 10, "number of user samples in layer 2")
flags.DEFINE_integer(
    "dim_1", 128, "Size of output dim (final is 2x this, if using concat)")
flags.DEFINE_integer(
    "dim_2", 128, "Size of output dim (final is 2x this, if using concat)")
flags.DEFINE_integer("neg_sample_size", 20, "number of negative samples")
flags.DEFINE_integer("batch_size", 512, "minibatch size.")


def load_data(datadir="../graph_data", dataset="CTDNE-fb-forum"):
    # ensure that node_id is stored as string format
    edges = pd.read_csv("{}/{}.edges".format(datadir, dataset))
    nodes = pd.read_csv("{}/{}.nodes".format(datadir, dataset))
    return edges, nodes


# SAGEInfo is a namedtuple that specifies the parameters
# of the recursive GraphSAGE layers
SAGEInfo = namedtuple("SAGEInfo",
                      ["layer_name",  # name of the layer (to get feature embedding etc.)
                       "num_samples",
                       "output_dim"])


class BulkTemporalEdgeBatchIterator(object):
    """Given layer_infos, generate a batch of support_fields: 
        Firstly, generate a batch of temporal edges (batch_size, from_node_id, to_node_id, timestamp)

        Secondly, according to layer_infos, 
            generate support_fields for (from_node_id, timestamp) and (to_node_id, timestamp)
            generate support_fields for temporal context edges:
            each consists of (from_node_id, timestamp) and (to_node_id, timestamp)                   

        Finally, a batch of support_fields has a size of :
            Layer i: (batch_size, window_size, support_sizes[i])
    """

    def __init__(self, edges, placeholders, layer_infos, context_layer_infos, batch_size=512, context_window=5):
        self.Edge = namedtuple(
            "Edge", ["from_node_id", "to_node_id", "timestamp"])
        self.edges, self.nodes_id = self._node_prune(edges)
        self.Adj = namedtuple("Adj", ["to_node_id", "timestamp"])
        self.adj = self.construct_adj()
        self.placeholders = placeholders
        self.layer_infos = layer_infos
        self.context_layer_infos = context_layer_infos
        self.batch_size = batch_size
        self.context_window = context_window
        self.support_sizes = compute_support_sizes(layer_infos)
        self.context_support_sizes = compute_support_sizes(context_layer_infos)

        self.train_test_split()
        self.batch_num = 0

    def _node_prune(self, edges, min_score=5):
        edges = edges.sort_values(by="timestamp")
        from_id_counts = dict(edges["from_node_id"].value_counts())
        to_id_counts = dict(edges["to_node_id"].value_counts())
        keys = set(from_id_counts.keys()).union(set(to_id_counts.keys()))
        id_counts = {k: from_id_counts.get(
            k, 0) + to_id_counts.get(k, 0) for k in keys}
        prune_ids = set(filter(lambda s: id_counts[s] < min_score, id_counts))
        print("********Remove nodes less than 5-score********")
        edges = edges[edges["from_node_id"].apply(
            lambda s: s not in prune_ids)]
        edges = edges[edges["to_node_id"].apply(lambda s: s not in prune_ids)]

        new_edges = []
        print("********Discard edge attributes if exists.********")
        for _, row in edges.iterrows():
            new_edges.append(
                self.Edge(row["from_node_id"], row["to_node_id"], row["timestamp"]))

        nodes_id = set(e.from_node_id for e in new_edges).union(
            set(e.from_node_id for e in new_edges))
        print("********Finally, we get %d edges and %d nodes" %
              (len(new_edges), len(nodes_id)))
        return pd.Series(new_edges), nodes_id

    def construct_adj(self, bidirection=True):
        # construct adjacency list: node_id: Edge(neighbor_node_id, timestamp),
        adj = {k: [] for k in self.nodes_id}
        adj[0] = []  # add a setinel node
        if bidirection:
            print("********By default, graph is set as bidirectional.********")
        for from_node_id, to_node_id, timestamp in self.edges:
            adj[from_node_id].append(self.Adj(to_node_id, timestamp))
            if bidirection:
                adj[to_node_id].append(self.Adj(from_node_id, timestamp))

        return adj

    def train_test_split(self, val_ratio=0.1, test_ratio=0.2):
        train_ratio = 1 - val_ratio - test_ratio

        train_end_idx = int(len(self.edges) * train_ratio)
        val_end_idx = int(len(self.edges) * val_ratio) + train_end_idx
        self.train_idx = list(range(train_end_idx))
        self.val_idx = list(range(train_end_idx, val_end_idx))
        self.test_idx = list(range(val_end_idx, len(self.edges)))

    def neg_edges(self, edges, adj, nodes):
        pass

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_idx)

    def next_train_batch(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_idx))
        batch_idx = self.train_idx[start_idx:end_idx]
        return self.batch_feed_dict(batch_idx)

    # extract node_id and eliminate timestamp
    def extract_node_id(self, arr):
        if isinstance(arr, tuple):
            return arr[0]
        elif isinstance(arr, list) or isinstance(arr, np.ndarray):
            return [self.extract_node_id(i) for i in arr]
        else:
            return arr

    def batch_feed_dict(self, batch_idx):
        # [layer_i, batch_size, 1, support_sizes[i]]
        batch_edges = self.edges[batch_idx]
        batch_from = [[] for _ in range(len(self.layer_infos))]
        batch_to = [[] for _ in range(len(self.layer_infos))]
        for from_id, to_id, timestamp in batch_edges:
            from_samples = self.sample(from_id, timestamp)
            to_samples = self.sample(to_id, timestamp)
            for k in range(len(self.layer_infos)):
                batch_from[k].append(from_samples[k])
                batch_to[k].append(to_samples[k])

        # [layer_i, batch_size, context_window, support_sizes[i]]
        batch_context_edges = []
        for edge_idx in batch_idx:
            start_idx = max(edge_idx - self.context_window, 0)
            tmp = self.edges[start_idx: edge_idx]
            if len(tmp) < self.context_window:
                pad = pd.Series([self.Edge(0, 0, 0)] *
                                (self.context_window - len(tmp)))
                if len(tmp) == 0:
                    tmp = pad
                else:
                    tmp = np.hstack([pad, tmp])
            batch_context_edges.append(tmp)

        batch_context_from = [[] for _ in range(len(self.context_layer_infos))]
        batch_context_to = [[] for _ in range(len(self.context_layer_infos))]
        for context_edges in batch_context_edges:
            # [context_window, layer_i ,support_sizes[i]]
            context_from = [self.sample(from_id, ts)
                            for from_id, _, ts in context_edges]
            context_to = [self.sample(to_id, ts)
                          for _, to_id, ts in context_edges]
            for k in range(len(self.context_layer_infos)):
                tmp_from = [arr[k] for arr in context_from]
                batch_context_from[k].append(tmp_from)
                tmp_to = [arr[k] for arr in context_to]
                batch_context_to[k].append(tmp_to)

        batch_from = self.extract_node_id(batch_from)
        batch_to = self.extract_node_id(batch_to)

        batch_context_from = self.extract_node_id(batch_context_from)
        batch_context_to = self.extract_node_id(batch_context_to)

        feed_dict = dict()
        for k in range(len(self.layer_infos)):
            feed_dict.update(
                {self.placeholders["batch_from_%d" % k]: batch_from[k]})
            feed_dict.update(
                {self.placeholders["batch_to_%d" % k]: batch_to[k]})
        for k in range(len(self.context_layer_infos)):
            feed_dict.update(
                {self.placeholders["batch_context_from_%d" % k]: batch_context_from[k]})
            feed_dict.update(
                {self.placeholders["batch_context_to_%d" % k]: batch_context_to[k]})
        feed_dict[self.placeholders["label"]] = [1] * self.batch_size
        return feed_dict

    def sample(self, node_id, timestamp):
        samples = [[(node_id, timestamp)]]
        for k in range(len(self.layer_infos)):
            t = len(self.layer_infos) - k - 1
            num_samples = self.layer_infos[t].num_samples
            nodes = []
            for itid, ts in samples[k]:
                adj_list = [el for el in self.adj[itid] if el[1] < ts]
                replace = len(adj_list) < num_samples
                if len(adj_list) == 0:
                    adj_list = [self.Adj(0, 0)]
                adj_list = pd.Series(adj_list)
                nodes.extend(np.random.choice(
                    adj_list, size=num_samples, replace=replace))
            samples.append(nodes)
        return samples

    def val_feed_dict(self, size=None):
        edge_list = self.val_idx
        if size is None:
            return self.batch_feed_dict(edge_list)
        else:
            ind = np.random.permutation(len(edge_list))
            val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
            return self.batch_feed_dict(val_edges)

    def test_feed_dict(self):
        return self.batch_feed_dict(self.test_idx)

    def shuffle(self):
        # self.train_edges = np.random.permutation(self.train_edges)
        self.train_idx = np.random.permutation(self.train_idx)
        self.batch_num = 0


def compute_support_sizes(layer_infos):
    support_size = 1
    support_sizes = [support_size]
    for k in range(len(layer_infos)):
        t = len(layer_infos) - k - 1
        support_size *= layer_infos[t].num_samples
        support_sizes.append(support_size)
    return support_sizes


if __name__ == "__main__":
    layer_infos = [SAGEInfo("node", FLAGS.samples_1, FLAGS.dim_1),
                   SAGEInfo("node", FLAGS.samples_2, FLAGS.dim_2)]
    support_sizes = compute_support_sizes(layer_infos)
    context_window = 5
    context_layer_infos = layer_infos
    context_support_sizes = support_sizes

    placeholders = {}
    for i, layer in enumerate(layer_infos):
        placeholders["batch_from_%d" % i] = tf.placeholder(
            tf.int32, shape=(None, support_sizes[i]), name="batch_from_%d" % i)
        placeholders["batch_to_%d" % i] = tf.placeholder(
            tf.int32, shape=(None, support_sizes[i]), name="batch_to_%d" % i)
        placeholders["batch_context_from_%d" % i] = tf.placeholder(tf.int32, shape=(
            None, context_window, support_sizes[i]), name="batch_from_%d" % i)
        placeholders["batch_context_to_%d" % i] = tf.placeholder(tf.int32, shape=(
            None, context_window, support_sizes[i]), name="batch_to_%d" % i)
        placeholders["label"] = tf.placeholder(
            tf.int32, shape=(None), name="label")
    placeholders["dropout"] = tf.placeholder_with_default(
        0., shape=(), name="dropout")

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True  # pylint: disable=no-member
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    megred = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("../log/", sess.graph)
    sess.run(tf.global_variables_initializer())

    edges, nodes = load_data(dataset="CTDNE-fb-forum")
    batch = BulkTemporalEdgeBatchIterator(
        edges, placeholders, layer_infos, context_layer_infos, batch_size=FLAGS.batch_size)
    batch.shuffle()

    t = 0
    start_time = datetime.now()
    while not batch.end() and t < 10:
        last_time = datetime.now()
        feed_dict = batch.next_train_batch()
        feed_dict.update({placeholders["dropout"]: 1.0})
        values = sess.run(list(placeholders.values()), feed_dict=feed_dict)
        print("********Batch: %d time: %d secs********" %
              (batch.batch_num, (datetime.now() - last_time).seconds))
        t += 1
        # pprint(values)
    print(datetime.now() - start_time)
