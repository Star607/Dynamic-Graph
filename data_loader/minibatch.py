"""Because we want to provide accessbility over temporal graphs, we use a new
store format:
    edges: (from_node_id, to_node_id, timestamp) for batch generation
    adjacency list: from_node_id: (to_node_id, timestamp) for support computation
"""
import os
from collections import namedtuple
from datetime import datetime

import numpy as np
import pandas as pd

import tensorflow.compat.v1 as tf
from tensorflow import keras
import tensorflow.keras.backend as K  # pylint: disable=import-error
from data_loader.neigh_samplers import UniformNeighborSampler, MaskNeighborSampler, TemporalNeighborSampler


flags = tf.app.flags
FLAGS = flags.FLAGS


def load_data(datadir="/nfs/zty/Graph/Dynamic-Graph/graph_data", dataset="CTDNE-fb-forum"):
    # ensure that node_id is stored as string format
    edges = pd.read_csv("{}/{}.edges".format(datadir, dataset))
    nodes = pd.read_csv("{}/{}.nodes".format(datadir, dataset))
    return edges, nodes


# SAGEInfo is a namedtuple that specifies the parameters
# of the recursive GraphSAGE layers
SAGEInfo = namedtuple("SAGEInfo",
                      ["layer_name",  # name of the layer (to get feature embedding etc.)
                       "num_samples",  # sampling neighbors
                       "output_dim"])


class TruncatedTemporalEdgeBatchIterator(object):
    """We use a dummy node with index 0, and use id2idx to transform all nodes into continuous integers.

    Edges are sorted in temporal ascending order.

    Adjacency list are stored in two arrays, one for neighbors, another for the corresponding timestamps.
    """

    def __init__(self, edges, nodes, placeholders, batch_size=512, context_size=25, max_degree=100, neg_sample_size=25, **kwargs):
        """Nodes is padded with a dummy node: 0, neg_samples are implemented in models.
        """
        # id start from 1, avoiding collides with the dummy node 0
        self.id2idx = {row["node_id"]: index + 1
                       for index, row in nodes.iterrows()}
        self.id2idx["padding_node"] = 0

        edges_full = self._node_prune(edges)
        edges_full["from_node_id"] = edges_full["from_node_id"].map(
            self.id2idx)
        edges_full["to_node_id"] = edges_full["to_node_id"].map(self.id2idx)

        # padding with an edge (0, 0, 0.0, 0.0, ...)
        dtypes = edges_full.dtypes
        # Assure the dtype consistency
        dtypes[["from_node_id", "to_node_id"]] = int
        edges_full.loc[len(edges_full)] = [0] * len(edges_full.columns)
        edges_full = edges_full.astype(dtypes).sort_values(by="timestamp")

        # assert timestamp is increasing
        ts_delta = edges_full["timestamp"].shift(-1) - edges_full["timestamp"]
        assert (np.all(ts_delta[:len(ts_delta) - 1] >= 0))

        self.id_name = ["from_node_id", "to_node_id", "timestamp"]
        self.edges = edges_full[self.id_name]
        # print("edges dataframe info:")
        # print(edges.info())
        if len(self.id_name) == len(edges.columns):
            self.edge_features = None
        else:
            self.edge_features = edges_full[[
                f for f in edges_full.columns if f not in self.id_name]].to_numpy()
        assert(not np.any(self.edges.isnull()))

        self.placeholders = placeholders
        self.batch_size = batch_size
        self.context_size = context_size
        self.max_degree = max_degree
        self.adj_ids, self.adj_tss, self.deg = self.construct_adj()
        self.neg_sample_size = neg_sample_size
        self.train_test_split()
        self.batch_num = 0

    def _node_prune(self, edges, min_score=5):
        edges = edges.sort_values(by="timestamp")
        from_id_counts = dict(edges["from_node_id"].value_counts())
        to_id_counts = dict(edges["to_node_id"].value_counts())
        ids = set(from_id_counts.keys()).union(set(to_id_counts.keys()))
        id_counts = {k: from_id_counts.get(
            k, 0) + to_id_counts.get(k, 0) for k in ids}
        prune_ids = set(filter(lambda s: id_counts[s] < min_score, id_counts))
        print("********Remove %d nodes less than 5-score********" %
              len(prune_ids))
        edges = edges[edges["from_node_id"].apply(
            lambda s: s not in prune_ids)]
        edges = edges[edges["to_node_id"].apply(lambda s: s not in prune_ids)]

        reserve_ids = ids - prune_ids
        print("********Finally, we get %d edges and %d nodes********" %
              (len(edges), len(reserve_ids)))
        return edges

    def construct_adj(self):
        """Construct truncated adjacency list and each entry is sorted in temporal order except padding zeros.

        Return:
            a truncated adjacency list containing neighbor ids
            a truncated adjacency list containing edge timestamps
        """
        adj_ids_list = [[] for _ in range(len(self.id2idx))]
        adj_tss_list = [[] for _ in range(len(self.id2idx))]
        # print(len(self.id2idx), len(self.id2idx))
        # Attention! df.iterrows will change dtype for each column!
        for row in self.edges.itertuples():
            from_id, to_id, ts = row.from_node_id, row.to_node_id, row.timestamp
            if from_id >= len(self.id2idx) or to_id >= len(self.id2idx):
                print(row)
            adj_ids_list[from_id].append(to_id)
            adj_tss_list[from_id].append(ts)
            adj_ids_list[to_id].append(from_id)
            adj_tss_list[to_id].append(ts)

        adj_ids = np.zeros((len(self.id2idx), self.max_degree), dtype=np.int32)
        adj_tss = np.zeros(
            (len(self.id2idx), self.max_degree), dtype=np.float64)
        deg = np.zeros(len(self.id2idx), dtype=np.int64)
        for i in range(len(self.id2idx)):
            deg[i] = len(adj_ids_list[i])
            if deg[i] == 0:
                continue
            replace = deg[i] < self.max_degree
            # print(deg[i])
            indices = sorted(np.random.choice(
                deg[i], self.max_degree, replace=replace))
            # to-do: padding with zero?
            adj_ids[i, :] = np.array(adj_ids_list[i])[indices]
            adj_tss[i, :] = np.array(adj_tss_list[i])[indices]
        return adj_ids, adj_tss, deg

    def train_test_split(self, val_ratio=0.1, test_ratio=0.2):
        train_ratio = 1 - val_ratio - test_ratio

        train_end_idx = int(len(self.edges) * train_ratio)
        val_end_idx = int(len(self.edges) * val_ratio) + train_end_idx
        self.train_idx = list(range(train_end_idx))
        self.val_idx = list(range(train_end_idx, val_end_idx))
        self.test_idx = list(range(val_end_idx, len(self.edges)))

    def neg_edges(self, batch_edges):
        pass

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_idx)

    def next_train_batch(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_idx))
        batch_idx = self.train_idx[start_idx:end_idx]
        return self.batch_feed_dict(batch_idx)

    def get_context_edges(self, batch_idx):
        """
        Returns:
            context_edges: (batch_size, context_size)
        """
        context_idx = [np.arange(idx-self.context_size, idx)
                       for idx in batch_idx]
        context_idx = np.maximum(context_idx, 0)
        context_edges = [self.edges.iloc[indices] for indices in context_idx]
        return context_edges

    def batch_feed_dict(self, batch_idx):
        # edges has been dropped, but their indices didn't change, so it is suitable to use DataFrame.iloc[]
        batch_edges = self.edges.iloc[batch_idx]
        batch_from = batch_edges["from_node_id"].tolist()
        batch_to = batch_edges["to_node_id"].tolist()
        timestamp = batch_edges["timestamp"].tolist()

        context_edges = self.get_context_edges(batch_idx)
        context_from = [df["from_node_id"].tolist() for df in context_edges]
        context_to = [df["to_node_id"].tolist() for df in context_edges]
        context_timestamp = [df["timestamp"].tolist() for df in context_edges]

        feed_dict = dict()
        # for eager execution debug
        # feed_dict["batch_from"] = batch_from
        # feed_dict["timestamp"] = timestamp
        # feed_dict.update({self.placeholders["batch_size"]: self.batch_size})
        feed_dict.update({self.placeholders["batch_from"]: batch_from})
        feed_dict.update({self.placeholders["batch_to"]: batch_to})
        feed_dict.update({self.placeholders["timestamp"]: timestamp})

        # feed_dict.update(
        # {self.placeholders["context_size"]: self.context_size})
        feed_dict.update({self.placeholders["context_from"]: context_from})
        feed_dict.update({self.placeholders["context_to"]: context_to})
        feed_dict.update(
            {self.placeholders["context_timestamp"]: context_timestamp})

        return feed_dict

    def val_feed_dict(self, size=None):
        val_idx = np.random.permutation(self.val_idx)
        return self.batch_feed_dict(val_idx)

    def test_feed_dict(self):
        test_idx = np.random.permutation(self.test_idx)
        return self.batch_feed_dict(test_idx)

    def shuffle(self):
        self.train_idx = np.random.permutation(self.train_idx)
        self.batch_num = 0


class SparseTemporalEdgeBatchIterator(object):
    pass


class TemporalNodeBatchIterator(object):
    pass


def compute_support_sizes(batch_size, layer_infos):
    support_size = batch_size
    support_sizes = [support_size]
    for k in range(len(layer_infos)):
        t = len(layer_infos) - k - 1
        support_size *= layer_infos[t].num_samples
        support_sizes.append(support_size)
    return support_sizes


if __name__ == "__main__":
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True  # pylint: disable=no-member
    config.allow_soft_placement = True
    # tf.enable_eager_execution(config=config)

    np.random.seed(42)
    tf.set_random_seed(42)

    start_time = datetime.now()
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    placeholders = {
        "batch_from": tf.placeholder(tf.int32, shape=(None), name="batch_from"),
        "batch_to": tf.placeholder(tf.int32, shape=(None), name="batch_to"),
        "timestamp": tf.placeholder(tf.float64, shape=(None), name="timestamp"),
        # "batch_size": tf.placeholder(tf.int32, name="batch_size"),
        "context_from": tf.placeholder(tf.int32, shape=(None), name="context_from"),
        "context_to": tf.placeholder(tf.int32, shape=(None), name="context_to"),
        "context_timestamp": tf.placeholder(tf.float64, shape=(None), name="timestamp"),
        # "context_size": tf.placeholder(tf.int32, name="context_size")
    }

    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("../log", sess.graph)

    # edges, nodes = load_data(dataset="CTDNE-fb-forum")
    edges, nodes = load_data(dataset="JODIE-reddit")
    batch = TruncatedTemporalEdgeBatchIterator(edges, nodes, placeholders)
    batch.shuffle()
    print("Preprocessing time:", datetime.now() - start_time)

    # construct graphsage computation graph
    sampler = MaskNeighborSampler(
        adj_info=batch.adj_ids, ts_info=batch.adj_tss)
    # sampler = TemporalNeighborSampler(
    #     adj_info=batch.adj_ids, ts_info=batch.adj_tss)
    layer_infos = [
        SAGEInfo("sample_1", 10, 128),
        SAGEInfo("sample_2", 5, 128)
    ]

    support_sizes = compute_support_sizes(FLAGS.batch_size, layer_infos)
    print("support_sizes", support_sizes)
    sample_1, sample_ts_1 = sampler(
        (placeholders["batch_from"], placeholders["timestamp"], support_sizes[0], layer_infos[1].num_samples))
    sample_2, sample_ts_2 = sampler(
        (sample_1, sample_ts_1, support_sizes[1], layer_infos[0].num_samples))

    print("Computation graph time:", datetime.now() - start_time)
    it = 0
    start_time = datetime.now()
    sess.run(tf.global_variables_initializer())
    while not batch.end():  # and it <= 10:
        it += 1
        last_time = datetime.now()
        feed_dict = batch.next_train_batch()
        # if (batch.batch_num + 1) * batch.batch_size < len(batch.train_idx):
        #     continue
        # print("tf running")

        # values = sess.run(list(placeholders.values()), feed_dict=feed_dict)
        # values = sess.run([sample_1, sample_ts_1], feed_dict=feed_dict)
        values = sess.run([sample_2, sample_ts_2], feed_dict=feed_dict)
        # if it % 10 == 0:
        print("********Batch: %d time: %d secs********" %
              (batch.batch_num, (datetime.now() - last_time).seconds))
        # it += 1
    print(datetime.now() - start_time)
