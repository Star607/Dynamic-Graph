from __future__ import division
from __future__ import print_function

from model.layers import Layer

import numpy as np
import tensorflow as tf
# flags = tf.app.flags
# FLAGS = flags.FLAGSs


"""
Classes that are used to sample node neighborhoods
"""


class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """

    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def _call(self, inputs):
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        adj_lists = tf.slice(adj_lists, [0, 0], [-1, num_samples])
        return adj_lists


# class MaskNeighborSampler(Layer):
#     """Mask edges of adjacency list later than current edge timestamp.
#     """

#     def __init__(self, adj_info, ts_info, **kwargs):
#         super(MaskNeighborSampler, self).__init__(**kwargs)
#         self.adj_info = tf.Variable(adj_info, trainable=False, name="adj_info")
#         self.ts_info = tf.Variable(ts_info, trainable=False, name="ts_info")

#     def _call(self, inputs):
#         ids, tss, batch_size, num_samples = inputs
#         adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
#         # adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
#         ts_lists = tf.nn.embedding_lookup(self.ts_info, ids)
#         # shape: (num_ids, max_degree) < (num_ids, 1)
#         # expand dims according to numpy broadcast rules
#         mask = tf.less(ts_lists, tf.expand_dims(tss, axis=1))
#         return neighbors, tf.cast(mask, dtype=tf.float32)

class MaskNeighborSampler(Layer):
    """Mask edges of adjacency list later than current edge timestamp.
    """

    def __init__(self, adj_info, ts_info, **kwargs):
        super(MaskNeighborSampler, self).__init__(**kwargs)
        # self.adj_info = adj_info
        # self.ts_info = ts_info
        self.adj_info = tf.Variable(adj_info, trainable=False, name="adj_info")
        self.ts_info = tf.Variable(ts_info, trainable=False, name="ts_info")

    def _call(self, inputs):
        # Attention! the last batch is always smaller than a normal batch
        # batch_size >= tf.shape(ids)[0]
        ids, tss, batch_size, num_samples = inputs
        print("batch_size:", batch_size)
        num_ids = tf.shape(ids)[0]
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
        ts_lists = tf.nn.embedding_lookup(self.ts_info, ids)
        # shape: (num_ids, max_degree) < (num_ids, 1)
        # expand dims according to numpy broadcast rules
        mask = tf.less(ts_lists, tf.expand_dims(tss, axis=1))
        assert_op = tf.debugging.Assert(
            tf.equal(tf.shape(tf.shape(mask))[-1], 2), [tf.shape(ids), tf.shape(adj_lists), tf.shape(mask)])
        with tf.control_dependencies([assert_op]):
            indices = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
        # Attention! the last batch is always smaller than a normal batch
        # Padding indices to a vector of batch_size length
        indices = tf.concat(
            [indices, tf.zeros([batch_size-num_ids], dtype=tf.int32)], axis=0)
        # avoid tf.random.uniform ```minval < maxval```
        err_indices = tf.maximum(indices, 1)
        # shape: (batch_size * num_samples,)
        col_idx = tf.reshape([tf.random.uniform(shape=(1, num_samples), minval=0,
                                                maxval=err_indices[i], dtype=tf.int32)
                              for i in range(batch_size)], shape=[-1])
        # Attention! tf.map_fn is too slow!
        # def uniform_fn(maxval): return tf.random.uniform(shape=(1, num_samples), minval=0,
        #                                                  maxval=maxval, dtype=tf.int32)
        # col_idx = tf.reshape(
        #     tf.vectorized_map(uniform_fn, err_indices), shape=[-1])
        row_idx = tf.tile(tf.range(batch_size), [num_samples])
        row_idx = tf.reshape(tf.transpose(tf.reshape(
            row_idx, shape=[-1, batch_size])), shape=[-1])
        # shape: (batch_size * num_samples, 2)
        ids = tf.stack([row_idx, col_idx], axis=1)
        # Attention! the last batch is always smaller than a normal batch
        ids = tf.reshape(ids, [-1, num_samples, 2])[:num_ids]
        ids = tf.reshape(ids, [-1, 2])
        # replace those indices at 0 with node 0 and ts 0
        mask_ids = tf.tile(tf.greater(indices[:num_ids], 0), [num_samples])
        mask_ids = tf.reshape(tf.transpose(
            tf.reshape(mask_ids, shape=[num_samples, -1])), shape=[-1])
        neighbors = tf.cast(mask_ids, adj_lists.dtype) * \
            tf.gather_nd(adj_lists, ids)
        tss = tf.cast(mask_ids, ts_lists.dtype) * tf.gather_nd(ts_lists, ids)
        return neighbors, tss


class TemporalNeighborSampler(Layer):
    """Slice a window of adjacency lists of nodes on different timestamps.
    """

    def __init__(self, adj_info, ts_info, **kwargs):
        super(TemporalNeighborSampler, self).__init__(**kwargs)
        # self.adj_info = adj_info
        # self.ts_info = ts_info
        self.adj_info = tf.Variable(adj_info, trainable=False, name="adj_info")
        self.ts_info = tf.Variable(ts_info, trainable=False, name="ts_info")

    def _call(self, inputs):
        ids, tss, batch_size, num_samples = inputs
        num_ids = tf.shape(ids)[0]

        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
        ts_lists = tf.nn.embedding_lookup(self.ts_info, ids)
        # expand dims of tss according to numpy broadcast rule
        mask = tf.less(ts_lists, tf.expand_dims(tss, axis=1))
        indices = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
        # Attention! the last batch is always smaller than a normal batch
        # Padding indices to a vector of batch_size length
        indices = tf.concat(
            [indices, tf.zeros([batch_size-num_ids], dtype=tf.int32)], axis=0)
        col_idx = tf.concat([tf.range(indices[i]-num_samples, indices[i])
                             for i in range(batch_size)], axis=0)
        # Attention! tf.map_fn is too slow
        # def range_fn(idx): return tf.range(idx-num_samples, idx)
        # col_idx = tf.reshape(tf.map_fn(range_fn, indices), shape=[-1])
        row_idx = tf.tile(tf.range(batch_size), [num_samples])
        row_idx = tf.reshape(tf.transpose(tf.reshape(
            row_idx, shape=[-1, batch_size])), shape=[-1])
        ids = tf.stack([row_idx, col_idx], axis=1)
        # Attention! the last batch is always smaller than a normal batch
        ids = tf.reshape(ids, [-1, num_samples, 2])[:num_ids]
        ids = tf.reshape(ids, [-1, 2])
        neighbors = tf.gather_nd(adj_lists, ids)
        tss = tf.gather_nd(ts_lists, ids)
        return neighbors, tss


def check(adj_info, ts_info, input_ids, input_tss, neighbors, tss):
    adj_info = adj_info[input_ids]
    ts_info = ts_info[input_ids]
    neighbors = np.reshape(neighbors, (len(input_ids), -1))
    tss = np.reshape(tss, (len(input_ids), -1))
    print("input_tss", input_tss)
    print("neigh_tss", tss)
    for i in range(len(input_ids)):
        for neigh, ts in zip(neighbors[i], tss[i]):
            if input_tss[i] > 0:
                assert ts < input_tss[i], print(
                    "ts, input_tss:", ts, input_tss[i])
            find = np.any([neigh == n and ts == t for n,
                           t in zip(adj_info[i], ts_info[i])])
            if not find and neigh != 0:
                raise NotImplementedError(
                    "Not find neighbor {} and ts {} from id {}.".format(str(neigh), str(ts), str(input_ids[i])))


if __name__ == "__main__":
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # tf.Variable() doesnt work in eager mode, so comment ```self.adj_info=tf.Variable``` in __init__ function
    tf.enable_eager_execution()
    tf.random.set_random_seed(42)
    np.random.seed(42)
    adj_info = np.array(
        [np.random.choice(5, size=4) for i in range(5)])
    ts_info = np.array([np.arange(1, 5) for _ in range(5)])
    print("adj_info", adj_info)
    print("ts_info", ts_info)
    mask_sampler = MaskNeighborSampler(adj_info, ts_info)
    ids = np.arange(5, dtype=np.int64)
    tss = np.array(np.random.randint(1, 5, size=5))
    num_samples = 10
    neighbors, neigh_tss = mask_sampler(
        (tf.constant(ids), tf.constant(tss), len(ids), num_samples))
    print("MaskNeighborSampler:")
    print("neighbors:", neighbors)
    print("neigh_tss:", neigh_tss)
    check(adj_info, ts_info, ids, tss, neighbors.numpy(), neigh_tss.numpy())

    print("TemporalNeighborSampler:")
    temporal_sampler = TemporalNeighborSampler(adj_info, ts_info)
    neighbors, neigh_tss = temporal_sampler(
        (tf.constant(ids), tf.constant(tss), len(ids), num_samples))
    # print("neighbors shape", neighbors.shape)
    # print("neigh_tss shape", neigh_tss.shape)
    print("neighbors:", neighbors)
    print("neigh_tss:", neigh_tss)
    check(adj_info, ts_info, ids, tss, neighbors.numpy(), neigh_tss.numpy())
