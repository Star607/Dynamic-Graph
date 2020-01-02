import os
import time
import numpy as np
import tensorflow.compat.v1 as tf

from data_loader.minibatch import load_data, TruncatedTemporalEdgeBatchIterator
from data_loader.neigh_samplers import MaskNeighborSampler, TemporalNeighborSampler
from model.gta import GraphTemporalAttention, SAGEInfo
from main import *
# from tensorflow.python.util import deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.logging.set_verbosity(tf.logging.ERROR)

flags = tf.app.flags
FLAGS = flags.FLAGS


def run(argv=None):
    print("Loading training data...")
    edges, nodes = load_data(datadir="./graph_data", dataset=FLAGS.dataset)
    print("Done loading training data...")
    placeholders = construct_placeholders()
    batch = TruncatedTemporalEdgeBatchIterator(
        edges, nodes, placeholders, batch_size=FLAGS.batch_size, max_degree=FLAGS.max_degree, context_size=FLAGS.context_size)
    adj_info, ts_info = batch.adj_ids, batch.adj_tss
    sampler = MaskNeighborSampler(adj_info, ts_info)
    layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                   SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
    ctx_layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]
    model = GraphTemporalAttention(
        placeholders, None, adj_info, ts_info, batch.degrees, layer_infos,
        ctx_layer_infos, sampler, bipart=batch.bipartite, n_users=batch.n_users)

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)
    sess.run(tf.global_variables_initializer())

    t = time.time()
    for epoch in range(FLAGS.epochs):
        batch.shuffle()
        print("Epoch %04d" % (epoch+1))
        while not batch.end():
            t = time.time()
            feed_dict = batch.next_train_batch()
            cnt = sess.run(model.count_nonzero_ops, feed_dict)
            print("non-zero elements")
            print(cnt)


if __name__ == "__main__":
    tf.app.run(main=run)
