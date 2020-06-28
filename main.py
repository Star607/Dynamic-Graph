import csv
import os
import time
from io import StringIO
import subprocess
import gpustat
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import trange

from data_loader.minibatch import TemporalEdgeBatchIterator, load_data
from data_loader.neigh_samplers import (MaskNeighborSampler,
                                        TemporalNeighborSampler)
from model.gta import GraphTemporalAttention, SAGEInfo
from model.trainer import ModelTrainer
from model.utils import EarlyStopMonitor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.logging.set_verbosity(tf.logging.ERROR)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'ia-contact', 'experiment dataset')
flags.DEFINE_string('method', 'GTA', 'experiment method')
flags.DEFINE_string('sampler', 'mask', 'neighbor sampler')
flags.DEFINE_boolean('dynamic_neighbor', True, 'use dynamic neighbors')
flags.DEFINE_boolean('use_context', False, 'use temporal context mechanism')
flags.DEFINE_integer('context_size', 1, 'number of temporal context samples')
flags.DEFINE_string('loss', 'xent', 'loss function')
flags.DEFINE_float('learning_rate', 0.0001, 'initial learning rate')
flags.DEFINE_integer('epochs', 1, 'number of epochs to train')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability)')
flags.DEFINE_float('weight_decay', 0.0,
                   'weight for l2 loss on embedding matrix')

flags.DEFINE_integer('max_degree', 100, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of users samples in layer 2')
flags.DEFINE_integer(
    'dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer(
    'dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('batch_size', 128, 'minibatch size.')
flags.DEFINE_boolean(
    'concat', False, 'Use concat between neighbor features and self features.')

# logging, saving, validation settings etc.
flags.DEFINE_boolean('save_embeddings', False,
                     'whether to save embeddings for all nodes after training')
flags.DEFINE_string('base_log_dir', 'log/',
                    'base directory for logging and saving embeddings')
flags.DEFINE_boolean('pretrain', False, 'if use last training models')
flags.DEFINE_string('save_dir', 'saved_models/', 'directory for saving models')
flags.DEFINE_string('result_dir', 'results/', 'directory for saving results')
flags.DEFINE_integer('gpu', 1, "which gpu to use.")
flags.DEFINE_integer('print_every', 50, "How often to print training info.")
flags.DEFINE_boolean('display', True, "Whether to display progress bar.")


def get_free_gpu():
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(
        gpu.entry['memory.total']) - float(gpu.entry['memory.used']), stats)
    pairs = list(zip(ids, ratios))
    random.shuffle(pairs)
    bestGPU = max(pairs, key=lambda x: x[1])[0]
    print("setGPU: Setting GPU to: {}".format(bestGPU))
    return str(bestGPU)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = get_free_gpu()


def write_result(label, preds, params):
    acc = accuracy_score(label, preds > 0.5)
    f1 = f1_score(label, preds > 0.5)
    auc = roc_auc_score(label, preds)
    res_path = "{}/{}-{}.csv".format(FLAGS.result_dir,
                                     FLAGS.dataset, FLAGS.method)
    headers = ["method", "dataset", "accuracy", "f1", "auc", "params"]
    if not os.path.exists(res_path):
        f = open(res_path, 'w')
        f.write(",".join(headers) + "\r\n")
        f.close()
        os.chmod(res_path, 0o777)
    with open(res_path, 'a') as f:
        result_str = "{},{},{:.4f},{:.4f},{:.4f}".format(
            FLAGS.method, FLAGS.dataset, acc, f1, auc)
        # params = {"epochs":FLAGS.epochs, "sampler":FLAGS.sampler, "learning_rate":FLAGS.learning_rate,
        #           "dropout":FLAGS.dropout, "weight_decay":FLAGS.weight_decay,
        #           "use_context":FLAGS.use_context, "context_size":FLAGS.context_size}
        params_str = ",".join(["{}={}".format(k, v)
                               for k, v in params.items()])
        params_str = "\"{}\"".format(params_str)
        row = result_str + "," + params_str + "\r\n"
        print("{}-{}: acc {:.3f}, f1 {:.3f}, auc {:.3f}".format(FLAGS.dataset,
                                                                FLAGS.method, acc, f1, auc))
        f.write(row)


def main(argv=None):
    print("Loading training data {}.".format(FLAGS.dataset))
    edges, nodes = load_data(datadir="./ctdne_data/", dataset=FLAGS.dataset)
    train_edges = pd.read_csv("../train_data/{}.csv".format(FLAGS.dataset))
    test_edges = pd.read_csv("../test_data/{}.csv".format(FLAGS.dataset))
    print("Done loading training data.")
    # test_ratio is consistent with the comparison experiment
    trainer = ModelTrainer(edges, nodes, val_ratio=0.05, test_ratio=0.25)
    print(len(train_edges), len(test_edges), 2 * len(trainer.batch.edges))
    assert(len(train_edges)+len(test_edges) ==
           2 * (len(trainer.batch.edges)-1))
    early_stopper = EarlyStopMonitor()
    if FLAGS.pretrain:
        trainer.restore_models()
    for epoch in range(FLAGS.epochs):
        trainer.train(epoch=epoch)
        val_auc = trainer.valid()
        print(f"val_auc: {val_auc}")
        # trainer.save_models(epoch=epoch)
        if early_stopper.early_stop_check(val_auc):
            print(f"No improvement over {early_stopper.max_round} epochs")
            trainer.params["epochs"] = epoch
            # trainer.restore(epoch=epoch-2)
            break
    y = trainer.test(test_edges)
    if FLAGS.epochs > 1:
        write_result(test_edges["label"], y, trainer.params)


if __name__ == "__main__":
    tf.app.run()
