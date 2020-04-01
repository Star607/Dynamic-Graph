import csv
import os
import time
import numpy as np
import tensorflow as tf
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data_loader.minibatch import load_data, TemporalEdgeBatchIterator
from data_loader.neigh_samplers import MaskNeighborSampler, TemporalNeighborSampler
from model.gta import GraphTemporalAttention, SAGEInfo

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.logging.set_verbosity(tf.logging.ERROR)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'CTDNE-fb-forum', 'experiment dataset')
flags.DEFINE_string('method', 'GTA', 'experiment method')
flags.DEFINE_string('sampler', 'mask', 'neighbor sampler')
flags.DEFINE_boolean('use_context', True, 'use temporal context mechanism')
flags.DEFINE_string('loss', 'xent', 'loss function')
flags.DEFINE_float('learning_rate', 0.0001, 'initial learning rate')
flags.DEFINE_integer('epochs', 1, 'number of epochs to train')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability)')
flags.DEFINE_float('weight_decay', 0.0,
                   'weight for l2 loss on embedding matrix')
flags.DEFINE_integer('val_batches', 50, 'evaluate a batch of validation set after some batches')
flags.DEFINE_integer('val_epochs', 5, 'evaluate a full validation set after some epochs')

flags.DEFINE_integer('max_degree', 100, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of users samples in layer 2')
flags.DEFINE_integer(
    'dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer(
    'dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('context_size', 20, 'number of temporal context samples')
flags.DEFINE_integer('batch_size', 128, 'minibatch size.')
flags.DEFINE_boolean(
    'concat', False, 'Use concat between neighbor features and self features.')

# logging, saving, validation settings etc.
flags.DEFINE_boolean('save_embeddings', True,
                     'whether to save embeddings for all nodes after training')
flags.DEFINE_string('base_log_dir', 'log/',
                    'base directory for logging and saving embeddings')
flags.DEFINE_string('save_dir', 'saved_models/', 'directory for saving models')
flags.DEFINE_string('result_dir', 'results/', 'directory for saving results')
flags.DEFINE_integer('gpu', 0, "which gpu to use.")
flags.DEFINE_integer('print_every', 50, "How often to print training info.")

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

def write_result(loss, acc, f1, auc):
    res_path = FLAGS.result_dir + "{}-{}.csv".format(FLAGS.dataset, FLAGS.method)
    headers = ["method", "dataset", "accuracy", "f1", "auc", "params"]
    if not os.path.exists(res_path):        
        f = open(res_path, 'w')
        f.write(",".join(headers) + "\r\n")
        f.close()
        os.chmod(res_path, 0o777)
    with open(res_path, 'a') as f:
        result_str = "{},{},{:.4f},{:.4f},{:.4f}".format(FLAGS.method, FLAGS.dataset, acc, f1, auc)
        params = {"epochs":FLAGS.epochs, "sampler":FLAGS.sampler, "learning_rate":FLAGS.learning_rate, 
                  "dropout":FLAGS.dropout, "weight_decay":FLAGS.weight_decay,
                  "use_context":FLAGS.use_context, "context_size":FLAGS.context_size}
        params_str = ",".join(["{}={}".format(k, v) for k, v in params.items()])
        params_str = "\"{}\"".format(params_str)
        row = result_str + "," + params_str + "\r\n"
        f.write(row)

def main(argv=None):
    


if __name__ == "__main__":
    tf.app.run()
