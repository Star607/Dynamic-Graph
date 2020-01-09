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


def log_dir():
    log_dir = FLAGS.base_log_dir + "{}-{}".format(FLAGS.dataset, FLAGS.method)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        os.chmod(log_dir, 0o777)
    return log_dir


def valid_batch(sess, model, batch_iter, mode="valid"):
    if mode == "valid":
        feed_dict = batch_iter.val_feed_dict()
    elif mode == "test":
        feed_dict = batch_iter.test_feed_dict()
    else:
        raise Exception("Not supported mode ", mode)
    outs = sess.run([model.loss, model.acc], feed_dict=feed_dict)
    return outs


def valid_full(sess, model, batch_iter, placeholders, mode="valid"):
    if mode == "valid":
        batches = batch_iter.val_batch()
        num_samples = len(batch_iter.val_idx)
    elif mode == "test":
        batches = batch_iter.test_batch()
        num_samples = len(batch_iter.test_idx)
    else:
        raise Exception("Not supported mode ", mode)
    tot_loss = 0
    probs = []
    labels = []
    for feed_dict in batches:
        outs = sess.run([model.loss, model.pred_op], feed_dict=feed_dict)
        tot_loss += outs[0] * feed_dict[placeholders["batch_size"]]
        probs.extend(outs[1])
        batch_size = feed_dict[placeholders["batch_size"]]
        labels.extend([1] * batch_size + [0] * batch_size)
    tot_loss /= num_samples
    preds = np.array(probs) >= 0.5
    return tot_loss, accuracy_score(labels, preds), f1_score(labels, preds), roc_auc_score(labels, probs)


def construct_placeholders():
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
    }
    return placeholders


def config_tensorflow():
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)
    return merged, sess, summary_writer

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
    print("Loading training data...")
    edges, nodes = load_data(datadir="./graph_data", dataset=FLAGS.dataset)
    print("Done loading training data...")

    placeholders = construct_placeholders()
    batch = TemporalEdgeBatchIterator(edges, nodes, placeholders,
                                      batch_size=FLAGS.batch_size, max_degree=FLAGS.max_degree, context_size=FLAGS.context_size)

    adj_info, ts_info = batch.adj_ids, batch.adj_tss
    if FLAGS.sampler == "mask":
        sampler = MaskNeighborSampler(adj_info, ts_info)
    elif FLAGS.sampler == "temporal":
        sampler = TemporalNeighborSampler(adj_info, ts_info)
    else:
        raise NotImplementedError("Sampler %s not supported." % FLAGS.sampler)

    layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                   SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
    ctx_layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
    model = GraphTemporalAttention(
        placeholders, None, adj_info, ts_info, batch.degrees, layer_infos,
        ctx_layer_infos, sampler, bipart=batch.bipartite, n_users=batch.n_users)

    merged, sess, summary_writer = config_tensorflow()
    sess.run(tf.global_variables_initializer())
    batch_num = len(batch.train_idx) // FLAGS.batch_size
    
    for epoch in range(FLAGS.epochs):
        print("Epoch %04d" % (epoch + 1))
        batch.shuffle()
        tot_loss = tot_acc = val_loss = val_acc = 0
        sess.run(tf.local_variables_initializer())
        with trange(batch_num) as batch_bar:
            while not batch.end():
                batch_bar.set_description("batch %04d" % batch_num)
                feed_dict = batch.next_train_batch()
                outs = sess.run(
                    [merged, model.opt_op, model.loss, model.acc, model.auc, model.acc_update, model.auc_update], feed_dict)
                tot_loss += outs[2] * feed_dict[placeholders["batch_size"]]
                tot_acc += outs[3] * feed_dict[placeholders["batch_size"]]
                loss, acc = outs[2], outs[3]
                if batch.batch_num % FLAGS.val_batches == 0:
                    val_loss, val_acc = valid_batch(sess, model, batch)
                batch_bar.update()
                batch_bar.set_postfix(
                    loss=loss, acc=acc, val_loss=val_loss, val_acc=val_acc)
                summary_writer.add_summary(
                    outs[0], epoch * batch_num + batch.batch_num)
        tot_loss /= len(batch.train_idx)
        tot_acc /= len(batch.train_idx)
        # print("Full train_loss %.4f train_acc %.4f" % (tot_loss, tot_acc))
        # if epoch % FLAGS.val_epochs == 0:
        val_loss, acc, f1, auc = valid_full(sess, model, batch, placeholders)
        print("Full train_loss %.4f train_acc %.4f val_loss %.4f val_acc %.4f val_f1 %.4f val_auc %.4f" %(tot_loss, tot_acc, val_loss, acc, f1, auc))
        # epoch_bar.set_postfix(loss=tot_loss, acc=tot_acc,
                            # val_loss=val_loss, val_f1=f1, val_auc=auc)
    print("Optimized finishied!")
    loss, acc, f1, auc = valid_full(sess, model, batch, placeholders, mode="test")
    write_result(loss, acc, f1, auc)
    print("Dataset %s Test set statistics: loss %.4f acc %.4f f1 %.4f auc %.4f" % (FLAGS.dataset, loss, acc, f1, auc))


if __name__ == "__main__":
    tf.app.run()
