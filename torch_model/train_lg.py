from datetime import datetime
import math
from multiprocessing import Process, Queue
import logging
import time
import random
import os
import sys
import argparse
from numba import jit
from torch.utils import data

from tqdm import trange
import torch
import pandas as pd
import numpy as np
#import numba

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from data_loader.data_util import load_graph, load_label_data
from torch_model.sampling_tgat import TGAN, SamplingFusion, LGFusion
from tgat.sampling import NeighborFinder, BiSamplingNFinder, NeighborLoader, global_anchors
from model.utils import EarlyStopMonitor, RandEdgeSampler, get_free_gpu


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--data',
                        type=str,
                        help='data sources to use',
                        default='JODIE-wikipedia')
    parser.add_argument("--display", dest="display", action="store_true")
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument('-f', '--freeze', action='store_true')
    parser.add_argument('--model',
                        default='SamplingFusion',
                        choices=['TGAT', 'SamplingFusion', 'LG'])
    parser.add_argument('--bs', type=int, default=200, help='batch_size')
    parser.add_argument('--prefix',
                        type=str,
                        default='',
                        help='prefix to name the checkpoints')
    parser.add_argument('--n_degree',
                        type=int,
                        default=20,
                        help='number of neighbors to sample')
    parser.add_argument('--n_head',
                        type=int,
                        default=2,
                        help='number of heads used in attention layer')
    parser.add_argument('--n_epoch',
                        type=int,
                        default=50,
                        help='number of epochs')
    parser.add_argument('--n_layer',
                        type=int,
                        default=2,
                        help='number of network layers')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='learning rate')
    parser.add_argument('--drop_out',
                        type=float,
                        default=0.1,
                        help='dropout probability')
    parser.add_argument('--gpu',
                        type=int,
                        default=3,
                        help='idx for the gpu to use')
    parser.add_argument('--node_dim',
                        type=int,
                        default=128,
                        help='Dimentions of the node embedding')
    parser.add_argument('--time_dim',
                        type=int,
                        default=128,
                        help='Dimentions of the time embedding')
    parser.add_argument('--agg_method',
                        type=str,
                        choices=['attn', 'lstm', 'mean'],
                        help='local aggregation method',
                        default='attn')
    parser.add_argument('--attn_mode',
                        type=str,
                        choices=['prod', 'map'],
                        default='prod',
                        help='use dot product attention or mapping based')
    parser.add_argument('--time',
                        type=str,
                        choices=['time', 'pos', 'empty'],
                        help='how to use time information',
                        default='time')
    parser.add_argument('--uniform',
                        action='store_true',
                        help='take uniform sampling from temporal neighbors')
    parser.add_argument("-s",
                        "--sampling",
                        default="bisampling",
                        choices=["ngh_finder", "bisampling"],
                        help="Whether use normal sampling or binary sampling.")
    parser.add_argument("-n-anchors", type=int, default=20)
    parser.add_argument("--metric", default="pagerank", choices=["pagerank", "degree", "closeness", "betweenness"])
    return parser


args = config_parser().parse_args()
# Arguments
if True:
    BATCH_SIZE = args.bs
    NUM_NEIGHBORS = args.n_degree
    NUM_NEG = 1
    NUM_EPOCH = args.n_epoch
    NUM_HEADS = args.n_head
    DROP_OUT = args.drop_out
    # GPU = get_free_gpu()
    # GPU = str(args.gpu)
    if args.gpu >= 0:
        GPU = str(args.gpu)
    else:
        GPU = get_free_gpu()
    UNIFORM = args.uniform
    USE_TIME = args.time
    AGG_METHOD = args.agg_method
    ATTN_MODE = args.attn_mode
    SEQ_LEN = NUM_NEIGHBORS
    DATA = args.data
    NUM_LAYER = args.n_layer
    LEARNING_RATE = args.lr
    NODE_DIM = args.node_dim
    TIME_DIM = args.time_dim

    MODEL_SAVE_PATH = f'./saved_models/{args.model}-{args.agg_method}-{args.attn_mode}-{args.data}.pth'

    def get_checkpoint_path(epoch):
        return f'./ckpt/{args.model}-{args.agg_method}-{args.attn_mode}-{args.data}-{epoch}.pth'


def set_logger():
    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)

    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.FileHandler('log/Fusion-{}.log'.format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


logger = set_logger()
logger.info(args)

def eval_one_epoch(hint, model, data_loader, label):
    data_loader.reset()
    with torch.no_grad():
        model = model.eval()
        TEST_BATCH_SIZE = args.bs
        scores = []
        while not data_loader.end():
            src_list, dst_list = data_loader.next_batch()
            if args.model == "LG":
                model.set_anchors(data_loader.batch_anchor)
            prob_score = model(src_list, dst_list).sigmoid()
            scores.extend(list(prob_score.cpu().numpy()))
        pred_label = np.array(scores) > 0.5
        pred_prob = np.array(scores)
    return accuracy_score(label, pred_label), \
            average_precision_score(label, pred_label), \
            f1_score(label, pred_label), roc_auc_score(label, pred_prob)


if True:
    edges, n_nodes, val_time, test_time = load_graph(dataset=args.data)
    g_df = edges[["from_node_id", "to_node_id", "timestamp"]].copy()
    g_df["idx"] = np.arange(1, len(g_df) + 1)
    g_df.columns = ["u", "i", "ts", "idx"]

    if len(edges.columns) > 4:
        e_feat = edges.iloc[:, 4:].to_numpy()
        padding = np.zeros((1, e_feat.shape[1]))
        e_feat = np.concatenate((padding, e_feat))
    else:
        e_feat = np.zeros((len(g_df) + 1, NODE_DIM))

    if args.freeze:
        # e_feat = edges.iloc[:, 4:].to_numpy()
        # NODE_DIM = e_feat.shape[1]
        n_feat = np.zeros((n_nodes + 1, NODE_DIM))
    else:
        # e_feat = np.zeros((len(g_df) + 1, NODE_DIM))
        bound = np.sqrt(6 / (2 * NODE_DIM))
        n_feat = np.random.uniform(-bound, bound, (n_nodes + 1, NODE_DIM))

    src_l = g_df.u.values
    dst_l = g_df.i.values
    e_idx_l = g_df.idx.values
    ts_l = g_df.ts.values

    max_src_index = src_l.max()
    max_idx = max(src_l.max(), dst_l.max())

# set train, validation, test datasets
if True:
    valid_train_flag = (ts_l < val_time)

    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]

    _, val_data, test_data = load_label_data(dataset=DATA)

    val_src_l = val_data.u.values
    val_dst_l = val_data.i.values
    val_ts_l = val_data.ts.values
    val_label_l = val_data.label.values

    test_src_l = test_data.u.values
    test_dst_l = test_data.i.values
    test_ts_l = test_data.ts.values
    test_label_l = test_data.label.values


def set_random_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed()
# Initialize the data structure for graph and edge sampling
# build the graph for fast query
# @jit
def build_graph(src_l, dst_l, e_idx_l, ts_l):
    adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
        adj_list[src].append((dst, eidx, ts))
        adj_list[dst].append((src, eidx, ts))
    return adj_list
adj_list = build_graph(train_src_l, train_dst_l, train_e_idx_l, train_ts_l)
full_adj_list = build_graph(src_l, dst_l, e_idx_l, ts_l)

if args.sampling == "ngh_finder":
    train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)
    full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)
elif args.sampling == "bisampling":
    train_ngh_finder = BiSamplingNFinder(adj_list)
    full_ngh_finder = BiSamplingNFinder(full_adj_list)
else:
    raise NotImplementedError(args.sampling)

train_rand_sampler = RandEdgeSampler(train_src_l, train_dst_l)

# Model initialize
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device('cuda:{}'.format(GPU))
if args.model == "SamplingFusion":
    Model = SamplingFusion
elif args.model == "LG":
    Model = LGFusion
elif args.model == "TGAT":
    Model = TGAN
else:
    raise NotImplementedError(args.model)

model = Model(train_ngh_finder,
              n_feat,
              e_feat,
              n_feat_freeze=args.freeze,
              num_layers=NUM_LAYER,
              use_time=USE_TIME,
              agg_method=AGG_METHOD,
              attn_mode=ATTN_MODE,
              seq_len=SEQ_LEN,
              n_head=NUM_HEADS,
              drop_out=DROP_OUT,
              node_dim=NODE_DIM,
              time_dim=TIME_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
model = model.to(device)
num_instance = len(train_src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)

logger.info('num of training instances: {}'.format(num_instance))
logger.info('num of batches per epoch: {}'.format(num_batch))
idx_list = np.arange(num_instance)
np.random.shuffle(idx_list)
train_nodes = (train_src_l, train_dst_l, train_src_l)
train_loader = NeighborLoader(train_ngh_finder,
                              NUM_LAYER,
                              train_nodes,
                              train_ts_l,
                              device,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
val_nodes = (val_src_l, val_dst_l)
val_loader = NeighborLoader(full_ngh_finder,
                            NUM_LAYER,
                            val_nodes,
                            val_ts_l,
                            device,
                            batch_size=BATCH_SIZE)
test_nodes = (test_src_l, test_dst_l)
test_loader = NeighborLoader(full_ngh_finder,
                             NUM_LAYER,
                             test_nodes,
                             test_ts_l,
                             device,
                             batch_size=BATCH_SIZE)

# set anchors for both data_loaders and models
if args.model == "LG":
    edges = list(zip(train_src_l, train_dst_l))
    anchors = global_anchors(edges, n_anchors=args.n_anchors, metric=args.metric)
    for loader in [train_loader, val_loader, test_loader]:
        loader.set_anchors(anchors)

early_stopper = EarlyStopMonitor()
epoch_bar = trange(NUM_EPOCH, disable=(not args.display))
for epoch in epoch_bar:
    # Training
    # training use only training graph
    np.random.shuffle(idx_list)
    batch_bar = trange(num_batch, disable=(not args.display))

    src_l_fake, dst_l_fake = train_rand_sampler.sample(len(train_src_l))
    src_nodes = (train_src_l, train_dst_l, dst_l_fake)
    train_loader.src_nodes = src_nodes
    train_loader.reset()
    for k in batch_bar:
        src_l_cut, dst_l_cut, dst_l_fake = train_loader.next_batch()
        src_nodes, src_eids, src_time = src_l_cut
        size = len(src_nodes[0])
        with torch.no_grad():
            pos_label = torch.ones(size, dtype=torch.float, device=device)
            neg_label = torch.zeros(size, dtype=torch.float, device=device)

        optimizer.zero_grad()
        model = model.train()
        if args.model == "LG":
            model.set_anchors(train_loader.batch_anchor)
        pos_prob, neg_prob = model.contrast(src_l_cut, dst_l_cut, dst_l_fake,
                                            NUM_NEIGHBORS)
        loss = criterion(pos_prob, pos_label)
        loss += criterion(neg_prob, neg_label)

        loss.backward()
        optimizer.step()
        # get training results
        with torch.no_grad():
            model = model.eval()
            pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(),
                                         (neg_prob).cpu().detach().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            acc = accuracy_score(true_label, pred_label)
            ap = average_precision_score(true_label, pred_label)
            f1 = f1_score(true_label, pred_label)
            auc = roc_auc_score(true_label, pred_score)
            batch_bar.set_postfix(loss=loss.item(), acc=acc, f1=f1, auc=auc)
    # validation phase use all information
    val_acc, val_ap, val_f1, val_auc = eval_one_epoch('val for old nodes',
                                                      model, val_loader,
                                                      val_label_l)
    epoch_bar.update()
    epoch_bar.set_postfix(acc=val_acc, f1=val_f1, auc=val_auc)

    if early_stopper.early_stop_check(val_auc):
        break
    else:
        torch.save(model.state_dict(), get_checkpoint_path(epoch))

logger.info('No improvment over {} epochs, stop training'.format(
    early_stopper.max_round))
logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
best_model_path = get_checkpoint_path(early_stopper.best_epoch)
model.load_state_dict(torch.load(best_model_path))
logger.info(
    f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
model.eval()
# testing phase use all information
_, _, _, val_auc = eval_one_epoch('val for old nodes', model, val_loader,
                                  val_label_l)
test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for old nodes',
                                                      model, test_loader,
                                                      test_label_l)

logger.info('Test statistics: acc: {:.4f}, f1:{:.4f} auc: {:.4f}'.format(
    test_acc, test_f1, test_auc))

logger.info('Saving model')
torch.save(model.state_dict(), MODEL_SAVE_PATH)
logger.info('Models saved')

res_path = "results/{}-fusion.csv".format(DATA)
headers = ["method", "dataset", "valid_auc", "accuracy", "f1", "auc", "params"]
if not os.path.exists(res_path):
    f = open(res_path, 'w+')
    f.write(",".join(headers) + "\r\n")
    f.close()
    os.chmod(res_path, 0o777)
config = f"freeze={args.freeze},sampling={args.sampling},model={args.model}"
config += f",num_anchors={args.n_anchors},metric={args.metric}"
with open(res_path, "a") as file:
    file.write("{},{},{:.4f},{:.4f},{:.4f},{:.4f},\"{}\"".format(
        args.model, DATA, val_auc, test_acc, test_f1, test_auc, config))
    file.write("\n")
