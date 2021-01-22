import numpy as np
import argparse
import logging
import os
from datetime import datetime
import time
from collections import defaultdict
from datetime import datetime
from itertools import product

import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data_loader.data_util import load_data, load_split_edges, load_label_edges, _iterate_datasets as iterate_datasets
from experiment.adaptors import run_node2vec, run_triad, run_htne, run_tnode, run_gta, run_ctdne

parser = argparse.ArgumentParser(
    description="Perform contrastive experiments.")
parser.add_argument("--method", type=str, default="node2vec",
                    help="Contrastive method name.")
parser.add_argument("--n_jobs", type=int, default=16,
                    help="Job numbers for joblib Parallel function.")
parser.add_argument("-d", "--dataset", type=str,
                    default="all", help="Specific dataset for experiments; default is all datasets.")
parser.add_argument("--start", type=int, default=0, help="Datset start index.")
parser.add_argument("--end", type=int, default=14,
                    help="Datset end index (exclusive).")
parser.add_argument("--gid", type=int, default=0)
parser.add_argument("--run", action="store_true", default=False,
                    help="Whether running embeddings.")
parser.add_argument("--times", type=int, default=1,
                    help="Experiment repetition times.")

args = parser.parse_args()


def set_logger():
    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(
        'log/{}-{}-{}-{}.log'.format(args.method, args.start, args.end, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = set_logger()
logger.info(args)

data_dir = "/nfs/zty/Graph/format_data/"
train_data_dir = "/nfs/zty/Graph/train_data/"
valid_data_dir = "/nfs/zty/Graph/valid_data/"
test_data_dir = "/nfs/zty/Graph/test_data/"


def load_embeddings(path, skiprows=0, sep=" "):
    df = pd.read_csv(path, skiprows=skiprows, sep=sep, header=None)
    # nodes = df[0]
    df[0] = df[0].astype("int64")
    id2idx = {int(row[0]): index for index, row in df.iterrows()}
    # id2idx = defaultdict(lambda: 0, id2idx)
    embs = df.loc[:, 1:].to_numpy()
    # print(embs.shape)
    assert(embs.shape[1] == 128)
    # Padding 0 row as embeddings all zero
    # embs = np.concatenate([np.zeros((1, 128)), embs])
    return id2idx, embs


def iterate_times(method):
    def worker(*ar, **kw):
        results = []
        for _ in range(args.times):
            logger.info("Times {} begins.".format(_))
            res = method(*ar, **kw)
            results.append(res)
        return results
    return worker


def id_map(edges, nodes):
    train_edges, valid_edges, test_edges = edges
    id2idx = {row.node_id: row.id_map for row in nodes.itertuples()}

    def _f(edges):
        edges["from_node_id"] = edges["from_node_id"].map(id2idx)
        edges["to_node_id"] = edges["to_node_id"].map(id2idx)
        return edges
    return _f(train_edges), _f(valid_edges), _f(test_edges)


@iterate_times
def evaluate_node2vec(project_dir="/nfs/zty/Graph/0-node2vec/emb"):
    fname = iterate_datasets()
    fname = fname[args.start: args.end]
    if args.run:
        logger.info("Running {} embedding programs.".format("node2vec"))
        run_node2vec(dataset=args.dataset, n_jobs=args.n_jobs,
                     fname=fname, start=args.start, end=args.end, times=args.times)
        logger.info("Done node2vec embedding.")
    else:
        logger.info("Use pretrained {} embeddings.".format("node2vec"))
    for name, p, q in product(fname, [0.25, 0.5, 1, 2, 4], [0.25, 0.5, 1, 2, 4]):
        logger.info("dataset={}, p={:.2f}, q={:.2f}".format(name, p, q))

        edges, nodes = load_label_edges(dataset=name)
        train_edges, valid_edges, test_edges = id_map(edges[0], nodes[0])

        fpath = "{}/{}-{p:.2f}-{q:.2f}.emb".format(
            project_dir, name, p=p, q=q)
        id2idx, embs = load_embeddings(fpath, skiprows=1)
        X_train = edge2tabular(train_edges, id2idx, embs)
        y_train = train_edges["label"]
        X_valid = edge2tabular(valid_edges, id2idx, embs)
        y_valid = valid_edges["label"]
        X_test = edge2tabular(test_edges, id2idx, embs)
        y_test = test_edges["label"]
        # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        vauc, acc, f1, auc = lr_evaluate(X_train, y_train, X_valid, y_valid, X_test, y_test)
        write_result(name, "node2vec", {"p": p, "q": q}, (vauc, acc, f1, auc))
    pass


@iterate_times
def evaluate_triad(project_dir="/nfs/zty/Graph/2-DynamicTriad/output"):
    fname = iterate_datasets(dataset=args.dataset)
    fname = fname[args.start: args.end]
    if args.run:
        logger.info("Running {} embedding programs.".format(args.method))
        run_triad(dataset=args.dataset, n_jobs=args.n_jobs,
                  fname=fname, start=args.start, end=args.end, times=args.times)
        logger.info("Done training embedding.")
    else:
        logger.info("Use pretrained {} embeddings.".format(args.method))
    for name, stepsize in product(fname, [1, 4, 8]):
        logger.info(name)

        edgel, nodel = load_label_edges(dataset=name)
        train_edges, valid_edges, test_edges = id_map(edgel[0], nodel[0])

        fdir = "{}/{}-{}/".format(project_dir, name, stepsize)
        step_embeds = [load_embeddings(fdir + f, skiprows=0)
                       for f in os.listdir(fdir)]
        id2idx, embeds = step_embeds[-1]
        X_train = edge2tabular(train_edges, id2idx, embeds)
        y_train = train_edges["label"]
        X_valid = edge2tabular(valid_edges, id2idx, embeds)
        y_valid = valid_edges["label"]
        X_test = edge2tabular(test_edges, id2idx, embeds)
        y_test = test_edges["label"]
        # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        vauc, acc, f1, auc = lr_evaluate(X_train, y_train, X_valid, y_valid, X_test, y_test)
        write_result(name, "triad", {"beta_1": 0.1,
                                     "beta_2": 0.1, "stepsize": stepsize}, (vauc, acc, f1, auc))


@iterate_times
def evaluate_tnode():
    fname = iterate_datasets(dataset=args.dataset)
    fname = fname[args.start: args.end]
    logger.info("Running {} embedding programs.".format(args.method))
    run_tnode(dataset=args.dataset, n_jobs=args.n_jobs,
              start=args.start, end=args.end)
    logger.info("Done {}.".format(args.method))


@iterate_times
def evaluate_ctdne(project_dir="/nfs/zty/Graph/Dynamic-Graph/ctdne_embs"):
    fname = iterate_datasets(dataset=args.dataset)
    fname = fname[args.start: args.end]
    if args.run:
        logger.info("Running {} embedding programs.".format(args.method))
        Parallel(n_jobs=args.n_jobs)(
            delayed(run_ctdne)(fname=[name]) for name in fname)
        logger.info("Done {} embeddings.".format(args.method))
    for name in fname:
        logger.info(
            "dataset={}, num_walk=10, walk_length=80, context_window=10".format(name))

        fpath = "{}/{}.emb".format(project_dir, name)
        id2idx, embeds = load_embeddings(fpath, skiprows=0, sep=" ")

        edgel, nodel = load_label_edges(dataset=name)
        train_edges, valid_edges, test_edges = id_map(edgel[0], nodel[0])

        X_train = edge2tabular(train_edges, id2idx, embeds)
        y_train = train_edges["label"]
        X_valid = edge2tabular(valid_edges, id2idx, embeds)
        y_valid = valid_edges["label"]
        X_test = edge2tabular(test_edges, id2idx, embeds)
        y_test = test_edges["label"]
        # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        vauc, acc, f1, auc = lr_evaluate(X_train, y_train, X_valid, y_valid, X_test, y_test)
        write_result(name, "ctdne", {
                     "num_walk": 10, "walk_length": 80, "context_window": 10}, (vauc, acc, f1, auc))


@iterate_times
def evaluate_htne(project_dir="/nfs/zty/Graph/4-htne/emb"):
    fname = iterate_datasets(dataset=args.dataset)
    fname = fname[args.start: args.end]
    if args.run:
        logger.info("Running {} embedding programs.".format(args.method))
        run_htne(dataset=args.dataset, n_jobs=args.n_jobs, fname=fname)
        logger.info("Done training embedding.")
    else:
        logger.info("Use pretrained {} embeddings.".format(args.method))

    for name in fname:
        logger.info(name)

        edgel, nodel = load_label_edges(dataset=name)
        train_edges, valid_edges, test_edges = id_map(edgel[0], nodel[0])

        for hist_len in [20]:
            fpath = "{}/{}.emb{}".format(project_dir, name, hist_len)
            id2idx, embeds = load_embeddings(fpath, skiprows=1, sep=" ")
            X_train = edge2tabular(train_edges, id2idx, embeds)
            y_train = train_edges["label"]
            X_valid = edge2tabular(valid_edges, id2idx, embeds)
            y_valid = valid_edges["label"]
            X_test = edge2tabular(test_edges, id2idx, embeds)
            y_test = test_edges["label"]
            # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
            vauc, acc, f1, auc = lr_evaluate(X_train, y_train, X_valid, y_valid, X_test, y_test)
            write_result(name, "htne", {"hist_len": hist_len, "epoch": 50}, (vauc, acc, f1, auc))


def edge2tabular(edges, id2idx, embs):
    unseen_from_nodes = set(edges["from_node_id"]) - set(id2idx.keys())
    unseen_to_nodes = set(edges["to_node_id"]) - set(id2idx.keys())
    logger.info("{} unseen from_nodes, {} unseen to_nodes in id2idx.".format(
        len(unseen_from_nodes), len(unseen_to_nodes)))

    edges["from_node_id"] = edges["from_node_id"].map(id2idx)
    edges["to_node_id"] = edges["to_node_id"].map(id2idx).astype('int64')
    # print(edges.dtypes)
    X = np.zeros((len(edges), 2 * embs.shape[1]))
    X_from = embs[edges["from_node_id"]]
    X_to = embs[edges["to_node_id"]]
    return np.concatenate((X_from, X_to), axis=1)


def lr_evaluate(X_train, y_train, X_valid, y_valid, X_test, y_test):
    clf = LogisticRegression(random_state=42).fit(X_train, y_train)
    yp_valid = clf.predict_proba(X_valid)[:, 1]
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    vauc = roc_auc_score(y_valid, yp_valid)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    return vauc, acc, f1, auc


def write_result(dataset, method, params, metrics, result_dir="/nfs/zty/Graph/Dynamic-Graph/comp_results"):
    val_auc, acc, f1, auc = metrics
    res_path = "{}/{}-{}.csv".format(result_dir, dataset, args.method)
    headers = ["method", "dataset", "valid_auc", "accuracy", "f1", "auc", "params"]
    if not os.path.exists(res_path):
        f = open(res_path, 'w+')
        f.write(",".join(headers) + "\r\n")
        f.close()
        os.chmod(res_path, 0o777)
    with open(res_path, 'a') as f:
        result_str = "{},{},{:.4f},{:.4f},{:.4f},{:.4f}".format(
            method, dataset, val_auc, acc, f1, auc)
        params_str = ",".join(["{}={}".format(k, v)
                               for k, v in params.items()])
        params_str = "\"{}\"".format(params_str)
        row = result_str + "," + params_str + "\r\n"
        logger.info(
            "{}-{}: {:.3f}, {:.3f}, {:.3f}".format(method, dataset, acc, f1, auc))
        # print("{}-{}: {:.3f}, {:.3f}, {:.3f}".format(method, dataset, acc, f1, auc))
        f.write(row)


@iterate_times
def evaluate_gta():
    run_gta(dataset=args.dataset, start=args.start, end=args.end)


@iterate_times
def evaluate_tgat(project_dir="/nfs/zty/Graph/TGAT-bk"):
    fname = iterate_datasets(dataset=args.dataset)
    fname = fname[args.start: args.end]
    # command = "python {}/exper_edge.py -d {} --gpu {} -f --uniform "
    # command = "python {}/exper_edge.py -d {} --gpu {} -f"
    # command = "python {}/exper_edge.py -d {} --gpu {} --uniform"
    command = "python {}/exper_edge.py -d {} --gpu {} --time empty  "
    commands = []
    for name in fname:
        commands.append(command.format(project_dir, name, args.gid))
    os.chdir(project_dir)
    print("Preprocessing finished.")
    for cmd in commands:
        os.system(cmd)


@iterate_times
def evaluate_sage():
    fname = iterate_datasets()
    os.chdir("/nfs/zty/Graph/Dynamic-Graph")
    cmd = "python -m experiment.graphsage --dataset {dataset} --epochs 50 --dropout 0.2 --weight_decay 1e-5 --learning_rate=0.0001 --nodisplay "
    commands = [cmd.format(dataset=name) for name in fname]
    print("Preprocessing finished.")
    Parallel(n_jobs=args.n_jobs)(delayed(os.system)(cmd) for cmd in commands)


if __name__ == "__main__":
    if args.method == "node2vec":
        evaluate = evaluate_node2vec
    elif args.method == "triad":
        evaluate = evaluate_triad
    elif args.method == "ctdne":
        evaluate = evaluate_ctdne
    elif args.method == "tnode":
        evaluate = evaluate_tnode
    elif args.method == "gta":
        evaluate = evaluate_gta
    elif args.method == "sage":
        evaluate = evaluate_sage
    elif args.method == "htne":
        evaluate = evaluate_htne
    elif args.method == "tgat":
        evaluate = evaluate_tgat
    else:
        raise NotImplementedError(
            "Method {} not implemented!".format(args.method))
    # fname, _ = iterate_datasets()
    # for name in fname[args.start: args.end]:
    #     print(name)
    #     evaluate(name)
    evaluate()
    # evaluate(fname[0])
    # Parallel(n_jobs=16)(delayed(evaluate)(name) for name in fname[args.start: args.end])
