import argparse
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from experiment.adaptors import iterate_datasets
from experiment.adaptors import run_node2vec, run_triad, run_htne, run_tnode, run_gta

parser = argparse.ArgumentParser(
    description="Perform contrastive experiments.")
parser.add_argument("--method", type=str, default="node2vec",
                    help="Contrastive method name.")
parser.add_argument("--n_jobs", type=int, default=16,
                    help="Job numbers for joblib Parallel function.")
parser.add_argument("--dataset", type=str,
                    default="all", help="Specific dataset for experiments; default is all datasets.")
parser.add_argument("--start", type=int, default=0, help="Datset start index.")
parser.add_argument("--end", type=int, default=100,
                    help="Datset end index (exclusive).")
parser.add_argument("--run", type=bool, default=False,
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
        'log/{}-{}-{}-{}.log'.format(args.method, args.start, args.end, str(time.time())))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = set_logger()
logger.info(args)

data_dir = "/nfs/zty/Graph/ctdne_data/"
train_data_dir = "/nfs/zty/Graph/train_data/"
test_data_dir = "/nfs/zty/Graph/test_data/"


def load_embeddings(path, skiprows=0, sep=" "):
    df = pd.read_csv(path, skiprows=skiprows, sep=sep, header=None)
    # nodes = df[0]
    df[0] = df[0].astype("int64")
    id2idx = {row[0]: index for index, row in df.iterrows()}
    id2idx = defaultdict(lambda: 0, id2idx)
    embs = df.loc[:, 1:].to_numpy()
    # print(embs.shape)
    assert(embs.shape[1] == 128)
    # Padding 0 row as embeddings all zero
    embs = np.concatenate([np.zeros((1, 128)), embs])
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


@iterate_times
def evaluate_node2vec(project_dir="/nfs/zty/Graph/0-node2vec/emb"):
    fname, _ = iterate_datasets()
    fname = fname[args.start: args.end]
    if args.run:
        logger.info("Running {} embedding programs.".format("node2vec"))
        run_node2vec(dataset=args.dataset, n_jobs=args.n_jobs,
                     start=args.start, end=args.end, times=args.times)
        logger.info("Done node2vec embedding.")
    else:
        logger.info("Use pretrained {} embeddings.".format("node2vec"))
    for name, p, q in product(fname, [0.25, 0.5, 1, 2, 4], [0.25, 0.5, 1, 2, 4]):
        sname = name[:-4]
        logger.info("dataset={}, p={:.2f}, q={:.2f}".format(sname, p, q))
        # print(name, p, q)
        train_path = os.path.join(train_data_dir, name)
        train_edges = pd.read_csv(train_path)
        test_path = os.path.join(test_data_dir, name)
        test_edges = pd.read_csv(test_path)

        fpath = "{}/{}-{p:.2f}-{q:.2f}.emb".format(
            project_dir, sname, p=p, q=q)
        id2idx, embs = load_embeddings(fpath, skiprows=1)
        X_train = edge2tabular(train_edges, id2idx, embs)
        y_train = train_edges["label"]
        X_test = edge2tabular(test_edges, id2idx, embs)
        y_test = test_edges["label"]
        # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        acc, f1, auc = lr_evaluate(X_train, y_train, X_test, y_test)
        write_result(sname, "node2vec", {"p": p, "q": q}, (acc, f1, auc))
    pass


@iterate_times
def evaluate_triad(project_dir="/nfs/zty/Graph/2-DynamicTriad/output"):
    fname, _ = iterate_datasets(dataset=args.dataset)
    fname = fname[args.start: args.end]
    if args.run:
        logger.info("Running {} embedding programs.".format(args.method))
        run_triad(dataset=args.dataset, n_jobs=args.n_jobs,
                  start=args.start, end=args.end, times=args.times)
        logger.info("Done training embedding.")
    else:
        logger.info("Use pretrained {} embeddings.".format(args.method))
    for name, stepsize in product(fname, [1, 4, 8]):
        sname = name[:-4]
        logger.info(sname)
        # print(name)
        edges = pd.read_csv(os.path.join(data_dir, name))

        def tswrapper(tss, steps=32):
            stride = len(tss) // steps
            timeloc = [tss.iloc[i * stride] for i in range(steps)]

            def f(ts):
                for i in range(steps):
                    if ts < timeloc[i]:
                        return max(i - 2, 0)
                return max(steps - 2, 0)
            return f
        ts2step = tswrapper(edges["timestamp"], 32 // stepsize)
        train_path = os.path.join(train_data_dir, name)
        train_edges = pd.read_csv(train_path)
        train_edges["step"] = train_edges["timestamp"].apply(ts2step)
        test_path = os.path.join(test_data_dir, name)
        test_edges = pd.read_csv(test_path)
        test_edges["step"] = test_edges["timestamp"].apply(ts2step)

        fdir = "{}/{}-{}/".format(project_dir, sname, stepsize)
        # fdir = "{project_dir}/{sname}-{stepsize}/".format()
        step_embeds = [load_embeddings(fdir + f, skiprows=0)
                       for f in os.listdir(fdir)]

        X_train = [edge2tabular(train_edges[train_edges["step"] == i],
                                step_embeds[i][0], step_embeds[i][1]) for i in range(32 // stepsize)]
        X_train = np.concatenate(X_train, axis=0)
        y_train = train_edges["label"]
        X_test = [edge2tabular(test_edges[test_edges["step"] == i],
                               step_embeds[i][0], step_embeds[i][1]) for i in range(32 // stepsize)]
        X_test = np.concatenate(X_test, axis=0)
        y_test = test_edges["label"]
        # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        acc, f1, auc = lr_evaluate(X_train, y_train, X_test, y_test)
        write_result(sname, "triad", {"beta_1": 0.1,
                                      "beta_2": 0.1, "stepsize": stepsize}, (acc, f1, auc))


@iterate_times
def evaluate_tnode():
    fname, _ = iterate_datasets(dataset=args.dataset)
    fname = fname[args.start: args.end]
    logger.info("Running {} embedding programs.".format(args.method))
    run_tnode(dataset=args.dataset, n_jobs=args.n_jobs,
              start=args.start, end=args.end, times=args.times)
    logger.info("Done {}.".format(args.method))


@iterate_times
def evaluate_ctdne(project_dir="/nfs/zty/Graph/Dynamic-Graph/ctdne_output"):
    from .ctdne import ctdne
    fname, _ = iterate_datasets(dataset=args.dataset)
    fname = fname[args.start: args.end]
    if args.run:
        logger.info("Running {} embedding programs.".format(args.method))
        Parallel(n_jobs=args.n_jobs)(
            delayed(ctdne)(name[:-4]) for name in fname)
        logger.info("Done {} embeddings.".format(args.method))
    for name in fname:
        sname = name[:-4]
        logger.info(
            "dataset={}, num_walk=10, walk_length=80, context_window=10".format(sname))
        # print(name, p, q)
        train_path = os.path.join(train_data_dir, name)
        train_edges = pd.read_csv(train_path)
        test_path = os.path.join(test_data_dir, name)
        test_edges = pd.read_csv(test_path)

        fpath = "{}/{}.emb".format(project_dir, sname)
        id2idx, embs = load_embeddings(fpath, skiprows=0, sep=",")
        # TemporalRandomWalk loses many nodes, so we remove unseen nodes from train_edges.
        from_mask = train_edges["from_idx"].apply(lambda s: s in id2idx.keys())
        to_mask = train_edges["to_idx"].apply(lambda s: s in id2idx.keys())
        mask = np.logical_and(from_mask, to_mask)
        if sum(mask) > 1:
            logger.info(
                "{} {} edges removed from train edges.".format(sname, len(mask)))
            train_edges = train_edges[mask].reset_index(drop=True)

        X_train = edge2tabular(train_edges, id2idx, embs)
        y_train = train_edges["label"]
        X_test = edge2tabular(test_edges, id2idx, embs)
        y_test = test_edges["label"]
        # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        acc, f1, auc = lr_evaluate(X_train, y_train, X_test, y_test)
        write_result(sname, "ctdne", {
                     "num_walk": 10, "walk_length": 80, "context_window": 10}, (acc, f1, auc))


def edge2tabular(edges, id2idx, embs):
    unseen_from_nodes = set(edges["from_idx"]) - set(id2idx.keys())
    unseen_to_nodes = set(edges["to_idx"]) - set(id2idx.keys())
    logger.info("{} unseen from_nodes, {} unseen to_nodes in id2idx.".format(
        len(unseen_from_nodes), len(unseen_to_nodes)))

    edges["from_idx"] = edges["from_idx"].map(id2idx)
    edges["to_idx"] = edges["to_idx"].map(id2idx).astype('int64')
    # print(edges.dtypes)
    X = np.zeros((len(edges), 2 * embs.shape[1]))
    X_from = embs[edges["from_idx"]]
    X_to = embs[edges["to_idx"]]
    return np.concatenate((X_from, X_to), axis=1)


def lr_evaluate(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(random_state=42).fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    return acc, f1, auc


def write_result(dataset, method, params, metrics, result_dir="/nfs/zty/Graph/Dynamic-Graph/comp_results"):
    acc, f1, auc = metrics
    res_path = "{}/{}-{}.csv".format(result_dir, dataset, args.method)
    headers = ["method", "dataset", "accuracy", "f1", "auc", "params"]
    if not os.path.exists(res_path):
        f = open(res_path, 'w+')
        f.write(",".join(headers) + "\r\n")
        f.close()
        os.chmod(res_path, 0o777)
    with open(res_path, 'a') as f:
        result_str = "{},{},{:.4f},{:.4f},{:.4f}".format(
            method, dataset, acc, f1, auc)
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
def evaluate_tgat():
    fname, _ = iterate_datasets(dataset=args.dataset)
    fname = fname[args.start: args.end]
    if args.run:
        for name in fname:
            pass


@iterate_times
def evaluate_sage():
    fname, _ = iterate_datasets()
    fname = [name[:-4] for name in fname[args.start:args.end]]
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
