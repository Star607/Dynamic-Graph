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

train_data_dir = "/nfs/zty/Graph/train_data/"
test_data_dir = "/nfs/zty/Graph/test_data/"


def load_embeddings(path, skiprows=0, sep=" "):
    df = pd.read_csv(path, skiprows=skiprows, sep=sep, header=None)
    nodes = df[0]
    id2idx = {row[0]: index for index, row in df.iterrows()}
    id2idx = defaultdict(lambda: 0, id2idx)
    embs = df.loc[:, 1:].to_numpy()
    # print(embs.shape)
    assert(embs.shape[1] == 128)
    return id2idx, embs


def iterate_times(method):
    def worker(*ar, **kw):
        results = []
        for _ in range(args.times):
            logging.info("Times {} begins.".format(_))
            res = method(*ar, **kw)
            results.append(res)
        return results
    return worker


@iterate_times
def evaluate_node2vec(project_dir="/nfs/zty/Graph/0-node2vec/emb"):
    fname, _ = iterate_datasets()[args.start: args.end]
    if args.run:
        logging.info("Running {} embedding programs.".format("node2vec"))
        run_node2vec(dataset=args.dataset, n_jobs=args.n_jobs,
                     start=args.start, end=args.end, times=args.times)
        logging.info("Done node2vec embedding.")
    else:
        logging.info("Use pretrained {} embeddings.".format("node2vec"))
    for name, p, q in product(fname, [0.25, 0.5, 1, 2, 4], [0.25, 0.5, 1, 2, 4]):
        sname = name[:-4]
        logging.info("dataset={}, p={:.2f}, q={:.2f}".format(sname, p, q))
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
    sname = name[:-4]
    logging.info(sname)
    # print(name)
    train_path = os.path.join(train_data_dir, name)
    train_edges = pd.read_csv(train_path)
    test_path = os.path.join(test_data_dir, name)
    test_edges = pd.read_csv(test_path)

    fpath = "{}/{}-0.1-0.1/31.out".format(project_dir, sname)
    id2idx, embs = load_embeddings(fpath, skiprows=0)
    X_train = edge2tabular(train_edges, id2idx, embs)
    y_train = train_edges["label"]
    X_test = edge2tabular(test_edges, id2idx, embs)
    y_test = test_edges["label"]
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # acc, f1, auc = lr_evaluate(X_train, y_train, X_test, y_test)
    # write_result(sname, "triad", {"beta_1": 0.1,
    #                               "beta_2": 0.1}, (acc, f1, auc))


def edge2tabular(edges, id2idx, embs):
    edges["from_idx"] = edges["from_idx"].map(id2idx)
    logger.info("{} unseen from_nodes in id2idx".format(
        len(set(edges["from_idx"]) - set(id2idx.keys()))))
    edges["to_idx"] = edges["to_idx"].map(id2idx).astype('int64')
    logger.info("{} unseen to_nodes in id2idx".format(
        len(set(edges["to_idx"]) - set(id2idx.keys()))))
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
    res_path = "{}/{}.csv".format(result_dir, dataset)
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
        logging.info(
            "{}-{}: {:.3f}, {:.3f}, {:.3f}".format(method, dataset, acc, f1, auc))
        # print("{}-{}: {:.3f}, {:.3f}, {:.3f}".format(method, dataset, acc, f1, auc))
        f.write(row)


@iterate_times
def evaluate_gta():
    run_gta(dataset=args.dataset, start=args.start, end=args.end)


if __name__ == "__main__":
    if args.method == "node2vec":
        evaluate = evaluate_node2vec
    elif args.method == "triad":
        evaluate = evaluate_triad
    elif args.method == "htne":
        evaluate = evaluate_htne
    elif args.method == "tnode":
        evaluate = evaluate_tnode
    elif args.method == "gta":
        evaluate = evaluate_gta
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
