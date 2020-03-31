"""For those models generating embeddings for nodes, we use a LR classifiers to 
classify if the edge exists. For those end2end models, we abstract a predict 
operation to provide a probability for existence of the edge.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

import argparse

parser = argparse.ArgumentParser(description="Perform contrastive experiments.")
parser.add_argument("--method", type=str, default="node2vec", help="Contrastive method name.")
parser.add_argument("--n_jobs", type=int, default=16, help="Job numbers for joblib Parallel function.")
parser.add_argument("--dataset", type=str, 
default="all", help="Specific dataset for experiments; default is all datasets.")

args = parser.parse_args()

from datetime import datetime
from collections import defaultdict 
import time

from adaptors import run_node2vec, run_triad, run_htne, run_tnode

def data_stats(project_dir="/nfs/zty/Graph/ctdne_data/"):
    # nodes, edges, d_avg, d_max, timespan(days) 
    fname = [f for f in os.listdir(project_dir) if f.endswith("csv")]
    fname = sorted(fname)
    fpath = [project_dir + f for f in fname]
    for name, path in zip(fname, fpath):
        print("*****{}*****".format(name))
        df = pd.read_csv(path)
        nodes = list(set(df["from_node_id"]).union(df["to_node_id"]))
        adj = defaultdict(list)
        for row in df.itertuples():
            adj[row.from_node_id].append(row.to_node_id)
            adj[row.to_node_id].append(row.from_node_id)
        degrees = [len(v) for k, v in adj.items()]
        begin = datetime.fromtimestamp(min(df["timestamp"]))
        end = datetime.fromtimestamp(max(df["timestamp"]))
        delta = (end - begin).total_seconds() / 86400
        print("nodes:{} edges:{} d_max:{} d_avg:{:.2f} timestamps:{:.2f}".format(len(nodes), len(df), max(degrees), len(df)/len(nodes), delta))

def to_dataframe(fname=["ia-contact.edges"]):
    # project_dir = "../../ctdne_data/"
    # fname = ["ia-contact.edges", "ia-enron-employees.edges", "ia-escorts-dynamic.edges", "ia-frwikinews-user-edits.edges", "ia-movielens-user2tags-10m.edges", "ia-radoslaw-email.edges", "ia-slashdot-reply-dir.edges"]
    # oname = [f for f in os.listdir(project_dir) if f .endswith(".edges") and f not in set(fname)]
    # # files = [project_dir + f for f in fname]
    # files = [project_dir + f for f in oname]
    # header = ['from_node_id', 'to_node_id', 'timestamp']
    # header2 = ['from_node_id', 'to_node_id', 'state_label', 'timestamp']
    # # for f, name in zip(files, fname):
    # for f, name in zip(files, oname):
    #     name = name[:name.find('.')]
    #     print("*****{}*****".format(name))
    #     # df = pd.read_csv(f, header=None, sep="\s+")
    #     df = pd.read_csv(f, header=None)
    #     if len(df.columns) == 3:
    #         df.columns = header
    #         df["state_label"] = 0
    #         df = df[header2]
    #     else:
    #         df.columns = header2
    #     df.to_csv("{}/{}.csv".format(project_dir, name), index=None)
    pass


def train_test_split(train_ratio=0.75, project_dir="/nfs/zty/Graph/ctdne_data/", store_dir="/nfs/zty/Graph/Dynamic-Graph/"):
    # nodes, edges, d_avg, d_max, timespan(days) 
    fname = [f for f in os.listdir(project_dir) if f.endswith("csv")]
    fname = sorted(fname)
    fpath = [project_dir + f for f in fname]
    for name, path in zip(fname, fpath):
        print("*****{}*****".format(name))
        start = time.time()
        df = pd.read_csv(path)
        df = df.drop("state_label", axis=1)
        df = df.sort_values(by="timestamp").reset_index(drop=True)
        train_end_idx = int(len(df) * train_ratio)
        # df.loc contains both start and stop
        train_df = df.iloc[:train_end_idx]#.reset_index(drop=True)
        test_df = df.iloc[train_end_idx:].reset_index(drop=True)
        test_df["label"] = 1
        
        train_nodes = set(train_df["from_node_id"]).union(train_df["to_node_id"])
        test_nodes = set(test_df["from_node_id"]).union(test_df["to_node_id"])
        unseen_nodes = test_nodes - train_nodes
        from_edges = test_df["from_node_id"].apply(lambda x: x in unseen_nodes)
        to_edges = test_df["to_node_id"].apply(lambda x: x in unseen_nodes)
        edges = np.logical_or(from_edges, to_edges)
        print("unseen {} nodes, {} edges in test set".format(len(unseen_nodes), sum(edges)))
        # remove those edges containing unseen nodes
        test_df = test_df[np.logical_not(edges)]

        nodes = list(set(df["from_node_id"]).union(df["to_node_id"]))
        node2id = {key: idx for idx, key in enumerate(nodes)}
        neg_test_df = test_df.copy().reset_index(drop=True)
        neg_test_df["label"] = 0
        neg_toids = np.zeros(len(neg_test_df), dtype=np.int32)
        for index, row in enumerate(test_df.itertuples()):
            pos_idx = node2id[row.to_node_id]
            # swap the positive index with the last element
            nodes[-1], nodes[pos_idx] = nodes[pos_idx], nodes[-1]
            neg_idx = np.random.choice(len(nodes) - 1)
            neg_toids[index] = nodes[neg_idx]
            # neg_test_df.loc[index, "to_node_id"] = nodes[neg_idx]
            # swap the positive index with the last element
            nodes[-1], nodes[pos_idx] = nodes[pos_idx], nodes[-1]
        neg_test_df["to_node_id"] = neg_toids
        test_df = test_df.append(neg_test_df, ignore_index=True)
        test_df = test_df.sort_values(by="timestamp")
        train_df.to_csv("{}/train_data/{}".format(store_dir, name), index=None)
        test_df.to_csv("{}/test_data/{}".format(store_dir, name), index=None)
        end = time.time()
        print("test edges: {} time: {:.2f} seconds".format(len(test_df), end - start))
    pass


def iterate_datasets(project_dir="/nfs/zty/Graph/", method="node2vec"):
    fname = os.listdir(os.path.join(project_dir, "train_data"))
    fpath = [os.path.join(project_dir, "train_data/{}".format(f)) for f in fname]
    lines = [len(open(f, "r").readlines()) for f in fpath]
    # sort the dataset by train data size
    forder = [f for l, f in sorted(zip(lines, fname))]
    fpath = [os.path.join(project_dir, "train_data/{}".format(f)) for f in forder]
    for name, file in zip(fname, fpath):
        df = pd.read_csv(file)
        if method == "node2vec":
            run_node2vec(df, name[:-4])


def load_embeddings():
    pass


def get_model():
    pass


def predict(X, y):
    pass

if __name__ == "__main__":
    # data_stats()
    # train_test_split()
    # df = pd.read_csv("../../train_data/ia-contact.csv")
    # run_node2vec(df, "ia-contact")
    if args.method == "node2vec":
        run_node2vec(dataset=args.dataset, n_jobs=args.n_jobs)
    elif args.method == "triad":
        run_triad(dataset=args.dataset, n_jobs=args.n_jobs)
    elif args.method == "htne":
        run_htne(dataset=args.dataset, n_jobs=args.n_jobs)
    elif args.method == "tnode":
        run_tnode(dataset=args.dataset, n_jobs=args.n_jobs)
    else:
        raise NotImplementedError("Method {} not implemented!".format(args.method))
    # iterate_datasets()
    
    
        