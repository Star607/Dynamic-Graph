"""For those models generating embeddings for nodes, we use a LR classifiers to 
classify if the edge exists. For those end2end models, we abstract a predict 
operation to provide a probability for existence of the edge.
"""
import argparse
import os
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from experiment.adaptors import (run_gta, run_htne, run_node2vec, run_tnode,
                                 run_triad, iterate_datasets)

parser = argparse.ArgumentParser(
    description="Perform contrastive experiments.")
parser.add_argument("--method", type=str, default="node2vec",
                    help="Contrastive method name.")
parser.add_argument("--n_jobs", type=int, default=16,
                    help="Job numbers for joblib Parallel function.")
parser.add_argument("--dataset", type=str,
                    default="all", help="Specific dataset for experiments; default is all datasets.")
parser.add_argument("--times", type=int, default=1,
                    help="Experiment repetition times.")
parser.add_argument("--start", type=int, default=0, help="Datset start index.")
parser.add_argument("--end", type=int, default=4, help="Datset end index.")

args = parser.parse_args()


def data_stats(project_dir="/nfs/zty/Graph/ctdne_data/"):
    # nodes, edges, d_avg, d_max, timespan(days)
    fname = [f for f in os.listdir(project_dir) if f.endswith("csv")]
    fname = sorted(fname)
    fpath = [os.path.join(project_dir, f) for f in fname]
    for name, path in zip(fname, fpath):
        print("*****{}*****".format(name))
        df = pd.read_csv(path)
        nodes = list(set(df["from_node_id"]).union(df["to_node_id"]))
        noint = sum([not isinstance(nid, int) for nid in nodes])
        print("{} node ids are not integer.".format(noint))

        adj = defaultdict(list)
        for row in df.itertuples():
            adj[row.from_node_id].append(row.to_node_id)
            adj[row.to_node_id].append(row.from_node_id)
        degrees = [len(v) for k, v in adj.items()]
        begin = datetime.fromtimestamp(min(df["timestamp"]))
        end = datetime.fromtimestamp(max(df["timestamp"]))
        delta = (end - begin).total_seconds() / 86400
        print("nodes:{} edges:{} d_max:{} d_avg:{:.2f} timestamps:{:.2f}".format(
            len(nodes), len(df), max(degrees), len(df)/len(nodes), delta))


def to_dataframe():
    '''
    Transform all edges files inside ctdne_data into five-column csv: from_node_id, to_node_id, timestamp, from_node_idx, to_node_idx
    '''
    project_dir = "/nfs/zty/Graph/ctdne_data/"
    oname = [f for f in os.listdir(project_dir) if f .endswith(".csv")]
    files = [os.path.join(project_dir, f) for f in oname]
    # header = ['from_node_id', 'to_node_id', 'timestamp']
    # header2 = ['from_node_id', 'to_node_id', 'state_label', 'timestamp']
    for f, name in zip(files, oname):
        name = name[:name.find('.')]
        print("*****{}*****".format(name))
        # df = pd.read_csv(f, header=None, sep="\s+")
        df = pd.read_csv(f)
        if len(df) >= 5:
            continue
        df = df.drop("state_label", axis=1)
        ids = set(df["from_node_id"]).union(set(df["to_node_id"]))
        id2idx = {idt: idx + 1 for idx, idt in enumerate(ids)}
        df["from_idx"] = df["from_node_id"].map(id2idx).astype("int64")
        df["to_idx"] = df["to_node_id"].map(id2idx).astype("int64")
        df.to_csv("{}/{}.csv".format(project_dir, name), index=None)
        nodes = pd.DataFrame()
        nodes["node_id"] = id2idx.keys()
        nodes["id_map"] = nodes["node_id"].map(id2idx).astype("int64")
        nodes.to_csv("{}/{}.nodes".format(project_dir, name), index=None)
    pass


def negative_sampling(df, nodes, node2id):
    df["label"] = 1
    neg_df = df.copy().reset_index(drop=True)
    neg_df["label"] = 0
    neg_toids = np.zeros(len(df), dtype=np.int32)
    for index, row in enumerate(df.itertuples()):
        pos_idx = node2id[row.to_node_id]
        # swap the positive index with the last element
        nodes[-1], nodes[pos_idx] = nodes[pos_idx], nodes[-1]
        neg_idx = np.random.choice(len(nodes) - 1)
        neg_toids[index] = nodes[neg_idx]
        # neg_test_df.loc[index, "to_node_id"] = nodes[neg_idx]
        # swap the positive index with the last element
        nodes[-1], nodes[pos_idx] = nodes[pos_idx], nodes[-1]
    neg_df["to_node_id"] = neg_toids
    df = df.append(neg_df, ignore_index=True)
    df = df.sort_values(by="timestamp")
    return df


def train_test_split(train_ratio=0.75, project_dir="/nfs/zty/Graph/ctdne_data/", store_dir="/nfs/zty/Graph/"):
    # nodes, edges, d_avg, d_max, timespan(days)
    fname = [f for f in os.listdir(project_dir) if f.endswith("csv")]
    fname = sorted(fname)
    fpath = [os.path.join(project_dir, f) for f in fname]
    for name, path in zip(fname, fpath):
        print("*****{}*****".format(name))
        start = time.time()
        df = pd.read_csv(path)
        df = df.drop("state_label", axis=1)
        df = df.sort_values(by="timestamp").reset_index(drop=True)
        train_end_idx = int(len(df) * train_ratio)
        # df.loc contains both start and stop
        train_df = df.iloc[:train_end_idx]  # .reset_index(drop=True)
        test_df = df.iloc[train_end_idx:].reset_index(drop=True)
        train_nodes = set(train_df["from_node_id"]).union(
            train_df["to_node_id"])
        test_nodes = set(test_df["from_node_id"]).union(test_df["to_node_id"])
        unseen_nodes = test_nodes - train_nodes
        from_edges = test_df["from_node_id"].apply(lambda x: x in unseen_nodes)
        to_edges = test_df["to_node_id"].apply(lambda x: x in unseen_nodes)
        edges = np.logical_or(from_edges, to_edges)
        print("unseen {} nodes, {} edges in test set".format(
            len(unseen_nodes), sum(edges)))
        # remove those edges containing unseen nodes
        test_df = test_df[np.logical_not(edges)]
        nodes = list(train_nodes)
        node2id = {key: idx for idx, key in enumerate(nodes)}
        train_df = negative_sampling(train_df, nodes, node2id)
        test_df = negative_sampling(test_df, nodes, node2id)

        # train_df.to_csv("{}/train_data/{}".format(store_dir, name), index=None)
        # test_df.to_csv("{}/test_data/{}".format(store_dir, name), index=None)
        end = time.time()
        print("test edges: {} time: {:.2f} seconds".format(
            len(test_df), end - start))
    pass


def train_test2idx(root_dir="/nfs/zty/Graph"):
    for fname in os.listdir(os.path.join(root_dir, "old_train_data")):
        name = fname[:-4]
        nodes = pd.read_csv(os.path.join(
            root_dir, "ctdne_data", name+".nodes"))
        id2idx = {row["node_id"]:  row["id_map"]
                  for index, row in nodes.iterrows()}
        path = os.path.join(root_dir, "old_train_data", fname)
        df = pd.read_csv(path)
        df["from_idx"] = df["from_node_id"].map(id2idx).astype("int64")
        df["to_idx"] = df["to_node_id"].map(id2idx).astype("int64")
        df.to_csv(os.path.join(root_dir, "train_data", name+".csv"), index=None)

    for fname in os.listdir(os.path.join(root_dir, "old_test_data")):
        name = fname[:-4]
        nodes = pd.read_csv(os.path.join(
            root_dir, "ctdne_data", name+".nodes"))
        id2idx = {row["node_id"]:  row["id_map"]
                  for index, row in nodes.iterrows()}
        path = os.path.join(root_dir, "old_test_data", fname)
        df = pd.read_csv(path)
        df["from_idx"] = df["from_node_id"].map(id2idx).astype("int64")
        df["to_idx"] = df["to_node_id"].map(id2idx).astype("int64")
        df.to_csv(os.path.join(root_dir, "test_data", name+".csv"), index=None)


if __name__ == "__main__":
    data_stats(project_dir="/nfs/zty/Graph/train_data")
    # to_dataframe()
    # train_test2idx()
    # train_test_split()
    # df = pd.read_csv("../../train_data/ia-contact.csv")
    # if args.method == "node2vec":
    #     run_method = run_node2vec
    # elif args.method == "triad":
    #     run_method = run_triad
    # elif args.method == "htne":
    #     run_method = run_htne
    # elif args.method == "tnode":
    #     run_method = run_tnode
    # elif args.method == "gta":
    #     run_method = run_gta
    # else:
    #     raise NotImplementedError(
    #         "Method {} not implemented!".format(args.method))
    # run_method(dataset=args.dataset, n_jobs=args.n_jobs,
    #            start=args.start, end=args.end, times=args.times)
