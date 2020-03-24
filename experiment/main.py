"""For those models generating embeddings for nodes, we use a LR classifiers to 
classify if the edge exists. For those end2end models, we abstract a predict 
operation to provide a probability for existence of the edge.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

from datetime import datetime
from collections import defaultdict 

def data_stats(project_dir="../../ctdne_data/"):
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


def train_test_split():
    pass


def iterate_datasets():
    pass


def load_embeddings():
    pass


def get_model():
    pass


def predict(X, y):
    pass

if __name__ == "__main__":
    data_stats()
    pass
        
    
        