import argparse
import logging
import os
import time
from collections import defaultdict
from datetime import datetime

import dgl
import numpy as np
import pandas as pd

from data_loader.data_util import _iterate_datasets, _load_data


def data_stats(project_dir="/nfs/zty/Graph/format_data/"):
    # nodes, edges, d_avg, d_max, timespan(days)
    fname = [f for f in os.listdir(project_dir) if f.endswith("edges")]
    fname = sorted(fname)
    fpath = [os.path.join(project_dir, f) for f in fname]
    for name, path in zip(fname, fpath):
        name = name[:-6]
        print("*****{}*****".format(name))
        edges = pd.read_csv(path)
        nodes = pd.read_csv(os.path.join(project_dir, f"{name}.nodes"))
        enodes = list(set(edges["from_node_id"]).union(edges["to_node_id"]))
        assert len(nodes) == len(enodes), "The number of nodes is not the same as that of edges."
        noint = sum([not isinstance(nid, int) for nid in enodes])
        print("{} node ids are not integer.".format(noint))

        id2idx = {row.node_id: row.id_map for row in nodes.itertuples()}
        edges["from_node_id"] = edges["from_node_id"].map(id2idx)
        edges["to_node_id"] = edges["to_node_id"].map(id2idx)
        g = dgl.DGLGraph((edges["from_node_id"], edges["to_node_id"]))
        g.add_edges(edges["to_node_id"], edges["from_node_id"])
        degrees = g.in_degrees()
        begin = datetime.fromtimestamp(edges["timestamp"].min())
        end = datetime.fromtimestamp(edges["timestamp"].max())
        delta = (end - begin).total_seconds() / 86400
        print("density:{:.4f}, nodes:{} edges:{} d_max:{} d_avg:{:.2f} timestamps:{:.2f}".format(
            len(edges) * 2.0 / (len(nodes) * len(nodes) - 1), len(nodes), len(edges), max(degrees), len(edges)/len(nodes), delta))


def train_test_split(args, root_dir="/nfs/zty/Graph"):
    """We split the original data into three datasets along the time dimension: train, valid and test, according to `args.train_ratio` and `args.val_ratio`. Further, we remove nodes in valid and test datasets but that are unseen in train datasets. We re-index the node ids and save them in dataset.nodes files.
    """
    input_dir = os.path.join(root_dir, "format_data")
    train_dir = os.path.join(root_dir, "train_data")
    val_dir = os.path.join(root_dir, "valid_data")
    test_dir = os.path.join(root_dir, "test_data")
    for name in _iterate_datasets():
        edges, nodes = _load_data(dataset=name, mode="format_data")
        print("*****{}*****".format(name))
        start = time.time()
        edges = edges.sort_values(by="timestamp").reset_index(drop=True)
        ts = sorted(edges["timestamp"].unique())
        train_ts = ts[int(args.train_ratio * len(ts))]
        val_ts = ts[int((args.train_ratio + args.valid_ratio) * len(ts))]
        print("Timestamp {} slots, train timestamp cut at {} slot.".format(
            len(ts), int(args.train_ratio * len(ts))))
        train_edges = edges[edges["timestamp"] < train_ts]
        val_edges = edges[np.logical_and(
            edges["timestamp"] >= train_ts, edges["timestamp"] < val_ts)]
        test_edges = edges[edges["timestamp"] >= val_ts]
        train_nodes = set(train_edges["from_node_id"]).union(set(train_edges["to_node_id"]))
        print("Total {} nodes, Train/Unseen {}/{} nodes.".format(len(nodes),
                                                                 len(train_nodes), len(nodes)-len(train_nodes)))
        val_edges = val_edges[val_edges["from_node_id"].isin(train_nodes)]
        val_edges = val_edges[val_edges["to_node_id"].isin(train_nodes)]
        test_edges = test_edges[test_edges["from_node_id"].isin(train_nodes)]
        test_edges = test_edges[test_edges["to_node_id"].isin(train_nodes)]
        train_edges.to_csv(os.path.join(train_dir, f"{name}.edges"), index=None)
        val_edges.to_csv(os.path.join(val_dir, f"{name}.edges"), index=None)
        test_edges.to_csv(os.path.join(test_dir, f"{name}.edges"), index=None)
        nodes = nodes[nodes["node_id"].isin(train_nodes)]
        nodes["id_map"] = list(range(len(nodes)))
        for output in [train_dir, val_dir, test_dir]:
            nodes.to_csv(os.path.join(output, f"{name}.nodes"), index=None)
        print("Train {} edges, valid {} edges, test {} edges.".format(
            len(train_edges), len(val_edges), len(test_edges)))
        print("Time {:.2f}".format(time.time() - start))
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


def train_test_label(root_dir="/nfs/zty/Graph/"):
    # nodes, edges, d_avg, d_max, timespan(days)
    fname = _iterate_datasets()
    input_dirs = ["train_data", "valid_data", "test_data"]
    for name in fname:
        print("*****{}*****".format(name))
        start = time.time()
        for indir in input_dirs:
            outdir = "label_" + indir
            print("*****{}*****".format(indir))
            edges, nodes = _load_data(dataset=name, mode=indir)
            id2idx = {row.node_id: row.id_map for row in nodes.itertuples()}
            label_df = negative_sampling(edges, nodes["node_id"].to_numpy(), id2idx)
            path = os.path.join(root_dir, outdir, f"{name}.edges")
            label_df.to_csv(path, index=None)
        end = time.time()
        print("Time: {:.2f} secs".format(end - start))


def config_parser():
    parser = argparse.ArgumentParser("Configuration for a unified train-valid-test preprocesser.")
    parser.add_argument(
        "--task", "-t", choices=["datastat", "datasplit", "datalabel"], required=True)
    parser.add_argument("--train-ratio", "-tr", type=float,
                        default=0.70, help="Train dataset ratio.")
    parser.add_argument("--valid-ratio", "-vr", type=float, default=0.05,
                        help="Valid dataset ratio, and test ratio will be computed by (1-train_ratio-valid_ratio).")
    parser.add_argument("--label", dest="label", action="store_true", default=False,
                        help="Whether to generate negative samples for datasets. Each labeled dataset will have a suffix xxx_label.edges.")
    return parser.parse_args()


if __name__ == "__main__":
    import sys
    args = config_parser()
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.DEBUG)
    logger.info(args)
    if args.task == "datastat":
        data_stats(project_dir="/nfs/zty/Graph/format_data")
    elif args.task == "datasplit":
        train_test_split(args)
    elif args.task == "datalabel":
        train_test_label()
    # data_stats(project_dir="/nfs/zty/Graph/test_data")
    # to_dataframe()
    # train_test2idx()
    # train_test_split()
