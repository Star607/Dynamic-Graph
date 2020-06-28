from gensim.models import Word2Vec
from stellargraph.data import BiasedRandomWalk
from stellargraph.data import TemporalRandomWalk
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from stellargraph import StellarGraph
from stellargraph.datasets import IAEnronEmployees
import os
import argparse


def ctdne(dataset="ia-contact", output="/nfs/zty/Graph/Dynamic-Graph/ctdne_output"):
    # if os.path.exists("{}/{}.emb".format(output, dataset)):
    #     return
    print("Begin CTDNE {} embeddings.".format(dataset))
    train_dir = "/nfs/zty/Graph/train_data"
    df = pd.read_csv("{}/{}.csv".format(train_dir, dataset))
    edges_df = df[df["label"] == 1].copy()
    print("original edges: {} filtering edges: {}".format(len(df), len(edges_df)))
    edges_df = edges_df[["from_idx", "to_idx", "timestamp"]].copy()
    edges_df.columns = ["source", "target", "time"]
    edges_df["source"] = edges_df["source"].astype(str)
    edges_df["target"] = edges_df["target"].astype(str)
    # nodes = list(set(edges_df["source"]).union(set(edges_df["target"])))
    graph = StellarGraph(
        edges=edges_df,
        edge_weight_column="time",
    )

    num_walks_per_node = 10
    walk_length = 80
    context_window_size = 10
    num_cw = len(graph.nodes()) * num_walks_per_node *\
        (walk_length - context_window_size + 1)

    print("Begin CTDNE TemporalRandomWalk.")
    temporal_rw = TemporalRandomWalk(graph)
    temporal_walks = temporal_rw.run(
        num_cw=num_cw,
        cw_size=context_window_size,
        max_walk_length=walk_length,
        walk_bias="exponential",
    )
    print("End CTDNE TemporalRandomWalk.")
    embedding_size = 128
    temporal_model = Word2Vec(temporal_walks, size=embedding_size,
                              window=context_window_size, min_count=0, sg=1, workers=16, iter=1)

    print("Done CTDNE {} embeddings.".format(dataset))

    # if not os.path.exists("{}/{}.emb".format(output, dataset)):
    wv = temporal_model.wv
    vecs = np.array([wv[u] for u in wv.vocab])
    df = pd.DataFrame(vecs, index=wv.vocab)
    walks_nodes = set([s for l in temporal_walks for s in l])
    embed_nodes = set(wv.vocab.keys())
    nodes = set(edges_df["source"]).union(set(edges_df["target"]))
    print("{} nodes not exist in walk_nodes.".format(len(nodes - walks_nodes)))
    print("{} nodes not exist in embed_nodes.".format(len(nodes - embed_nodes)))

    emb_path = "{}/{}.emb".format(output, dataset)
    if not os.path.exists(emb_path):
        f = open(emb_path, "w")
        f.close()
        os.chmod(emb_path, 0o777)
    df.to_csv("{}/{}.emb".format(output, dataset), header=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="ia-contact", help="Specific dataset for experiments.")

    args = parser.parse_args()

    ctdne(args.dataset)
