import os
import pandas as pd
import numpy as np
import dgl
import collections
import networkx as nx
import matplotlib.pyplot as plt
import torch


def degree_of_graph(degrees, name, type):
    root_dir = "/new_disk_1/wy/MyCode/Dynamic-Graph/images_degrees_distribution/"
    x, y = np.unique(degrees, return_counts=True)

    plt.yscale('log')
    plt.xscale('log')
    # plt.scatter(x, y, c="black", s=10, marker='o', alpha=0.3)
    plt.scatter(x, y, c="red", s=10, marker='o', alpha=0.5)
    title = name + ' ' + type + "-Degree Distribution"
    plt.title(title)

    plt.savefig(os.path.join(root_dir, f"{name + '_' + type + '_degree_distribution'}.png"), dpi=600)
    plt.show()


project_dir = "/nfs/zty/Graph/format_data/"
fname = [f for f in os.listdir(project_dir) if f.endswith("edges")]
fname = sorted(fname)
fpath = [os.path.join(project_dir, f) for f in fname]
for name, path in zip(fname, fpath):
    name = name[:-6]
    print("*****{}*****".format(name))
    edges = pd.read_csv(path)
    nodes = pd.read_csv(os.path.join(project_dir, f"{name}.nodes"))
    enodes = list(set(edges["from_node_id"]).union(edges["to_node_id"]))  # 取并集
    assert len(nodes) == len(enodes), "The number of nodes is not the same as that of edges."
    noint = sum([not isinstance(nid, int) for nid in enodes])
    print("{} node ids are not integer.".format(noint))

    id2idx = {row.node_id: row.id_map for row in nodes.itertuples()}  # 花括号内是键值对，表示字典
    edges["from_node_id"] = edges["from_node_id"].map(id2idx)
    edges["to_node_id"] = edges["to_node_id"].map(id2idx)
    g = dgl.DGLGraph((edges["from_node_id"], edges["to_node_id"]))
    # g.add_edges(edges["to_node_id"], edges["from_node_id"])  #添加反向边

    in_degrees = g.in_degrees()  # tensor([0, 2, 1, 1, ...])
    in_degrees = pd.Series(in_degrees.numpy())
    in_degrees = in_degrees[in_degrees > 0]
    degree_of_graph(in_degrees, name, type="In")
    print('*******************' + name + 'In Degree Distribution' + '*******************')
    print(in_degrees.describe())

    out_degrees = g.out_degrees()
    out_degrees = pd.Series(out_degrees.numpy())
    out_degrees = out_degrees[out_degrees > 0]
    degree_of_graph(out_degrees, name, type="Out")
    print('*******************' + name + 'Out Degree Distribution' + '*******************')
    print(out_degrees.describe())
