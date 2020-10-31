from collections import defaultdict
import numpy as np
import pandas as pd
import os
from itertools import product
from joblib import Parallel, delayed
import subprocess
from data_loader.data_util import load_data, load_split_edges,  _iterate_datasets as iterate_datasets
import networkx as nx


def run_node2vec(dataset="all", n_jobs=16, project_dir="/nfs/zty/Graph/0-node2vec/", **kwargs):
    # :padataframe columns: from_node_id, to_node_id, timestamp
    # train embeddings for nodes
    # python src/main.py --input graph/karate.edgelist --output emb/karate.emd
    # input format: node1_id_int node2_id_int
    # output format:
    # first line: num_of_nodes dim_of_representation
    # node_id dim1 dim2 ... dimd
    fname = kwargs["fname"]
    command = "python {project_dir}/src/main.py --input {input} --output {output}.emb --p {p} --q {q}"
    commands = []
    for name in fname:
        input_path = os.path.join(project_dir, "graph/{}.csv".format(name))
        output_path = os.path.join(project_dir, "emb/{}".format(name))
        if not os.path.exists(input_path):
            edges, nodes = load_data(dataset=name, mode="train")[0]
            id2idx = {row.node_id: row.id_map for row in nodes.itertuples()}
            edges["from_node_id"] = edges["from_node_id"].map(id2idx)
            edges["to_node_id"] = edges["to_node_id"].map(id2idx)
            edges = edges[["from_node_id", "to_node_id"]]
            edges.to_csv(input_path, index=None, header=None, sep=" ")

        for p, q in product([0.25, 0.5, 1, 2, 4], [0.25, 0.5, 1, 2, 4]):
            opath = "{}-{p:.2f}-{q:.2f}".format(output_path, p=p, q=q)
            cmd = command.format(project_dir=project_dir,
                                 input=input_path, output=opath, p=p, q=q)
            # if not os.path.exists(opath):
            # print(cmd)
            commands.append(cmd)
    print("Preprocessing finished.")
    Parallel(n_jobs=n_jobs)(delayed(os.system)(cmd) for cmd in commands)


def run_triad(dataset="all", project_dir="/nfs/zty/Graph/2-DynamicTriad/", n_jobs=16, **kwargs):
    # input format: adjacency list
    #   1. undirected
    #   2. all vertices exist in all graphs
    #   3. leave no out-degree vertex only its node name
    #   4. unweighted edge uses 1.0 as its weight
    # output format: node_name d_1 d_2 ... d_k
    # equally divided into 32 snapshots
    def edges2adj(df, nodes):
        # df: DataFrame, nodes: dict(node_name, list)
        rdf = df[["to_node_id", "from_node_id", "timestamp"]]
        rdf.columns = ["from_node_id", "to_node_id", "timestamp"]
        df = pd.concat([df, rdf], sort=True).sort_values(
            by="timestamp").reset_index(drop=True)
        adj_ids = {nid: [] for nid in nodes}
        adj_cnt = {nid: [] for nid in nodes}
        for name, group in df.groupby("from_node_id"):
            vals = group["to_node_id"].value_counts().keys().tolist()
            cnts = group["to_node_id"].value_counts().tolist()
            adj_ids[name].extend(vals)
            adj_cnt[name].extend(cnts)
        return adj_ids, adj_cnt

    def write_adj(input_dir, i, adj_ids, adj_cnt):
        file = open("{}/{}".format(input_dir, i), "w")
        for nid in adj_ids.keys():
            vals = adj_ids[nid]
            cnts = adj_cnt[nid]
            line = "{} ".format(nid)
            line += " ".join(["{} {:.1f}".format(v, c)
                              for v, c in zip(vals, cnts)])
            # for v, c in zip(vals, cnts):
            #     line += "{} {:.1f} ".format(v, c)
            line += "\n"
            file.write(line)
        file.close()

    fname = kwargs["fname"]
    command = "python {project_dir} -n {timestep} -d {input_dir} -o {output_dir} -l {stepsize} -s {stepsize} --beta-smooth {b_1} --beta-triad {b_2} -K 128 --cachefn cache"
    commands = []
    for name in fname:
        input_dir = os.path.join(project_dir, "data/{}".format(name))
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
            df, nodes = load_data(dataset=name, mode="train")[0]
            id2idx = {row.node_id: row.id_map for row in nodes.itertuples()}
            df["from_node_id"] = df["from_node_id"].map(id2idx)
            df["to_node_id"] = df["to_node_id"].map(id2idx)

            nodes = set(id2idx.values())
            stride = len(df) // 128
            adjs = [edges2adj(df.iloc[i*stride:(i+1)*stride], nodes)
                    for i in range(128)]
            # for i, (adj_ids, adj_cnt) in enumerate(adjs):
            Parallel(n_jobs=32)(delayed(write_adj)(input_dir, i, adj_ids, adj_cnt)
                                for i, (adj_ids, adj_cnt) in enumerate(adjs))
        for stepsize in [1, 4, 16]:
            b_1 = b_2 = 0.1
            output_dir = os.path.join(
                project_dir, "output/{}-{}".format(name, stepsize))
            if not os.path.exists(output_dir) or len(os.listdir(output_dir)) < 32:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    os.chmod(output_dir, 0o777)
            cmd = command.format(
                project_dir=project_dir, timestep=32//stepsize, input_dir=input_dir, output_dir=output_dir, stepsize=stepsize, b_1=b_1, b_2=b_2)
            commands.append(cmd)
    print("Preprocessing finished.")
    Parallel(n_jobs=n_jobs)(delayed(os.system)(cmd) for cmd in commands)


def run_htne(dataset="all", project_dir="/nfs/zty/Graph/4-htne/", n_jobs=4, **kwargs):
    fname = iterate_datasets(dataset=dataset)
    command = "python {project_dir}/HTNE.py -d {input_path} -o {output_path} --hist-len {hist_len}"
    commands = []
    for name in fname:
        input_path = os.path.join(project_dir, "data/{}.edges".format(name))
        if not os.path.exists(input_path):
            df, nodes = load_data(dataset=name, mode="train")[0]
            df["timestamp"] = (df["timestamp"] - df["timestamp"].min()) / \
                (df["timestamp"].max() - df["timestamp"].min())
            id2idx = {row.node_id: row.id_map for row in nodes.itertuples()}
            df["from_node_id"] = df["from_node_id"].map(id2idx)
            df["to_node_id"] = df["to_node_id"].map(id2idx)
            df = df[["from_node_id", "to_node_id", "timestamp"]]
            df.to_csv(input_path, index=None, header=None, sep=" ")
        output_path = os.path.join(project_dir, "emb/{}.emb".format(name))
        for hist_len in [20]:
            hist_path = output_path + str(hist_len)
            commands.append(command.format(project_dir=project_dir,
                                           input_path=input_path, output_path=hist_path, hist_len=hist_len))
    print("Preprocessing finished.")
    Parallel(n_jobs=n_jobs)(delayed(os.system)(cmd) for cmd in commands)


def run_tnode(dataset="all", project_dir="/nfs/zty/Graph/5-tNodeEmbed/", n_jobs=16, **kwargs):
    fname = iterate_datasets(dataset=dataset)
    fname = fname[kwargs["start"]: kwargs["end"]]
    command = "python {project_dir}/src/main.py -d {dataset} -n {nstep}"
    commands = []
    for name, nstep in product(fname, [128, 32, 8]):
        dump_foler = os.path.join(project_dir, "data/{}".format(name))
        if not os.path.exists(dump_foler):
            os.makedirs(dump_foler)
            os.chmod(dump_foler, 0o777)
        cmd = command.format(project_dir=project_dir,
                             dataset=name, nstep=nstep)
        commands.append(cmd)
    os.chdir(project_dir)
    print("Preprocessing finished.")
    Parallel(n_jobs=n_jobs)(delayed(os.system)(cmd) for cmd in commands)


def run_gta(dataset="all", project_dir="/nfs/zty/Graph/Dynamic-Graph/", n_jobs=4, **kwargs):
    start, end = kwargs["start"], kwargs["end"]
    fname = iterate_datasets(dataset=dataset)[start: end]
    os.chdir(project_dir)

    command = "python main.py --dataset {dataset} --epochs 50 --dropout 0.2 --weight_decay 1e-5 --learning_rate=0.0001 --nodisplay "
    commands = []
    comps = []
    for name in fname:
        cmd = command.format(dataset=name)
        commands.append(cmd)
        commands.append(cmd + " --nodynamic_neighbor")
        commands.append(cmd + " --sampler temporal")

        commands.append(
            cmd + " --sampler temporal --use_context --context_size 20")
        commands.append(
            cmd + " --sampler temporal --use_context --context_size 40")
        commands.append(
            cmd + " --sampler temporal --use_context --context_size 80")
        commands.append(
            cmd + " --sampler temporal --use_context --context_size 160")
        commands.append(
            cmd + " --sampler mask --use_context --context_size 20")
        commands.append(
            cmd + " --sampler mask --use_context --context_size 40")
        commands.append(
            cmd + " --sampler mask --use_context --context_size 80")
        commands.append(
            cmd + " --sampler mask --use_context --context_size 160")
    # comps = repeat_string(comps)
    # commands = repeat_string(commands, times=1)
    print("Preprocessing finished.")
    # Parallel(n_jobs=n_jobs)(delayed(os.system)(cmd) for cmd in comps)
    Parallel(n_jobs=n_jobs)(delayed(os.system)(cmd) for cmd in commands)


def run_ctdne(dataset="all", project_dir="/nfs/zty/Graph/Dynamic-Graph/ctdne_embs/", n_jobs=4, **kwargs):
    from CTDNE import CTDNE
    fname = kwargs["fname"]
    for name in fname:
        edges, nodes = load_data(dataset=name, mode="train")[0]
        id2idx = {row.node_id: row.id_map for row in nodes.itertuples()}
        edges["from_node_id"] = edges["from_node_id"].map(id2idx)
        edges["to_node_id"] = edges["to_node_id"].map(id2idx)
        edges["time"] = edges["timestamp"]
        mg = nx.from_pandas_edgelist(edges, "from_node_id", "to_node_id", "time", nx.MultiGraph)
        CTDNE_model = CTDNE(mg, dimensions=128, workers=4)
        model = CTDNE_model.fit()
        wv = model.wv
        vecs = np.array([wv[u] for u in wv.vocab])
        df = pd.DataFrame(vecs, index=wv.vocab)
        real_nodes = set([str(i) for i in id2idx.values()])
        embed_nodes = set(wv.vocab.keys())
        print("{} nodes not exist in embed_nodes.".format(len(real_nodes - embed_nodes)))
        df.to_csv("{}/{}.emb".format(project_dir, name), header=None, sep=" ")


def repeat_string(cmds, times=5):
    return [cmd for cmd in cmds for _ in range(times)]
