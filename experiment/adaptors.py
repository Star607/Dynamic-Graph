import numpy as np
import pandas as pd
import os
from itertools import product
from joblib import Parallel, delayed
import subprocess


def iterate_datasets(dataset="all", project_dir="/nfs/zty/Graph/"):
    fname = os.listdir(os.path.join(project_dir, "train_data"))
    fpath = [os.path.join(project_dir, "train_data/{}".format(f))
             for f in fname]
    lines = [len(open(f, "r").readlines()) for f in fpath]
    # sort the dataset by train data size
    forder = [f for l, f in sorted(zip(lines, fname))]
    fpath = [os.path.join(project_dir, "train_data/{}".format(f))
             for f in forder]
    if dataset != "all":
        forder = [name for name, file in zip(
            forder, fpath) if name[:-4] == dataset]
        fpath = [file for name, file in zip(
            forder, fpath) if name[:-4] == dataset]
    return forder, fpath


def run_node2vec(dataset="all", n_jobs=16, project_dir="/nfs/zty/Graph/0-node2vec/", **kwargs):
    # :padataframe columns: from_node_id, to_node_id, timestamp
    # train embeddings for nodes
    # python src/main.py --input graph/karate.edgelist --output emb/karate.emd
    # input format: node1_id_int node2_id_int
    # output format:
    # first line: num_of_nodes dim_of_representation
    # node_id dim1 dim2 ... dimd
    fname, fpath = iterate_datasets(dataset=dataset)
    command = "python {project_dir}/src/main.py --input {input} --output {output}.emb --p {p} --q {q}"
    commands = []
    for name, file in zip(fname, fpath):
        name = name[:-4]
        input_path = os.path.join(project_dir, "graph/{}.csv".format(name))
        output_path = os.path.join(project_dir, "emb/{}".format(name))
        if not os.path.exists(input_path):
            df = pd.read_csv(file)
            df = df[["from_idx", "to_idx"]]
            df.to_csv(input_path, index=None, header=None, sep=" ")

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
        rdf = df[["to_idx", "from_idx", "timestamp"]]
        rdf.columns = ["from_idx", "to_idx", "timestamp"]
        df = pd.concat([df, rdf], sort=True).sort_values(
            by="timestamp").reset_index(drop=True)
        adj_ids = {nid: [] for nid in nodes}
        adj_cnt = {nid: [] for nid in nodes}
        for name, group in df.groupby("from_idx"):
            vals = group["to_idx"].value_counts().keys().tolist()
            cnts = group["to_idx"].value_counts().tolist()
            adj_ids[name].extend(vals)
            adj_cnt[name].extend(cnts)
        # for name, group in df.groupby("to_node_id"):
        #     vals = group["from_node_id"].value_counts().keys().tolist()
        #     cnts = group["from_node_id"].value_counts().tolist()
        #     adj_ids[name].extend(vals)
        #     adj_cnt[name].extend(cnts)
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

    fname, fpath = iterate_datasets(dataset=dataset)
    fname = fname[kwargs["start"]: kwargs["end"]]
    fpath = fpath[kwargs["start"]: kwargs["end"]]
    command = "python {project_dir} -n {timestep} -d {input_dir} -o {output_dir} -l {stepsize} -s {stepsize} --beta-smooth {b_1} --beta-triad {b_2} -K 128 --cachefn cache"
    commands = []
    for name, file in zip(fname, fpath):
        name = name[:-4]
        input_dir = os.path.join(project_dir, "data/{}".format(name))
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
            df = pd.read_csv(file)

            nodes = set(df["from_idx"]).union(df["to_idx"])
            stride = len(df) // 32
            adjs = [edges2adj(df.iloc[i*stride:(i+1)*stride], nodes)
                    for i in range(32)]
            # for i, (adj_ids, adj_cnt) in enumerate(adjs):
            Parallel(n_jobs=32)(delayed(write_adj)(input_dir, i, adj_ids, adj_cnt)
                                for i, (adj_ids, adj_cnt) in enumerate(adjs))
        for stepsize in [1, 4, 8]:
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


def run_htne(dataset="all", project_dir="/nfs/zty/Graph/4-htne/", n_jobs=16, **kwargs):
    fname, fpath = iterate_datasets(dataset=dataset)
    command = "python {project_dir}/HTNE.py -d {input_path} -o {output_path}"
    commands = []
    for name, file in zip(fname, fpath):
        name = name[:-4]
        input_path = os.path.join(project_dir, "data/{}.csv".format(name))
        if not os.path.exists(input_path):
            df = pd.read_csv(os.path.join(
                project_dir, "../train_data/{}.csv".format(name)))
            df["timestamp"] = (df["timestamp"] - df["timestamp"].min()) / \
                (df["timestamp"].max() - df["timestamp"].min())
            df = df["from_idx", "to_idx", "timestamp"]  # the 3rd column is time
            nodes = set(df.from_idx).union(set(df.to_idx))
            id2idx = {nid: idx for idx, nid in enumerate(nodes)}
            df.from_idx = df.from_idx.map(id2idx)
            df.to_idx = df.to_idx.map(id2idx)
            df.to_csv(input_path, index=None, header=None, sep=" ")
        output_path = os.path.join(project_dir, "emb/{}.emb".format(name))
        if not os.path.exists(output_path):
            commands.append(command.format(project_dir=project_dir,
                                           input_path=input_path, output_path=output_path))
    print("Preprocessing finished.")
    Parallel(n_jobs=n_jobs)(delayed(os.system)(cmd) for cmd in commands)


def run_tnode(dataset="all", project_dir="/nfs/zty/Graph/5-tNodeEmbed/", n_jobs=16, **kwargs):
    fname, fpath = iterate_datasets(dataset=dataset)
    fname = fname[kwargs["start"]: kwargs["end"]]
    command = "python {project_dir}/src/main.py -d {dataset} -n {nstep}"
    commands = []
    for name, nstep in product(fname, [32]):
        name = name[:-4]
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


def run_gta(dataset="all", project_dir="/nfs/zty/Graph/Dynamic-Graph/", n_jobs=1, **kwargs):
    start, end = kwargs["start"], kwargs["end"]
    fname, _ = iterate_datasets(dataset=dataset)
    fname = [name[:-4] for name in fname[start:end]]
    os.chdir(project_dir)

    command = "python main.py --dataset {dataset} --epochs 50 --dropout 0.2 --weight_decay 1e-5 --learning_rate=0.0001 --nodisplay "
    commands = []
    comps = []
    for name in fname:
        cmd = command.format(dataset=name)
        # commands.append(cmd)
        commands.append(cmd + " --nodynamic_neighbor")
        commands.append(cmd + " --sampler temporal")
        # commands.append(
        # cmd + " --sampler mask --use_context --context_size 10")
        # commands.append(
        #     cmd + " --sampler temporal --use_context --context_size 10")
        commands.append(
            cmd + " --sampler temporal --use_context --context_size 1")
        commands.append(
            cmd + " --sampler temporal --use_context --context_size 5")
        commands.append(
            cmd + " --sampler temporal --use_context --context_size 20")
        commands.append(
            cmd + " --sampler mask --use_context --context_size 1")
        commands.append(
            cmd + " --sampler mask --use_context --context_size 5")
        commands.append(
            cmd + " --sampler mask --use_context --context_size 20")
        # comps.append(cmd + " --sampler mask --use_context --context_size 20")
    # comps = repeat_string(comps)
    # commands = repeat_string(commands, times=1)
    print("Preprocessing finished.")
    # Parallel(n_jobs=n_jobs)(delayed(os.system)(cmd) for cmd in comps)
    Parallel(n_jobs=n_jobs)(delayed(os.system)(cmd) for cmd in commands)


def repeat_string(cmds, times=5):
    return [cmd for cmd in cmds for _ in range(times)]
