import numpy as np
import pandas as pd
import os
from itertools import product
from joblib import Parallel, delayed
import subprocess

def iterate_datasets(project_dir="/nfs/zty/Graph/"):
    fname = os.listdir(os.path.join(project_dir, "train_data"))
    fpath = [os.path.join(project_dir, "train_data/{}".format(f)) for f in fname]
    lines = [len(open(f, "r").readlines()) for f in fpath]
    # sort the dataset by train data size
    forder = [f for l, f in sorted(zip(lines, fname))]
    fpath = [os.path.join(project_dir, "train_data/{}".format(f)) for f in forder]
    return forder, fpath

def run_node2vec(project_dir="/nfs/zty/Graph/0-node2vec/", n_jobs=16):
    # :padataframe columns: from_node_id, to_node_id, timestamp
    # train embeddings for nodes
    # python src/main.py --input graph/karate.edgelist --output emb/karate.emd
    # input format: node1_id_int node2_id_int
    # output format: 
    # first line: num_of_nodes dim_of_representation
    # node_id dim1 dim2 ... dimd
    fname, fpath = iterate_datasets()
    command = "python {project_dir}/src/main.py --input {input} --output {output}.emb --p {p} --q {q} --iter 100"
    commands = []
    for name, file in zip(fname, fpath):
        name = name[:-4]
        input_path = os.path.join(project_dir, "graph/{}.csv".format(name))
        output_path = os.path.join(project_dir, "emb/{}".format(name))
        if not os.path.exists(input_path):
            df = pd.read_csv(file)
            df = df[["from_node_id", "to_node_id"]]
            df.to_csv(input_path, index=None, header=None, sep=" ")
    
        for p, q in product([0.25, 0.5, 1, 2, 4], [0.25, 0.5, 1, 2, 4]):
            opath = "{}-{p:.2f}-{q:.2f}".format(output_path, p=p, q=q) 
            cmd = command.format(project_dir=project_dir, input=input_path, output=opath, p=p, q=q)
            if not os.path.exists(opath):
                # print(cmd)
                commands.append(cmd)
                # os.system(cmd)
                # subprocess.Popen(cmd)
    Parallel(n_jobs=n_jobs)(delayed(os.system)(cmd) for cmd in commands)

def run_triad(project_dir="/nfs/zty/Graph/2-DynamicTriad/", n_jobs=16):
    pass
def run_htn2(project_dir="/nfs/zty/Graph/4-htne/", n_jobs=16):
    pass
def run_tnode(project_dir="/nfs/zty/Graph/5-tNodeEmbed/", n_jobs=16):
    pass
