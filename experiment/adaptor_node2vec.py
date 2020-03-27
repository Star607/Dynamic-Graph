import numpy as np
import pandas as pd
import os
from itertools import product

def run_node2vec(df, name, project_dir="/nfs/zty/Graph/0-node2vec/"):
    # :padataframe columns: from_node_id, to_node_id, timestamp
    # train embeddings for nodes
    # python src/main.py --input graph/karate.edgelist --output emb/karate.emd
    # input format: node1_id_int node2_id_int
    # output format: 
    # first line: num_of_nodes dim_of_representation
    # node_id dim1 dim2 ... dimd
    input_path = os.path.join(project_dir, "graph/{}".format(name))
    output_path = os.path.join(project_dir, "emb/{}".format(name))
    df = df[["from_node_id", "to_node_id"]]
    df.to_csv(input_path, index=None, header=None, sep=" ")
    for p, q in product([0.25, 0.5, 1, 2, 4], [0.25, 0.5, 1, 2, 4]):
        opath = "{}-{p:.2f}-{q:.2f}".format(output_path, p=p, q=q) 
        command = "nohup python {project_dir}/src/main.py --input {input} --output {output} --p {p} --q {q} --iter 100 &> {output}.out &".format(project_dir=project_dir, input=input_path, output=opath, p=p, q=q)
        # command = "python {project_dir}/src/main.py --input {input} --output {output} --p {p} --q {q} --iter 100 &> {output}.out".format(project_dir=project_dir, input=input_path, output=opath, p=p, q=q)
        if not os.path.exists(opath):
            print(command)
            os.system(command)
