import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time


def box_analysis(data):
    q_up = data.quantile(q=0.75)
    q_low = data.quantile(q=0.25)
    iqr = q_up - q_low
    up = q_up + 1.5 * iqr
    low = q_low - 1.5 * iqr
    bool_index = (data <= up) & (data >= low)
    return bool_index


def cal_time(stamp1, stamp2):
    t1 = time.localtime(stamp1)
    t2 = time.localtime(stamp2)
    t1 = time.strftime("%Y-%m-%d %H:%M:%S", t1)
    t2 = time.strftime("%Y-%m-%d %H:%M:%S", t2)
    time1 = datetime.datetime.strptime(t1, "%Y-%m-%d %H:%M:%S")
    time2 = datetime.datetime.strptime(t2, "%Y-%m-%d %H:%M:%S")
    return (time2 - time1).days


project_dir = "/nfs/zty/Graph/format_data/"
root_dir = "/new_disk_1/wy/MyCode/Dynamic-Graph/images_edges_distribution/"

fname = [f for f in os.listdir(project_dir) if f.endswith("edges")]
fname = sorted(fname)
fpath = [os.path.join(project_dir, f) for f in fname]
for name, path in zip(fname, fpath):
    name = name[:-6]
    print("*****{}*****".format(name))
    edges = pd.read_csv(path)
    timestamp = edges['timestamp']
    timestamp_c = timestamp[box_analysis(timestamp)]
    tmin = timestamp_c.min()
    timestamp_c = [cal_time(tmin, i) for i in timestamp_c]

    x, y = np.unique(timestamp_c, return_counts=True)
    plt.bar(x, y)
    plt.title(name)
    plt.savefig(os.path.join(root_dir, f"{name}.png"), dpi=600)
    plt.show()
