import argparse
import os
import logging
import time

import numpy as np
import pandas as pd
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch
from torch import nn
import dgl

from data_loader.minibatch import load_data, TemporalEdgeBatchIterator


class GTA(nn.Module):
    pass


def config_parser():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


if __name__ == "__main__":
    pass
