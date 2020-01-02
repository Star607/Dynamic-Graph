"""For those models generating embeddings for nodes, we use a LR classifiers to 
classify if the edge exists. For those end2end models, we abstract a predict 
operation to provide a probability for existence of the edge.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


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
