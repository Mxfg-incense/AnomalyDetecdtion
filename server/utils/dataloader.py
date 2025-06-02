import os
import json
import math
import torch
import numpy
import pandas
import argparse
import time 
import warnings
from sklearn.exceptions import ConvergenceWarning
import utils.scikit_wrappers as scikit_wrappers



def load_dataset(path, dataset, train=True):
    """
    Loads the UCR dataset given in input in numpy arrays.

    @param path Path where the UCR dataset is located.
    @param dataset Name of the UCR dataset.

    @return Quadruplet containing the training set, the corresponding training
            labels, the testing set and the corresponding testing labels.
    """
    if train:
        train_file = os.path.join(path, dataset + "_train.txt")
    else:
        train_file = os.path.join(path, dataset + "_test.txt")
    # 读取csv文件,每行可能列数不同
    with open(train_file, 'r') as f:
        lines = f.readlines()
    # 找到最短的列数
    min_cols = min(len(line.split()) for line in lines)
    # 截断每行到最短长度并转换为DataFrame
    train_df = [line.split()[:min_cols] for line in lines]
    train_array = numpy.array(train_df)

    # Move the labels to {0, ..., L-1}
    labels = numpy.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = numpy.expand_dims(train_array[:, 1:], 1).astype(numpy.float64)
    train_labels = numpy.vectorize(transform.get)(train_array[:, 0])

    return train, train_labels
