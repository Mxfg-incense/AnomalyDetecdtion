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
from utils.dataloader import load_dataset

def train_classifier(dataset, data_path, save_path, encoder: scikit_wrappers.CausalCNNEncoder):
    """
    训练分类器的主函数

    参数:
        dataset (str): 数据集名称
        path (str): 数据集路径
        save_path (str): 模型保存路径
        encoder (CausalCNNEncoder): 编码器名称

    返回:
        dict: 包含训练结果的字典
    """
    start_time = time.time()

    # 加载数据集
    train, train_labels = load_dataset(data_path, dataset)

    train_encoded = encoder.encode(train)

    # 训练分类器
    classifier = scikit_wrappers.SVMClassifier()
    with open("./config/default_hyperparameters_classifier.json", 'r') as hf:
        hp_dict = json.load(hf)

    classifier.set_params(**hp_dict)


    with warnings.catch_warnings(record=True) as w:
        classifier.fit(train_encoded, train_labels)

        # 保存模型
        model_path = os.path.join(save_path, dataset)
        classifier.save(model_path)

        end_time = time.time()
        training_time = end_time - start_time

        for warning_message in w:
            if issubclass(warning_message.category, ConvergenceWarning):
                return {
                    'status': 'success',
                    'message': 'Warning: Solver terminated early. Please make sure training data is labeled carefully.',
                    'training_time': training_time,
                }
            
        return {
            'status': 'success',
            'message': 'Training completed successfully',
            'training_time': training_time,
        }
    