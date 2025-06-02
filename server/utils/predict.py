import os
import json
import torch
import numpy as np
import pandas
import time
import utils.scikit_wrappers as scikit_wrappers
from utils.dataloader import load_dataset
from config import Config

def calculate_anomaly_values(test_data, normal_train):
    """
    计算异常值
    """
    diff = np.abs(test_data - normal_train)
    range = np.max(diff) - np.min(diff)
    threshold = 0.3 * range
    anomaly_values = (diff > threshold).astype(int).tolist()
    return anomaly_values

def load_data(data):
    """
    将输入数据转换为模型所需的格式
    """
    min_length = min(len(d['values']) for d in data)
    values_array = np.array([d['values'][:min_length] for d in data])
    values_array = np.expand_dims(values_array, 1).astype(np.float64)
    return values_array

def predict(data, model_path, encoder, cuda=True, gpu=0, label_mode = 0):
    """
    使用保存的模型进行预测

    参数:
        data: 预处理后的晶圆数据
        model_path: 模型路径
        encoder: 编码器
        cuda: 是否使用CUDA
        gpu: GPU设备号
        label_mode: 标签模式 0:未打标，只输出预测结果 1：打标，输出预测准确率，以及判断错误的样本


    返回:
        dict: 包含预测结果的字典
    """
    start_time = time.time()

    # 检查CUDA可用性
    if cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")
        cuda = False

    # 准备数据
    test_data = load_data(data)

    test_encoded = encoder.encode(test_data)

    # 加载分类器
    classifier = scikit_wrappers.SVMClassifier()
    classifier.load(model_path)

    # 进行预测
    predictions = classifier.predict(test_encoded)

    # load train data 
    model_name = os.path.basename(model_path)
    train, train_labels = load_dataset(Config.DATA_DIR, model_name, train=True)
    # extract the train data with train label = 0
    normal_train = []
    for train_data, train_label in zip(train, train_labels):
        if train_label == 0:
            normal_train.append(train_data)
    normal_train = np.array(normal_train)
    normal_train = np.mean(normal_train, axis=0)
    test_data = test_data[:, :, :normal_train.shape[1]]
    if label_mode == "0":  # 修改为数字比较
        anomalies = []
        for i, pred in enumerate(predictions):
            if int(pred) == 1:
                anomaly = {}
                anomaly['WaferName'] = data[i]['WaferName']
                anomaly['values'] = calculate_anomaly_values(test_data[i], normal_train)
                anomalies.append(anomaly)
    else:
        correct = sum(1 for i, pred in enumerate(predictions) if data[i]['label'] == int(pred))
        wrong_predictions = []
        for i, pred in enumerate(predictions):
            if data[i]['label'] != int(pred):
                wrong_prediction = {}
                wrong_prediction['WaferName'] = data[i]['WaferName']
                if int(pred) == 1:
                    wrong_prediction['values'] = calculate_anomaly_values(test_data[i], normal_train)
                else:
                    wrong_prediction['values'] = [0] * normal_train.shape[1]
                wrong_predictions.append(wrong_prediction)


    end_time = time.time()
    inference_time = end_time - start_time

    if label_mode == "0":
        return {
            'status': 'success',
            'message': 'Prediction completed successfully',
            'anomalies': anomalies,
            'inference_time': inference_time
        }
    else:
        return {
            'status': 'success',
            'message': 'Validation completed successfully',
            'accuracy': correct / len(predictions),
            'wrong': wrong_predictions,
            'inference_time': inference_time
        }
