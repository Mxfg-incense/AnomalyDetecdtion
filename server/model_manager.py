import os
from pathlib import Path
from utils.train import train_classifier
from utils.predict import predict
from config import Config
import utils.scikit_wrappers as scikit_wrappers

class ModelManager:
    def __init__(self):
        self.encoder_model = None
        self._load_encoder()

    def _load_encoder(self):
        """加载 encoder 模型"""

        self.encoder_model = scikit_wrappers.CausalCNNEncoder()
        self.encoder_model.set_params(**Config.DEFAULT_ENCODER_PARAMS)
        self.encoder_model.load(str(Config.ENCODER_PATH) + "/Wafer")


    def get_encoder(self):
        """获取全局 encoder 模型"""
        if self.encoder_model is None:
            self._load_encoder()
        return self.encoder_model

    def run_training(self, config):
        """运行训练流程"""
        try:
            
            result = train_classifier(
                dataset=config["modelName"],
                data_path=Config.DATA_DIR,
                save_path=Config.MODEL_DIR,
                encoder=self.get_encoder()
            )
            
            return result

        except Exception as e:
            raise Exception(f"Training failed: {str(e)}")

    def run_prediction(self, processed_data, dataset_name, label_mode):
        """运行预测流程"""
        try:
            model_path = Config.MODEL_DIR / dataset_name
            
            result = predict(
                data=processed_data,
                model_path=str(model_path),
                encoder=self.get_encoder(),
                label_mode=label_mode
            )
            
            return result

        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}") 