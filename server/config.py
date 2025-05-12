from pathlib import Path
import json

class Config:
    DATA_DIR = Path("./data/")
    LOG_DIR = Path("./log")
    MODEL_DIR = Path("./models/classifier")
    ENCODER_PATH = Path("./models/encoder")
    
    REQUIRED_TRAIN_FIELDS = [
        "eqID", "chamberID", "recipe", "parameterName", 
        "trainingMode", "labelMode", "Wafers", "modelName"
    ]
    
    REQUIRED_PREDICT_FIELDS = [
        "eqID", "chamberID", "recipe", "parameterName",
        "Wafers", "labelMode", "modelName"
    ]
    
    DEFAULT_ENCODER_PARAMS = json.load(open("./config/default_hyperparameters_encoder.json", 'r'))
    
    
    @classmethod
    def ensure_directories(cls):
        """确保必要的目录存在"""
        for directory in [cls.DATA_DIR, cls.LOG_DIR, cls.MODEL_DIR, cls.ENCODER_PATH]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_config(cls, config: dict, mode: str) -> str:
        """验证配置参数
        
        Args:
            config: 配置字典
            mode: 验证模式 ("train" 或 "predict")
            
        Returns:
            str: 错误信息，如果验证通过则返回 None
        """
        required_fields = cls.REQUIRED_TRAIN_FIELDS if mode == "train" else cls.REQUIRED_PREDICT_FIELDS
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        if not config.get("Wafers"):
            raise ValueError("Wafer data cannot be empty")
            
        if mode == "train":
            if config["trainingMode"] not in ["0", "1"]:
                raise ValueError("Invalid training mode, must be '0' or '1'")
            if config["labelMode"] not in ["0", "1"]:
                raise ValueError("Invalid label mode, must be '0' or '1'")
            # 增量训练模式验证
            if config["trainingMode"] == "1" and config["labelMode"] == "0":
                raise ValueError("Incremental training mode does not accept unlabelled data")
        # replace the '/' with '_' in modelName
        config["modelName"] = config["modelName"].replace("/", "_")
    
    @classmethod
    def get_default_model_name(cls, config):
        """生成数据集名称"""
        return f"{config['eqID']}_{config['chamberID']}_{config['recipe']}_{config['parameterName']}" 