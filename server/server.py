from flask import Flask, request, jsonify, render_template, redirect
from config import Config
from data_processor import DataProcessor
from logger import Logger
from model_manager import ModelManager

app = Flask(__name__)

# 初始化组件
Config.ensure_directories()
data_processor = DataProcessor(Config.DATA_DIR)
logger = Logger(Config.LOG_DIR)
model_manager = ModelManager()

@app.route("/train", methods=["POST"])
def train():
    """处理训练请求"""
    try:   
        config = request.get_json()
        Config.validate_config(config, "train")

        processed_data = data_processor.process_wafer_data(
            config["Wafers"], 
            config["trainingMode"], 
            config["labelMode"]
        )

        data_processor.save_training_data(
            processed_data, 
            f"{config['modelName']}_train.txt", 
            config["trainingMode"]
        )

        result = model_manager.run_training(config)
        logger.save_log(result, "training")
        return jsonify(result)
        
    except Exception as e:
        print("Error:", str(e))
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route("/predict", methods=["POST"])
def predict():
    """处理预测请求"""
    try:
        config = request.get_json()
        Config.validate_config(config, "predict")

        processed_data = data_processor.process_wafer_data(
            config["Wafers"], 
            None, 
            label_mode=config["labelMode"]
        )

        result = model_manager.run_prediction(
            processed_data, 
            config["modelName"],
            config["labelMode"]
        )
        
        logger.save_log(result, "prediction")
        return jsonify(result)
        
    except Exception as e:
        print("Error:", str(e))
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
    

# 从根目录直接重定向到训练页面
@app.route("/", methods=["GET"])
def redirect_to_train():
    """重定向到训练页面"""
    try:
        return redirect("/train")
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# 修改前端页面路由
@app.route("/predict", methods=["GET"])
def predict_ui():
    """提供预测请求的前端界面"""
    return render_template('predict.html')

@app.route("/train", methods=["GET"])
def train_ui():
    """提供训练请求的前端界面"""
    return render_template('train.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
