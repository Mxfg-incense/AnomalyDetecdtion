import requests
import json

def test_training_api(json_path):
    # API端点
    url = "http://localhost:5000/train"
    
    # 准备请求数据
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    try:
        # 发送POST请求
        response = requests.post(
            url,
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        # 检查响应
        print("Training Response:")
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
    except json.JSONDecodeError:
        print("Error decoding response JSON")
        print("Raw response:", response.text)

def test_prediction_api(json_path):
    # API端点
    url = "http://localhost:5000/predict"
    
    # 准备预测请求数据
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    try:
        # 发送POST请求
        response = requests.post(
            url,
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        # 检查响应
        print("\nPrediction Response:")
        print(f"Status Code: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
    except json.JSONDecodeError:
        print("Error decoding response JSON")
        print("Raw response:", response.text)

if __name__ == "__main__":
    # 测试训练接口
    # test_training_api('./mock_json/training_1_0.json')
    test_training_api('./mock_json/training_0_1.json')
    test_training_api('./mock_json/training_0_0.json')
    test_prediction_api('./mock_json/predict_0.json')
    test_prediction_api('./mock_json/predict_1.json')
    # # 测试预测接口
    test_training_api('./mock_json/training_1_1.json')
    test_prediction_api('./mock_json/predict_1.json')
