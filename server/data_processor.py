import numpy as np
from scipy import interpolate
from pathlib import Path

class DataProcessor:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def interpolate_timeseries(self, time_points, values, last_timestep):
        """对时间序列数据进行插值处理"""
        # 创建插值函数
        f = interpolate.interp1d(time_points, values, kind='linear', fill_value='extrapolate')
        
        # 计算新的时间点（每timestep秒一个点）
        # 计算时间点之间的差值，并取整数
        time_diff = np.diff(time_points)
        timestep = np.round(np.mean(time_diff), 0)
        new_time = np.arange(timestep, last_timestep, timestep)
        
        # 进行插值
        new_values = f(new_time)
        return new_values
    

    # mock some data with different shape 

    def generate_low_frequency_signal(self, length, frequency, amplitude=1, noise_std=1):
        t = np.linspace(0, 10, length)  # 时间轴
        signal = amplitude * np.sin(2 * np.pi * frequency * t)  # 低频正弦波
        noise = np.random.normal(0, noise_std, length)  # 高斯噪声
        return signal + noise  # 添加噪声
    
        # add low frequency signal at a random interval
    def add_low_frequency_signal(self, time_series):
        amplitude = np.ptp(time_series) / 2
        length = np.random.randint(len(time_series) // 4, len(time_series))
        start = np.random.randint(0, len(time_series) - length)
        signal = self.generate_low_frequency_signal(length, 0.05, amplitude, 1)
        new_time_series = time_series.copy()
        new_time_series[start:start + length] += signal
        return new_time_series

    def add_bias(self, time_series):
        """添加偏置到时间序列数据"""
        range = np.ptp(time_series)
        bias = np.random.uniform(-range / 2, range / 2)
        return time_series + bias
    
    def add_spike(self, time_series):
        """在时间序列数据中添加尖峰"""
        length = len(time_series)
        spike_position = np.random.randint(0, length)
        max_time = np.max(time_series)
        spike_value = np.random.uniform(max_time, max_time * 2)
        new_time_series = time_series.copy()
        new_time_series[spike_position] = spike_value
        return new_time_series

    def process_wafer_data(self, wafers, training_mode = None, label_mode = None):
        """处理晶圆数据，转换为训练格式
        training_mode: "0":全量训练, 可接受打标数据和未打标数据
                       "1":增量训练，仅接受打标数据
        label_mode: "0":未打标数据
                     "1":打标数据
        """
        processed_data = []
        
        # 统计 wafer ["label"] 的数量是否等于1 
        set_labels = set(wafer["label"] for wafer in wafers if "label" in wafer)
        if set_labels == {0} and training_mode == "0":
            label_mode = "0"
        elif set_labels == {1} and training_mode == "0":
            raise ValueError("All wafer are labelled as abnormal, cannot be used for training.")
            
        # find the minimum of last processTime of all wafer
        min_last_timestep = min(wafer["processTime"][-1] for wafer in wafers)
        timestep = np.round(np.mean(np.diff(wafers[0]["processTime"])), 5)
        print(f"Using timestep: {timestep} seconds, minimum last timestep: {min_last_timestep} seconds")
        for wafer in wafers:
            # 确保时间和值的长度相同

            # 进行插值处理
            assert len(wafer["processTime"]) == len(wafer["values"]), \
            f"Time points and values must have the same length for wafer {wafer['WaferName']}"
            
            # 检查长度是否过短
            if len(wafer["processTime"]) < 10:
                raise ValueError(f"Wafer {wafer['WaferName']} has too few data points for training or prediction.")

            time_points = np.array(wafer["processTime"])
            values = np.array(wafer["values"])
            f = interpolate.interp1d(time_points, values, kind='linear', fill_value='extrapolate')
            interpolated_values = f(np.arange(timestep, min_last_timestep, timestep))
            if len(interpolated_values) == 0:
                raise ValueError(f"Wafer {wafer['WaferName']} has no data after interpolation. Check the processTime and values.")
            # 预测模式
            if training_mode is None:
                processed_data.append({
                    "WaferName": wafer["WaferName"],
                    "values": interpolated_values,
                    "label": 0 if label_mode == "0" else wafer["label"]
                })
                continue
            
            # 处理未标记数据（仅全量训练模式）
            if training_mode == "0":
                processed_data.append({"values": interpolated_values, "label": 0 if label_mode == "0" else wafer["label"]})
                rand_val = np.random.rand()
                if rand_val < 1/3:
                    processed_data.append({"values": self.add_low_frequency_signal(interpolated_values), "label": 1})
                elif rand_val < 2/3:
                    processed_data.append({"values": self.add_bias(interpolated_values), "label": 1})
                else:
                    processed_data.append({"values": self.add_spike(interpolated_values), "label": 1})
                continue

            # 处理已标记数据（增量训练模式）
            if "label" not in wafer:
                raise ValueError(f"Wafer {wafer['WaferName']} is missing label information.")
            print(f"Processing wafer {interpolated_values} with label {wafer['label']}")
            processed_data.append({
                "values": interpolated_values,
                "label": wafer["label"]
            })
        print(f"Processed {len(processed_data)} wafers with timestep {timestep} seconds.")
        return processed_data, timestep

    def save_training_data(self, data, filename, training_mode):
        """保存处理后的数据为UCR格式的训练文件"""
        file_path = self.data_dir / filename
        # 如果文件不存在且训练模式为增量训练，则抛出异常
        if not file_path.exists() and training_mode == "1":
            raise ValueError(f"You are trying to incrementally train a model on a file that does not exist.")
        # 根据训练模式选择文件打开方式：'w' 覆盖写入，'a' 追加写入
        file_mode = 'w' if training_mode == "0" else 'a'
        with open(file_path, file_mode) as f:
            for wafer in data:
                # UCR格式：第一列为标签，后面为时间序列值
                values_str = [str(wafer["label"])] + [f"{v:.4f}" for v in wafer["values"]]
                f.write(' '.join(values_str) + '\n')
        return file_path 