import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import copy
import random
import uuid
from datetime import datetime

class TimeSeriesGenerator:
    def __init__(self):
        self.base_timeseries = None
        self.time_points = None
        self.normal_samples = []
        self.abnormal_samples = []
        self.current_sample = None
        self.is_editing_abnormal = False
        self.fig = None
        self.ax = None
        self.dragging = False
        self.drag_point_index = None
        
    def load_base_timeseries(self, timePoints, values):
        """加载基础时序数据"""
        self.time_points = np.array(timePoints)
        self.base_timeseries = np.array(values)
        self.current_sample = self.base_timeseries.copy()
        
    def generate_normal_samples(self, count=10, noise_level=0.05):
        """生成正常样本（添加少量噪声）"""
        self.normal_samples = []
        base_range = np.ptp(self.base_timeseries)  # 峰峰值
        
        for i in range(count):
            # 添加随机噪声
            noise = np.random.normal(0, noise_level * base_range, len(self.base_timeseries))
            sample = self.base_timeseries + noise
            
            # 随机小幅度时间偏移（不改变形状，只是整体前后移动一点）
            time_shift = random.uniform(-0.5, 0.5)  # 时间轴上的小偏移
            shifted_time = self.time_points + time_shift
            
            # 确保时间点仍然是递增的
            shifted_time = np.sort(shifted_time)
            
            self.normal_samples.append({
                "time": shifted_time.tolist(),
                "values": sample.tolist(),
                "label": 0
            })
            
        return self.normal_samples
    
    def generate_abnormal_patterns(self, count=5):
        """生成几种预定义的异常模式"""
        self.abnormal_samples = []
        base_range = np.ptp(self.base_timeseries)
        
        for i in range(count):
            # 随机选择异常类型
            abnormal_type = random.choice([
                'spike', 'drop', 'shift', 'trend', 'oscillation'
            ])
            
            sample = self.base_timeseries.copy()
            
            if abnormal_type == 'spike':
                # 随机位置出现尖峰
                pos = random.randint(0, len(sample)-1)
                sample[pos] += random.uniform(0.5, 1.5) * base_range
                
            elif abnormal_type == 'drop':
                # 随机位置出现下降
                pos = random.randint(0, len(sample)-1)
                sample[pos] -= random.uniform(0.5, 1.5) * base_range
                
            elif abnormal_type == 'shift':
                # 从某点开始整体偏移
                pos = random.randint(1, len(sample)-2)
                shift_value = random.uniform(0.3, 0.7) * base_range
                if random.random() > 0.5:
                    shift_value = -shift_value
                sample[pos:] += shift_value
                
            elif abnormal_type == 'trend':
                # 添加趋势
                trend = np.linspace(0, random.uniform(0.5, 1.0) * base_range, len(sample))
                if random.random() > 0.5:
                    trend = -trend
                sample += trend
                
            elif abnormal_type == 'oscillation':
                # 添加振荡
                freq = random.uniform(0.5, 2.0)
                amp = random.uniform(0.2, 0.5) * base_range
                oscillation = amp * np.sin(freq * np.pi * np.arange(len(sample)) / len(sample))
                sample += oscillation
            
            self.abnormal_samples.append({
                "time": self.time_points.tolist(),
                "values": sample.tolist(),
                "type": abnormal_type
            })
            
        return self.abnormal_samples
    
    def setup_interactive_editor(self):
        """设置交互式编辑界面"""
        plt.close('all')  # 关闭之前的图形
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25)
        
        # 绘制基础时序
        self.base_line, = self.ax.plot(self.time_points, self.base_timeseries, 'b-', alpha=0.5, label='基础时序')
        
        # 绘制当前编辑的样本
        self.current_line, = self.ax.plot(self.time_points, self.current_sample, 'r-', marker='o', label='当前编辑')
        
        # 添加控制按钮
        ax_reset = plt.axes([0.7, 0.05, 0.1, 0.075])
        ax_save = plt.axes([0.81, 0.05, 0.1, 0.075])
        ax_toggle = plt.axes([0.59, 0.05, 0.1, 0.075])
        
        self.btn_reset = Button(ax_reset, 'reset')
        self.btn_save = Button(ax_save, 'save')
        self.btn_toggle = Button(ax_toggle, 'toggle')
        
        self.btn_reset.on_clicked(self.reset_current_sample)
        self.btn_save.on_clicked(self.save_current_sample)
        self.btn_toggle.on_clicked(self.toggle_mode)
        
        # 设置标题和图例
        self.update_title()
        plt.legend()
        
        # 设置鼠标事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        plt.show()
    
    def update_title(self):
        """更新图表标题"""
        mode = "abnormal" if self.is_editing_abnormal else "normal"
        self.ax.set_title(f'edit {mode} sample - drag data point to modify')
    
    def reset_current_sample(self, event):
        """重置当前样本为基础时序"""
        self.current_sample = self.base_timeseries.copy()
        self.current_line.set_ydata(self.current_sample)
        self.fig.canvas.draw_idle()
    
    def save_current_sample(self, event):
        """保存当前编辑的样本"""
        sample = {
            "time": self.time_points.tolist(),
            "values": self.current_sample.tolist(),
            "label": 1 if self.is_editing_abnormal else 0
        }
        
        if self.is_editing_abnormal:
            self.abnormal_samples.append(sample)
            print(f"已保存异常样本 #{len(self.abnormal_samples)}")
        else:
            self.normal_samples.append(sample)
            print(f"已保存正常样本 #{len(self.normal_samples)}")
            
        # 重置为基础样本的轻微变形
        noise = np.random.normal(0, 0.05 * np.ptp(self.base_timeseries), len(self.base_timeseries))
        self.current_sample = self.base_timeseries + noise
        self.current_line.set_ydata(self.current_sample)
        self.fig.canvas.draw_idle()
    
    def toggle_mode(self, event):
        """切换正常/异常编辑模式"""
        self.is_editing_abnormal = not self.is_editing_abnormal
        self.update_title()
        
        # 重置当前样本
        if self.is_editing_abnormal:
            # 异常模式下，从基础样本开始编辑
            self.current_sample = self.base_timeseries.copy()
        else:
            # 正常模式下，添加小噪声
            noise = np.random.normal(0, 0.05 * np.ptp(self.base_timeseries), len(self.base_timeseries))
            self.current_sample = self.base_timeseries + noise
            
        self.current_line.set_ydata(self.current_sample)
        self.fig.canvas.draw_idle()
    
    def find_nearest_point(self, x, y):
        """找到最接近鼠标位置的数据点"""
        distances = np.sqrt((self.time_points - x)**2 + (self.current_sample - y)**2)
        return np.argmin(distances)
    
    def on_press(self, event):
        """鼠标按下事件"""
        if event.inaxes != self.ax:
            return
            
        # 找到最近的点
        self.drag_point_index = self.find_nearest_point(event.xdata, event.ydata)
        self.dragging = True
    
    def on_release(self, event):
        """鼠标释放事件"""
        self.dragging = False
        self.drag_point_index = None
    
    def on_motion(self, event):
        """鼠标移动事件"""
        if not self.dragging or event.inaxes != self.ax or self.drag_point_index is None:
            return
            
        # 更新拖拽点的值
        self.current_sample[self.drag_point_index] = event.ydata
        self.current_line.set_ydata(self.current_sample)
        self.fig.canvas.draw_idle()
    
    def export_to_json(self, filename, eq_id="EQ01", chamber_id="CH01", recipe="Recipe1", parameter_name="Param1"):
        """导出为训练格式的JSON文件"""
        # 合并正常和异常样本
        all_samples = []
        
        # 添加正常样本
        for i, sample in enumerate(self.normal_samples):
            wafer = {
                "WaferName": f"Normal_{uuid.uuid4().hex[:8]}",
                "processTime": sample["time"],
                "values": sample["values"],
                "label": sample["label"]
            }
            all_samples.append(wafer)
        
        # 添加异常样本
        for i, sample in enumerate(self.abnormal_samples):
            wafer = {
                "WaferName": f"Abnormal_{uuid.uuid4().hex[:8]}",
                "processTime": sample["time"],
                "values": sample["values"],
                "label": sample["label"]
            }
            all_samples.append(wafer)
        
        # 创建完整的训练数据格式
        training_data = {
            "eqID": eq_id,
            "chamberID": chamber_id,
            "recipe": recipe,
            "parameterName": parameter_name,
            "trainingMode": "0",
            "labelMode": "0",
            "Wafers": all_samples
        }
        
        # 保存到文件
        with open(filename, 'w') as f:
            json.dump(training_data, f, indent=4)
            
        print(f"数据已导出到 {filename}")
        print(f"共 {len(self.normal_samples)} 个正常样本和 {len(self.abnormal_samples)} 个异常样本")

# 使用示例
if __name__ == "__main__":
    # 创建生成器
    generator = TimeSeriesGenerator()
    
    # 示例：加载基础时序数据
    # 可以从文件加载或手动指定
    values = [0.0075927735, 0.0068359375, 0.006689453, 0.006689453, 0.0066650393, 0.0066650393, 0.0066650393, 0.0066650393, 0.0066406252, 0.0066406252, 0.0066406252, 0.0066406252, 0.0066406252, 0.0066406252, 0.0066406252, 0.0066406252, 0.0066406252, 0.0066406252, 0.0066406252, 0.0066406252, 0.0066406252, 0.0066406252, 0.006616211, 0.006616211, 0.006616211, 0.006616211, 0.006616211, 0.006616211, 0.006616211, 0.006616211, 0.006616211, 0.006616211, 0.006616211, 0.006616211, 0.006616211, 0.006616211, 0.006616211, 0.006616211, 0.0065917969, 0.006616211, 0.006616211, 0.006616211, 0.0065917969, 0.0065917969, 0.0065917969, 0.0065917969, 0.0065917969, 0.0065917969, 0.0065917969, 0.0065917969]
    # 从 0 开始间隔为2 
    timePoints = np.linspace(0, len(values) * 2, len(values))
    # 将values缩放到0-10的范围
    min_val = min(values)
    max_val = max(values)
    values = [(x - min_val) / (max_val - min_val) * 10 for x in values]

    # 加载基础时序
    generator.load_base_timeseries(timePoints, values)
    
    # 自动生成一些正常样本
    generator.generate_normal_samples(count=5)
    
    # 自动生成一些异常样本
    # generator.generate_abnormal_patterns(count=3)
    
    # 启动交互式编辑器
    generator.setup_interactive_editor()
    
    # 导出数据（在交互式编辑完成后调用）
    # 可以在交互界面关闭后手动调用这个函数
    generator.export_to_json('./test_request/predict_1.json') 