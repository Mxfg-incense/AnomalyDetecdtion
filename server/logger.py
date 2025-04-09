import json
from datetime import datetime
from pathlib import Path

class Logger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        
    def save_log(self, result, prefix="training"):
        """保存日志"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{prefix}_log_{timestamp}.json"
        
        with open(log_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'result': result
            }, f, indent=4)
        
        return log_file 