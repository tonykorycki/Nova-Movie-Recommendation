import json
import re
import os
import time
from datetime import datetime
import hashlib

def create_directories():
    """Create necessary directories for the app"""
    for folder in ["profiles", "feedback", "watchlists", "cache"]:
        os.makedirs(folder, exist_ok=True)
    
    # Create cache subdirectories
    os.makedirs(os.path.join("cache", "joblib"), exist_ok=True)

def sanitize_username(username_input):
    """Sanitize username to prevent file system issues"""
    if not username_input:
        return ""
    return re.sub(r'[^\w]', '_', username_input)

def extract_list(json_str, key="name"):
    """Extract a list of items from a JSON string"""
    try:
        data = json.loads(json_str.replace("'", '"'))
        return [item[key] for item in data]
    except:
        return []

def extract_director(crew_str):
    """Extract director name from crew JSON string"""
    try:
        crew_list = json.loads(crew_str.replace("'", '"'))
        for item in crew_list:
            if item.get("job") == "Director":
                return item.get("name")
    except:
        return ""
    return ""

def get_date_based_seed():
    """Generate a deterministic seed based on today's date"""
    today = datetime.now().strftime("%Y-%m-%d")
    return int(hashlib.md5(today.encode()).hexdigest(), 16) % 10000

def load_user_data(filepath, default_data=None):
    """Load user data from JSON file with error handling"""
    if default_data is None:
        default_data = {}
        
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return default_data
    return default_data

def save_user_data(filepath, data):
    """Save user data to JSON file with error handling"""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f)
        return True, None
    except Exception as e:
        return False, str(e)

class PerformanceTracker:
    """Track performance metrics for the app"""
    
    def __init__(self):
        self.metrics = {
            'load_time': 0,
            'search_time': 0,
            'rec_time': 0,
            'total_time': 0
        }
        self.start_time = None
    
    def start(self, metric_name):
        """Start timing a specific metric"""
        self.start_time = time.time()
        return self
    
    def stop(self, metric_name):
        """Stop timing and record the metric"""
        if self.start_time is not None:
            self.metrics[metric_name] = time.time() - self.start_time
            self.start_time = None
        return self.metrics[metric_name]
    
    def get_metrics(self):
        """Return all metrics"""
        return self.metrics