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

def collect_user_ratings():
    """Collect all user ratings from feedback directory
    
    Returns:
    --------
    dict
        Dictionary with username as key and another dict of {movie: rating} as value
    """
    ratings = {}
    feedback_dir = "feedback"
    
    if os.path.exists(feedback_dir):
        for filename in os.listdir(feedback_dir):
            if filename.endswith("_ratings.json"):
                username = filename.replace("_ratings.json", "")
                filepath = os.path.join(feedback_dir, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        user_ratings = json.load(f)
                        if user_ratings and isinstance(user_ratings, dict):
                            ratings[username] = user_ratings
                except Exception as e:
                    print(f"Error loading ratings for {username}: {e}")
    
    return ratings

def get_user_genre_preferences(username, df):
    """Extract genre preferences from user ratings"""
    ratings_file = f"feedback/{username}_ratings.json"
    ratings = load_user_data(ratings_file, {})
    
    if not ratings:
        return []
    
    # Get movies the user rated highly (4+ stars)
    liked_movies = [movie for movie, rating in ratings.items() if rating >= 4]
    
    # Get genres from these movies
    genre_counts = {}
    for movie in liked_movies:
        movie_data = df[df['title'] == movie]
        if not movie_data.empty and 'genres_list' in movie_data.columns:
            genres = movie_data['genres_list'].iloc[0]
            if isinstance(genres, list):
                for genre in genres:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    # Return top genres (those that appear in at least 2 liked movies)
    return [genre for genre, count in genre_counts.items() if count >= 2]

def ensure_recommendation_fields(rec_list):
    """Ensure all recommendation objects have required fields and proper formats for display"""
    if rec_list is None:
        return []
        
    for rec in rec_list:
        # Set default values for any missing keys
        rec["Title"] = rec.get("Title", "Unknown")
        rec["Year"] = rec.get("Year", "N/A") 
        rec["Rating"] = rec.get("Rating", 0.0)
        rec["Genres"] = rec.get("Genres", "")
        rec["Similarity"] = rec.get("Similarity", 0.0)
        rec["Description"] = rec.get("Description", "No description available")
        
        # Convert non-string values to strings for safe display
        for key in rec:
            if rec[key] is None:
                rec[key] = "N/A"
            elif isinstance(rec[key], (int, float)) and key != "Similarity" and key != "Rating":
                rec[key] = str(rec[key])
    
    return rec_list

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