import os
import time
import pandas as pd
import numpy as np
from datetime import datetime

from data_loader import load_data, create_title_index_mapping
from utils import collect_user_ratings, load_user_data, save_user_data

# Import recommender models
from recommenders.ml_recommender import MLRecommender, get_recommender as get_ml_recommender
from recommenders.collaborative import TrueCollaborativeRecommender, get_recommender as get_collab_recommender
from recommenders.matrix_factorization import MatrixFactorizationRecommender, get_recommender as get_mf_recommender

def train_all_models():
    """Train all recommendation models that require training"""
    print("\n===== NOVA RECOMMENDATION SYSTEM MODEL TRAINING =====")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("evaluation", exist_ok=True)
    
    print("\nLoading movie data...")
    df = load_data()
    title_to_index = create_title_index_mapping(df)
    
    # Collect all user ratings
    print("\nCollecting user ratings...")
    user_ratings = collect_user_ratings()
    
    # Check if we have enough data to train
    if not user_ratings:
        print("No user ratings found. Cannot train models.")
        return False
    
    user_count = len(user_ratings)
    rating_count = sum(len(ratings) for ratings in user_ratings.values())
    avg_ratings = rating_count / user_count if user_count > 0 else 0
    
    print(f"Found {rating_count} total ratings from {user_count} users")
    print(f"Average of {avg_ratings:.1f} ratings per user")
    
    # Collect user feedback (likes/dislikes) if available
    feedback_data = {}
    feedback_dir = "feedback"
    if os.path.exists(feedback_dir):
        for filename in os.listdir(feedback_dir):
            if filename.endswith("_movie_feedback.json"):
                username = filename.split("_")[0]
                filepath = os.path.join(feedback_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        feedback = json.load(f)
                        feedback_data[username] = feedback
                except Exception as e:
                    print(f"Warning: Could not load feedback file {filename}: {str(e)}")
    
    # 1. Train ML Recommender
    print("\n[1/3] Training ML Recommender...")
    ml_start = time.time()
    try:
        ml_recommender = get_ml_recommender(df, title_to_index)
        ml_recommender.train(user_ratings, feedback_data)
        print(f"✓ ML Recommender trained successfully in {time.time() - ml_start:.2f} seconds")
    except Exception as e:
        print(f"✗ Error training ML Recommender: {str(e)}")
    
    # 2. Train Collaborative Filtering Recommender
    print("\n[2/3] Training Collaborative Filtering Recommender...")
    collab_start = time.time()
    try:
        collab_recommender = get_collab_recommender(df, "true", title_to_index)
        collab_recommender.fit(user_ratings)
        print(f"✓ Collaborative Filtering Recommender trained successfully in {time.time() - collab_start:.2f} seconds")
    except Exception as e:
        print(f"✗ Error training Collaborative Filtering Recommender: {str(e)}")
    
    # 3. Train Matrix Factorization Recommender
    print("\n[3/3] Training Matrix Factorization Recommender...")
    mf_start = time.time()
    try:
        mf_recommender = get_mf_recommender(df, title_to_index)
        # If the recommender has a fit method, call it
        if hasattr(mf_recommender, 'fit') and callable(getattr(mf_recommender, 'fit')):
            mf_recommender.fit(user_ratings)
            print(f"✓ Matrix Factorization Recommender trained successfully in {time.time() - mf_start:.2f} seconds")
        else:
            print("✗ Matrix Factorization Recommender doesn't have a fit method")
    except Exception as e:
        print(f"✗ Error training Matrix Factorization Recommender: {str(e)}")
    
    # Save training timestamp
    training_info = {
        "last_trained": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "user_count": user_count,
        "rating_count": rating_count,
        "avg_ratings_per_user": avg_ratings
    }
    
    os.makedirs("models", exist_ok=True)
    try:
        with open("models/training_info.json", "w") as f:
            json.dump(training_info, f)
    except Exception as e:
        print(f"Warning: Could not save training info: {str(e)}")
    
    total_time = time.time() - start_time
    print(f"\nAll models trained in {total_time:.2f} seconds")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("===============================================")
    
    return True

def collect_user_feedback():
    """Collect user feedback (likes/dislikes) from feedback files"""
    feedback_data = {}
    feedback_dir = "feedback"
    
    if not os.path.exists(feedback_dir):
        return feedback_data
    
    for filename in os.listdir(feedback_dir):
        if filename.endswith("_movie_feedback.json"):
            username = filename.split("_")[0]
            filepath = os.path.join(feedback_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    import json
                    feedback = json.load(f)
                    feedback_data[username] = feedback
            except Exception as e:
                print(f"Warning: Could not load feedback file {filename}: {str(e)}")
    
    return feedback_data

if __name__ == "__main__":
    import json
    train_all_models()