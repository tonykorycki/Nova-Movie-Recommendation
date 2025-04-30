import pandas as pd
import os
import json
from data_loader import load_data, create_title_index_mapping
from recommenders.ml_recommender import MLRecommender
from utils import collect_user_ratings

def main():
    """Train the ML recommender model with all available user data"""
    print("Loading data...")
    df = load_data()
    title_to_index = create_title_index_mapping(df)
    
    print("Collecting user ratings...")
    user_ratings = collect_user_ratings()
    
    if not user_ratings:
        print("No user ratings found. Cannot train model.")
        return
    
    print(f"Found ratings from {len(user_ratings)} users")
    
    print("Creating and training ML recommender...")
    recommender = MLRecommender(df, title_to_index)
    recommender.train(user_ratings)
    
    print("Model trained and saved successfully!")
    
if __name__ == "__main__":
    main()