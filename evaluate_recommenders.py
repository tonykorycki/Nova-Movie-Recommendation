import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

from data_loader import load_data, create_title_index_mapping
from utils import collect_user_ratings
from evaluation import RecommenderEvaluator

# Import recommenders
from recommenders.content_based import get_recommender as get_content_recommender
from recommenders.collaborative import get_recommender as get_collab_recommender
from recommenders.matrix_factorization import get_recommender as get_mf_recommender
from recommenders.ml_recommender import get_recommender as get_ml_recommender

def evaluate_all_recommenders():
    """Evaluate all recommender systems"""
    print("Loading data...")
    df = load_data()
    title_to_index = create_title_index_mapping(df)
    
    print("Collecting user ratings...")
    user_ratings = collect_user_ratings()
    
    if not user_ratings:
        print("No user ratings found. Cannot evaluate recommenders.")
        return
    
    print(f"Found ratings from {len(user_ratings)} users")
    
    # Create evaluator
    evaluator = RecommenderEvaluator(df)
    
    # Prepare train/test splits
    print("Preparing data splits...")
    user_splits = evaluator.prepare_data(user_ratings, min_ratings=5)
    
    if not user_splits:
        print("No users have enough ratings for evaluation.")
        return
    
    print(f"Prepared splits for {len(user_splits)} users")
    
    # Define recommenders to evaluate
    recommenders = [
        ("Content-Based", get_content_recommender(df, "content", title_to_index)),
        ("Enhanced Content-Based", get_content_recommender(df, "enhanced", title_to_index)),
        ("Collaborative Filtering", get_collab_recommender(df, "true", title_to_index)),
        ("Matrix Factorization", get_mf_recommender(df, title_to_index)),
        ("ML Recommender", get_ml_recommender(df, title_to_index)),
    ]
    
    # Evaluate each recommender
    for name, recommender in recommenders:
        print(f"Evaluating {name}...")
        try:
            results = evaluator.evaluate_recommender(recommender, user_splits, n=10, name=name)
            print(f"Results for {name}:")
            for metric, value in results.items():
                print(f"  {metric}: {value:.4f}")
        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")
    
    # Save results
    results_file = evaluator.save_results()
    print(f"Results saved to {results_file}")
    
    # Plot comparison
    plt = evaluator.plot_comparison()
    plt.savefig(os.path.join("evaluation", "recommender_comparison.png"))
    print("Comparison plot saved to evaluation/recommender_comparison.png")
    
    return evaluator

if __name__ == "__main__":
    evaluate_all_recommenders()