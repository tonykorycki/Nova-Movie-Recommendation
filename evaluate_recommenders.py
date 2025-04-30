import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import traceback

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
    
    # Results dictionary to accumulate evaluation metrics
    all_results = {}
    
    # Define recommenders to evaluate
    recommenders = [
        ("Content-Based", get_content_recommender(df, "content", title_to_index)),
        ("Enhanced Content-Based", get_content_recommender(df, "enhanced", title_to_index)),
        ("Collaborative Filtering", get_collab_recommender(df, "true", title_to_index)),
        # Matrix Factorization will be handled separately due to its requirements
        ("ML Recommender", get_ml_recommender(df, title_to_index)),
    ]
    
    # Evaluate each recommender
    for name, recommender in recommenders:
        print(f"Evaluating {name}...")
        try:
            results = evaluator.evaluate_recommender(recommender, user_splits, n=10, name=name)
            all_results[name] = results
            print(f"Results for {name}:")
            for metric, value in results.items():
                print(f"  {metric}: {value:.4f}")
        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")
            traceback.print_exc()
    
    # Handle Matrix Factorization separately as it needs special handling
    try:
        print("Evaluating Matrix Factorization...")
        mf_results = {"precision": 0, "recall": 0, "f1_score": 0, 
                     "hit_rate": 0, "coverage": 0, "diversity": 0, "runtime": 0}
        
        count = 0
        for username, (train, test) in user_splits.items():
            try:
                # Create a fresh recommender for each user
                mf_recommender = get_mf_recommender(df, title_to_index)
                
                # Format the ratings appropriately for this recommender
                liked_movies = [movie for movie, rating in train.items() if rating >= 3]
                
                if liked_movies:
                    # Evaluate this user
                    user_result = evaluator.evaluate_for_user(
                        mf_recommender, train, test, n=10, name="Matrix Factorization"
                    )
                    
                    # Accumulate results
                    for key in mf_results:
                        if key in user_result:
                            mf_results[key] += user_result[key]
                    count += 1
            except Exception as e:
                print(f"  Error evaluating Matrix Factorization for user {username}: {str(e)}")
        
        # Average the results
        if count > 0:
            for key in mf_results:
                mf_results[key] /= count
            
            all_results["Matrix Factorization"] = mf_results
            print(f"Results for Matrix Factorization:")
            for metric, value in mf_results.items():
                print(f"  {metric}: {value:.4f}")
    except Exception as e:
        print(f"Error evaluating Matrix Factorization: {str(e)}")
        traceback.print_exc()
    
    # Save all results
    os.makedirs("evaluation", exist_ok=True)
    results_file = os.path.join("evaluation", "recommender_evaluation.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_file}")
    
    # Plot comparison if we have results
    if all_results:
        try:
            plt = evaluator.plot_comparison(all_results)
            plt.savefig(os.path.join("evaluation", "recommender_comparison.png"))
            print("Comparison plot saved to evaluation/recommender_comparison.png")
        except Exception as e:
            print(f"Error plotting comparison: {str(e)}")
            traceback.print_exc()
    
    return evaluator

if __name__ == "__main__":
    evaluate_all_recommenders()