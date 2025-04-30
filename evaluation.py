import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import json
import os

class RecommenderEvaluator:
    """Framework for evaluating and comparing recommender systems"""
    
    def __init__(self, df, test_size=0.2, random_state=42):
        """Initialize evaluator with dataset
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Movie dataframe
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
        """
        self.df = df
        self.test_size = test_size
        self.random_state = random_state
        self.results = {}
        self.metrics = ["precision", "recall", "f1_score", "hit_rate", "coverage", "diversity", "runtime"]
    
    def _split_user_data(self, ratings, user_id=None):
        """Split a user's ratings into train and test sets
        
        Parameters:
        -----------
        ratings : dict
            Dict of {movie: rating} for a user
        user_id : str, optional
            User ID for tracking
        
        Returns:
        --------
        train_ratings, test_ratings : tuple of dicts
            Train and test rating dictionaries
        """
        # Convert ratings to list of (movie, rating) tuples
        rating_items = list(ratings.items())
        
        # If user has very few ratings, use leave-one-out
        if len(rating_items) <= 5:
            if len(rating_items) <= 1:
                return ratings, {}  # Can't split if only one rating
            
            train = dict(rating_items[:-1])
            test = dict([rating_items[-1]])
            return train, test
        
        # Otherwise do regular train_test_split
        train_items, test_items = train_test_split(
            rating_items,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        train = dict(train_items)
        test = dict(test_items)
        
        return train, test
    
    def prepare_data(self, all_ratings, min_ratings=5):
        """Prepare train/test splits for all users
        
        Parameters:
        -----------
        all_ratings : dict
            Dictionary with username as key and another dict of {movie: rating} as value
        min_ratings : int
            Minimum number of ratings required for a user to be included
            
        Returns:
        --------
        dict
            Dictionary with username as key and tuple of (train_ratings, test_ratings) as value
        """
        user_splits = {}
        
        for username, ratings in all_ratings.items():
            if len(ratings) >= min_ratings:
                train, test = self._split_user_data(ratings, username)
                if train and test:  # Only include if both train and test have data
                    user_splits[username] = (train, test)
        
        return user_splits
    
    def evaluate_recommender(self, recommender, user_splits, n=10, name=None):
        """Evaluate a recommender system on split user data
        
        Parameters:
        -----------
        recommender : BaseRecommender
            Recommender to evaluate
        user_splits : dict
            Dictionary from prepare_data with train/test splits
        n : int
            Number of recommendations to generate
        name : str, optional
            Name to use for this evaluation run
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        # Initialize metrics
        total_precision = 0
        total_recall = 0
        total_hit_rate = 0
        all_recommendations = set()
        diversity_scores = []
        runtimes = []
        
        # Get name of recommender if not specified
        if name is None:
            name = recommender.name
        
        num_users = len(user_splits)
        
        # Evaluate for each user
        for username, (train_ratings, test_ratings) in user_splits.items():
            # Convert ratings to a list of liked movies
            liked_movies = list(train_ratings.keys())
            
            # Time the recommendation generation
            start_time = time.time()
            
            # Generate recommendations - different recommenders might need different parameters
            try:
                if hasattr(recommender, 'fit'):
                    # Some recommenders need to be fit with the training data
                    recommender.fit(train_ratings)
                
                # Generate recommendations - handle different parameter requirements
                if "user_id" in recommender.recommend.__code__.co_varnames:
                    recommendations = recommender.recommend(liked_movies, n=n, user_id=username)
                else:
                    recommendations = recommender.recommend(liked_movies, n=n)
                
                runtime = time.time() - start_time
                runtimes.append(runtime)
            
                # Extract just the movie titles from recommendations
                rec_titles = [r["Title"] for r in recommendations]
                
                # Update coverage tracking
                all_recommendations.update(rec_titles)
                
                # Calculate recommendation diversity (unique genres)
                unique_genres = set()
                for title in rec_titles:
                    movie_data = self.df[self.df['title'] == title]
                    if not movie_data.empty and 'genres_list' in movie_data.columns:
                        genres = movie_data['genres_list'].iloc[0]
                        if isinstance(genres, list):
                            unique_genres.update(genres)
                
                diversity = len(unique_genres) / max(1, len(rec_titles))
                diversity_scores.append(diversity)
                
                # Calculate precision and recall
                test_set = set(test_ratings.keys())
                recommended_set = set(rec_titles)
                
                # Find the intersection of recommended items and test items
                relevant_retrieved = len(test_set.intersection(recommended_set))
                
                # Calculate metrics
                precision = relevant_retrieved / max(1, len(recommended_set))
                recall = relevant_retrieved / max(1, len(test_set))
                
                # Hit rate (did we recommend at least one relevant item?)
                hit_rate = 1 if relevant_retrieved > 0 else 0
                
                # Accumulate metrics
                total_precision += precision
                total_recall += recall
                total_hit_rate += hit_rate
            
            except Exception as e:
                print(f"Error evaluating for user {username}: {str(e)}")
                continue
        
        # Calculate final metrics
        avg_precision = total_precision / num_users if num_users > 0 else 0
        avg_recall = total_recall / num_users if num_users > 0 else 0
        avg_hit_rate = total_hit_rate / num_users if num_users > 0 else 0
        
        # F1 score
        avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        # Coverage - what percentage of all movies does the recommender cover?
        total_movies = len(self.df)
        coverage = len(all_recommendations) / total_movies
        
        # Average diversity
        avg_diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0
        
        # Average runtime
        avg_runtime = sum(runtimes) / len(runtimes) if runtimes else 0
        
        # Compile results
        results = {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1_score": avg_f1,
            "hit_rate": avg_hit_rate,
            "coverage": coverage,
            "diversity": avg_diversity,
            "runtime": avg_runtime
        }
        
        # Store results
        self.results[name] = results
        
        return results
    
    def save_results(self, filename="recommender_evaluation.json"):
        """Save evaluation results to a file"""
        os.makedirs("evaluation", exist_ok=True)
        filepath = os.path.join("evaluation", filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        return filepath
    
    def plot_comparison(self, metrics=None):
        """Plot comparison of recommenders
        
        Parameters:
        -----------
        metrics : list, optional
            List of metrics to compare. If None, uses all metrics.
        """
        if not self.results:
            print("No evaluation results available")
            return
        
        if metrics is None:
            metrics = self.metrics
        
        # Filter to only existing metrics
        metrics = [m for m in metrics if m in self.metrics]
        
        # Number of recommenders and metrics
        n_recommenders = len(self.results)
        n_metrics = len(metrics)
        
        # Prepare plot
        plt.figure(figsize=(12, 8))
        
        # Draw bars for each recommender and metric
        bar_width = 0.8 / n_recommenders
        
        for i, (name, results) in enumerate(self.results.items()):
            values = [results.get(metric, 0) for metric in metrics]
            positions = np.arange(n_metrics) + i * bar_width
            
            plt.bar(positions, values, width=bar_width, label=name)
        
        # Set labels and title
        plt.xlabel("Metric")
        plt.ylabel("Score")
        plt.title("Recommender System Comparison")
        plt.xticks(np.arange(n_metrics) + (n_recommenders - 1) * bar_width / 2, metrics)
        plt.legend()
        
        # Show grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return plt