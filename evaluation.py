import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time
from sklearn.metrics import precision_score, recall_score, f1_score

class RecommenderEvaluator:
    """Class to evaluate and compare recommender systems"""
    
    def __init__(self, df):
        """
        Initialize evaluator
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Movie dataframe
        """
        self.df = df
        self.results = {}
    
    def prepare_data(self, user_ratings, min_ratings=5):
        """
        Create train/test splits from user ratings
        
        Parameters:
        -----------
        user_ratings : dict
            Dictionary of user_id -> dict(movie_title -> rating)
        min_ratings : int
            Minimum number of ratings required for a user
        
        Returns:
        --------
        dict
            Dictionary of user_id -> (train_ratings, test_ratings)
        """
        splits = {}
        
        for username, ratings in user_ratings.items():
            if len(ratings) < min_ratings:
                continue
            
            # Convert to list of (movie, rating) tuples
            ratings_list = list(ratings.items())
            
            # Shuffle ratings
            np.random.seed(42)  # for reproducibility
            np.random.shuffle(ratings_list)
            
            # Split into train (80%) and test (20%)
            split_idx = int(0.8 * len(ratings_list))
            train = dict(ratings_list[:split_idx])
            test = dict(ratings_list[split_idx:])
            
            splits[username] = (train, test)
        
        return splits
    
    def evaluate_for_user(self, recommender, train_ratings, test_ratings, n=10, name="Recommender"):
        """
        Evaluate a recommender for a single user
        
        Parameters:
        -----------
        recommender : BaseRecommender
            Recommender instance to evaluate
        train_ratings : dict
            Dictionary of movie_title -> rating for training
        test_ratings : dict
            Dictionary of movie_title -> rating for testing
        n : int
            Number of recommendations to generate
        name : str
            Name of the recommender
        
        Returns:
        --------
        dict
            Evaluation metrics for this user
        """
        try:
            # Reset recommender for this user
            recommender.__init__(self.df, None)
            
            # Time the recommendation process
            start_time = time.time()
            
            # Generate recommendations based on train set
            liked_movies = [movie for movie, rating in train_ratings.items() if rating >= 3]
            
            # Skip if no liked movies in training set
            if not liked_movies:
                return {}
            
            # Get recommendations
            recommendations = recommender.recommend(liked_movies, n=n)
            
            runtime = time.time() - start_time
            
            # Extract recommended movie titles
            rec_titles = [r.get('Title', '') for r in recommendations if r.get('Title')]
            
            # Consider movies with ratings >= 3 as relevant in test set
            relevant = [movie for movie, rating in test_ratings.items() if rating >= 3]
            
            # Calculate precision and recall
            true_positives = len(set(rec_titles) & set(relevant))
            
            precision = true_positives / len(rec_titles) if rec_titles else 0
            recall = true_positives / len(relevant) if relevant else 0
            
            # Calculate F1 score
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate hit rate (1 if at least one relevant item is recommended)
            hit_rate = 1 if true_positives > 0 else 0
            
            # Calculate coverage (proportion of movies that can be recommended)
            # This is an approximation since we only have a sample
            all_movies = set(self.df['title'])
            coverage = len(set(rec_titles)) / len(all_movies) if all_movies else 0
            
            # Calculate diversity (average number of different genres)
            genre_sets = []
            for title in rec_titles:
                movie_data = self.df[self.df['title'] == title]
                if not movie_data.empty and 'genres_list' in movie_data.columns:
                    genres = movie_data.iloc[0].get('genres_list', [])
                    if isinstance(genres, list):
                        genre_sets.append(set(genres))
            
            diversity = len(set().union(*genre_sets)) if genre_sets else 0
            
            # Return metrics
            return {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "hit_rate": hit_rate,
                "coverage": coverage,
                "diversity": diversity,
                "runtime": runtime
            }
        except Exception as e:
            print(f"Error evaluating {name} for user: {str(e)}")
            return {}
    
    def evaluate_recommender(self, recommender, user_splits, n=10, name="Recommender"):
        """
        Evaluate a recommender across multiple users
        
        Parameters:
        -----------
        recommender : BaseRecommender
            Recommender instance to evaluate
        user_splits : dict
            Dictionary of user_id -> (train_ratings, test_ratings)
        n : int
            Number of recommendations to generate
        name : str
            Name of the recommender
        
        Returns:
        --------
        dict
            Average evaluation metrics across users
        """
        # Initialize metrics
        metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "hit_rate": 0.0,
            "coverage": 0.0,
            "diversity": 0,
            "runtime": 0.0
        }
        
        valid_users = 0
        
        # Evaluate for each user
        for username, (train, test) in user_splits.items():
            try:
                user_metrics = self.evaluate_for_user(
                    recommender, train, test, n=n, name=name
                )
                
                if not user_metrics:
                    continue
                
                # Accumulate metrics
                for key in metrics:
                    metrics[key] += user_metrics.get(key, 0)
                
                valid_users += 1
            except Exception as e:
                print(f"Error evaluating {name} for user {username}: {str(e)}")
        
        # Calculate averages
        if valid_users > 0:
            for key in metrics:
                metrics[key] /= valid_users
        
        # Store results
        self.results[name] = metrics
        
        return metrics
    
    def save_results(self):
        """
        Save evaluation results to a JSON file
        
        Returns:
        --------
        str
            Path to the saved file
        """
        os.makedirs("evaluation", exist_ok=True)
        results_file = os.path.join("evaluation", "recommender_evaluation.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        return results_file
    
    def plot_comparison(self, results=None):
        """
        Create a comparison plot of recommenders
        
        Parameters:
        -----------
        results : dict, optional
            Results dictionary to use, defaults to self.results
            
        Returns:
        --------
        matplotlib.pyplot
            Plot object
        """
        if results is None:
            results = self.results
        
        if not results:
            raise ValueError("No results available for plotting")
        
        # Create a table of metrics for all recommenders
        recommender_names = list(results.keys())
        metrics = ["precision", "recall", "f1_score", "hit_rate"]
        
        # Create a dataframe for the plot
        data = []
        for metric in metrics:
            metric_values = [results[name].get(metric, 0) for name in recommender_names]
            data.append(metric_values)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set position of bars on x-axis
        x = np.arange(len(metrics))
        width = 0.15  # Width of bars
        
        # Create bars
        for i, name in enumerate(recommender_names):
            values = [results[name].get(metric, 0) for metric in metrics]
            ax.bar(x + i*width - (len(recommender_names)-1)*width/2, values, width, label=name)
        
        # Add labels and title
        ax.set_ylabel('Score')
        ax.set_title('Recommender System Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)
        
        # Add value labels on bars
        for i, name in enumerate(recommender_names):
            for j, metric in enumerate(metrics):
                value = results[name].get(metric, 0)
                ax.text(
                    j + i*width - (len(recommender_names)-1)*width/2,
                    value + 0.01,
                    f"{value:.2f}",
                    ha='center',
                    va='bottom',
                    rotation=90,
                    fontsize=8
                )
        
        plt.tight_layout()
        return plt