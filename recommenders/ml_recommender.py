import pandas as pd
import numpy as np
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseRecommender

class MLRecommender(BaseRecommender):
    """Machine Learning based recommender using user ratings and movie features"""
    
    def __init__(self, df, title_to_index):
        super().__init__(df, title_to_index)
        self.model_path = "models/ml_recommender_model.pkl"
        self.user_vectors = {}
        self.movie_features = None
        self.prepare_movie_features()
        self.load_model()
    
    def prepare_movie_features(self):
        """Prepare movie features for similarity computation"""
        # Combine relevant features for content representation
        self.df['feature_text'] = (
            self.df['genres_clean'].fillna('') + ' ' + 
            self.df['overview'].fillna('') + ' ' +
            self.df['director'].fillna('') + ' ' +
            self.df['cast_clean'].fillna('')
        )
        
        # Create TF-IDF representation of movies
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.movie_features = tfidf.fit_transform(self.df['feature_text'])
    
    def load_model(self):
        """Load trained model if exists, otherwise initialize empty model"""
        try:
            if os.path.exists(self.model_path):
                self.user_vectors = joblib.load(self.model_path)
            else:
                # Create directory if needed
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def train(self, user_ratings, user_feedback=None):
        """Train/update the model with user ratings and feedback
        
        Parameters:
        -----------
        user_ratings : dict
            Dictionary with username as key and another dict of {movie: rating} as value
        user_feedback : dict, optional
            Dictionary with username as key and another dict with 'liked' and 'disliked' lists
        """
        # Process ratings data
        for username, ratings in user_ratings.items():
            # Skip users with no ratings
            if not ratings:
                continue
                
            # Get movie indices for rated movies
            rated_indices = [self.title_to_index.get(title) for title in ratings.keys() 
                           if title in self.title_to_index]
            
            if not rated_indices:
                continue
                
            # Extract features for rated movies
            user_movies_features = self.movie_features[rated_indices]
            
            # Create weighted average of movie features based on ratings
            weights = np.array([float(r) for r in ratings.values()])
            weights = weights / weights.sum()  # Normalize
            
            # Compute user vector as weighted combination of movie features
            self.user_vectors[username] = user_movies_features.multiply(weights.reshape(-1, 1)).mean(axis=0)
        
        # Process "not interested" feedback if provided
        if user_feedback:
            for username, feedback in user_feedback.items():
                # Only process if we already have a user vector from ratings
                if username not in self.user_vectors:
                    continue
                    
                # Get disliked movies
                disliked = feedback.get('disliked', [])
                if not disliked:
                    continue
                    
                # Get indices for disliked movies
                disliked_indices = [self.title_to_index.get(title) for title in disliked 
                                 if title in self.title_to_index]
                
                if not disliked_indices:
                    continue
                    
                # Extract features for disliked movies
                disliked_features = self.movie_features[disliked_indices]
                
                # Adjust user vector to move away from disliked movies
                # Use a smaller weight for negative feedback to avoid over-correction
                negative_weight = 0.3
                negative_vector = disliked_features.mean(axis=0) * negative_weight
                
                # Update user vector (move away from disliked items)
                self.user_vectors[username] = self.user_vectors[username] - negative_vector
        
        # Save model
        joblib.dump(self.user_vectors, self.model_path)
        
    def recommend(self, liked_movies, n=10, user_id=None, favorite_genres=None):
        """Recommend movies based on user profile and liked movies
        
        Parameters:
        -----------
        liked_movies : list
            List of movie titles the user likes
        n : int
            Number of recommendations to return
        user_id : str, optional
            User ID for personalized recommendations
        favorite_genres : list, optional
            List of user's favorite genres
            
        Returns:
        --------
        list
            List of movie recommendation dictionaries
        """
        # If we have the user's vector, use it
        if user_id and user_id in self.user_vectors:
            user_vector = self.user_vectors[user_id]
            similarities = cosine_similarity(user_vector, self.movie_features)[0]
        # Otherwise use liked_movies to compute recommendation
        elif liked_movies:
            # Get indices for liked movies
            liked_indices = [self.title_to_index.get(title) for title in liked_movies 
                           if title in self.title_to_index]
            
            if not liked_indices:
                return []
                
            # Compute similarity
            liked_features = self.movie_features[liked_indices]
            user_profile = liked_features.mean(axis=0)
            similarities = cosine_similarity(user_profile, self.movie_features)[0]
        else:
            return []
        
        # Get indices of movies with highest similarity scores
        # Skip movies that are already in liked_movies
        already_liked_indices = set(self.title_to_index.get(title) 
                                  for title in liked_movies 
                                  if title in self.title_to_index)
        
        indices = np.argsort(similarities)[::-1]
        indices = [idx for idx in indices if idx not in already_liked_indices][:n]
        
        # Get actual movies and return as list of dictionaries
        recommendations = []
        for idx in indices:
            movie = self.df.iloc[idx]
            recommendations.append({
                "Title": movie["title"],
                "Year": int(movie.get("year", 0)) if "year" in movie else "N/A",
                "Rating": round(float(movie.get("vote_average", 0)), 1),
                "Genres": movie.get("genres_clean", ""),
                "Similarity": float(similarities[idx]),
                "Description": movie.get("overview", "")[:100] + "..." if len(movie.get("overview", "")) > 100 else movie.get("overview", "")
            })
        
        return recommendations

def get_recommender(df, title_to_index):
    """Factory function to create and return a MLRecommender instance"""
    return MLRecommender(df, title_to_index)