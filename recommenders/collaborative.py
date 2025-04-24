import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
from .base import BaseRecommender

class CollaborativeFilteringRecommender(BaseRecommender):
    """Collaborative filtering recommender using user-item interaction matrix"""
    
    def __init__(self, df, title_to_index=None):
        """
        Initialize the collaborative filtering recommender
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Movie dataframe
        title_to_index : pandas.Series, optional
            Mapping from title to dataframe index
        """
        super().__init__(df, title_to_index)
        self.name = "Collaborative Filtering Recommender"
        self.description = "Recommends movies based on what similar users have enjoyed"
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
    
    def create_dummy_users(self, n_users=20, n_movies_per_user=10):
        """
        Create dummy users for demonstration purposes
        
        Parameters:
        -----------
        n_users : int
            Number of dummy users to create
        n_movies_per_user : int
            Number of movies per user
            
        Returns:
        --------
        dict
            Dictionary of user_id -> list of movie titles
        """
        dummy_users = {}
        
        # Group movies by genre for more realistic user preferences
        genre_groups = {}
        for _, row in self.df.iterrows():
            for genre in row.get('genres_list', []):
                if genre not in genre_groups:
                    genre_groups[genre] = []
                genre_groups[genre].append(row['title'])
        
        # Create users with genre-based preferences
        genres = list(genre_groups.keys())
        for i in range(n_users):
            # Give each user 1-3 preferred genres
            n_genres = min(3, np.random.randint(1, 4))
            user_genres = np.random.choice(genres, n_genres, replace=False)
            
            # Get movies from those genres
            user_movies = []
            for genre in user_genres:
                n_from_genre = n_movies_per_user // len(user_genres)
                if genre in genre_groups and len(genre_groups[genre]) >= n_from_genre:
                    genre_movies = np.random.choice(genre_groups[genre], n_from_genre, replace=False)
                    user_movies.extend(genre_movies)
            
            # Add some random movies if needed
            remaining = n_movies_per_user - len(user_movies)
            if remaining > 0:
                random_movies = np.random.choice(self.df['title'].values, remaining, replace=False)
                user_movies.extend(random_movies)
                
            dummy_users[f'user{i+1}'] = list(set(user_movies))
        
        return dummy_users
    
    def create_user_item_matrix(self, user_ratings=None, n_dummy_users=20):
        """
        Create user-item matrix from real and dummy user ratings
        
        Parameters:
        -----------
        user_ratings : dict
            Dictionary of movie_title -> user rating
        n_dummy_users : int
            Number of dummy users to create
            
        Returns:
        --------
        pandas.DataFrame
            User-item matrix
        """
        # Create dummy users
        dummy_users = self.create_dummy_users(n_dummy_users)
        
        # Add current user if ratings provided
        if user_ratings:
            dummy_users['current_user'] = list(user_ratings.keys())
        
        # Get all unique movies
        all_movies = set()
        for movies in dummy_users.values():
            all_movies.update(movies)
        
        # Create matrix
        matrix = pd.DataFrame(0, index=dummy_users.keys(), columns=list(all_movies))
        
        # Fill matrix with binary interactions
        for user, movies in dummy_users.items():
            for movie in movies:
                # Use ratings if available for current user, otherwise just mark as watched (1)
                if user == 'current_user' and movie in user_ratings:
                    matrix.loc[user, movie] = user_ratings[movie]
                else:
                    matrix.loc[user, movie] = 1
        
        return matrix
    
    def fit(self, user_ratings=None, n_dummy_users=20):
        """
        Fit the collaborative filtering model
        
        Parameters:
        -----------
        user_ratings : dict, optional
            Dictionary of movie_title -> user rating
        n_dummy_users : int
            Number of dummy users to create
        """
        # Create user-item matrix
        self.user_item_matrix = self.create_user_item_matrix(user_ratings, n_dummy_users)
        
        # Calculate user similarity
        self.user_similarity = pd.DataFrame(
            cosine_similarity(self.user_item_matrix),
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        # Calculate item similarity (movie-movie)
        self.item_similarity = pd.DataFrame(
            cosine_similarity(self.user_item_matrix.T),
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        
        return self
    
    def recommend_user_based(self, liked_movies, n=10):
        """
        Generate user-based collaborative filtering recommendations
        
        Parameters:
        -----------
        liked_movies : list
            List of movie titles the user likes
        n : int
            Number of recommendations to return
            
        Returns:
        --------
        list
            List of recommended movie dictionaries
        """
        if self.user_similarity is None:
            self.fit()
            
        if not liked_movies:
            return []
            
        if 'current_user' not in self.user_similarity.index:
            # Current user not in the matrix, can't make user-based recommendations
            return []
            
        # Find similar users
        current_user_similarities = self.user_similarity.loc['current_user'].drop('current_user')
        most_similar_users = current_user_similarities.sort_values(ascending=False).head(5)
        
        # Get recommended movies from similar users
        recommendations = {}
        for user, similarity in most_similar_users.items():
            if similarity <= 0:
                continue
                
            user_liked = self.user_item_matrix.loc[user]
            user_movies = user_liked[user_liked > 0].index.tolist()
            
            for movie in user_movies:
                if movie not in liked_movies:
                    if movie in recommendations:
                        recommendations[movie] += similarity
                    else:
                        recommendations[movie] = similarity
        
        # Sort recommendations by score
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        
        # Get top N movie information
        top_movies = []
        scores = []
        for movie, score in sorted_recs[:n]:
            if movie in self.df['title'].values:
                idx = self.df[self.df['title'] == movie].index[0]
                top_movies.append(idx)
                scores.append(round(score, 3))
        
        # Format recommendations
        return self._format_recommendations(top_movies, scores)
    
    def recommend_item_based(self, liked_movies, n=10):
        """
        Generate item-based collaborative filtering recommendations
        
        Parameters:
        -----------
        liked_movies : list
            List of movie titles the user likes
        n : int
            Number of recommendations to return
            
        Returns:
        --------
        list
            List of recommended movie dictionaries
        """
        if self.item_similarity is None:
            self.fit()
            
        if not liked_movies:
            return []
        
        # Filter for valid movies
        valid_movies = [movie for movie in liked_movies if movie in self.item_similarity.index]
        if not valid_movies:
            return []
        
        # Calculate similarity scores for all movies based on user's liked movies
        similarity_scores = pd.Series(0, index=self.item_similarity.columns)
        for movie in valid_movies:
            similarity_scores += self.item_similarity[movie]
        
        # Remove already liked movies
        similarity_scores = similarity_scores.drop(valid_movies, errors='ignore')
        
        # Get top N recommendations
        top_recs = similarity_scores.sort_values(ascending=False).head(n)
        
        # Get movie indices and scores
        top_movies = []
        scores = []
        for movie, score in top_recs.items():
            if movie in self.df['title'].values:
                idx = self.df[self.df['title'] == movie].index[0]
                top_movies.append(idx)
                scores.append(round(score, 3))
        
        # Format recommendations
        return self._format_recommendations(top_movies, scores)
    
    def recommend(self, liked_movies, n=10, method='hybrid'):
        """
        Generate collaborative filtering recommendations
        
        Parameters:
        -----------
        liked_movies : list
            List of movie titles the user likes
        n : int
            Number of recommendations to return
        method : str
            Collaborative filtering method: 'user', 'item', or 'hybrid'
            
        Returns:
        --------
        list
            List of recommended movie dictionaries
        """
        # Create binary ratings from liked movies
        ratings = {movie: 1 for movie in liked_movies}
        
        # Fit model with current user's ratings
        self.fit(ratings)
        
        if method == 'user':
            return self.recommend_user_based(liked_movies, n)
        elif method == 'item':
            return self.recommend_item_based(liked_movies, n)
        else:  # hybrid
            # Get recommendations from both methods
            user_recs = self.recommend_user_based(liked_movies, n)
            item_recs = self.recommend_item_based(liked_movies, n)
            
            # Combine and deduplicate recommendations
            combined_recs = {}
            for rec in user_recs + item_recs:
                title = rec['Title']
                if title in combined_recs:
                    combined_recs[title]['Score'] = max(combined_recs[title]['Score'], rec['Score'])
                else:
                    combined_recs[title] = rec
            
            # Sort by score and return top N
            result = list(combined_recs.values())
            result.sort(key=lambda x: x.get('Score', 0), reverse=True)
            return result[:n]


class TrueCollaborativeRecommender(CollaborativeFilteringRecommender):
    """True collaborative filtering using user profiles and ratings"""
    
    def __init__(self, df, title_to_index=None, profiles_path="profiles", feedback_path="feedback"):
        """
        Initialize with paths to user profiles and feedback
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Movie dataframe
        title_to_index : pandas.Series, optional
            Mapping from title to dataframe index
        profiles_path : str
            Path to user profiles directory
        feedback_path : str
            Path to user feedback directory
        """
        super().__init__(df, title_to_index)
        self.name = "True Collaborative Filtering"
        self.description = "Recommends based on actual user profiles and ratings"
        self.profiles_path = profiles_path
        self.feedback_path = feedback_path
        
    def load_real_user_data(self):
        """
        Load real user data from profiles and feedback
        
        Returns:
        --------
        dict
            Dictionary of user_id -> list of liked/rated movies
        """
        import os
        import json
        
        user_data = {}
        
        # Load from profiles (liked movies)
        if os.path.exists(self.profiles_path):
            for filename in os.listdir(self.profiles_path):
                if filename.endswith('.json'):
                    try:
                        username = filename.split('.')[0]
                        with open(os.path.join(self.profiles_path, filename), 'r') as f:
                            profile = json.load(f)
                            if 'liked_movies' in profile and profile['liked_movies']:
                                user_data[username] = profile['liked_movies']
                    except:
                        continue
        
        # Load from feedback (movie ratings)
        if os.path.exists(self.feedback_path):
            for filename in os.listdir(self.feedback_path):
                if filename.endswith('_feedback.json') or filename.endswith('_ratings.json'):
                    try:
                        username = filename.split('_')[0]
                        with open(os.path.join(self.feedback_path, filename), 'r') as f:
                            ratings = json.load(f)
                            
                            # Add to existing user data or create new entry
                            if username in user_data:
                                # If it's a dictionary of ratings, get movies rated >= 3
                                if isinstance(ratings, dict):
                                    for movie, rating in ratings.items():
                                        if rating >= 3 and movie not in user_data[username]:
                                            user_data[username].append(movie)
                            else:
                                # Create new user data
                                if isinstance(ratings, dict):
                                    user_data[username] = [
                                        movie for movie, rating in ratings.items() if rating >= 3
                                    ]
                    except:
                        continue
        
        return user_data
    
    def fit(self, user_ratings=None, n_dummy_users=10):
        """
        Fit model with real user data and dummy users as needed
        
        Parameters:
        -----------
        user_ratings : dict, optional
            Current user's movie ratings
        n_dummy_users : int
            Number of dummy users to add if real users are insufficient
        """
        # Load real user data
        real_users = self.load_real_user_data()
        
        # Ensure we have enough users - add dummy users if needed
        if len(real_users) < 5:
            dummy_users = self.create_dummy_users(n_dummy_users)
            real_users.update(dummy_users)
        
        # Create user-item matrix with all movies
        all_movies = set()
        for movies in real_users.values():
            all_movies.update(movies)
            
        if user_ratings:
            all_movies.update(user_ratings.keys())
            real_users['current_user'] = list(user_ratings.keys())
        
        # Create matrix
        self.user_item_matrix = pd.DataFrame(0, index=real_users.keys(), columns=list(all_movies))
        
        # Fill matrix with binary interactions (1 for liked/rated)
        for user, movies in real_users.items():
            for movie in movies:
                self.user_item_matrix.loc[user, movie] = 1
                
        # Add current user's ratings if available
        if user_ratings and 'current_user' in self.user_item_matrix.index:
            for movie, rating in user_ratings.items():
                # Normalize rating to 0-1 scale
                norm_rating = rating / 5.0
                self.user_item_matrix.loc['current_user', movie] = norm_rating
        
        # Calculate similarities
        self.user_similarity = pd.DataFrame(
            cosine_similarity(self.user_item_matrix),
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        self.item_similarity = pd.DataFrame(
            cosine_similarity(self.user_item_matrix.T),
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        
        return self


def get_recommender(df, recommender_type="user_based", title_to_index=None):
    """
    Factory function to get the appropriate collaborative recommender
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Movie dataframe
    recommender_type : str
        Type of recommender to create: 'user_based', 'item_based', 'hybrid', 'true'
    title_to_index : pandas.Series, optional
        Mapping from title to dataframe index
        
    Returns:
    --------
    CollaborativeFilteringRecommender
        Initialized recommender object
    """
    if recommender_type == "true":
        recommender = TrueCollaborativeRecommender(df, title_to_index)
    else:  # Standard collaborative filtering
        recommender = CollaborativeFilteringRecommender(df, title_to_index)
        
    return recommender