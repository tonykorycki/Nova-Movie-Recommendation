import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sp
from .base import BaseRecommender

class MatrixFactorizationRecommender(BaseRecommender):
    """Matrix Factorization recommendation system using SVD"""
    
    def __init__(self, df, title_to_index=None):
        """
        Initialize the Matrix Factorization recommender
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Movie dataframe
        title_to_index : pandas.Series, optional
            Mapping from title to dataframe index
        """
        super().__init__(df, title_to_index)
        self.name = "Matrix Factorization Recommender"
        self.description = "Recommends movies using matrix factorization (SVD)"
        self.user_item_matrix = None
        self.user_factors = None
        self.item_factors = None
        self.item_bias = None
        self.global_mean = None
    
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
        Create user-item matrix for matrix factorization
        
        Parameters:
        -----------
        user_ratings : dict, optional
            Dictionary of movie_title -> user rating
        n_dummy_users : int
            Number of dummy users to create
            
        Returns:
        --------
        scipy.sparse.csr_matrix
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
        
        # Create mappings for users and items
        users = list(dummy_users.keys())
        movies = list(all_movies)
        
        self.user_mapping = {user: i for i, user in enumerate(users)}
        self.item_mapping = {movie: i for i, movie in enumerate(movies)}
        
        # Prepare data for sparse matrix
        row_ind = []
        col_ind = []
        data = []
        
        for user, movies_list in dummy_users.items():
            user_idx = self.user_mapping[user]
            for movie in movies_list:
                if movie in self.item_mapping:
                    movie_idx = self.item_mapping[movie]
                    
                    # Use actual ratings if available for current user
                    if user == 'current_user' and user_ratings and movie in user_ratings:
                        rating = float(user_ratings[movie])
                    else:
                        # Generate a synthetic rating (3-5 range for liked movies)
                        rating = np.random.uniform(3.0, 5.0)
                    
                    row_ind.append(user_idx)
                    col_ind.append(movie_idx)
                    data.append(rating)
        
        # Convert to numpy arrays
        row_ind = np.array(row_ind, dtype=np.int32)
        col_ind = np.array(col_ind, dtype=np.int32)
        data = np.array(data, dtype=np.float32)
        
        # Create sparse matrix
        shape = (len(users), len(movies))
        matrix = sp.csr_matrix((data, (row_ind, col_ind)), shape=shape)
        
        return matrix
    
    def fit(self, user_ratings=None, n_factors=20, n_dummy_users=20):
        """
        Fit the matrix factorization model using SVD
        
        Parameters:
        -----------
        user_ratings : dict, optional
            Dictionary of movie_title -> user_rating
        n_factors : int
            Number of latent factors to use
        n_dummy_users : int
            Number of dummy users to create
        """
        # Handle empty user ratings
        if user_ratings is None:
            user_ratings = {}
        
        # Create user-item matrix
        self.user_item_matrix = self.create_user_item_matrix(user_ratings, n_dummy_users)
        
        # Check if matrix is empty
        if self.user_item_matrix.nnz == 0:
            # Matrix is empty, can't factorize
            return self
        
        # Use sparse SVD if the matrix is large
        if self.user_item_matrix.shape[1] > 1000:
            # Use TruncatedSVD for large sparse matrices
            svd = TruncatedSVD(n_components=min(n_factors, min(self.user_item_matrix.shape)-1))
            item_features = svd.fit_transform(self.user_item_matrix.T)
            user_features = (self.user_item_matrix @ item_features) @ np.diag(1/svd.singular_values_)
            
            self.user_factors = user_features
            self.item_factors = item_features
        else:
            # Fill missing values with zeros for SVD
            dense_matrix = self.user_item_matrix.toarray()
            
            # Perform SVD (adjust n_factors to avoid errors)
            n_factors = min(n_factors, min(dense_matrix.shape)-1)
            u, sigma, vt = svds(dense_matrix, k=n_factors)
            
            # Store the decomposition
            self.user_factors = u
            self.item_factors = vt.T
        
        return self
    
    def recommend(self, liked_movies, n=10):
        """
        Generate recommendations based on matrix factorization
        
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
        if self.user_factors is None or self.item_factors is None:
            # Create user ratings from liked movies
            user_ratings = {movie: 5.0 for movie in liked_movies}
            self.fit(user_ratings)
            
        if self.user_factors is None or self.item_factors is None:
            return []
        
        # Get the user's predicted ratings
        # Last user is the current user
        user_id = len(self.user_mapping) - 1
        user_predictions = self.user_factors[user_id] @ self.item_factors.T
        
        # Create a list of (movie_idx, predicted_rating)
        predictions = []
        for movie, movie_idx in self.item_mapping.items():
            if movie not in liked_movies:
                predictions.append((movie_idx, user_predictions[movie_idx]))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        top_indices = [idx for idx, _ in predictions[:n]]
        top_scores = [score for _, score in predictions[:n]]
        
        # Format recommendations
        return self._format_recommendations(top_indices, [round(score, 3) for score in top_scores])

def get_recommender(df, title_to_index=None):
    """
    Create and return a Matrix Factorization recommender
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Movie dataframe
    title_to_index : pandas.Series, optional
        Mapping from title to dataframe index
        
    Returns:
    --------
    MatrixFactorizationRecommender
        Initialized recommender
    """
    recommender = MatrixFactorizationRecommender(df, title_to_index)
    return recommender  # Don't pre-fit the model to avoid issues