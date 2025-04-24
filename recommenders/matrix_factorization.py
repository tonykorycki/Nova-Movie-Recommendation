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
        self.description = "Uses Singular Value Decomposition to identify latent factors in movies"
        self.user_item_matrix = None
        self.user_features = None
        self.item_features = None
        self.sigma = None
        self.reconstructed_matrix = None
    
    def create_user_item_matrix(self, user_ratings, n_dummy_users=20):
        """
        Create user-item matrix for factorization
        
        Parameters:
        -----------
        user_ratings : dict
            Dictionary with movie titles as keys and ratings as values
        n_dummy_users : int
            Number of dummy users to create
            
        Returns:
        --------
        scipy.sparse.csr_matrix
            User-item sparse matrix
        """
        # Create dummy users first - group by genre for realism
        genres = {}
        for _, row in self.df.iterrows():
            for genre in row.get('genres_list', []):
                if genre not in genres:
                    genres[genre] = []
                genres[genre].append(row['title'])
        
        # Create dummy users with genre preferences
        all_genres = list(genres.keys())
        ratings = []
        
        # Create row indices, column indices and data for sparse matrix
        row_ind = []
        col_ind = []
        data = []
        
        # Add dummy users (users 0 to n_dummy_users-1)
        for user_id in range(n_dummy_users):
            # Select 1-3 preferred genres for this user
            n_genres = np.random.randint(1, 4)
            if len(all_genres) > n_genres:
                user_genres = np.random.choice(all_genres, n_genres, replace=False)
                
                # Rate 10-20 movies from those genres
                for genre in user_genres:
                    if genre in genres:
                        # How many movies to rate from this genre
                        n_movies = min(len(genres[genre]), np.random.randint(5, 10))
                        
                        if n_movies > 0:
                            # Sample movies from this genre
                            genre_movies = np.random.choice(genres[genre], n_movies, replace=False)
                            
                            # Add ratings (3.5-5.0 for preferred genres)
                            for movie in genre_movies:
                                if movie in self.title_to_index.index:
                                    movie_idx = self.title_to_index[movie]
                                    rating = np.random.uniform(3.5, 5.0)
                                    
                                    row_ind.append(user_id)
                                    col_ind.append(movie_idx)
                                    data.append(rating)
                
                # Add some random low-rated movies
                n_random = np.random.randint(5, 15)
                random_movies = np.random.choice(self.df['title'].values, n_random, replace=False)
                for movie in random_movies:
                    if movie in self.title_to_index.index:
                        movie_idx = self.title_to_index[movie]
                        # Low rating (1.0-3.0)
                        rating = np.random.uniform(1.0, 3.0)
                        
                        row_ind.append(user_id)
                        col_ind.append(movie_idx)
                        data.append(rating)
        
        # Add the current user (as the last user)
        current_user_id = n_dummy_users
        for movie, rating in user_ratings.items():
            if movie in self.title_to_index.index:
                movie_idx = self.title_to_index[movie]
                
                row_ind.append(current_user_id)
                col_ind.append(movie_idx)
                data.append(rating)
        
        # Create sparse matrix
        shape = (n_dummy_users + 1, len(self.df))
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
            
            self.user_features = user_features
            self.item_features = item_features
            self.sigma = svd.singular_values_
        else:
            # Fill missing values with zeros for SVD
            dense_matrix = self.user_item_matrix.toarray()
            
            # Perform SVD (adjust n_factors to avoid errors)
            n_factors = min(n_factors, min(dense_matrix.shape)-1)
            u, sigma, vt = svds(dense_matrix, k=n_factors)
            
            # Store the decomposition
            self.user_features = u
            self.item_features = vt.T
            self.sigma = sigma
        
        # Reconstruct matrix for predictions
        sigma_diag = np.diag(self.sigma)
        self.reconstructed_matrix = self.user_features @ sigma_diag @ self.item_features.T
        
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
        if self.reconstructed_matrix is None:
            # Create user ratings from liked movies
            user_ratings = {movie: 5.0 for movie in liked_movies}
            self.fit(user_ratings)
            
        if self.reconstructed_matrix is None:
            return []
        
        # Get the user's predicted ratings
        # Last user is the current user
        user_id = self.reconstructed_matrix.shape[0] - 1
        user_predictions = self.reconstructed_matrix[user_id]
        
        # Create a mapping from movie index to original dataframe index
        movie_mapping = {}
        for title, idx in self.title_to_index.items():
            movie_mapping[idx] = title
        
        # Create a list of (movie_idx, predicted_rating)
        predictions = []
        for movie_idx in range(len(user_predictions)):
            movie_title = movie_mapping.get(movie_idx)
            if movie_title and movie_title not in liked_movies:
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
    return MatrixFactorizationRecommender(df, title_to_index)