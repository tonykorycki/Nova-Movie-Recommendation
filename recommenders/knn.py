import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from .base import BaseRecommender

class KNNRecommender(BaseRecommender):
    """K-Nearest Neighbors recommendation system"""
    
    def __init__(self, df, title_to_index=None):
        """
        Initialize the KNN recommender
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Movie dataframe
        title_to_index : pandas.Series, optional
            Mapping from title to dataframe index
        """
        super().__init__(df, title_to_index)
        self.name = "K-Nearest Neighbors Recommender"
        self.description = "Uses KNN algorithm to find movies with similar features"
        self.feature_matrix = None
        self.knn_model = None
        self.scaler = None
    
    def create_feature_matrix(self):
        """
        Create feature matrix for KNN algorithm
        
        Returns:
        --------
        numpy.ndarray
            Feature matrix
        """
        # Extract features for KNN
        features = []
        
        # 1. TF-IDF features would be ideal but can be computationally expensive
        # Instead, we'll use simpler numeric features
        
        # Identify numeric columns we can use
        numeric_cols = ['vote_average', 'popularity', 'vote_count', 'runtime']
        numeric_cols = [col for col in numeric_cols if col in self.df.columns]
        
        if not numeric_cols:
            # Fallback if no numeric columns are available
            return None
        
        # Fill missing values to avoid NaN issues
        feature_df = self.df[numeric_cols].fillna(0)
        
        # 2. Add encoded genre information (one-hot encoding would be ideal)
        # For simplicity, we'll use genre count features
        if 'genres_list' in self.df.columns:
            # Count genres per movie
            feature_df['genre_count'] = self.df['genres_list'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
            
            # Get top genres and create binary features
            all_genres = []
            for genres in self.df['genres_list']:
                if isinstance(genres, list):
                    all_genres.extend(genres)
            
            top_genres = pd.Series(all_genres).value_counts().head(10).index
            
            # Create binary features for top genres
            for genre in top_genres:
                feature_df[f'genre_{genre}'] = self.df['genres_list'].apply(
                    lambda x: 1 if isinstance(x, list) and genre in x else 0
                )
                
        # 3. Add year as a feature (normalized by decade)
        if 'year' in self.df.columns:
            feature_df['decade'] = (self.df['year'] // 10) * 10
        
        # Scale the features
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(feature_df)
        
        return scaled_features
    
    def fit(self, n_neighbors=10):
        """
        Fit the KNN model
        
        Parameters:
        -----------
        n_neighbors : int
            Number of neighbors to consider
        """
        # Create feature matrix
        self.feature_matrix = self.create_feature_matrix()
        
        if self.feature_matrix is None:
            return None
        
        # Initialize and fit KNN model
        self.knn_model = NearestNeighbors(
            n_neighbors=min(n_neighbors + 1, len(self.feature_matrix)),  # +1 because the movie itself will be included
            algorithm='auto',
            metric='euclidean'
        )
        self.knn_model.fit(self.feature_matrix)
        
        return self
    
    def get_movie_neighbors(self, movie_idx, n_neighbors=10):
        """
        Find neighbors for a specific movie
        
        Parameters:
        -----------
        movie_idx : int
            Index of the movie in the dataframe
        n_neighbors : int
            Number of neighbors to return
            
        Returns:
        --------
        list
            List of movie indices
        list
            List of distances
        """
        if self.knn_model is None or self.feature_matrix is None:
            return [], []
            
        # Get movie feature vector
        movie_vector = self.feature_matrix[movie_idx].reshape(1, -1)
        
        # Find k neighbors
        distances, indices = self.knn_model.kneighbors(
            movie_vector, 
            n_neighbors=min(n_neighbors + 1, len(self.feature_matrix))
        )
        
        # Remove the movie itself (first result with distance 0)
        return indices[0][1:], distances[0][1:]
    
    def recommend(self, liked_movies, n=10):
        """
        Generate recommendations based on liked movies using KNN
        
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
        if self.knn_model is None:
            self.fit()
            
        if not liked_movies or self.feature_matrix is None:
            return []
        
        # Get valid indices for selected movies
        valid_movies = [movie for movie in liked_movies if movie in self.title_to_index.index]
        if not valid_movies:
            return []
        
        # Get movie indices
        movie_indices = [self.title_to_index[movie] for movie in valid_movies]
        
        # Find neighbors for each movie
        all_neighbors = []
        all_distances = []
        
        for idx in movie_indices:
            neighbors, distances = self.get_movie_neighbors(idx, n_neighbors=n*2)  # Get more neighbors than needed
            all_neighbors.extend(neighbors)
            all_distances.extend(distances)
        
        # Create a dictionary to store movie index -> distance
        neighbor_dict = {}
        for idx, dist in zip(all_neighbors, all_distances):
            # For movies recommended multiple times, use the minimum distance
            if idx not in neighbor_dict:
                neighbor_dict[idx] = dist
            else:
                neighbor_dict[idx] = min(neighbor_dict[idx], dist)
        
        # Remove movies that the user already likes
        for idx in movie_indices:
            if idx in neighbor_dict:
                del neighbor_dict[idx]
        
        # Sort by distance (ascending)
        sorted_neighbors = sorted(neighbor_dict.items(), key=lambda x: x[1])
        
        # Get top N movie indices and convert distances to similarity scores
        top_indices = [idx for idx, _ in sorted_neighbors[:n]]
        top_scores = [1 / (1 + dist) for _, dist in sorted_neighbors[:n]]  # Convert distance to similarity
        
        # Format recommendations
        return self._format_recommendations(top_indices, [round(score, 3) for score in top_scores])

def get_recommender(df, title_to_index=None):
    """
    Create and return a KNN recommender
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Movie dataframe
    title_to_index : pandas.Series, optional
        Mapping from title to dataframe index
        
    Returns:
    --------
    KNNRecommender
        Initialized recommender
    """
    recommender = KNNRecommender(df, title_to_index)
    recommender.fit()  # Pre-fit the model
    return recommender