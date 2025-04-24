import streamlit as st
import pandas as pd
import numpy as np
from .base import BaseRecommender, compute_similarity_matrix

class ContentBasedRecommender(BaseRecommender):
    """Content-based movie recommendation system using TF-IDF and cosine similarity"""
    
    def __init__(self, df, title_to_index=None, content_columns=None):
        """
        Initialize the content-based recommender
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Movie dataframe
        title_to_index : pandas.Series, optional
            Mapping from title to dataframe index
        content_columns : list, optional
            List of columns to use for content representation
        """
        super().__init__(df, title_to_index)
        self.name = "Content-Based Recommender"
        self.description = "Recommends movies similar to ones you like based on features like overview, genres, director, etc."
        self.content_columns = content_columns or ['overview']
        self.similarity_matrix = None
    
    def fit(self):
        """Compute similarity matrix for content-based recommendations"""
        # Compute enhanced similarity matrix with all relevant columns
        self.similarity_matrix = compute_similarity_matrix(self.df, self.content_columns)
        return self
        
    def recommend(self, liked_movies, n=10, favorite_genres=None):
        """
        Generate content-based recommendations
        
        Parameters:
        -----------
        liked_movies : list
            List of movie titles the user likes
        n : int
            Number of recommendations to return
        favorite_genres : list, optional
            User's favorite genres for additional filtering
            
        Returns:
        --------
        list
            List of recommended movie dictionaries
        """
        if not self.similarity_matrix is not None:
            self.fit()
            
        if not liked_movies:
            return []
        
        # Get valid indices for selected movies
        valid_movies = [movie for movie in liked_movies if movie in self.title_to_index.index]
        if not valid_movies:
            return []
            
        # Get indices of valid movies
        movie_indices = [self.title_to_index[movie] for movie in valid_movies]
        
        # Sum similarity scores for all liked movies
        sim_scores = np.sum(self.similarity_matrix[movie_indices], axis=0)
        
        # Create a list of (index, similarity score) tuples
        sim_scores = list(enumerate(sim_scores))
        
        # Sort movies by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Filter out movies that are already liked
        sim_scores = [i for i in sim_scores if i[0] not in movie_indices]
        
        # Apply genre-based weighting if favorite genres are provided
        if favorite_genres and len(favorite_genres) > 0:
            # Convert to set for faster lookups
            genre_set = set(favorite_genres)
            
            # Add genre weight to each movie score
            weighted_scores = []
            for idx, score in sim_scores[:50]:  # Only process top 50 for better performance
                movie = self.df.iloc[idx]
                # Calculate genre match
                movie_genres = movie.get('genres_list', [])
                genre_matches = sum(1 for genre in movie_genres if genre in genre_set)
                
                if len(favorite_genres) > 0:
                    genre_weight = genre_matches / len(favorite_genres)
                else:
                    genre_weight = 0
                
                # Apply weights: content similarity (50%), genre match (30%), rating (20%)
                final_score = (
                    0.5 * score + 
                    0.3 * genre_weight + 
                    0.2 * (movie.get('vote_average', 0) / 10)
                )
                
                weighted_scores.append((idx, final_score, score))  # Store original score for reference
                
            # Sort by final weighted score
            weighted_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get top N movie indices and scores
            movie_indices = [i[0] for i in weighted_scores[:n]]
            scores = [round(i[1], 3) for i in weighted_scores[:n]]
        else:
            # If no genre preferences, just use similarity scores
            movie_indices = [i[0] for i in sim_scores[:n]]
            scores = [round(i[1], 3) for i in sim_scores[:n]]
        
        # Format recommendations
        return self._format_recommendations(movie_indices, scores)

class EnhancedContentRecommender(ContentBasedRecommender):
    """Enhanced content-based recommender with more features and weighting"""
    
    def __init__(self, df, title_to_index=None):
        """Initialize with all relevant content columns"""
        content_columns = ['overview', 'genres_clean', 'director', 'cast_clean']
        super().__init__(df, title_to_index, content_columns)
        self.name = "Enhanced Content-Based Recommender"
        self.description = "Advanced content analysis with overview, genres, director, and cast"
    
    def prepare_content_fields(self):
        """Prepare enhanced content fields for better recommendations"""
        # Create a rich combined content field
        self.df['combined_content'] = ''
        
        # Add overview with higher weight (repeat to increase importance)
        self.df['combined_content'] = self.df['overview'].fillna('') + ' ' + self.df['overview'].fillna('')
        
        # Add genres (convert list to string with spaces)
        self.df['combined_content'] += self.df['genres_list'].apply(
            lambda x: ' ' + ' '.join([f"genre_{genre}" for genre in x]) if isinstance(x, list) else ''
        )
        
        # Add director (with prefix to distinguish from actor names)
        self.df['combined_content'] += ' director_' + self.df['director'].fillna('').str.replace(' ', '_')
        
        # Add main cast (first 3 actors)
        self.df['combined_content'] += self.df['cast_list'].apply(
            lambda x: ' ' + ' '.join([f"actor_{actor}" for actor in x[:3]]) if isinstance(x, list) else ''
        )
        
        # Add year information as a feature (binned by decade)
        self.df['combined_content'] += self.df['year'].fillna(0).astype(int).apply(
            lambda x: f' decade_{int(x/10)*10}' if x > 0 else ''
        )
        
        # Return the dataframe with the new column
        return self.df
    
    def fit(self):
        """Prepare content fields and compute similarity matrix"""
        # First prepare the enhanced content fields
        self.prepare_content_fields()
        
        # Then compute similarity matrix
        return super().fit()
        
def get_recommender(df, recommender_type="content", title_to_index=None):
    """
    Factory function to get the appropriate recommender
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Movie dataframe
    recommender_type : str
        Type of recommender to create
    title_to_index : pandas.Series, optional
        Mapping from title to dataframe index
        
    Returns:
    --------
    BaseRecommender
        Initialized recommender object
    """
    if recommender_type == "enhanced":
        recommender = EnhancedContentRecommender(df, title_to_index)
    else:  # Default to basic content-based
        recommender = ContentBasedRecommender(df, title_to_index)
        
    # Pre-compute the similarity matrix
    recommender.fit()
    return recommender