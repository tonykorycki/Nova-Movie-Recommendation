import numpy as np
import pandas as pd
import streamlit as st
import os
from .base import BaseRecommender

# Check if TensorFlow/Keras is available
try:
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    

class NeuralRecommender(BaseRecommender):
    """Neural network–based recommendation system"""
    def __init__(self, df, title_to_index=None):
        """
        Initialize the Neural Network recommender
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Movie dataframe
        title_to_index : pandas.Series, optional
            Mapping from title to dataframe index
        """
        super().__init__(df, title_to_index)
        self.name = "Neural Network Recommender"
        self.description = "Uses deep learning to model user-item interactions"
        self.model = None
        self.movie_embedding = None
        self.movie_bias = None
        self.movie_embeddings_df = None
        self.movie_factors = None
        
        # Neural embeddings path for caching
        self.cache_dir = "cache"
        self.embeddings_path = os.path.join(self.cache_dir, "movie_embeddings.npy")
        self.bias_path = os.path.join(self.cache_dir, "movie_bias.npy")
    
    def build_simple_model(self, n_movies, n_factors=20):
        """
        Build a simple neural embedding model
        
        Parameters:
        -----------
        n_movies : int
            Number of movies in the dataset
        n_factors : int
            Number of latent factors
            
        Returns:
        --------
        tf.keras.Model
            Neural network model
        """
        if not TF_AVAILABLE:
            st.error("TensorFlow is not available. Please install TensorFlow to use the Neural Network recommender.")
            return None
            
        # Movie input and embedding
        movie_input = Input(shape=(1,), name='movie_input')
        movie_embedding = Embedding(n_movies, n_factors, name='movie_embedding')(movie_input)
        movie_flatten = Flatten()(movie_embedding)
        
        # Movie bias terms
        movie_bias = Embedding(n_movies, 1, name='movie_bias')(movie_input)
        movie_bias_flatten = Flatten()(movie_bias)
        
        # Create a simple MLP
        concat = Concatenate()([movie_flatten, movie_bias_flatten])
        dense1 = Dense(20, activation='relu')(concat)
        dense2 = Dense(10, activation='relu')(dense1)
        output = Dense(1)(dense2)
        
        # Build model
        model = Model(inputs=movie_input, outputs=output)
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        
        return model
        
    def create_dummy_training_data(self, n_samples=10000):
        """
        Create dummy training data for the neural model
        
        Parameters:
        -----------
        n_samples : int
            Number of training samples to create
            
        Returns:
        --------
        tuple
            (movies, ratings) for training
        """
        # Sample random movie indices
        n_movies = len(self.df)
        movies = np.random.randint(0, n_movies, n_samples)
        
        # Create ratings - higher for popular and well-rated movies
        ratings = []
        for idx in movies:
            movie = self.df.iloc[idx]
            # Base rating influenced by vote_average and popularity
            base_rating = movie.get('vote_average', 5) / 2  # Convert 0-10 to 0-5 scale
            
            # Add some noise for variability
            noise = np.random.normal(0, 0.5)
            rating = base_rating + noise
            
            # Clamp to valid rating range
            rating = max(0.5, min(5, rating))
            ratings.append(rating)
            
        return movies, np.array(ratings)
    
    def load_or_train_embeddings(self, n_factors=20):
        """
        Load cached embeddings or train new ones
        
        Parameters:
        -----------
        n_factors : int
            Number of latent factors
            
        Returns:
        --------
        tuple
            (movie_embeddings, movie_bias) arrays
        """
        # Try to load cached embeddings
        try:
            if os.path.exists(self.embeddings_path) and os.path.exists(self.bias_path):
                movie_embeddings = np.load(self.embeddings_path)
                movie_bias = np.load(self.bias_path)
                
                if movie_embeddings.shape[0] == len(self.df) and movie_embeddings.shape[1] == n_factors:
                    return movie_embeddings, movie_bias
        except Exception as e:
            st.warning(f"Error loading cached embeddings: {e}. Training new embeddings.")
        
        # If not available, train new embeddings
        if not TF_AVAILABLE:
            return self._create_fallback_embeddings(n_factors)
            
        # Build model
        model = self.build_simple_model(len(self.df), n_factors)
        if model is None:
            return self._create_fallback_embeddings(n_factors)
            
        # Create training data
        movie_indices, ratings = self.create_dummy_training_data()
        
        # Train model
        model.fit(movie_indices, ratings, epochs=5, batch_size=64, verbose=0)
        
        # Extract embeddings
        movie_embedding_layer = model.get_layer('movie_embedding')
        movie_embeddings = movie_embedding_layer.get_weights()[0]
        
        movie_bias_layer = model.get_layer('movie_bias')
        movie_bias = movie_bias_layer.get_weights()[0]
        
        # Cache embeddings
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(self.embeddings_path, movie_embeddings)
            np.save(self.bias_path, movie_bias)
        except Exception as e:
            st.warning(f"Couldn't save embeddings to cache: {e}")
        
        return movie_embeddings, movie_bias
    
    def _create_fallback_embeddings(self, n_factors):
        """
        Create fallback embeddings when TensorFlow is not available
        
        Parameters:
        -----------
        n_factors : int
            Number of latent factors
            
        Returns:
        --------
        tuple
            (movie_embeddings, movie_bias) arrays
        """
        # Create SVD-like embeddings using features
        movie_embeddings = np.random.normal(0, 0.1, size=(len(self.df), n_factors))
        
        # Initialize with some feature-based values
        if 'vote_average' in self.df.columns:
            movie_embeddings[:, 0] = (self.df['vote_average'].fillna(5.0) / 10.0) * 2 - 1
            
        if 'popularity' in self.df.columns:
            # Normalize popularity (which can be very skewed)
            pop = self.df['popularity'].fillna(0)
            pop = (pop - pop.min()) / (pop.max() - pop.min() + 1e-6)
            movie_embeddings[:, 1] = pop * 2 - 1
        
        # Create separate bias term from ratings
        movie_bias = np.zeros((len(self.df), 1))
        if 'vote_average' in self.df.columns:
            movie_bias[:, 0] = (self.df['vote_average'].fillna(5.0) - 5.0) / 5.0
        
        return movie_embeddings, movie_bias
    
    def fit(self, n_factors=20):
        """
        Fit the neural recommendation model
        
        Parameters:
        -----------
        n_factors : int
            Number of latent factors
        """
        # Load or train embeddings
        self.movie_embedding, self.movie_bias = self.load_or_train_embeddings(n_factors)
        
        # Create a dataframe with movie embeddings for faster lookups
        self.movie_factors = n_factors
        
        return self
    
    def _calculate_similarity(self, movie_idx, n=100):
        """
        Calculate cosine similarity between a movie and all others
        
        Parameters:
        -----------
        movie_idx : int
            Index of the movie
        n : int
            Number of similar movies to return
            
        Returns:
        --------
        list
            List of (movie_idx, similarity_score) tuples
        """
        if self.movie_embedding is None:
            return []
        
        # Get embedding for the movie
        embedding = self.movie_embedding[movie_idx]
        
        # Calculate cosine similarity
        norm = np.linalg.norm(self.movie_embedding, axis=1) * np.linalg.norm(embedding)
        norm = np.where(norm == 0, 1e-10, norm)  # Avoid division by zero
        
        similarity = np.dot(self.movie_embedding, embedding) / norm
        
        # Create (idx, similarity) pairs
        sim_scores = list(enumerate(similarity))
        
        # Sort by similarity
        sim_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Remove the movie itself
        sim_scores = [s for s in sim_scores if s[0] != movie_idx]
        
        # Return top N
        return sim_scores[:n]
    
    def recommend(self, liked_movies, n=10):
        """
        Generate recommendations based on neural embeddings
        
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
        if self.movie_embedding is None:
            self.fit()
            
        if not liked_movies:
            return []
        
        # Get valid indices for selected movies
        valid_movies = [movie for movie in liked_movies if movie in self.title_to_index.index]
        if not valid_movies:
            return []
        
        movie_indices = [self.title_to_index[movie] for movie in valid_movies]
        
        # Get similar movies for each liked movie
        all_similar = []
        for idx in movie_indices:
            similar = self._calculate_similarity(idx, n=n*2)
            all_similar.extend(similar)
        
        # Create a dictionary to store movie index -> score
        movie_scores = {}
        for idx, score in all_similar:
            if idx not in movie_scores:
                movie_scores[idx] = score
            else:
                movie_scores[idx] = max(movie_scores[idx], score)
        
        # Remove movies that are already liked
        for idx in movie_indices:
            if idx in movie_scores:
                del movie_scores[idx]
        
        # Sort by score
        sorted_scores = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top N
        top_indices = [idx for idx, _ in sorted_scores[:n]]
        top_scores = [score for _, score in sorted_scores[:n]]
        
        # Format recommendations
        return self._format_recommendations(top_indices, [round(score, 3) for score in top_scores])

def get_recommender(df, title_to_index=None):
    """
    Create and return a Neural Network recommender
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Movie dataframe
    title_to_index : pandas.Series, optional
        Mapping from title to dataframe index
        
    Returns:
    --------
    NeuralRecommender
        Initialized recommender
    """
    return NeuralRecommender(df, title_to_index)