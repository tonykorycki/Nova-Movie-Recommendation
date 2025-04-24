import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import scipy.sparse as sp

class BaseRecommender:
    """Base class for all recommendation algorithms"""
    
    def __init__(self, df, title_to_index=None):
        """
        Initialize the recommender
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Movie dataframe
        title_to_index : pandas.Series, optional
            Mapping from title to dataframe index
        """
        self.df = df
        self.title_to_index = title_to_index if title_to_index is not None else pd.Series(df.index, index=df['title']).drop_duplicates()
        self.name = "Base Recommender"
        self.description = "Base recommendation system"
        
    def fit(self):
        """Train the recommendation model"""
        # To be implemented by subclasses
        pass
        
    def recommend(self, liked_movies, n=10):
        """
        Generate recommendations based on liked movies
        
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
        # To be implemented by subclasses
        return []
    
    def _format_recommendations(self, indices, scores=None):
        """
        Format recommendations for display
        
        Parameters:
        -----------
        indices : list
            List of movie indices
        scores : list, optional
            List of recommendation scores
            
        Returns:
        --------
        list
            List of recommendation dictionaries
        """
        recommendations = []
        
        for i, idx in enumerate(indices):
            if idx >= len(self.df):
                continue
                
            movie = self.df.iloc[idx]
            rec = {
                'Title': movie.get('title', 'Unknown'),
                'Genres': movie.get('genres_clean', ''),
                'Year': movie.get('year', 'Unknown'),
                'Rating': movie.get('vote_average', 0),
            }
            
            if scores is not None and i < len(scores):
                rec['Score'] = scores[i]
                
            recommendations.append(rec)
            
        return recommendations

@st.cache_data
def compute_tfidf_matrix(df, content_columns=None):
    """
    Compute TF-IDF matrix for content-based filtering
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Movie dataframe
    content_columns : list, optional
        List of columns to use for content representation
        
    Returns:
    --------
    scipy.sparse.csr_matrix
        TF-IDF matrix
    """
    # Default to using only overview if no columns specified
    if content_columns is None:
        content_columns = ['overview']
    
    # Create combined content field
    df['combined_content'] = ''
    for col in content_columns:
        if col in df.columns:
            df['combined_content'] += df[col].fillna('') + ' '
    
    # Initialize TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Return the TF-IDF matrix
    return tfidf.fit_transform(df['combined_content'])

@st.cache_data
def compute_similarity_matrix(df, content_columns=None):
    """
    Compute cosine similarity matrix based on TF-IDF
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Movie dataframe
    content_columns : list, optional
        List of columns to use for content representation
        
    Returns:
    --------
    numpy.ndarray
        Cosine similarity matrix
    """
    try:
        # Get TF-IDF matrix
        tfidf_matrix = compute_tfidf_matrix(df, content_columns)
        
        # Compute cosine similarity matrix
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        return cosine_sim
    except Exception as e:
        st.error(f"Error computing similarity matrix: {e}")
        return None

def combine_recommendations(rec_lists, weights=None, n=10):
    """
    Combine multiple recommendation lists with optional weighting
    
    Parameters:
    -----------
    rec_lists : list of lists
        List of recommendation lists to combine
    weights : list, optional
        List of weights for each recommendation list
    n : int
        Number of recommendations to return
        
    Returns:
    --------
    list
        Combined and sorted list of recommendations
    """
    if not rec_lists:
        return []
        
    # Default to equal weights
    if weights is None:
        weights = [1] * len(rec_lists)
    
    # Normalize weights
    weights = [w / sum(weights) for w in weights]
    
    # Create a dictionary to store combined scores
    combined_scores = {}
    
    # Process each recommendation list with its weight
    for recs, weight in zip(rec_lists, weights):
        for i, rec in enumerate(recs):
            title = rec.get('Title')
            if not title:
                continue
                
            # Calculate score based on position and weight
            score = weight * (1 / (i + 1))
            
            # Add to combined scores
            if title in combined_scores:
                combined_scores[title]['Score'] += score
                # Keep other fields from highest weighted recommendation
                if score > combined_scores[title].get('OriginalScore', 0):
                    combined_scores[title].update({
                        k: v for k, v in rec.items() if k != 'Score'
                    })
                    combined_scores[title]['OriginalScore'] = score
            else:
                combined_scores[title] = rec.copy()
                combined_scores[title]['Score'] = score
                combined_scores[title]['OriginalScore'] = score
    
    # Convert back to list and sort by score
    result = list(combined_scores.values())
    result.sort(key=lambda x: x.get('Score', 0), reverse=True)
    
    # Remove temporary field and return top N
    for rec in result:
        if 'OriginalScore' in rec:
            del rec['OriginalScore']
            
    return result[:n]