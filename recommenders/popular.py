import pandas as pd
import numpy as np
from .base import BaseRecommender

class PopularMoviesRecommender(BaseRecommender):
    """Recommender for popular, highly-rated movies"""
    
    def __init__(self, df, title_to_index=None):
        super().__init__(df, title_to_index)
        self.name = "Popular Movies Recommender"
        self.description = "Recommends popular and highly-rated movies"
    
    def fit(self):
        """No training needed for popular recommender"""
        # Nothing to fit
        return self
    
    def recommend(self, liked_movies=None, n=10, recency_weight=0.3):
        """Generate recommendations based on popularity and ratings
        
        Parameters:
        -----------
        liked_movies : list, optional
            List of movie titles the user likes (used to filter out movies already seen)
        n : int
            Number of recommendations to return
        recency_weight : float
            Weight given to recency (0-1), higher means more recent movies preferred
            
        Returns:
        --------
        list
            List of recommended movie dictionaries
        """
        # Start with the full dataset
        df = self.df.copy()
        
        # Filter out already watched/liked movies if provided
        if liked_movies:
            df = df[~df['title'].isin(liked_movies)]
        
        # Ensure we have the needed columns
        if not all(col in df.columns for col in ['vote_average', 'vote_count', 'popularity', 'year']):
            raise ValueError("Required columns missing from dataframe")
        
        # Filter to movies with sufficient votes for reliable ratings (at least 100 votes)
        df = df[df['vote_count'] >= 100]
        
        if df.empty:
            return []
            
        # Create a score combining rating, popularity, and recency
        # Normalize each component for fair weighting
        
        # Normalize ratings (0-10 scale)
        max_rating = df['vote_average'].max()
        min_rating = df['vote_average'].min()
        df['rating_norm'] = (df['vote_average'] - min_rating) / (max_rating - min_rating)
        
        # Normalize popularity (can have very large values)
        max_pop = df['popularity'].max()
        min_pop = df['popularity'].min()
        df['popularity_norm'] = (df['popularity'] - min_pop) / (max_pop - min_pop)
        
        # Calculate recency score (0-1 scale, with newer movies higher)
        current_year = df['year'].max()
        earliest_year = df['year'].min()
        year_range = max(1, current_year - earliest_year)
        df['recency'] = (df['year'] - earliest_year) / year_range
        
        # Combine into final score: 40% rating, 40% popularity, 20% recency
        df['score'] = (0.4 * df['rating_norm'] + 
                      0.4 * df['popularity_norm'] + 
                      recency_weight * df['recency'])
        
        # Sort by final score and get top N
        top_movies = df.sort_values('score', ascending=False).head(n)
        
        # Format into recommendation dicts
        recommendations = []
        for _, movie in top_movies.iterrows():
            recommendations.append({
                "Title": movie["title"],
                "Year": int(movie["year"]) if not pd.isna(movie["year"]) else "N/A",
                "Rating": round(float(movie["vote_average"]), 1) if not pd.isna(movie["vote_average"]) else "N/A",
                "Genres": movie.get("genres_clean", ""),
                "Popularity": round(float(movie["popularity"]), 1) if not pd.isna(movie["popularity"]) else 0,
                "Vote Count": int(movie["vote_count"]) if not pd.isna(movie["vote_count"]) else 0,
                "Description": movie.get("overview", "")[:150] + "..." if len(movie.get("overview", "")) > 150 else movie.get("overview", ""),
            })
        
        return recommendations

# Factory function to create and return a recommender instance
def get_recommender(df, recommender_type="basic", title_to_index=None):
    """Create and return a popular movies recommender instance"""
    return PopularMoviesRecommender(df, title_to_index)