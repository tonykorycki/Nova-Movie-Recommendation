# recommenders/demographic_filtering.py

import pandas as pd

class DemographicRecommender:
    def __init__(self, df, min_percentile=0.90):
        """
        Initializes the recommender with the dataset and a vote count threshold.
        """
        self.df = df.copy()
        self.min_percentile = min_percentile
        self.C = self.df['vote_average'].mean()
        self.m = self.df['vote_count'].quantile(self.min_percentile)
        self.qualified = None

    def _calculate_weighted_rating(self, x):
        v = x['vote_count']
        R = x['vote_average']
        return (v / (v + self.m)) * R + (self.m / (v + self.m)) * self.C

    def filter_qualified_movies(self):
        """
        Filters movies that have vote_count >= m.
        """
        self.qualified = self.df[self.df['vote_count'] >= self.m].copy()
        if self.qualified.empty:
            print("⚠️ No qualified movies found with the current threshold. Consider lowering the percentile.")
        return self.qualified

    def get_top_movies(self, top_n=10):
        """
        Returns top N movies based on weighted rating.
        """
        if self.qualified is None:
            self.filter_qualified_movies()
        
        self.qualified['weighted_rating'] = self.qualified.apply(self._calculate_weighted_rating, axis=1)
        top_movies = self.qualified.sort_values('weighted_rating', ascending=False)
        return top_movies[['title_x', 'vote_count', 'vote_average', 'weighted_rating']].head(top_n)
