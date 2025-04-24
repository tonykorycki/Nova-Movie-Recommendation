import pandas as pd
import numpy as np
import os
from collections import defaultdict
import streamlit as st
from .base import BaseRecommender

class AssociationRulesRecommender(BaseRecommender):
    """Association rules based recommendation system"""
    
    def __init__(self, df, title_to_index=None):
        """
        Initialize the Association Rules recommender
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Movie dataframe
        title_to_index : pandas.Series, optional
            Mapping from title to dataframe index
        """
        super().__init__(df, title_to_index)
        self.name = "Association Rules Recommender"
        self.description = "Recommends movies frequently watched together"
        self.item_sets = {}
        self.association_rules = {}
        self.confidence_scores = {}
        self.support_dict = {}
        
        # Paths for caching
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.rules_path = os.path.join(self.cache_dir, "association_rules.pkl")
    
    def _create_transactions(self, min_transactions=1000):
        """
        Create synthetic user transactions for association rule mining
        
        Parameters:
        -----------
        min_transactions : int
            Minimum number of transactions to create
            
        Returns:
        --------
        list
            List of transaction lists, where each transaction is a list of movie titles
        """
        # Create simulated user transactions (movie watching sessions)
        # Group movies by genre for more realistic transactions
        genre_groups = {}
        for _, row in self.df.iterrows():
            for genre in row.get('genres_list', []):
                if genre not in genre_groups:
                    genre_groups[genre] = []
                genre_groups[genre].append(row['title'])
        
        # Create synthetic transactions
        transactions = []
        
        # First, create genre-based transactions (people tend to watch similar genres)
        all_genres = list(genre_groups.keys())
        for _ in range(min_transactions // 2):
            # Select 1-2 genres for this transaction
            n_genres = np.random.randint(1, 3)
            if len(all_genres) > 0:
                selected_genres = np.random.choice(all_genres, min(n_genres, len(all_genres)), replace=False)
                
                # Create a transaction with 2-5 movies from these genres
                transaction = []
                for genre in selected_genres:
                    if genre in genre_groups and len(genre_groups[genre]) > 0:
                        n_movies = min(np.random.randint(2, 6), len(genre_groups[genre]))
                        genre_movies = np.random.choice(genre_groups[genre], n_movies, replace=False)
                        transaction.extend(genre_movies)
                
                # Ensure transaction isn't empty or too large
                if transaction and len(transaction) <= 10:
                    transactions.append(list(set(transaction)))  # Remove duplicates
        
        # Add some random transactions for diversity
        for _ in range(min_transactions - len(transactions)):
            n_movies = np.random.randint(2, 6)
            if len(self.df) > 0:
                random_movies = np.random.choice(self.df['title'].values, 
                                               min(n_movies, len(self.df)), 
                                               replace=False)
                transactions.append(list(random_movies))
        
        return transactions
    
    def _load_real_transactions(self):
        """
        Load real user transactions from profiles and feedback
        
        Returns:
        --------
        list
            List of transaction lists from real user data
        """
        transactions = []
        
        # Try to load from profiles directory
        profiles_dir = "profiles"
        if os.path.exists(profiles_dir):
            import json
            for filename in os.listdir(profiles_dir):
                if filename.endswith('.json'):
                    try:
                        with open(os.path.join(profiles_dir, filename), 'r') as f:
                            profile = json.load(f)
                            if 'liked_movies' in profile and len(profile['liked_movies']) >= 2:
                                transactions.append(profile['liked_movies'])
                    except:
                        continue
        
        # Try to load from feedback directory
        feedback_dir = "feedback"
        if os.path.exists(feedback_dir):
            import json
            for filename in os.listdir(feedback_dir):
                if filename.endswith('.json'):
                    try:
                        with open(os.path.join(feedback_dir, filename), 'r') as f:
                            ratings = json.load(f)
                            if isinstance(ratings, dict) and len(ratings) >= 2:
                                # Only include movies with ratings >= 3.5
                                good_movies = [movie for movie, rating in ratings.items() 
                                              if rating >= 3.5]
                                if len(good_movies) >= 2:
                                    transactions.append(good_movies)
                    except:
                        continue
        
        # Try to load from watchlists directory
        watchlists_dir = "watchlists"
        if os.path.exists(watchlists_dir):
            import json
            for filename in os.listdir(watchlists_dir):
                if filename.endswith('.json'):
                    try:
                        with open(os.path.join(watchlists_dir, filename), 'r') as f:
                            watchlist = json.load(f)
                            if isinstance(watchlist, dict):
                                if 'watched' in watchlist and len(watchlist['watched']) >= 2:
                                    transactions.append(watchlist['watched'])
                                if 'to_watch' in watchlist and len(watchlist['to_watch']) >= 2:
                                    transactions.append(watchlist['to_watch'])
                    except:
                        continue
                        
        return transactions
        
    def _create_item_sets(self, transactions):
        """
        Create frequent itemsets from transactions
        
        Parameters:
        -----------
        transactions : list
            List of transactions
            
        Returns:
        --------
        dict
            Dictionary with itemsets and their support values
        """
        min_support = max(3, len(transactions) // 100)  # Adaptive minimum support
        
        # Count occurrences of individual items (1-itemsets)
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1
        
        # Filter by minimum support
        frequent_items = {item: count for item, count in item_counts.items() 
                         if count >= min_support}
        
        # Store in itemsets dictionary
        L1 = {frozenset([item]): count for item, count in frequent_items.items()}
        
        # Store support as percentage of transactions
        n_transactions = len(transactions)
        self.support_dict = {frozenset([item]): count/n_transactions 
                            for item, count in frequent_items.items()}
        
        # Generate 2-itemsets
        L2_candidates = defaultdict(int)
        for idx, transaction in enumerate(transactions):
            # Convert transaction to set of frequent items only
            frequent_trans = [item for item in transaction if item in frequent_items]
            
            # Generate all pairs
            for i in range(len(frequent_trans)):
                for j in range(i+1, len(frequent_trans)):
                    item_pair = frozenset([frequent_trans[i], frequent_trans[j]])
                    L2_candidates[item_pair] += 1
        
        # Filter 2-itemsets by minimum support
        L2 = {itemset: count for itemset, count in L2_candidates.items() 
             if count >= min_support}
        
        # Update support dict
        for itemset, count in L2.items():
            self.support_dict[itemset] = count/n_transactions
        
        # Combine into final itemsets
        all_itemsets = {**L1, **L2}
        
        return all_itemsets
    
    def _generate_rules(self, itemsets):
        """
        Generate association rules from frequent itemsets
        
        Parameters:
        -----------
        itemsets : dict
            Dictionary with itemsets and their counts
            
        Returns:
        --------
        dict
            Dictionary with rules and their confidence scores
        """
        rules = {}
        confidence_scores = {}
        
        # We only need rules from 2-itemsets
        for itemset, count in itemsets.items():
            if len(itemset) == 2:
                items = list(itemset)
                
                # Calculate confidence for both directions
                for i, item1 in enumerate(items):
                    item2 = items[1-i]  # The other item
                    
                    # Calculate confidence
                    confidence = count / itemsets[frozenset([item1])]
                    
                    # Store rule if confidence is meaningful
                    if confidence > 0.1:  # Minimum confidence threshold
                        rules.setdefault(item1, []).append(item2)
                        confidence_scores[(item1, item2)] = confidence
        
        return rules, confidence_scores
    
    def _load_cached_rules(self):
        """Load cached association rules if available"""
        try:
            rules_path = os.path.join(self.cache_dir, "association_rules.pkl")
            if os.path.exists(rules_path):
                cached_data = pd.read_pickle(rules_path)
                self.association_rules = cached_data.get('rules', {})
                self.confidence_scores = cached_data.get('confidence', {})
                self.support_dict = cached_data.get('support', {})
                return True
        except Exception as e:
            st.warning(f"Error loading cached rules: {e}")
        return False
    
    def _save_rules_to_cache(self):
        """Save association rules to cache"""
        try:
            rules_path = os.path.join(self.cache_dir, "association_rules.pkl")
            cached_data = {
                'rules': self.association_rules,
                'confidence': self.confidence_scores,
                'support': self.support_dict
            }
            pd.to_pickle(cached_data, rules_path)
        except Exception as e:
            st.warning(f"Error saving rules to cache: {e}")
    
    def fit(self, min_transactions=1000):
        """
        Fit the association rules model
        
        Parameters:
        -----------
        min_transactions : int
            Minimum number of transactions to create
        """
        # Try to load cached rules first
        if self._load_cached_rules():
            return self
            
        # Load real transactions from user data
        real_transactions = self._load_real_transactions()
        
        # If we don't have enough real transactions, create synthetic ones
        if len(real_transactions) < min_transactions:
            synthetic_transactions = self._create_transactions(
                min_transactions - len(real_transactions)
            )
            transactions = real_transactions + synthetic_transactions
        else:
            transactions = real_transactions
        
        # Create frequent itemsets
        self.item_sets = self._create_item_sets(transactions)
        
        # Generate association rules
        self.association_rules, self.confidence_scores = self._generate_rules(self.item_sets)
        
        # Save rules to cache
        self._save_rules_to_cache()
        
        return self
    
    def recommend(self, liked_movies, n=10):
        """
        Generate recommendations based on association rules
        
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
        if not self.association_rules:
            self.fit()
            
        if not liked_movies:
            return []
        
        # Get movies that are in our association rules
        valid_movies = [movie for movie in liked_movies if movie in self.association_rules]
        
        if not valid_movies:
            return []
            
        # Create a dictionary to score potential recommendations
        scores = defaultdict(float)
        
        # For each liked movie, find associated movies
        for movie in valid_movies:
            if movie in self.association_rules:
                for rec_movie in self.association_rules[movie]:
                    if rec_movie not in liked_movies:
                        # Score is the confidence of the rule
                        confidence = self.confidence_scores.get((movie, rec_movie), 0)
                        # Also consider support
                        support = self.support_dict.get(frozenset([movie, rec_movie]), 0)
                        # Combined score (can adjust weights)
                        score = 0.7 * confidence + 0.3 * support
                        scores[rec_movie] += score
        
        # Sort recommendations by score
        sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top N
        top_recs = sorted_recs[:n]
        
        # Format recommendations
        recommendations = []
        for movie_title, score in top_recs:
            if movie_title in self.df['title'].values:
                # Find the movie in the dataframe
                movie_idx = self.df[self.df['title'] == movie_title].index[0]
                
                recommendations.append({
                    'Title': movie_title,
                    'Genres': self.df.iloc[movie_idx].get('genres_clean', ''),
                    'Year': self.df.iloc[movie_idx].get('year', 'Unknown'),
                    'Rating': self.df.iloc[movie_idx].get('vote_average', 0),
                    'Score': round(score, 3)
                })
                
        return recommendations

def get_recommender(df, title_to_index=None):
    """
    Create and return an Association Rules recommender
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Movie dataframe
    title_to_index : pandas.Series, optional
        Mapping from title to dataframe index
        
    Returns:
    --------
    AssociationRulesRecommender
        Initialized recommender
    """
    return AssociationRulesRecommender(df, title_to_index)