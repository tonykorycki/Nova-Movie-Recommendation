# recommenders/content_based.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, dataframe):
        """
        Initialize with the merged preprocessed dataframe.
        """
        self.df = dataframe.copy()
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None

    def prepare_features(self):
        """
        Create a 'combined_features' column by combining metadata for each movie.
        """
        def combine_features(x):
            return ' '.join(str(x.get(col, '')) for col in ['genres_clean', 'cast_clean', 'director', 'keywords'])
        
        self.df['combined_features'] = self.df.apply(combine_features, axis=1)

    def vectorize(self):
        """
        Vectorize the combined features using TF-IDF.
        """
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.df['combined_features'])

    def compute_similarity(self):
        """
        Compute cosine similarity matrix for all movies.
        """
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def create_indices(self):
        """
        Create a mapping from movie title to index.
        """
        self.indices = pd.Series(self.df.index, index=self.df['title_x']).drop_duplicates()

    def fit(self):
        """
        Fit the content-based filtering model.
        """
        self.prepare_features()
        self.vectorize()
        self.compute_similarity()
        self.create_indices()

    def recommend(self, movie_title, top_n=10):
        """
        Recommend top N movies similar to the given movie title.

        Parameters:
            movie_title (str): Title of the movie user likes.
            top_n (int): Number of similar movies to recommend.

        Returns:
            pd.DataFrame: Top N similar movies.
        """
        if movie_title not in self.indices:
            print(f"[ERROR] Movie '{movie_title}' not found in dataset.")
            return pd.DataFrame()

        idx = self.indices[movie_title]

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        sim_scores = sim_scores[1:top_n+1]  # Skip the movie itself at 0

        movie_indices = [i[0] for i in sim_scores]

        return self.df.iloc[movie_indices][['title_x', 'genres_clean', 'vote_average', 'popularity']]
