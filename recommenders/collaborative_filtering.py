# recommenders/collaborative.py

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate, train_test_split

class CollaborativeFilteringRecommender:
    def __init__(self, ratings_csv):
        """
        Initialize with path to ratings CSV file.
        """
        self.ratings_csv = ratings_csv
        self.data = None
        self.model = None
        self.trainset = None

    def load_data(self):
        """
        Load the ratings data into Surprise format.
        """
        ratings = pd.read_csv(self.ratings_csv)

        reader = Reader(rating_scale=(0.5, 5.0))  # Assuming rating scale
        self.data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    def train_model(self):
        """
        Train the SVD model on the full dataset.
        """
        self.trainset = self.data.build_full_trainset()
        self.model = SVD()
        self.model.fit(self.trainset)

    def cross_validate_model(self, folds=5):
        """
        Perform cross-validation and print RMSE, MAE.
        """
        print("[INFO] Cross-validating SVD model...")
        results = cross_validate(SVD(), self.data, measures=['RMSE', 'MAE'], cv=folds, verbose=True)
        return results

    def predict_rating(self, user_id, movie_id):
        """
        Predict rating for a given user and movie.
        """
        if self.model is None:
            print("[ERROR] Model is not trained yet!")
            return None
        
        prediction = self.model.predict(uid=user_id, iid=movie_id)
        return prediction.est  # The estimated rating

    def recommend_movies(self, user_id, movie_mapping_df, top_n=10):
        """
        Recommend top N movies for a given user that they haven't rated yet.
        
        Parameters:
            user_id (int): ID of the user
            movie_mapping_df (pd.DataFrame): DataFrame with ['movieId', 'title_x']
            top_n (int): Number of recommendations
        
        Returns:
            pd.DataFrame: Recommended movie titles and predicted ratings.
        """
        all_movie_ids = movie_mapping_df['movieId'].unique()
        user_rated_movies = set([j for (j, _) in self.trainset.ur[self.trainset.to_inner_uid(user_id)]])
        user_rated_movie_ids = set([self.trainset.to_raw_iid(j) for j in user_rated_movies])

        movies_to_predict = [movie for movie in all_movie_ids if str(movie) not in user_rated_movie_ids]

        predictions = [self.predict_rating(user_id, movie_id) for movie_id in movies_to_predict]
        movie_preds = pd.DataFrame({
            'movieId': movies_to_predict,
            'predicted_rating': predictions
        })

        top_movies = movie_preds.sort_values('predicted_rating', ascending=False).head(top_n)
        top_movies = top_movies.merge(movie_mapping_df, on='movieId', how='left')

        return top_movies[['title_x', 'predicted_rating']]
