# Nova Movie Recommendation System

Nova is a comprehensive movie recommendation system built with Streamlit that offers personalized movie recommendations using various machine learning algorithms.

## Features

- **Personalized Recommendations**: Content-based, collaborative, and hybrid recommendation models
- **Movie Exploration**: Search and browse the TMDB 5000 movie database
- **User Profiles**: Save your favorite genres and movies
- **Watchlist Management**: Track movies you want to watch and have watched
- **Analytics & Insights**: Visualize movie data and your viewing preferences
- **Daily Movie Recommendations**: Get a new movie suggestion every day

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/tonykorycki/NYU-DS-Bootcamp-Nova-movie-recommendation-project
   cd NYU-DS-Bootcamp-Nova-movie-recommendation-project
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

You can run Nova using either of two main scripts:

### Option 1: Using app4.py (Classic Interface)

This is a more lightweight, classic Streamlit interface:

```
streamlit run app4.py
```

### Option 2: Using main.py (Feature-Rich Interface)

This is the fully featured version with all recommendation algorithms and UI enhancements (work in progress):

```
streamlit run main.py
```

## Dataset

Nova requires the TMDB 5000 movies dataset. You should have the following CSV files in your project directory:
- tmdb_5000_movies.csv
- tmdb_5000_credits.csv

If you don't have these files, you can download them from [Kaggle](https://www.kaggle.com/tmdb/tmdb-movie-metadata).

## First-Time Use

1. Enter a username in the sidebar to enable personalized features
2. Select your favorite genres and movies you like
3. Explore recommendations or search for specific movies
4. Add movies to your watchlist
5. Rate movies you've watched to improve recommendations

## Project Structure

- app4.py: Classic interface application file
- main.py: Feature-rich main application file
- data_loader.py: Functions for loading and processing movie data
- utils.py: Utility functions for the application
- recommenders: Directory containing different recommendation algorithms
- profiles: User profiles (created automatically)
- watchlists: User watchlists (created automatically)
- feedback: User feedback and ratings (created automatically)