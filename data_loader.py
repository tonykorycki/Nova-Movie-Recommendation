import pandas as pd
import streamlit as st
from utils import extract_list, extract_director

@st.cache_data
def load_data():
    """
    Load and preprocess the movie dataset
    Returns a preprocessed DataFrame
    """
    try:
        movies_df = pd.read_csv("tmdb_5000_movies.csv")
        credits_df = pd.read_csv("tmdb_5000_credits.csv")
        
        # Merge dataframes once
        credits_df.rename(columns={'movie_id': 'id'}, inplace=True)
        merged_df = movies_df.merge(credits_df, on='id')
        merged_df.rename(columns={'title_x': 'title'}, inplace=True)
        
        # Fill null values
        merged_df['overview'] = merged_df['overview'].fillna('unknown')
        merged_df['homepage'] = merged_df['homepage'].fillna('none')
        merged_df['tagline'] = merged_df['tagline'].fillna('none')
        merged_df.dropna(axis=0, how='any', inplace=True)
        
        # Pre-process JSON columns once
        merged_df['genres_list'] = merged_df['genres'].apply(lambda x: extract_list(x))
        merged_df['genres_clean'] = merged_df['genres_list'].apply(lambda x: ", ".join([item for item in x[:3]]))
        merged_df['cast_list'] = merged_df['cast'].apply(lambda x: extract_list(x))
        merged_df['cast_clean'] = merged_df['cast_list'].apply(lambda x: ", ".join([item for item in x[:3]]))
        merged_df['director'] = merged_df['crew'].apply(extract_director)
        
        # Convert release_date to year
        merged_df['year'] = pd.to_datetime(merged_df['release_date'], errors='coerce').dt.year
        
        # Prepare DataFrame for caching by handling unhashable types
        from utils import prepare_df_for_caching
        cacheable_df = prepare_df_for_caching(merged_df)
        
        return cacheable_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def filter_dataframe(df, year_range, runtime_range, languages, actors=None):
    """
    Apply filters to the dataframe based on user selections
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to filter
    year_range : tuple
        (min_year, max_year)
    runtime_range : tuple
        (min_runtime, max_runtime)
    languages : list
        List of selected languages
    actors : list, optional
        List of selected actors
        
    Returns:
    --------
    pandas.DataFrame
        Filtered dataframe
    """
    filtered_df = df.copy()
    
    # Apply basic filters
    mask = (
        (filtered_df['year'] >= year_range[0]) &
        (filtered_df['year'] <= year_range[1]) &
        (filtered_df['runtime'] >= runtime_range[0]) &
        (filtered_df['runtime'] <= runtime_range[1]) &
        (filtered_df['original_language'].isin(languages))
    )
    
    # Actor filtering - only apply if actors are selected
    if actors:
        actor_mask = filtered_df['cast_list'].apply(
            lambda cast: any(actor in cast for actor in actors)
        )
        mask = mask & actor_mask
    
    return filtered_df[mask]

def get_top_actors(df, limit=1000):
    """
    Extract top actors from the dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The movie dataframe
    limit : int
        Maximum number of actors to return
        
    Returns:
    --------
    list
        List of top actors
    """
    # Build frequency count of actors
    actor_counts = {}
    
    # Process only the first 100 movies for performance
    for cast in df['cast_list'].head(100):
        for actor in cast[:5]:  # Only use top 3 actors per movie
            if actor in actor_counts:
                actor_counts[actor] += 1
            else:
                actor_counts[actor] = 1
    
    # Sort by count and return top 'limit' actors
    sorted_actors = sorted(actor_counts.items(), key=lambda x: x[1], reverse=True)
    return [actor for actor, _ in sorted_actors[:limit]]

def create_title_index_mapping(df):
    """
    Create a mapping from movie titles to DataFrame indices
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The movie dataframe
        
    Returns:
    --------
    pandas.Series
        Mapping from title to index
    """
    return pd.Series(df.index, index=df['title']).drop_duplicates()

def get_movie_details(df, title):
    """
    Get detailed information about a movie
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The movie dataframe
    title : str
        Movie title
        
    Returns:
    --------
    dict or None
        Dictionary of movie details or None if not found
    """
    movie_data = df[df['title'] == title]
    if movie_data.empty:
        return None
        
    movie = movie_data.iloc[0]
    return {
        'title': movie.get('title', 'Unknown'),
        'year': movie.get('year', 'N/A'),
        'genres': movie.get('genres_clean', 'N/A'),
        'director': movie.get('director', 'N/A'),
        'cast': movie.get('cast_clean', 'N/A'),
        'rating': movie.get('vote_average', 'N/A'),
        'overview': movie.get('overview', 'No overview available'),
        'homepage': movie.get('homepage', None),
        'tagline': movie.get('tagline', None),
        'popularity': movie.get('popularity', 0)
    }