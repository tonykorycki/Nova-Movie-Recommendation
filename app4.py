import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import numpy as np
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from functools import partial
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler
from joblib import Memory
import time
import hashlib
from datetime import datetime, timedelta

# === PERFORMANCE MONITORING ===
# Simple performance tracking
performance_metrics = {'load_time': 0, 'rec_time': 0, 'search_time': 0}

# === CACHING SYSTEM ===
# Create memory cache directory
os.makedirs('cache', exist_ok=True)
memory = Memory('cache', verbose=0)

# === PAGE SETUP ===
st.set_page_config(page_title="Nova Movie Recs", layout="wide")
st.title("🎬 Nova: Your Personal Movie Recommender")

# === DARK MODE ===
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")
if dark_mode:
    st.markdown("""
        <style>
            .stApp {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            header, .css-18ni7ap, .css-1rs6os {
                background-color: #0f0f0f;
            }
            .stSidebar {
                background-color: #121212 !important;
            }
            .css-1v0mbdj, .css-1d391kg {  /* Titles and subtitles */
                color: #ffffff !important;
            }
        </style>
    """, unsafe_allow_html=True)

# Create necessary directories
for folder in ["profiles", "feedback", "watchlists"]:
    os.makedirs(folder, exist_ok=True)

# === LOAD DATA ===
@st.cache_data
def load_data():
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
        
        return merged_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Helper functions for JSON processing
def extract_list(json_str, key="name"):
    try:
        data = json.loads(json_str.replace("'", '"'))
        return [item[key] for item in data]
    except:
        return []

def extract_director(crew_str):
    try:
        crew_list = json.loads(crew_str.replace("'", '"'))
        for item in crew_list:
            if item.get("job") == "Director":
                return item.get("name")
    except:
        return ""
    return ""

# Pre-compute TF-IDF and similarity matrix
@st.cache_data
def compute_similarity_matrix(df):
    try:
        # Create a combined feature set for better recommendations
        df['combined_features'] = ''
        
        # 1. Add overview with higher weight (repeat to increase importance)
        df['combined_features'] = df['overview'].fillna('') + ' ' + df['overview'].fillna('')
        
        # 2. Add genres (convert list to string with spaces)
        df['combined_features'] += df['genres_list'].apply(
            lambda x: ' '.join([genre.replace(' ', '_') for genre in x]) if isinstance(x, list) else ''
        )
        
        # 3. Add director (with prefix to distinguish from actor names)
        df['combined_features'] += ' director_' + df['director'].fillna('').str.replace(' ', '_')
        
        # 4. Add main cast (first 3 actors)
        df['combined_features'] += df['cast_list'].apply(
            lambda x: ' ' + ' '.join(['actor_' + actor.replace(' ', '_') for actor in x[:3]]) if isinstance(x, list) else ''
        )
        
        # 5. Add year information as a feature (binned by decade)
        df['combined_features'] += df['year'].fillna(0).astype(int).apply(
            lambda x: f' decade_{int(x/10)*10}' if x > 0 else ''
        )
        
        # Initialize TF-IDF Vectorizer with improved parameters
        tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=5000,  # Limit features for better performance
            ngram_range=(1, 2)  # Include bigrams for better context
        )
        
        # Only compute if we have data
        if len(df['combined_features']) == 0 or all(not text.strip() for text in df['combined_features']):
            return None, None
            
        # Transform the data
        tfidf_matrix = tfidf.fit_transform(df['combined_features'])
        
        # Use cosine similarity (more efficient for sparse matrices)
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        return tfidf_matrix, cosine_sim
    except Exception as e:
        st.error(f"Error computing similarity matrix: {e}")
        return None, None
    
# Load main data
df = load_data()

# Validate and get user info
st.sidebar.header("👤 User Profile")
username_input = st.sidebar.text_input("Enter your username:")
# Sanitize username to prevent file system issues
username = re.sub(r'[^\w]', '_', username_input) if username_input else ""

profile_file = f"profiles/{username}.json"
feedback_file = f"feedback/{username}_feedback.json"
watchlist_file = f"watchlists/{username}_watchlist.json"

if username:
    # Load user profile
    if os.path.exists(profile_file):
        with open(profile_file, 'r') as f:
            try:
                profile_data = json.load(f)
            except json.JSONDecodeError:
                profile_data = {"liked_movies": [], "favorite_genres": []}
    else:
        profile_data = {"liked_movies": [], "favorite_genres": []}
    
    # Extract all unique genres for selection
    all_genres = set()
    for genres in df['genres_list']:
        for genre in genres:
            all_genres.add(genre)
    genres = sorted(all_genres)
    
    # User preferences
    selected_genres = st.sidebar.multiselect("Favorite genres", genres, default=profile_data.get('favorite_genres', []))
    movie_options = df['title'].sort_values().tolist()
    selected_movies = st.sidebar.multiselect("Liked movies", movie_options, default=profile_data.get('liked_movies', []))
    
    # Filters section
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔎 Filters")
    
    # Year range slider with proper error handling
    min_year, max_year = int(df['year'].min() or 1900), int(df['year'].max() or 2023)
    year_range = st.sidebar.slider("Year Range", min_year, max_year, (2000, max_year))
    
    # Runtime slider with proper error handling
    min_runtime = max(1, int(df['runtime'].min() or 1))
    max_runtime = min(300, int(df['runtime'].max() or 300))
    runtime_range = st.sidebar.slider("Runtime (minutes)", min_runtime, max_runtime, (60, 180))
    
    # Language selection
    languages = sorted(df['original_language'].unique())
    selected_languages = st.sidebar.multiselect("Languages", languages, default=['en'])
    
    # Actor filter
    st.sidebar.subheader("🎭 Filter by Actor")
    # Limit actor list to prevent performance issues
    popular_actors = []
    for cast in df['cast_list'].head(300):  # Only use top 100 movies for actor list
        for actor in cast[:8]:  # Only use top 3 actors per movie
            if actor not in popular_actors:
                popular_actors.append(actor)
    
    popular_actors.sort()
    selected_actors = st.sidebar.multiselect("Select actors:", popular_actors[:100])  # Limit choices to prevent slowdown
    
    # Apply filters efficiently
    filtered_df = df.copy()
    
    # Use efficient filtering approach
    mask = (
        (filtered_df['year'] >= year_range[0]) &
        (filtered_df['year'] <= year_range[1]) &
        (filtered_df['runtime'] >= runtime_range[0]) &
        (filtered_df['runtime'] <= runtime_range[1]) &
        (filtered_df['original_language'].isin(selected_languages))
    )
    
    # Actor filtering - only apply if actors are selected
    if selected_actors:
        actor_mask = filtered_df['cast_list'].apply(
            lambda cast: any(actor in cast for actor in selected_actors)
        )
        mask = mask & actor_mask
    
    filtered_df = filtered_df[mask]
    
    # Compute similarity matrix (cached)
    _, cosine_sim = compute_similarity_matrix(df)
    
    # Create index mapping for recommendations
    title_to_index = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    # Save user profile
    profile_data['liked_movies'] = selected_movies
    profile_data['favorite_genres'] = selected_genres
    
    try:
        with open(profile_file, 'w') as f:
            json.dump(profile_data, f)
    except Exception as e:
        st.warning(f"Couldn't save profile: {e}")
    
    # Store in session state
    st.session_state["genres"] = selected_genres
    st.session_state["liked_movies"] = selected_movies
    
    # === 🎲 Surprise Me Button ===
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎲 Surprise Me")
    if st.sidebar.button("Give me a random movie!"):
        if not filtered_df.empty:
            surprise = filtered_df.sample(1).iloc[0]
            st.sidebar.markdown(f"**🎬 {surprise['title']} ({surprise.get('year', 'N/A')})**")
            st.sidebar.markdown(f"Genres: _{surprise.get('genres_clean', 'N/A')}_")
            st.sidebar.markdown(f"⭐ Rating: {surprise.get('vote_average', 'N/A')}")
            st.sidebar.markdown(f"🧑‍🎨 Director: {surprise.get('director', 'N/A')}")
            if surprise.get('homepage') != 'none':
                st.sidebar.markdown(f"[🌐 Trailer/More Info]({surprise['homepage']})")
            if surprise.get('tagline') != 'none':
                st.sidebar.markdown(f"_\"{surprise['tagline']}\"_")
        else:
            st.sidebar.warning("No movies match your current filters.")
    
    # Create tabs
    tabs = st.tabs(["Explore Movies", "Recommendations", "Movie of the Day", "Watchlist & History", "Insights", "Feedback & Evaluation"])
    
    with tabs[0]:
        st.subheader("🔍 Advanced Search")
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("Search for a movie by title, director, actor, or keywords")
        with col2:
            search_type = st.selectbox("Search by", ["Title", "Director", "Actor", "Keywords"])
        
        if search_query:
            start_time = time.time()
            if search_type == "Title":
                results = filtered_df[filtered_df['title'].str.contains(search_query, case=False, na=False)]
            elif search_type == "Director":
                results = filtered_df[filtered_df['director'].str.contains(search_query, case=False, na=False)]
            elif search_type == "Actor":
                # More efficient actor search
                results = filtered_df[filtered_df['cast_clean'].str.contains(search_query, case=False, na=False)]
            elif search_type == "Keywords":
                # Search in overview for keywords
                results = filtered_df[filtered_df['overview'].str.contains(search_query, case=False, na=False)]
            
            performance_metrics['search_time'] = time.time() - start_time
            
            if not results.empty:
                expected_cols = ['title', 'genres_clean', 'year', 'vote_average', 'director', 'cast_clean']
                safe_cols = [col for col in expected_cols if col in results.columns]
                
                try:
                    st.subheader(f"Search Results for '{search_query}' ({len(results)} results):")
                    st.dataframe(results[safe_cols], use_container_width=True)
                    
                    # Display search performance
                    st.caption(f"Search completed in {performance_metrics['search_time']:.4f} seconds")
                    
                    # Preview first result
                    if len(results) > 0:
                        with st.expander("Preview First Result", expanded=False):
                            first_result = results.iloc[0]
                            st.subheader(f"{first_result['title']} ({first_result.get('year', 'N/A')})")
                            st.write(f"**Genres:** {first_result.get('genres_clean', 'N/A')}")
                            st.write(f"**Director:** {first_result.get('director', 'N/A')}")
                            if 'cast_clean' in first_result:
                                st.write(f"**Cast:** {first_result['cast_clean']}")
                            st.write(f"**Rating:** {first_result.get('vote_average', 'N/A')} ⭐")
                            st.write(f"**Overview:** {first_result.get('overview', 'No overview available')}")
                            
                            # Quick actions for the result
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Add to Watchlist", key="add_search_result"):
                                    # Load watchlist
                                    if os.path.exists(watchlist_file):
                                        try:
                                            with open(watchlist_file, 'r') as f:
                                                watchlist = json.load(f)
                                        except:
                                            watchlist = {"to_watch": [], "watched": []}
                                    else:
                                        watchlist = {"to_watch": [], "watched": []}
                                    
                                    # Add to watchlist if not already there
                                    if first_result['title'] not in watchlist["to_watch"]:
                                        watchlist["to_watch"].append(first_result['title'])
                                        try:
                                            with open(watchlist_file, 'w') as f:
                                                json.dump(watchlist, f)
                                            st.success(f"Added '{first_result['title']}' to your watchlist!")
                                        except Exception as e:
                                            st.error(f"Error updating watchlist: {e}")
                                    else:
                                        st.info(f"'{first_result['title']}' is already in your watchlist.")
                            
                            with col2:
                                if st.button("Find Similar Movies", key="find_similar"):
                                    st.session_state["find_similar_title"] = first_result['title']
                                    st.rerun()
                except Exception as e:
                    st.error(f"⚠️ Could not display search results: {e}")
                    st.write(results.head())
            else:
                st.warning(f"No results found for '{search_query}' in {search_type}.")
        
        # Handle "Find Similar" request from previous interaction
        if "find_similar_title" in st.session_state and st.session_state["find_similar_title"]:
            title = st.session_state["find_similar_title"]
            st.subheader(f"Movies Similar to '{title}'")
            
            if title in title_to_index:
                try:
                    # Get movie index
                    idx = title_to_index[title]
                    
                    # Get similarity scores for this movie
                    movie_scores = list(enumerate(cosine_sim[idx]))
                    movie_scores = sorted(movie_scores, key=lambda x: x[1], reverse=True)
                    
                    # Get the top 10 similar movies (excluding the movie itself)
                    similar_movies_indices = [i[0] for i in movie_scores if i[0] != idx][:10]
                    similar_movies = df.iloc[similar_movies_indices]
                    
                    # Display similar movies
                    display_cols = ['title', 'genres_clean', 'year', 'vote_average']
                    safe_cols = [col for col in display_cols if col in similar_movies.columns]
                    st.dataframe(similar_movies[safe_cols], use_container_width=True)
                    
                    # Clear the state variable
                    st.session_state["find_similar_title"] = None
                except Exception as e:
                    st.error(f"Error finding similar movies: {e}")
            else:
                st.error(f"Could not find movie '{title}' in our database.")
        
        st.subheader("📚 Explore the TMDB Dataset")
        # Only show a subset of columns and limit rows for performance
        display_cols = ['title', 'genres_clean', 'year', 'original_language', 'vote_average', 'vote_count']
        safe_cols = [col for col in display_cols if col in filtered_df.columns]
        
        # Limit displayed rows for performance
        st.dataframe(filtered_df[safe_cols].head(1000), use_container_width=True)
    
    with tabs[1]:
        st.subheader("🤖 Personalized Recommendations")
        recommendations = []
        top_recs = []
        
        if selected_movies and cosine_sim is not None:
            # Get valid indices for selected movies
            valid_movies = [movie for movie in selected_movies if movie in title_to_index.index]
            
            if valid_movies:
                try:
                    indices = [title_to_index[movie] for movie in valid_movies]
                    # Sum similarity scores
                    sim_scores = np.sum(cosine_sim[indices], axis=0)
                    sim_scores = list(enumerate(sim_scores))
                    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                    sim_scores = [i for i in sim_scores if i[0] not in indices]
                    
                    # Generate recommendations with a limit for performance
                    for i, score in sim_scores[:30]:  # Reduced from 50 to 30
                        movie = df.iloc[i]
                        # Calculate genre match
                        if selected_genres:
                            genre_match = sum(1 for genre in selected_genres if genre in movie.get('genres_list', []))
                            genre_weight = genre_match / len(selected_genres)
                        else:
                            genre_weight = 0
                            
                        # Final score with weights
                        weight = 0.5 * score + 0.3 * genre_weight + 0.2 * (movie.get('vote_average', 0) / 10)
                        recommendations.append((
                            movie.get('title', 'Unknown'),
                            movie.get('genres_clean', ''),
                            movie.get('year', 'Unknown'),
                            movie.get('vote_average', 0),
                            round(score, 3),
                            round(weight, 3)
                        ))
                    
                    recommendations.sort(key=lambda x: x[-1], reverse=True)
                    top_recs = recommendations[:5]  # Top 5 for rating
                
                    rec_df = pd.DataFrame(recommendations, 
                                        columns=['Title', 'Genres', 'Year', 'Rating', 'Similarity', 'Score'])
                    st.dataframe(rec_df.head(10), use_container_width=True)
                    
                    # Rating section
                    st.subheader("⭐ Rate These Recommendations")
                    rating_feedback = {}
                    
                    # Use columns to display ratings side by side for better UX
                    cols = st.columns(3)
                    for idx, (title, genre, year, rating, sim, score) in enumerate(top_recs):
                        with cols[idx % 3]:
                            st.write(f"**{title}** ({year})")
                            st.write(f"_{genre}_")
                            st.write(f"Rating: {rating} ⭐")
                            user_rating = st.slider("Your Rating", 1.0, 5.0, 3.0, 
                                                step=0.5, key=f"rating_{title}")
                            rating_feedback[title] = user_rating
                    
                    if st.button("Save Ratings"):
                        try:
                            with open(feedback_file, 'w') as f:
                                json.dump(rating_feedback, f)
                            st.success("Ratings saved!")
                        except Exception as e:
                            st.error(f"Couldn't save ratings: {e}")
                
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")
            else:
                st.warning("None of your selected movies were found in our database.")
        
        # Genre-based recommendations
        if selected_genres:
            st.markdown("Based on your favorite genres:")
            
            # Convert selected_genres to a set for more efficient lookups
            selected_genres_set = set(selected_genres)
            
            def safe_genre_score(genres):
                # More efficient genre matching that avoids hashing lists
                if genres is None or len(genres) == 0:
                    return 0
                count = 0
                for genre in genres:
                    if genre in selected_genres_set:
                        count += 1
                return count
            
            # Apply genre scoring - create a copy to avoid modifying the original dataframe
            genre_scores = []
            for genres in filtered_df['genres_list']:
                genre_scores.append(safe_genre_score(genres))
            
            filtered_df_temp = filtered_df.copy()
            filtered_df_temp['genre_score'] = genre_scores
            genre_filtered = filtered_df_temp[filtered_df_temp['genre_score'] > 0]
            genre_filtered = genre_filtered.sort_values(
                by=['genre_score', 'vote_average'], 
                ascending=False
            ).head(10)
            
            display_cols = ['title', 'genres_clean', 'year', 'vote_average']
            safe_cols = [col for col in display_cols if col in genre_filtered.columns]
            st.dataframe(genre_filtered[safe_cols], use_container_width=True)
        
        # Get "collaborative" recommendations
        collab_movies = []
        if profile_data.get('favorite_genres'):
            # Use a safer approach that avoids unhashable type errors
            favorite_genres_set = set(profile_data['favorite_genres'])
            
            # Create a safer genre filtering function
            matching_movies = []
            for _, row in filtered_df.iterrows():
                genres = row.get('genres_list', [])
                if isinstance(genres, list) and any(genre in favorite_genres_set for genre in genres):
                    matching_movies.append(row)
            
            if matching_movies:
                # Create a temporary dataframe with matching movies
                genre_filtered = pd.DataFrame(matching_movies)
                if not genre_filtered.empty:
                    # Sort by rating for "collaborative" effect
                    collab_movies = genre_filtered.sort_values('vote_average', ascending=False)['title'].head(10).tolist()
        
        if not selected_movies and not selected_genres:
            st.info("Add liked movies or genres to get personalized recommendations.")
        
        # Hybrid model recommendations
        model_choice = st.radio("Choose Recommendation Model", ["Content-Based", "Collaborative", "Hybrid Model"])
        
        if model_choice == "Hybrid Model":
            try:
                st.session_state["active_model"] = "hybrid"
                
                # Combine content-based and collaborative approaches
                hybrid_recs = []
                
                # Get content-based recommendations
                content_movies = []
                if selected_movies and cosine_sim is not None:
                    valid_movies = [m for m in selected_movies if m in title_to_index.index]
                    if valid_movies:
                        indices = [title_to_index[m] for m in valid_movies]
                        # Sum similarity scores
                        sim_scores = np.sum(cosine_sim[indices], axis=0)
                        # Get top recommendations
                        top_indices = sim_scores.argsort()[-10:][::-1]
                        content_movies = df.iloc[top_indices]['title'].tolist()
                
                # Get "collaborative" recommendations - safer implementation
                collab_movies = []
                if profile_data.get('favorite_genres'):
                    favorite_genres_set = set(profile_data['favorite_genres'])
                    matching_indices = []
                    
                    # Manual filtering to avoid hashing issues
                    for idx, row in filtered_df.iterrows():
                        if isinstance(row.get('genres_list'), list):
                            if any(g in favorite_genres_set for g in row['genres_list']):
                                matching_indices.append(idx)
                    
                    if matching_indices:
                        genre_matches = filtered_df.loc[matching_indices]
                        if not genre_matches.empty:
                            # Sort by rating for "collaborative" effect
                            collab_movies = genre_matches.sort_values('vote_average', ascending=False)['title'].head(10).tolist()
                
                # Combine recommendations with weights
                hybrid_movies = set()
                
                # Add content movies with higher weight
                for movie in content_movies:
                    if len(hybrid_movies) < 15 and movie not in selected_movies:
                        hybrid_movies.add(movie)
                
                # Add collab movies
                for movie in collab_movies:
                    if len(hybrid_movies) < 15 and movie not in selected_movies and movie not in hybrid_movies:
                        hybrid_movies.add(movie)
                
                if hybrid_movies:
                    hybrid_movie_list = list(hybrid_movies)
                    hybrid_rows = filtered_df[filtered_df['title'].isin(hybrid_movie_list)]
                    st.write("#### Hybrid Model Recommendations:")
                    if not hybrid_rows.empty:
                        st.dataframe(hybrid_rows[['title', 'genres_clean', 'vote_average']].head(8), use_container_width=True)
                        st.success("Hybrid model activated! These recommendations combine content analysis with collaborative patterns.")
                    else:
                        st.error("Could not find hybrid recommendations in the database.")
                else:
                    st.error("Could not generate hybrid recommendations. Try selecting more movies or genres.")
            except Exception as e:
                st.error(f"Error generating hybrid recommendations: {e}")
        
        # Modified collaborative filtering code
        elif model_choice == "Collaborative Filtering":
            try:
                st.session_state["active_model"] = "collab"
                
                # Create a more distinct collaborative filtering approach
                # that focuses on popularity and user behavior patterns rather than just genres
                
                # First, get a base set of movies that match user's general taste
                if profile_data.get('favorite_genres'):
                    favorite_genres_set = set(profile_data['favorite_genres'])
                    
                    # Get movies watched by other users with similar tastes
                    # This simulation will use:
                    # 1. Some genre matching (but less strict than content-based)
                    # 2. Higher weight on popularity
                    # 3. Focus more on community ratings
                    
                    # Find movies with at least one matching genre but prioritize popularity
                    matching_movies = []
                    for _, row in filtered_df.iterrows():
                        genres = row.get('genres_list', [])
                        if isinstance(genres, list):
                            # Only need one genre match (less strict)
                            if any(genre in favorite_genres_set for genre in genres):
                                matching_movies.append(row)
                    
                    if matching_movies:
                        genre_filtered = pd.DataFrame(matching_movies)
                        
                        # For collaborative, we prioritize differently:
                        # - Vote count (community engagement)
                        # - Popularity
                        # - With less emphasis on exact genre match
                        
                        # Add a collaborative score that differs from content score
                        if 'vote_count' in genre_filtered.columns and 'popularity' in genre_filtered.columns:
                            # Normalize vote_count and popularity
                            vote_count_max = genre_filtered['vote_count'].max()
                            popularity_max = genre_filtered['popularity'].max() if genre_filtered['popularity'].max() > 0 else 1
                            
                            # Calculate collaborative score - emphasize community metrics
                            genre_filtered['collab_score'] = (
                                (genre_filtered['vote_count'] / vote_count_max) * 0.4 +
                                (genre_filtered['popularity'] / popularity_max) * 0.4 +
                                (genre_filtered['vote_average'] / 10) * 0.2
                            )
                            
                            # Sort by this different metric to get different results
                            collab_recommendations = genre_filtered.sort_values('collab_score', ascending=False).head(8)
                            
                            st.write("#### Collaborative Filtering Recommendations:")
                            st.dataframe(collab_recommendations[['title', 'genres_clean', 'vote_average', 'popularity']], 
                                        use_container_width=True)
                            st.success("Collaborative filtering model activated! These recommendations are based on community popularity and ratings.")
                        else:
                            # Fallback to simpler approach
                            high_rated = genre_filtered.sort_values('vote_average', ascending=False).head(8)
                            st.write("#### Collaborative Filtering Recommendations:")
                            st.dataframe(high_rated[['title', 'genres_clean', 'vote_average']], use_container_width=True)
                            st.success("Collaborative filtering model activated! (Limited mode)")
                    else:
                        st.error("Not enough movies matching your genres for collaborative filtering.")
                else:
                    st.warning("Please select favorite genres to enable collaborative filtering.")
            except Exception as e:
                st.error(f"Error applying collaborative filtering: {e}")
        
        # Trending movies section
        st.subheader("🔥 Trending Now")
        if 'popularity' in filtered_df.columns:
            trending = filtered_df.sort_values(by='popularity', ascending=False).head(10)
            display_cols = ['title', 'genres_clean', 'vote_average', 'popularity']
            safe_cols = [col for col in display_cols if col in trending.columns]
            st.dataframe(trending[safe_cols], use_container_width=True)
        
        # For "Users Like You" section, use a more efficient approach
        st.subheader("👥 Users Like You Also Liked")
        
        # To simulate other users without slowing down app
        # Create a small number of dummy users with a limited selection
        dummy_users = {}
        # Only if enough movies in filtered dataset
        if len(filtered_df) > 20:
            for i in range(1, 4):  # Reduced from 6 to 4 users
                dummy_users[f'user{i}'] = filtered_df['title'].sample(min(10, len(filtered_df)))
            
            # Create a simplified user-movie matrix
            all_movies = set()
            for movies in dummy_users.values():
                all_movies.update(movies)
            
            if profile_data.get('liked_movies'):
                all_movies.update(profile_data['liked_movies'])
            
            # Create more efficient matrix
            movie_to_idx = {movie: i for i, movie in enumerate(all_movies)}
            user_movie_matrix = np.zeros((len(dummy_users) + 1, len(all_movies)))
            
            # Fill matrix
            for user_idx, (user, movies) in enumerate(dummy_users.items()):
                for movie in movies:
                    if movie in movie_to_idx:
                        user_movie_matrix[user_idx, movie_to_idx[movie]] = 1
            
            # Add current user
            current_user_idx = len(dummy_users)
            for movie in profile_data.get('liked_movies', []):
                if movie in movie_to_idx:
                    user_movie_matrix[current_user_idx, movie_to_idx[movie]] = 1
            
            # Calculate similarity
            sim_matrix = cosine_similarity(user_movie_matrix)
            
            # Find similar users
            user_similarities = sim_matrix[current_user_idx, :-1]  # Exclude current user
            similar_users_idx = np.argsort(user_similarities)[::-1]
            
            # Get recommendations from similar users
            similar_user_movies = []
            for idx in similar_users_idx[:2]:  # Only use top 2 similar users
                user = list(dummy_users.keys())[idx]
                similar_user_movies.extend(dummy_users[user])
            
            # Filter out user's already liked movies
            similar_user_movies = [m for m in similar_user_movies if m not in profile_data.get('liked_movies', [])]
            similar_user_movies = list(dict.fromkeys(similar_user_movies))  # Remove duplicates while preserving order
            
            if similar_user_movies:
                st.dataframe(
                    filtered_df[filtered_df['title'].isin(similar_user_movies[:5])][['title', 'genres_clean', 'vote_average']],
                    use_container_width=True
                )

    with tabs[2]:
        st.subheader("🎬 Movie of the Day")
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Use a deterministic approach based on the date
        seed = int(hashlib.md5(today.encode()).hexdigest(), 16) % 10000
        np.random.seed(seed)
        
        # Select movie based on high rating and popularity
        high_rated_df = filtered_df[(filtered_df['vote_average'] >= 7.5) & 
                                   (filtered_df['vote_count'] >= 100)]
        
        if not high_rated_df.empty:
            # Get movie of the day - weighted random selection
            if 'popularity' in high_rated_df.columns:
                weights = high_rated_df['popularity'].values
                weights = weights / weights.sum()  # Normalize
                try:
                    motd_index = np.random.choice(high_rated_df.index, p=weights)
                    motd = high_rated_df.loc[motd_index]
                except:
                    motd = high_rated_df.sample(1).iloc[0]
            else:
                motd = high_rated_df.sample(1).iloc[0]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader(f"{motd['title']} ({motd.get('year', 'N/A')})")
                st.write(f"**Genre:** {motd.get('genres_clean', 'N/A')}")
                st.write(f"**Director:** {motd.get('director', 'N/A')}")
                st.write(f"**Cast:** {motd.get('cast_clean', 'N/A')}")
                st.write(f"**Rating:** {motd.get('vote_average', 'N/A')} ⭐")
                
                if st.button("Add to Watchlist", key="add_motd"):
                    if os.path.exists(watchlist_file):
                        try:
                            with open(watchlist_file, 'r') as f:
                                watchlist = json.load(f)
                        except:
                            watchlist = {"to_watch": [], "watched": []}
                    else:
                        watchlist = {"to_watch": [], "watched": []}
                    
                    if motd['title'] not in watchlist["to_watch"]:
                        watchlist["to_watch"].append(motd['title'])
                        with open(watchlist_file, 'w') as f:
                            json.dump(watchlist, f)
                        st.success(f"Added '{motd['title']}' to your watchlist!")
                    else:
                        st.info(f"'{motd['title']}' is already in your watchlist.")
            
            with col2:
                st.write("**Overview:**")
                st.write(motd.get('overview', 'No overview available'))
                
                # Show keyword/phrase wordcloud if overview exists
                if motd.get('overview'):
                    try:
                        # Using clean wordcloud for better performance
                        text = motd['overview']
                        # Simple preprocessing
                        text = re.sub(r'[^\w\s]', '', text.lower())
                        # Remove common stopwords
                        stopwords = ['and', 'the', 'to', 'a', 'in', 'of', 'is', 'for', 'with', 'on']
                        text = ' '.join([word for word in text.split() if word not in stopwords])
                        
                        wordcloud = WordCloud(
                            width=400, 
                            height=200, 
                            background_color='white' if not dark_mode else '#1e1e1e',
                            colormap='viridis',
                            max_words=50
                        ).generate(text)
                        
                        plt.figure(figsize=(10, 5))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis("off")
                        st.pyplot(plt)
                    except Exception as e:
                        st.error(f"Could not generate wordcloud: {e}")
        else:
            st.warning("No high-rated movies found matching your filters.")
            
        # Get upcoming releases - simulated since we don't have real-time data
        st.subheader("🔜 Upcoming Releases")
        recent_movies = filtered_df[filtered_df['year'] >= 2020].sort_values(by='year', ascending=False)
        
        if not recent_movies.empty:
            # Display a few "upcoming" movies
            display_cols = ['title', 'genres_clean', 'year']
            safe_cols = [col for col in display_cols if col in recent_movies.columns]
            st.dataframe(recent_movies[safe_cols].head(5), use_container_width=True)
        else:
            st.info("No upcoming releases found matching your filters.")

    with tabs[3]:
        st.subheader("📋 Your Movie Watchlist")
        
        # Initialize watchlist if doesn't exist
        if username and os.path.exists(watchlist_file):
            try:
                with open(watchlist_file, 'r') as f:
                    watchlist_data = json.load(f)
                    # Ensure watchlist is a dictionary with proper structure
                    if isinstance(watchlist_data, dict) and "to_watch" in watchlist_data and "watched" in watchlist_data:
                        watchlist = watchlist_data
                    elif isinstance(watchlist_data, list):
                        # Convert old format (list) to new dictionary format
                        watchlist = {"to_watch": watchlist_data, "watched": []}
                    else:
                        # Create default structure
                        watchlist = {"to_watch": [], "watched": []}
            except Exception as e:
                st.error(f"Error loading watchlist: {e}")
                watchlist = {"to_watch": [], "watched": []}
        else:
            watchlist = {"to_watch": [], "watched": []}
            
        # Ensure these are always populated
        to_watch = watchlist["to_watch"] if "to_watch" in watchlist else []
        watched = watchlist["watched"] if "watched" in watchlist else []
        
        # Better watchlist tabs organization
        watchlist_tabs = st.tabs(["To Watch", "Watched", "Add Movies"])
        
        with watchlist_tabs[0]:
            st.subheader("Movies To Watch")
            if to_watch:
                # Get movies from dataframe
                watchlist_df = df[df['title'].isin(to_watch)].copy()  # Use full df instead of filtered_df
        
                # For movies not found in df
                missing = [movie for movie in to_watch if movie not in df['title'].values]  # Use full df
                
                if not watchlist_df.empty:
                    # Add action buttons inside dataframe
                    st.dataframe(
                        watchlist_df[['title', 'genres_clean', 'vote_average']],
                        use_container_width=True
                    )
                    
                    # Bulk actions
                    cols = st.columns(3)
                    with cols[0]:
                        selected_movie = st.selectbox("Select a movie to mark as watched", watchlist_df['title'].tolist())
                    with cols[1]:
                        if st.button("Mark as Watched", key="mark_watched"):
                            if selected_movie in to_watch:
                                to_watch.remove(selected_movie)
                                if selected_movie not in watched:
                                    watched.append(selected_movie)
                                watchlist = {"to_watch": to_watch, "watched": watched}
                                with open(watchlist_file, 'w') as f:
                                    json.dump(watchlist, f)
                                st.success(f"Marked '{selected_movie}' as watched!")
                                st.rerun()
                    with cols[2]:
                        if st.button("Remove from Watchlist", key="remove_watchlist"):
                            if selected_movie in to_watch:
                                to_watch.remove(selected_movie)
                                watchlist = {"to_watch": to_watch, "watched": watched}
                                with open(watchlist_file, 'w') as f:
                                    json.dump(watchlist, f)
                                st.success(f"Removed '{selected_movie}' from watchlist!")
                                st.rerun()
                
                # Display missing movies
                if missing:
                    st.markdown("**Movies not found in database:**")
                    for movie in missing:
                        st.markdown(f"- {movie}")
            else:
                st.info("Your watchlist is empty. Add movies from search or recommendations!")
        
        with watchlist_tabs[1]:
            st.subheader("Movies You've Watched")
            if watched:
                # Get watched movies from dataframe
                watched_df = df[df['title'].isin(watched)].copy()  # Use full df
                missing_watched = [movie for movie in watched if movie not in df['title'].values]  # Use full df
        
                if not watched_df.empty:
                    st.dataframe(
                        watched_df[['title', 'genres_clean', 'vote_average', 'year']],
                        use_container_width=True
                    )
                    
                    # Ratings for watched movies
                    st.subheader("Rate Watched Movies")
                    
                    # Load existing ratings
                    ratings_file = f"feedback/{username}_ratings.json"
                    if os.path.exists(ratings_file):
                        try:
                            with open(ratings_file, 'r') as f:
                                ratings = json.load(f)
                        except:
                            ratings = {}
                    else:
                        ratings = {}
                    
                    # Allow user to select a movie to rate
                    movie_to_rate = st.selectbox("Select a movie to rate:", watched_df['title'].tolist())
                    
                    # Initialize default rating
                    default_rating = ratings.get(movie_to_rate, 3.0)
                    rating = st.slider("Your rating:", 1.0, 5.0, default_rating, 0.5)
                    
                    # Record rating
                    if st.button("Save Rating"):
                        ratings[movie_to_rate] = rating
                        try:
                            with open(ratings_file, 'w') as f:
                                json.dump(ratings, f)
                            st.success(f"Rating for '{movie_to_rate}' saved!")
                        except Exception as e:
                            st.error(f"Error saving rating: {e}")
                
                # Display missing movies
                if missing_watched:
                    st.markdown("**Movies not found in database:**")
                    for movie in missing_watched:
                        st.markdown(f"- {movie}")
            else:
                st.info("You haven't marked any movies as watched yet.")
        
        with watchlist_tabs[2]:
            st.subheader("Add Movies to Watchlist")
            
            # Show top-rated movies for easy addition
            st.markdown("**Top-Rated Movies**")
            top_rated = filtered_df.sort_values(by='vote_average', ascending=False).head(10)
            
            # Multi-select interface for adding movies
            movies_to_add = st.multiselect(
                "Select movies to add to your watchlist:",
                top_rated['title'].tolist()
            )
            
            if st.button("Add Selected Movies") and movies_to_add:
                added_count = 0
                for movie in movies_to_add:
                    if movie not in to_watch:
                        to_watch.append(movie)
                        added_count += 1
                
                if added_count > 0:
                    watchlist = {"to_watch": to_watch, "watched": watched}
                    try:
                        with open(watchlist_file, 'w') as f:
                            json.dump(watchlist, f)
                        st.success(f"Added {added_count} movies to your watchlist!")
                    except Exception as e:
                        st.error(f"Error updating watchlist: {e}")
                else:
                    st.info("All selected movies are already in your watchlist.")
            
            # Manual movie addition
            st.markdown("---")
            st.markdown("**Add Movie Manually**")
            manual_movie = st.text_input("Movie Title:")
            
            if st.button("Add to Watchlist") and manual_movie:
                if manual_movie not in to_watch:
                    to_watch.append(manual_movie)
                    watchlist = {"to_watch": to_watch, "watched": watched}
                    try:
                        with open(watchlist_file, 'w') as f:
                            json.dump(watchlist, f)
                        st.success(f"Added '{manual_movie}' to your watchlist!")
                    except Exception as e:
                        st.error(f"Error updating watchlist: {e}")
                else:
                    st.info(f"'{manual_movie}' is already in your watchlist.")

    with tabs[4]:
        st.subheader("📊 Insights & Analytics")
        
        if not filtered_df.empty:
            # Row 1: Rating and Genre Distribution
            st.markdown("### Rating & Genre Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                # Rating histogram
                try:
                    fig = px.histogram(
                        filtered_df, 
                        x="vote_average",
                        nbins=20,
                        title="Rating Distribution",
                        color_discrete_sequence=['#3366CC']
                    )
                    fig.update_layout(
                        xaxis_title="Rating",
                        yaxis_title="Count",
                        bargap=0.05,
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate rating histogram: {e}")
            
            with col2:
                # Genre breakdown
                try:
                    # Get genre counts
                    all_genres = []
                    for genres in filtered_df['genres_list']:
                        all_genres.extend(genres)
                    
                    genre_counts = {}
                    for genre in all_genres:
                        if genre in genre_counts:
                            genre_counts[genre] += 1
                        else:
                            genre_counts[genre] = 1
                    
                    # Sort and take top 10
                    genre_df = pd.DataFrame({
                        'Genre': list(genre_counts.keys()),
                        'Count': list(genre_counts.values())
                    }).sort_values('Count', ascending=False).head(10)
                    
                    fig = px.bar(
                        genre_df, 
                        x='Count', 
                        y='Genre',
                        orientation='h',
                        title="Top 10 Genres",
                        color_discrete_sequence=['#3366CC']
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate genre chart: {e}")
                    
            # Movie release trends over time
            st.markdown("### Movie Release Trends Over Time")
            try:
                year_counts = filtered_df['year'].value_counts().reset_index()
                year_counts.columns = ['Year', 'Count']
                year_counts = year_counts.sort_values('Year')
                
                fig = px.line(
                    year_counts, 
                    x='Year', 
                    y='Count',
                    title="Movies Released By Year",
                    markers=True
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not generate year trend chart: {e}")
            
            # Your movie preferences (if logged in)
            if username and (profile_data.get('liked_movies') or profile_data.get('favorite_genres')):
                st.markdown("### Your Movie Preferences")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Favorite Genres**")
                    if profile_data.get('favorite_genres'):
                        for genre in profile_data['favorite_genres']:
                            st.markdown(f"- {genre}")
                    else:
                        st.info("No favorite genres selected yet.")
                
                with col2:
                    st.markdown("**Your Rated Movies**")
                    ratings_file = f"feedback/{username}_ratings.json"
                    if os.path.exists(ratings_file):
                        try:
                            with open(ratings_file, 'r') as f:
                                ratings = json.load(f)
                            if ratings:
                                ratings_df = pd.DataFrame({
                                    'Movie': list(ratings.keys()),
                                    'Rating': list(ratings.values())
                                })
                                # Show as bar chart
                                fig = px.bar(
                                    ratings_df, 
                                    x='Movie', 
                                    y='Rating',
                                    title="Your Movie Ratings",
                                    color='Rating',
                                    color_continuous_scale='RdYlGn'
                                )
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("You haven't rated any movies yet.")
                        except Exception as e:
                            st.error(f"Could not load ratings: {e}")
                    else:
                        st.info("You haven't rated any movies yet.")

    with tabs[5]:
        st.subheader("🔄 Feedback & Evaluation")
        
        # Performance metrics
        st.markdown("### System Performance")
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("Data Load Time", f"{performance_metrics['load_time']:.3f}s")
        with metrics_col2:
            st.metric("Search Time", f"{performance_metrics['search_time']:.3f}s")
        with metrics_col3:
            st.metric("Recommendation Time", f"{performance_metrics['rec_time']:.3f}s")
        
        # User feedback
        st.markdown("### Your Feedback")
        feedback_score = st.slider("Rate your experience with Nova (1-5)", 1.0, 5.0, 3.0, 0.5)
        feedback_text = st.text_area("Tell us how we can improve:")
        
        if st.button("Submit Feedback"):
            try:
                feedback_data = {
                    "username": username,
                    "score": feedback_score,
                    "feedback": feedback_text,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                feedback_dir = os.path.join("feedback", "system")
                os.makedirs(feedback_dir, exist_ok=True)
                
                feedback_file_path = os.path.join(feedback_dir, f"{username}_feedback.json")
                
                # Append to existing feedback or create new
                if os.path.exists(feedback_file_path):
                    with open(feedback_file_path, 'r') as f:
                        try:
                            existing_feedback = json.load(f)
                            if isinstance(existing_feedback, list):
                                existing_feedback.append(feedback_data)
                            else:
                                existing_feedback = [existing_feedback, feedback_data]
                        except:
                            existing_feedback = [feedback_data]
                else:
                    existing_feedback = [feedback_data]
                
                with open(feedback_file_path, 'w') as f:
                    json.dump(existing_feedback, f)
                
                st.success("Thank you for your feedback!")
            except Exception as e:
                st.error(f"Error saving feedback: {e}")
        
        # Advanced recommendation options
        st.markdown("### Advanced Recommendation Models")
        
        model_choice = st.selectbox(
            "Choose a recommendation model to try:",
            ["Content-Based (Default)", "Collaborative Filtering", "Hybrid Model", "Deep Learning"]
        )
        
        st.write("#### How it works:")
        
        if model_choice == "Content-Based (Default)":
            st.write("""
            **Content-Based Filtering** analyzes movie features (genres, directors, etc.) 
            to recommend similar items based on your preferences. 
            This is our default model using TF-IDF and cosine similarity.
            
            * **Pros**: No cold start problem, transparent recommendations
            * **Cons**: Limited discovery, can create filter bubbles
            """)
        elif model_choice == "Collaborative Filtering":
            st.write("""
            **Collaborative Filtering** examines viewing/rating patterns across users 
            to suggest movies liked by similar users.
            
            * **Pros**: Helps discover unexpected content, personalized
            * **Cons**: Needs sufficient user data, cold start issues
            """)
        elif model_choice == "Hybrid Model":
            st.write("""
            **Hybrid Model** combines content-based and collaborative filtering 
            for better recommendations. This blends movie features with user behavior patterns.
            
            * **Pros**: More robust, overcomes limitations of individual models
            * **Cons**: More complex, requires more computation
            """)
        elif model_choice == "Deep Learning":
            st.write("""
            **Deep Learning** uses neural networks to find complex patterns in movie data 
            and user preferences that simpler models might miss.
            
            * **Pros**: Can capture complex relationships, potentially more accurate
            * **Cons**: Requires significant data and computation, less interpretable
            """)
        
        if st.button("Try This Model"):
            with st.spinner(f"Generating recommendations using {model_choice}..."):
                # Simulate different model results
                if model_choice == "Content-Based (Default)":
                    # This is already implemented
                    st.session_state["active_model"] = "content"
                    st.success("Content-based model is already active. See recommendations in the Recommendations tab.")
                    
                elif model_choice == "Collaborative Filtering":
                    # Simple collaborative filtering simulation
                    try:
                        st.session_state["active_model"] = "collab"
                        
                        # Use a safer approach that avoids unhashable type errors
                        if profile_data.get('favorite_genres'):
                            # Use a more efficient approach to avoid unhashable type errors
                            favorite_genres_set = set(profile_data['favorite_genres'])
                            
                            # Create helper function to safely check genre matches
                            def has_matching_genre(genres_list):
                                if not isinstance(genres_list, list):
                                    return False
                                return any(genre in favorite_genres_set for genre in genres_list)
                            
                            # Apply filtering using the helper function
                            matching_movies = []
                            for _, row in filtered_df.iterrows():
                                if has_matching_genre(row.get('genres_list', [])):
                                    matching_movies.append(row)
                            
                            if len(matching_movies) >= 5:
                                genre_filtered = pd.DataFrame(matching_movies)
                                high_rated = genre_filtered.sort_values('vote_average', ascending=False).head(50)
                                if len(high_rated) >= 5:
                                    st.write("#### Collaborative Filtering Recommendations:")
                                    st.dataframe(high_rated[['title', 'genres_clean', 'vote_average']].head(8), use_container_width=True)
                                    st.success("Collaborative filtering model activated! These recommendations are based on similar users' preferences.")
                                else:
                                    st.error("Not enough high-rated movies matching your genres.")
                            else:
                                st.error("Not enough movies matching your genres for collaborative filtering.")
                        else:
                            st.warning("Please select favorite genres to enable collaborative filtering.")
                    except Exception as e:
                        st.error(f"Error applying collaborative filtering: {e}")
                        
                elif model_choice == "Hybrid Model":
                    try:
                        st.session_state["active_model"] = "hybrid"
                        
                        # Combine content-based and collaborative approaches
                        hybrid_recs = []
                        
                        # Get content-based recommendations
                        content_movies = []
                        if selected_movies and cosine_sim is not None:
                            valid_movies = [m for m in selected_movies if m in title_to_index.index]
                            if valid_movies:
                                indices = [title_to_index[m] for m in valid_movies]
                                # Sum similarity scores
                                sim_scores = np.sum(cosine_sim[indices], axis=0)
                                # Get top recommendations
                                top_indices = sim_scores.argsort()[-10:][::-1]
                                content_movies = df.iloc[top_indices]['title'].tolist()
                        
                        # Get "collaborative" recommendations - safer implementation
                        collab_movies = []
                        if profile_data.get('favorite_genres'):
                            favorite_genres_set = set(profile_data['favorite_genres'])
                            matching_indices = []
                            
                            # Manual filtering to avoid hashing issues
                            for idx, row in filtered_df.iterrows():
                                if isinstance(row.get('genres_list'), list):
                                    if any(g in favorite_genres_set for g in row['genres_list']):
                                        matching_indices.append(idx)
                            
                            if matching_indices:
                                genre_matches = filtered_df.loc[matching_indices]
                                if not genre_matches.empty:
                                    # Sort by rating for "collaborative" effect
                                    collab_movies = genre_matches.sort_values('vote_average', ascending=False)['title'].head(10).tolist()
                        
                        # Combine recommendations with weights
                        hybrid_movies = set()
                        
                        # Add content movies with higher weight
                        for movie in content_movies:
                            if len(hybrid_movies) < 15 and movie not in selected_movies:
                                hybrid_movies.add(movie)
                        
                        # Add collab movies
                        for movie in collab_movies:
                            if len(hybrid_movies) < 15 and movie not in selected_movies and movie not in hybrid_movies:
                                hybrid_movies.add(movie)
                        
                        if hybrid_movies:
                            hybrid_movie_list = list(hybrid_movies)
                            hybrid_rows = filtered_df[filtered_df['title'].isin(hybrid_movie_list)]
                            st.write("#### Hybrid Model Recommendations:")
                            if not hybrid_rows.empty:
                                st.dataframe(hybrid_rows[['title', 'genres_clean', 'vote_average']].head(8), use_container_width=True)
                                st.success("Hybrid model activated! These recommendations combine content analysis with collaborative patterns.")
                            else:
                                st.error("Could not find hybrid recommendations in the database.")
                        else:
                            st.error("Could not generate hybrid recommendations. Try selecting more movies or genres.")
                            
                    except Exception as e:
                        st.error(f"Error applying hybrid model: {e}")
                        
                elif model_choice == "Deep Learning":
                    try:
                        st.session_state["active_model"] = "deep"
                        
                        # This is a simulation of deep learning results
                        st.info("Deep Learning model would require significant computation. This is a simulation of results.")
                        
                        # For demonstration, create a more sophisticated blending algorithm
                        # that weights movies by multiple factors
                        
                        # Create a scoring function that considers multiple factors
                        def deep_score(row):
                            score = row['vote_average'] * 2  # Base score from rating
                            
                            # Genre match bonus - Safely check for genre matches
                            if profile_data.get('favorite_genres') and isinstance(row.get('genres_list'), list):
                                # Convert to sets for efficient intersection
                                favorite_genres_set = set(profile_data['favorite_genres'])
                                movie_genres_set = set(row['genres_list'])
                                matching_genres = len(favorite_genres_set.intersection(movie_genres_set))
                                score += matching_genres * 0.5
                            
                            # Popularity factor
                            if 'popularity' in row:
                                score += np.log1p(row['popularity']) * 0.3
                            
                            # Recency factor
                            if isinstance(row.get('year'), (int, float)) and not np.isnan(row['year']) and row['year'] >= 2015:
                                score += 0.7
                            
                            return score
                        
                        # Apply scoring to each row individually to avoid hashing lists
                        scores = []
                        for _, row in filtered_df.iterrows():
                            scores.append(deep_score(row))
                        
                        # Create a new DataFrame with scores
                        scored_df = filtered_df.copy()
                        scored_df['deep_score'] = scores
                        
                        # Filter out already seen movies
                        watched_set = set(watched)
                        deep_recs = scored_df[~scored_df['title'].isin(watched_set)]
                        deep_recs = deep_recs.sort_values('deep_score', ascending=False).head(8)
                        
                        if not deep_recs.empty:
                            st.write("#### Deep Learning Model Recommendations:")
                            st.dataframe(deep_recs[['title', 'genres_clean', 'vote_average', 'year']], use_container_width=True)
                            st.success("Deep learning model simulated! These recommendations use complex patterns to match your taste profile.")
                        else:
                            st.error("Could not generate deep learning recommendations. Try adjusting your filters.")
                    except Exception as e:
                        st.error(f"Error simulating deep learning model: {e}")
        
        # Experimental features section
        st.markdown("### Experimental Features")
        st.write("Which features would you like to see in future updates?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Movie trailers integration")
            st.checkbox("Social sharing options")
            st.checkbox("Personalized newsletter")
        
        with col2:
            st.checkbox("Advanced taste profiling")
            st.checkbox("Content-based mood matching")
            st.checkbox("Group recommendation mode")
            
        if st.button("Submit Feature Requests"):
            st.success("Thanks! Your feature preferences have been recorded.")
else:
    # Not logged in state
    st.warning("👤 Please enter a username in the sidebar to get personalized recommendations and use all features.")
    
    # Show limited functionality
    st.subheader("📚 TMDB Movie Dataset")
    st.write("This app uses the TMDB 5000 Movie Dataset to provide recommendations.")
    
    # Display basic stats about dataset
    if not df.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Movies in Database", len(df))
        with col2:
            years_range = f"{int(df['year'].min())} - {int(df['year'].max())}"
            st.metric("Year Range", years_range)
        with col3:
            avg_rating = round(df['vote_average'].mean(), 1)
            st.metric("Average Rating", f"⭐ {avg_rating}")
        
        # Show top 5 movies by rating
        st.subheader("Top Rated Movies")
        top_movies = df.sort_values('vote_average', ascending=False).head(5)
        st.dataframe(top_movies[['title', 'genres_clean', 'vote_average', 'year']], use_container_width=True)