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
        # Only use overview for recommendations
        tfidf = TfidfVectorizer(stop_words='english')
        # Fill empty overviews to prevent errors
        overview_data = df['overview'].fillna('').values
        
        # Only compute if we have data
        if len(overview_data) == 0 or all(not text.strip() for text in overview_data):
            return None, None
            
        tfidf_matrix = tfidf.fit_transform(overview_data)
        # Use sparse matrix to save memory
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
    for cast in df['cast_list'].head(100):  # Only use top 100 movies for actor list
        for actor in cast[:3]:  # Only use top 3 actors per movie
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
        search_query = st.text_input("🔍 Search for a movie by title")
        if search_query:
            results = filtered_df[filtered_df['title'].str.contains(search_query, case=False, na=False)]
            if not results.empty:
                expected_cols = ['title', 'genres_clean', 'year', 'vote_average', 'overview']
                safe_cols = [col for col in expected_cols if col in results.columns]
                
                try:
                    st.subheader(f"Search Results for '{search_query}':")
                    st.dataframe(results[safe_cols], use_container_width=True)
                except Exception as e:
                    st.error(f"⚠️ Could not display search results: {e}")
                    st.write(results.head())
            else:
                st.warning("No results found.")
        
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
            
            def genre_score(movie_genres_list):
                # More efficient genre matching
                return sum(1 for genre in selected_genres if genre in movie_genres_list)
            
            # Apply genre scoring
            filtered_df['genre_score'] = filtered_df['genres_list'].apply(genre_score)
            genre_filtered = filtered_df[filtered_df['genre_score'] > 0]
            genre_filtered = genre_filtered.sort_values(
                by=['genre_score', 'vote_average'], 
                ascending=False
            ).head(10)
            
            display_cols = ['title', 'genres_clean', 'year', 'vote_average']
            safe_cols = [col for col in display_cols if col in genre_filtered.columns]
            st.dataframe(genre_filtered[safe_cols], use_container_width=True)
        
        if not selected_movies and not selected_genres:
            st.info("Add liked movies or genres to get personalized recommendations.")
        
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
                dummy_users[f'user{i}'] = filtered_df['title'].sample(min(5, len(filtered_df))).tolist()
            
            if selected_movies:
                dummy_users[username] = selected_movies
                
                # Create a smaller set of titles for comparison
                # Use only popular movies to reduce matrix size
                popular_titles = filtered_df.sort_values('popularity', ascending=False)['title'].head(200).tolist()
                all_user_titles = set(popular_titles)
                for titles in dummy_users.values():
                    all_user_titles.update([t for t in titles if t in filtered_df['title'].values])
                
                all_titles = sorted(all_user_titles)
                
                # Create user-movie matrix
                user_movie_matrix = pd.DataFrame(0, index=dummy_users.keys(), columns=all_titles)
                for user, titles in dummy_users.items():
                    for title in titles:
                        if title in user_movie_matrix.columns:
                            user_movie_matrix.loc[user, title] = 1
                
                # Calculate similarity
                sim_matrix = cosine_similarity(user_movie_matrix)
                sim_df = pd.DataFrame(sim_matrix, index=user_movie_matrix.index, columns=user_movie_matrix.index)
                
                if username in sim_df.index:
                    similar_users = sim_df[username].drop(username).sort_values(ascending=False)
                    
                    if not similar_users.empty:
                        top_user = similar_users.idxmax()
                        liked_by_top_user = user_movie_matrix.loc[top_user]
                        liked_titles = liked_by_top_user[liked_by_top_user == 1].index.tolist()
                        suggested = [title for title in liked_titles if title not in selected_movies]
                        
                        suggestions_df = filtered_df[filtered_df['title'].isin(suggested)]
                        display_cols = ['title', 'genres_clean', 'vote_average']
                        safe_cols = [col for col in display_cols if col in suggestions_df.columns]
                        
                        if not suggestions_df.empty:
                            st.dataframe(suggestions_df[safe_cols].head(5), use_container_width=True)
                        else:
                            st.info("No additional suggestions found based on similar users.")
                    else:
                        st.info("Not enough data to find similar users.")
                else:
                    st.info("Add more liked movies to see what similar users enjoy.")
            else:
                st.info("Add liked movies to see what similar users enjoy.")
        else:
            st.info("Not enough movies match your filters to find similar users.")
    
    # Feedback tab
    with tabs[5]:
        st.subheader("📈 Evaluation Dashboard")
        if os.path.exists(feedback_file):
            try:
                with open(feedback_file, 'r') as f:
                    user_feedback = json.load(f)
                
                if user_feedback:
                    feedback_summary = pd.Series(user_feedback).value_counts()
                    
                    # Use Plotly for more efficient interactive charts
                    fig = px.bar(
                        x=feedback_summary.index, 
                        y=feedback_summary.values,
                        labels={'x': 'Rating', 'y': 'Count'},
                        title='Your Movie Ratings'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show actual ratings
                    st.write("Your movie ratings:")
                    feedback_df = pd.DataFrame({
                        'Movie': list(user_feedback.keys()),
                        'Your Rating': list(user_feedback.values())
                    }).sort_values('Your Rating', ascending=False)
                    
                    st.dataframe(feedback_df, use_container_width=True)
                else:
                    st.info("You have ratings saved, but they're empty. Try rating some movies!")
            except Exception as e:
                st.error(f"Error loading feedback: {e}")
        else:
            st.info("No ratings submitted yet.")
    
    # Movie of the Day tab
    with tabs[2]:
        st.subheader("🌟 Movie of the Day")
        
        # More efficiently find a good recommendation
        if not filtered_df.empty and 'popularity' in filtered_df.columns and 'vote_average' in filtered_df.columns:
            # Get 90th percentile for popularity
            mx = filtered_df['popularity'].quantile(0.90)
            
            # Find well-rated but not super popular movies
            try:
                lesser_known = filtered_df[(filtered_df['vote_average'] > 7) & (filtered_df['popularity'] < mx)]
                
                if not lesser_known.empty:
                    pick = lesser_known.sample(1).iloc[0]
                    
                    st.markdown(f"### **{pick.get('title', 'Unknown')}** ({pick.get('year', 'N/A')})")
                    st.markdown(f"**Genres:** {pick.get('genres_clean', 'N/A')}")
                    st.markdown(f"**Overview:** {pick.get('overview', 'No overview available')}")
                    st.markdown(f"**Rating:** {pick.get('vote_average', 'N/A')} ⭐")
                    
                    if pick.get('homepage') != 'none' and pick.get('homepage'):
                        st.markdown(f"[**Watch Trailer or More Info**]({pick['homepage']})")
                else:
                    st.info("No hidden gems found with current filters. Try adjusting your filters.")
            except Exception as e:
                st.error(f"Couldn't select movie of the day: {e}")
        else:
            st.info("Not enough movies match your current filters.")
    
    # Insights tab with lazy loading for performance
    with tabs[4]:
        st.subheader("📊 Visual Insights")
        
        # Use tabs within the insights tab to lazy load visualizations
        insight_tabs = st.tabs(["Genres", "Ratings", "Years", "Directors", "Languages", "Runtime"])
        
        with insight_tabs[0]:
            if st.button("Show Genre Distribution"):
                try:
                    # More efficient way to count genres
                    all_genres = []
                    for genres in filtered_df['genres_list'].head(1000):  # Limit to avoid slowdowns
                        all_genres.extend(genres)
                    
                    genre_counts = pd.Series(all_genres).value_counts().head(10)
                    
                    # Use Plotly instead of matplotlib
                    fig = px.bar(
                        x=genre_counts.index, 
                        y=genre_counts.values,
                        labels={'x': 'Genre', 'y': 'Count'},
                        title='Top Genres'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating genre chart: {e}")
        
        with insight_tabs[1]:
            if st.button("Show Rating vs Popularity"):
                try:
                    # Use a sample of the data for performance
                    if 'vote_average' in filtered_df.columns and 'popularity' in filtered_df.columns:
                        sample_df = filtered_df.sample(min(1000, len(filtered_df)))
                        
                        fig = px.scatter(
                            sample_df, 
                            x='vote_average', 
                            y='popularity',
                            hover_name='title',
                            opacity=0.5,
                            title='Rating vs Popularity'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Missing columns needed for this visualization.")
                except Exception as e:
                    st.error(f"Error generating ratings chart: {e}")
        
        with insight_tabs[2]:
            if st.button("Show Movies Per Year"):
                try:
                    if 'year' in filtered_df.columns:
                        year_counts = filtered_df['year'].value_counts().sort_index()
                        
                        fig = px.line(
                            x=year_counts.index,
                            y=year_counts.values,
                            labels={'x': 'Year', 'y': 'Number of Movies'},
                            title='Movies Released Per Year'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Year column not available.")
                except Exception as e:
                    st.error(f"Error generating year chart: {e}")
        
        with insight_tabs[3]:
            if st.button("Show Top Directors"):
                try:
                    if 'director' in filtered_df.columns and 'vote_average' in filtered_df.columns:
                        # Filter out unknown directors and those with few movies
                        director_counts = filtered_df['director'].value_counts()
                        valid_directors = director_counts[director_counts >= 3].index
                        
                        directors_df = filtered_df[filtered_df['director'].isin(valid_directors)]
                        top_directors = directors_df.groupby('director')['vote_average'].mean().sort_values(ascending=False).head(10)
                        
                        fig = px.bar(
                            x=top_directors.index,
                            y=top_directors.values,
                            labels={'x': 'Director', 'y': 'Average Rating'},
                            title='Top Directors by Average Rating'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Director or rating columns not available.")
                except Exception as e:
                    st.error(f"Error generating director chart: {e}")
        
        with insight_tabs[4]:
            if st.button("Show Language Distribution"):
                try:
                    if 'original_language' in filtered_df.columns:
                        lang_counts = filtered_df['original_language'].value_counts().head(10)
                        
                        fig = px.pie(
                            values=lang_counts.values,
                            names=lang_counts.index,
                            title='Most Common Languages'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Language column not available.")
                except Exception as e:
                    st.error(f"Error generating language chart: {e}")
        
        with insight_tabs[5]:
            if st.button("Show Runtime Distribution"):
                try:
                    if 'runtime' in filtered_df.columns:
                        fig = px.histogram(
                            filtered_df,
                            x='runtime',
                            nbins=30,
                            labels={'x': 'Runtime (minutes)', 'y': 'Count'},
                            title='Movie Runtime Distribution'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Runtime column not available.")
                except Exception as e:
                    st.error(f"Error generating runtime chart: {e}")
        
        # Word cloud section
        st.subheader("🔤 Word Cloud of Liked Movies")
        if st.button("Generate Word Cloud") and selected_movies:
            try:
                # Get overviews of liked movies
                liked_df = df[df['title'].isin(selected_movies)]
                if not liked_df.empty and 'overview' in liked_df.columns:
                    text = " ".join(liked_df['overview'].tolist())
                if text:
                    # Generate word cloud
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='white' if not dark_mode else '#121212',
                        max_words=100,
                        contour_width=3
                    ).generate(text)
                    
                    # Display word cloud
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.info("No overview text available for the selected movies.")
            except Exception as e:
                st.error(f"Error generating word cloud: {e}")
        else:
            st.info("Add some liked movies to generate a word cloud.")

    # Watchlist & History tab
    with tabs[3]:
        st.subheader("📋 Your Watchlist")
        
        # Load existing watchlist if it exists
        if os.path.exists(watchlist_file):
            try:
                with open(watchlist_file, 'r') as f:
                    watchlist = json.load(f)
            except json.JSONDecodeError:
                watchlist = {"to_watch": [], "watched": []}
        else:
            watchlist = {"to_watch": [], "watched": []}
        
        # Add to watchlist section
        col1, col2 = st.columns([3, 1])
        with col1:
            movie_to_add = st.selectbox("Select a movie to add", movie_options)
        with col2:
            if st.button("Add to Watchlist") and movie_to_add:
                if movie_to_add not in watchlist["to_watch"]:
                    watchlist["to_watch"].append(movie_to_add)
                    try:
                        with open(watchlist_file, 'w') as f:
                            json.dump(watchlist, f)
                        st.success(f"Added '{movie_to_add}' to your watchlist!")
                    except Exception as e:
                        st.error(f"Error updating watchlist: {e}")
                else:
                    st.info(f"'{movie_to_add}' is already in your watchlist.")
        
        # Display current watchlist
        if watchlist["to_watch"]:
            st.subheader("Movies to Watch")
            for i, movie in enumerate(watchlist["to_watch"]):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.write(f"{i+1}. {movie}")
                with col2:
                    if st.button("Watched", key=f"watched_{i}"):
                        watchlist["to_watch"].remove(movie)
                        if movie not in watchlist["watched"]:
                            watchlist["watched"].append(movie)
                        try:
                            with open(watchlist_file, 'w') as f:
                                json.dump(watchlist, f)
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error updating watchlist: {e}")
        else:
            st.info("Your watchlist is empty. Add some movies to get started!")
        
        # Watch history section
        st.subheader("📚 Watch History")
        if watchlist["watched"]:
            watched_df = pd.DataFrame({"Movies Watched": watchlist["watched"]})
            st.dataframe(watched_df, use_container_width=True)
            
            if st.button("Clear History"):
                watchlist["watched"] = []
                try:
                    with open(watchlist_file, 'w') as f:
                        json.dump(watchlist, f)
                    st.success("Watch history cleared!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error clearing history: {e}")
        else:
            st.info("You haven't marked any movies as watched yet.")
else:
    st.warning("👆 Please enter a username to continue.")
    st.info("This app uses your username to save your preferences and recommendations.")