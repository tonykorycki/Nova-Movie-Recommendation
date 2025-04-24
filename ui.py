import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud

def setup_page():
    """Configure page settings and title"""
    st.set_page_config(page_title="Nova Movie Recs", layout="wide")
    st.title("🎬 Nova: Your Personal Movie Recommender")

def apply_dark_mode():
    """Apply dark mode styling if enabled"""
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

def create_sidebar(df):
    """Create and render the sidebar with user inputs
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The movie dataframe
        
    Returns:
    --------
    dict
        Dictionary containing all sidebar selections
    """
    st.sidebar.header("👤 User Profile")
    username_input = st.sidebar.text_input("Enter your username:")
    
    # Dark mode toggle
    dark_mode = st.sidebar.checkbox("🌙 Dark Mode")
    if dark_mode:
        apply_dark_mode()
    
    # Extract genre list for selection
    all_genres = set()
    for genres in df['genres_list']:
        for genre in genres:
            all_genres.add(genre)
    genres = sorted(all_genres)
    
    # Movie options for selection
    movie_options = df['title'].sort_values().tolist()
    
    # User preferences (these will be populated by profile data later)
    selected_genres = st.sidebar.multiselect("Favorite genres", genres)
    selected_movies = st.sidebar.multiselect("Liked movies", movie_options)
    
    # Filters section
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔎 Filters")
    
    # Year range slider
    min_year, max_year = int(df['year'].min() or 1900), int(df['year'].max() or 2023)
    year_range = st.sidebar.slider("Year Range", min_year, max_year, (2000, max_year))
    
    # Runtime slider
    min_runtime = max(1, int(df['runtime'].min() or 1))
    max_runtime = min(300, int(df['runtime'].max() or 300))
    runtime_range = st.sidebar.slider("Runtime (minutes)", min_runtime, max_runtime, (60, 180))
    
    # Language selection
    languages = sorted(df['original_language'].unique())
    selected_languages = st.sidebar.multiselect("Languages", languages, default=['en'])
    
    # Create the surprise me button
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎲 Surprise Me")
    surprise_me = st.sidebar.button("Give me a random movie!")
    
    # Return all selections as a dictionary
    return {
        'username_input': username_input,
        'dark_mode': dark_mode,
        'selected_genres': selected_genres,
        'selected_movies': selected_movies,
        'year_range': year_range,
        'runtime_range': runtime_range,
        'selected_languages': selected_languages,
        'surprise_me': surprise_me
    }

def display_movie_details(movie, add_to_watchlist=None):
    """Display detailed information about a movie
    
    Parameters:
    -----------
    movie : dict
        Movie details dictionary
    add_to_watchlist : function, optional
        Callback function when "Add to Watchlist" is clicked
    """
    st.subheader(f"{movie['title']} ({movie['year']})")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write(f"**Genre:** {movie['genres']}")
        st.write(f"**Director:** {movie['director']}")
        st.write(f"**Cast:** {movie['cast']}")
        st.write(f"**Rating:** {movie['rating']} ⭐")
        
        if add_to_watchlist:
            if st.button("Add to Watchlist", key=f"add_{movie['title']}"):
                add_to_watchlist(movie['title'])
    
    with col2:
        st.write("**Overview:**")
        st.write(movie['overview'])
        
        if movie['homepage'] and movie['homepage'] != 'none':
            st.markdown(f"[**Trailer/More Info**]({movie['homepage']})")

def display_recommendations(recommendations, df, rating_callback=None):
    """Display movie recommendations with optional rating
    
    Parameters:
    -----------
    recommendations : list
        List of recommended movie details
    df : pandas.DataFrame
        The movie dataframe
    rating_callback : function, optional
        Callback function when rating is submitted
    """
    if not recommendations:
        st.info("No recommendations available. Try adding more movies or adjusting filters.")
        return
    
    # Display recommendation table
    rec_df = pd.DataFrame(recommendations)
    st.dataframe(rec_df, use_container_width=True)
    
    # Show rating interface if callback provided
    if rating_callback and len(recommendations) > 0:
        st.subheader("⭐ Rate These Recommendations")
        
        # Use columns to display ratings side by side for better UX
        top_recs = recommendations[:5]  # Only show rating UI for top 5
        cols = st.columns(3)
        rating_feedback = {}
        
        for idx, rec in enumerate(top_recs):
            with cols[idx % 3]:
                st.write(f"**{rec['Title']}** ({rec.get('Year', 'N/A')})")
                st.write(f"_{rec.get('Genres', 'N/A')}_")
                st.write(f"Rating: {rec.get('Rating', 'N/A')} ⭐")
                user_rating = st.slider("Your Rating", 1.0, 5.0, 3.0, 
                                      step=0.5, key=f"rating_{rec['Title']}")
                rating_feedback[rec['Title']] = user_rating
        
        if st.button("Save Ratings"):
            rating_callback(rating_feedback)

def display_watchlist(watchlist, mark_watched_callback=None, remove_callback=None):
    """Display user's watchlist
    
    Parameters:
    -----------
    watchlist : dict
        Dictionary containing 'to_watch' and 'watched' lists
    mark_watched_callback : function, optional
        Callback when "Mark as Watched" is clicked
    remove_callback : function, optional
        Callback when "Remove" is clicked
    """
    st.subheader("📋 Your Watchlist")
    
    if not watchlist.get("to_watch"):
        st.info("Your watchlist is empty. Add some movies to get started!")
        return
    
    # Display movies to watch
    for i, movie in enumerate(watchlist["to_watch"]):
        col1, col2, col3 = st.columns([5, 1, 1])
        with col1:
            st.write(f"{i+1}. {movie}")
        
        if mark_watched_callback:
            with col2:
                if st.button("✓ Watched", key=f"watched_{i}"):
                    mark_watched_callback(movie)
        
        if remove_callback:
            with col3:
                if st.button("🗑️ Remove", key=f"remove_{i}"):
                    remove_callback(movie)

def display_watch_history(watched_list, clear_history_callback=None):
    """Display user's watch history
    
    Parameters:
    -----------
    watched_list : list
        List of watched movies
    clear_history_callback : function, optional
        Callback when "Clear History" is clicked
    """
    st.subheader("📚 Watch History")
    
    if not watched_list:
        st.info("You haven't marked any movies as watched yet.")
        return
        
    watched_df = pd.DataFrame({"Movies Watched": watched_list})
    st.dataframe(watched_df, use_container_width=True)
    
    if clear_history_callback and st.button("Clear History"):
        clear_history_callback()

def create_wordcloud(text, dark_mode=False):
    """Generate and display a word cloud from text
    
    Parameters:
    -----------
    text : str
        Text to generate word cloud from
    dark_mode : bool
        Whether dark mode is enabled
    """
    if not text:
        st.info("Not enough text to generate a word cloud.")
        return
        
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

def display_metrics(metrics):
    """Display performance metrics
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of performance metrics
    """
    st.subheader("⏱️ Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Data Load Time", f"{metrics.get('load_time', 0):.3f}s")
    with col2:
        st.metric("Search Time", f"{metrics.get('search_time', 0):.3f}s")
    with col3:
        st.metric("Recommendation Time", f"{metrics.get('rec_time', 0):.3f}s")

def show_genre_distribution(df):
    """Generate and display genre distribution chart
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The movie dataframe
    """
    try:
        # More efficient way to count genres
        all_genres = []
        for genres in df['genres_list'].head(1000):  # Limit to avoid slowdowns
            all_genres.extend(genres)
        
        genre_counts = pd.Series(all_genres).value_counts().head(10)
        
        # Use Plotly for interactive charts
        fig = px.bar(
            x=genre_counts.index, 
            y=genre_counts.values,
            labels={'x': 'Genre', 'y': 'Count'},
            title='Top Genres'
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating genre chart: {e}")