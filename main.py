import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
import hashlib
from datetime import datetime
import re
import plotly.express as px

# Import modules from our modular structure
from data_loader import load_data, filter_dataframe, get_movie_details, create_title_index_mapping
from utils import create_directories, sanitize_username, load_user_data, save_user_data, PerformanceTracker, get_date_based_seed, collect_user_ratings, get_user_genre_preferences, ensure_recommendation_fields
import ui

# Import recommenders
from recommenders.content_based import ContentBasedRecommender, EnhancedContentRecommender, get_recommender as get_content_recommender
from recommenders.collaborative import CollaborativeFilteringRecommender, TrueCollaborativeRecommender, get_recommender as get_collab_recommender
from recommenders.knn import KNNRecommender, get_recommender as get_knn_recommender
from recommenders.matrix_factorization import MatrixFactorizationRecommender, get_recommender as get_mf_recommender
from recommenders.neural import NeuralRecommender, get_recommender as get_neural_recommender
from recommenders.association_rules import AssociationRulesRecommender, get_recommender as get_assoc_recommender
from recommenders.base import combine_recommendations
from recommenders.ml_recommender import MLRecommender, get_recommender as get_ml_recommender
from recommenders.popular import PopularMoviesRecommender, get_recommender as get_popular_recommender


def main():
    """Main function for the Nova movie recommendation system"""
    
    # Initialize performance tracker
    performance_tracker = PerformanceTracker()
    
    # Setup page configuration
    ui.setup_page()
    
    # Create necessary directories
    create_directories()
    
    # Load data with performance tracking
    performance_tracker.start('load_time')
    df = load_data()
    performance_tracker.stop('load_time')
    
    # Get user information from sidebar
    sidebar_selections = ui.create_sidebar(df)
    username_input = sidebar_selections['username_input']
    dark_mode = sidebar_selections['dark_mode']
    selected_genres = sidebar_selections['selected_genres']
    selected_movies = sidebar_selections['selected_movies']
    year_range = sidebar_selections['year_range']
    runtime_range = sidebar_selections['runtime_range']
    selected_languages = sidebar_selections['selected_languages']
    surprise_me = sidebar_selections['surprise_me']
    
    # Apply dark mode if selected
    if dark_mode:
        ui.apply_dark_mode()
    
    # Process user data if username is provided
    username = sanitize_username(username_input)
    if username:
        # Set file paths
        profile_file = f"profiles/{username}.json"
        feedback_file = f"feedback/{username}_movie_feedback.json"
        watchlist_file = f"watchlists/{username}_watchlist.json"
        
        # Check if this is a new user
        profile_data = load_user_data(profile_file, {"liked_movies": [], "favorite_genres": []})
        
        # If this is a new user or onboarding is not complete, run onboarding
        if not profile_data.get("onboarding_complete", False):
            onboarding_complete = onboard_new_user(username, df)
            if not onboarding_complete:
                return  # Stop here until onboarding is complete
        
        # Continue with the normal app flow once onboarding is complete...
        # Load or create user profile
        profile_data = load_user_data(
            profile_file, 
            {"liked_movies": [], "favorite_genres": []}
        )
        
        # Update profile with current selections
        profile_data['liked_movies'] = selected_movies
        profile_data['favorite_genres'] = selected_genres
        save_user_data(profile_file, profile_data)
        
        # Store in session state for persistence
        st.session_state["genres"] = selected_genres
        st.session_state["liked_movies"] = selected_movies
        
        # Filter dataframe based on user selections
        filtered_df = filter_dataframe(
            df, 
            year_range, 
            runtime_range, 
            selected_languages
        )
        
        # Create index mapping for recommendations
        title_to_index = create_title_index_mapping(df)
        
        # Handle "Surprise Me" button
        if surprise_me and not filtered_df.empty:
            surprise = filtered_df.sample(1).iloc[0]
            st.sidebar.markdown(f"**🎬 {surprise['title']} ({surprise.get('year', 'N/A')})**")
            st.sidebar.markdown(f"Genres: _{surprise.get('genres_clean', 'N/A')}_")
            st.sidebar.markdown(f"⭐ Rating: {surprise.get('vote_average', 'N/A')}")
            st.sidebar.markdown(f"🧑‍🎨 Director: {surprise.get('director', 'N/A')}")
            if surprise.get('homepage') != 'none' and surprise.get('homepage'):
                st.sidebar.markdown(f"[🌐 Trailer/More Info]({surprise['homepage']})")
            if surprise.get('tagline') != 'none' and surprise.get('tagline'):
                st.sidebar.markdown(f"_\"{surprise['tagline']}\"_")
        
        # Create main tabs
        tabs = st.tabs([
            "Explore Movies", 
            "Recommendations", 
            "Movie of the Day", 
            "Watchlist & History", 
            "Insights", 
            "Feedback & Evaluation"
        ])
        
        # Explore Movies tab
        with tabs[0]:
            # Create search mode options
            search_mode = st.radio("Search mode:", ["Direct Search", "Find Similar Movies"])
            
            if search_mode == "Direct Search":
                # Search functionality
                st.subheader("🔍 Advanced Search")
                col1, col2 = st.columns([3, 1])
                with col1:
                    search_query = st.text_input("Search for a movie by title, director, actor, or keywords")
                with col2:
                    search_type = st.selectbox("Search by", ["Title", "Director", "Actor", "Keywords"])
                
                # Fix for the search results when searching by Actor around line 115-140

                if search_query:
                    performance_tracker.start('search_time')
                    try:
                        if search_type == "Title":
                            results = filtered_df[filtered_df['title'].str.contains(search_query, case=False, na=False)]
                        elif search_type == "Director":
                            results = filtered_df[filtered_df['director'].str.contains(search_query, case=False, na=False)]
                        elif search_type == "Actor":
                            # Some entries might have NaN in cast_clean
                            cast_mask = filtered_df['cast_clean'].notna() & filtered_df['cast_clean'].str.contains(search_query, case=False, na=False)
                            results = filtered_df[cast_mask]
                        elif search_type == "Keywords":
                            results = filtered_df[filtered_df['overview'].str.contains(search_query, case=False, na=False)]
                        
                        search_time = performance_tracker.stop('search_time')
                        
                        if not results.empty:
                            # Define columns to display and ensure they exist
                            display_cols = ['title', 'genres_clean', 'year', 'vote_average', 'director', 'cast_clean']
                            safe_cols = [col for col in display_cols if col in results.columns]
                            
                            # Create a safe dataframe with all required columns for display
                            display_df = results[safe_cols].copy()
                            
                            # Ensure all columns have valid values to prevent JavaScript errors
                            for col in display_df.columns:
                                display_df[col] = display_df[col].fillna("N/A")
                            
                            st.subheader(f"Search Results for '{search_query}' ({len(results)} results):")
                            st.dataframe(display_df, use_container_width=True)
                            st.caption(f"Search completed in {search_time:.4f} seconds")
                            
                            # Preview first result
                            if len(results) > 0:
                                with st.expander("Preview First Result", expanded=False):
                                    first_result = results.iloc[0]
                                    try:
                                        movie_details = get_movie_details(df, first_result['title'])
                                        
                                        # Make sure movie_details has all required fields
                                        if movie_details is None:
                                            movie_details = {}
                                        
                                        # Ensure all keys exist with defaults
                                        required_keys = ['title', 'overview', 'vote_average', 'release_date', 
                                                        'runtime', 'genres_clean', 'director', 'cast_clean']
                                        for key in required_keys:
                                            if key not in movie_details:
                                                movie_details[key] = "N/A"
                                        
                                        # Display movie details
                                        ui.display_movie_details(
                                            movie_details,
                                            lambda title: add_to_watchlist(title, watchlist_file)
                                        )
                                    except Exception as e:
                                        st.error(f"Error displaying movie details: {str(e)}")
                        else:
                            st.warning(f"No results found for '{search_query}' in {search_type}.")
                    except Exception as e:
                        search_time = performance_tracker.stop('search_time')
                        st.error(f"Error during search: {str(e)}")
            else:  # Find Similar Movies mode
                st.subheader("🔍 Find Movies Similar To...")
                
                # Let user select a reference movie
                reference_movie = st.selectbox(
                    "Select a movie:",
                    df['title'].sort_values().tolist()
                )
                
                if reference_movie:
                    if st.button("Find Similar Movies"):
                        try:
                            # Create enhanced content recommender
                            recommender = get_content_recommender(df, "enhanced", title_to_index)
                            # Get recommendations based on this single movie
                            similar_movies = recommender.recommend([reference_movie], n=10)

                            similar_movies = ensure_recommendation_fields(similar_movies)                            
                            
                            if similar_movies:
                                st.subheader(f"Movies Similar to '{reference_movie}'")
                                
                                # Create a safer DataFrame with explicit string conversion of all values
                                display_data = []
                                for rec in similar_movies:
                                    display_data.append({
                                        "Title": str(rec.get("Title", "Unknown")),
                                        "Year": str(rec.get("Year", "N/A")),
                                        "Rating": float(rec.get("Rating", 0)),
                                        "Genres": str(rec.get("Genres", "")),
                                        "Similarity": float(rec.get("Similarity", 0))
                                    })
                                
                                # Create the DataFrame only if we have data
                                if display_data:
                                    display_df = pd.DataFrame(display_data)
                                    
                                    # Convert numeric columns to proper types
                                    if "Rating" in display_df.columns:
                                        display_df["Rating"] = pd.to_numeric(display_df["Rating"], errors='coerce').fillna(0)
                                    if "Similarity" in display_df.columns:
                                        display_df["Similarity"] = pd.to_numeric(display_df["Similarity"], errors='coerce').fillna(0)
                                        display_df["Similarity"] = display_df["Similarity"].map(lambda x: round(x, 2))
                                    
                                    st.dataframe(display_df, use_container_width=True)
                                else:
                                    st.warning("Unable to format recommendation data for display")
                        except Exception as e:
                            st.error(f"Error finding similar movies: {str(e)}")
                            st.write(f"Debug info: {type(e).__name__}")
                        
            
            # Dataset exploration
            st.subheader("📚 Explore the TMDB Dataset")
            display_cols = ['title', 'genres_clean', 'year', 'original_language', 'vote_average', 'vote_count']
            safe_cols = [col for col in display_cols if col in filtered_df.columns]
            st.dataframe(filtered_df[safe_cols].head(1000), use_container_width=True)
        
        # Recommendations tab
        with tabs[1]:
            st.subheader("🤖 Personalized Recommendations")
            
            # Create two types of recommendation approaches
            rec_approach = st.radio(
                "Recommendation approach:",
                ["Personalized For You", "Find Similar Movies"]
            )
            
            if rec_approach == "Personalized For You":
                # Load user data for recommendations
                watchlist = load_user_data(watchlist_file, {"to_watch": [], "watched": []})
                watched_movies = watchlist.get("watched", [])
                ratings_file = f"feedback/{username}_ratings.json"
                ratings = load_user_data(ratings_file, {})
                
                # Let user select recommendation algorithm
                algo_options = [
                    "ML Recommender",
                    "Content-Based", 
                    "Collaborative Filtering",
                    "Popular Movies",  # Changed from "Demographic"
                    "Hybrid Recommendations"
                ]
                #"Matrix Factorization",
                
                recommender_type = st.selectbox(
                    "Choose a recommendation algorithm:",
                    algo_options
                )
                
                # Check if we have enough data for recommendations
                if watched_movies and ratings:
                    performance_tracker.start('rec_time')
                    
                    try:
                        # ML Recommender is now the primary recommendation engine
                        if recommender_type == "ML Recommender (Best)":
                            # Collect all user ratings
                            user_ratings = collect_user_ratings()
                            
                            # Collect feedback data (liked/disliked)
                            user_feedback = {}
                            for filename in os.listdir("feedback"):
                                if filename.endswith("_movie_feedback.json"):
                                    username = filename.replace("_movie_feedback.json", "")
                                    feedback_file = os.path.join("feedback", filename)
                                    feedback_data = load_user_data(feedback_file, {"liked": [], "disliked": []})
                                    if feedback_data.get("liked") or feedback_data.get("disliked"):
                                        user_feedback[username] = feedback_data
                            
                            # Get recommender and train it
                            recommender = get_ml_recommender(df, title_to_index)
                            
                            # Train with both ratings and feedback
                            if len(user_ratings) >= 5 or not os.path.exists("models/ml_recommender_model.pkl"):
                                with st.spinner("Training ML model..."):
                                    recommender.train(user_ratings, user_feedback)
                            
                            # Generate recommendations
                            recommendations = recommender.recommend(watched_movies, n=10, user_id=username, favorite_genres=selected_genres)
                        
                        elif recommender_type == "Content-Based":
                            recommender = get_content_recommender(df, "enhanced", title_to_index)
                            recommendations = recommender.recommend(watched_movies, n=10, favorite_genres=selected_genres)
                        
                        elif recommender_type == "Collaborative Filtering":
                            # Get true collaborative filtering recommender
                            recommender = get_collab_recommender(df, "true", title_to_index)
                            # Convert ratings dict to format expected by recommender
                            user_ratings = load_user_data(ratings_file, {})
                            
                            # For TrueCollaborativeRecommender, the user_id is handled internally
                            # during the fit process rather than in the recommend method
                            if isinstance(recommender, TrueCollaborativeRecommender):
                                # First fit with user ratings to incorporate the current user
                                recommender.fit(user_ratings)
                                # Then generate recommendations without the user_id parameter
                                recommendations = recommender.recommend(watched_movies, n=10)
                            else:
                                # Standard collaborative filtering
                                recommendations = recommender.recommend(watched_movies, n=10)
                        
                        elif recommender_type == "Matrix Factorization":
                            # Get matrix factorization recommender
                            recommender = get_mf_recommender(df, title_to_index)
                            # Use ratings for this user
                            user_ratings = load_user_data(ratings_file, {})
                            recommendations = recommender.recommend(watched_movies, n=10)
                        
                        elif recommender_type == "Popular Movies":  # Changed from "Demographic"
                            # Get popular movies recommender
                            recommender = get_popular_recommender(df, title_to_index=title_to_index)
                            
                            # Option to prioritize recent movies
                            recency_preference = st.slider(
                                "Preference for recent movies:", 
                                0.0, 1.0, 0.3, 
                                help="Higher values will favor more recent releases"
                            )
                            
                            # Generate recommendations
                            recommendations = recommender.recommend(
                                liked_movies=watched_movies,
                                n=10,
                                recency_weight=recency_preference
                            )
                        
                        elif recommender_type == "Hybrid Recommendations":
                            # Combine multiple recommendation approaches
                            content_recommender = get_content_recommender(df, "enhanced", title_to_index)
                            collab_recommender = get_collab_recommender(df, "true", title_to_index)
                            
                            # Generate recommendations from both engines
                            content_recs = content_recommender.recommend(watched_movies, n=10, favorite_genres=selected_genres)
                            collab_recs = collab_recommender.recommend(watched_movies, n=10)
                            
                            # Combine with weighting (60% content, 40% collaborative)
                            recommendations = combine_recommendations([content_recs, collab_recs], weights=[0.6, 0.4], n=10)
                        
                        else:
                            # Fallback to content-based in case of unknown option
                            recommender = get_content_recommender(df, "enhanced", title_to_index)
                            recommendations = recommender.recommend(watched_movies, n=10)
                        
                        rec_time = performance_tracker.stop('rec_time')

                        # Display recommendations
                        if recommendations:
                            # Show the table view first for overview
                            st.dataframe(pd.DataFrame(recommendations), use_container_width=True)
                            st.caption(f"Recommendations generated in {rec_time:.4f} seconds")
                            
                            # Store user feedback
                            feedback_file = f"feedback/{username}_movie_feedback.json"
                            movie_feedback = load_user_data(feedback_file, {"liked": [], "disliked": []})
                            
                            # Then show detailed movie cards with interactive elements
                            st.markdown("### Explore These Recommendations")
                            
                            # Create expandable sections for each movie
                            for i, rec in enumerate(recommendations[:8]):  # Limit to top 8 recommendations
                                movie_title = rec["Title"]
                                
                                # Get full movie details
                                movie_data = df[df['title'] == movie_title]
                                
                                if not movie_data.empty:
                                    movie = movie_data.iloc[0]
                                    
                                    # Create an expander for each movie
                                    with st.expander(f"{i+1}. {movie_title} ({rec.get('Year', 'N/A')}) - {rec.get('Rating', 'N/A')}⭐"):
                                        # Two columns layout
                                        col1, col2 = st.columns([1, 2])
                                        
                                        with col1:
                                            # Movie details on the left
                                            st.markdown(f"**Year:** {movie.get('year', 'N/A')}")
                                            st.markdown(f"**Rating:** {movie.get('vote_average', 'N/A')}⭐ ({movie.get('vote_count', 'N/A')} votes)")
                                            st.markdown(f"**Genres:** {movie.get('genres_clean', 'N/A')}")
                                            st.markdown(f"**Director:** {movie.get('director', 'N/A')}")
                                            st.markdown(f"**Cast:** {movie.get('cast_clean', 'N/A')}")
                                        
                                        with col2:
                                            # Overview and actions on the right
                                            st.markdown("**Overview:**")
                                            st.markdown(f"*{movie.get('overview', 'No overview available')}*")
                                            
                                            # Interactive buttons
                                            action_col1, action_col2 = st.columns(2)
                                            with action_col1:
                                                watchlist = load_user_data(watchlist_file, {"to_watch": [], "watched": []})
                                                to_watch = watchlist.get("to_watch", [])
                                                watched = watchlist.get("watched", [])
                                                
                                                if movie_title not in to_watch and movie_title not in watched:
                                                    if st.button(f"Add to Watchlist", key=f"add_rec_{i}"):
                                                        add_to_watchlist(movie_title, watchlist_file)
                                                        
                                                        # Also add to positive feedback for model learning
                                                        if movie_title not in movie_feedback["liked"]:
                                                            movie_feedback["liked"].append(movie_title)
                                                            # Remove from disliked if present
                                                            if movie_title in movie_feedback["disliked"]:
                                                                movie_feedback["disliked"].remove(movie_title)
                                                        
                                                        save_user_data(feedback_file, movie_feedback)
                                                else:
                                                    st.info("Already in your watchlist or watched")
                                                    
                                            with action_col2:
                                                if movie_title not in movie_feedback["disliked"]:
                                                    if st.button(f"Not Interested", key=f"dislike_rec_{i}"):
                                                        # Add to negative feedback for model learning
                                                        if movie_title not in movie_feedback["disliked"]:
                                                            movie_feedback["disliked"].append(movie_title)
                                                            # Remove from liked if present
                                                            if movie_title in movie_feedback["liked"]:
                                                                movie_feedback["liked"].remove(movie_title)
                                                        
                                                        save_user_data(feedback_file, movie_feedback)
                                                        st.success(f"Noted: You're not interested in '{movie_title}'")
                                                else:
                                                    st.info("Already marked as 'Not Interested'")
                        
                    except Exception as e:
                        st.error(f"Error generating recommendations: {e}")
                        performance_tracker.stop('rec_time')
                else:
                    st.info("You need to watch and rate some movies to get personalized recommendations.")
            
            else:  # Find Similar Movies approach
                st.subheader("Find Movies Similar To...")
                
                # Let users select reference movies from the entire database
                reference_movies = st.multiselect(
                    "Find movies similar to:",
                    df['title'].tolist()
                )
                
                if reference_movies:
                    # Always use enhanced content recommender for similarity search
                    recommender = get_content_recommender(df, "enhanced", title_to_index)
                    recommendations = recommender.recommend(reference_movies, n=10)
                    
                    # Ensure recommendations have all required fields
                    recommendations = ensure_recommendation_fields(recommendations)
                    
                    # Display recommendations
                    if recommendations:
                        st.dataframe(pd.DataFrame(recommendations), use_container_width=True)
                        
                        # Allow adding to watchlist
                        st.subheader("Add to Watchlist")
                        movie_titles = [rec["Title"] for rec in recommendations]
                        to_add = st.multiselect("Select movies to add:", movie_titles, key="similar_add")
                        
                        if st.button("Add to Watchlist") and to_add:
                            for movie in to_add:
                                add_to_watchlist(movie, watchlist_file)
        
        # Movie of the Day tab
        with tabs[2]:
            st.subheader("🎬 Movie of the Day")
            
            # Use deterministic seed based on date for consistent daily recommendations
            seed = get_date_based_seed()
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
                        add_to_watchlist(motd['title'], watchlist_file)
                
                with col2:
                    st.write("**Overview:**")
                    st.write(motd.get('overview', 'No overview available'))
                    
                    # Show keyword/phrase wordcloud if overview exists
                    if motd.get('overview'):
                        ui.create_wordcloud(motd['overview'], dark_mode)
            else:
                st.warning("No high-rated movies found matching your filters.")
        
        # Watchlist & History tab
        with tabs[3]:
            st.subheader("📋 Your Movie Watchlist & History")
            
            # Load watchlist data
            watchlist_data = load_user_data(
                watchlist_file, 
                {"to_watch": [], "watched": []}
            )
            
            # Load ratings data
            ratings_file = f"feedback/{username}_ratings.json"
            ratings_data = load_user_data(ratings_file, {})
            
            watchlist_tabs = st.tabs(["To Watch", "Watched & Rated", "Add Movies"])
            
            with watchlist_tabs[0]:
                display_watchlist(
                    watchlist_data,
                    mark_watched_callback=lambda movie, rating=3.0, review="": mark_as_watched(movie, watchlist_file, username, rating, review),
                    remove_callback=lambda movie: remove_from_watchlist(movie, watchlist_file)
                )
            
            with watchlist_tabs[1]:
                st.subheader("Movies You've Watched")
                
                watched = watchlist_data.get("watched", [])
                if watched:
                    # Check for unrated movies
                    unrated = [movie for movie in watched if movie not in ratings_data]
                    
                    if unrated:
                        st.warning(f"You have {len(unrated)} watched movies that need ratings.")
                        
                        # Allow rating unrated movies
                        movie_to_rate = st.selectbox("Rate this watched movie:", unrated)
                        rating = st.slider("Your rating:", 1.0, 5.0, 3.0, 0.5)
                        
                        if st.button("Save Rating"):
                            ratings_data[movie_to_rate] = rating
                            save_user_data(ratings_file, ratings_data)
                            st.success(f"Rating for '{movie_to_rate}' saved!")
                            st.rerun()
                    
                    # Display watched and rated movies
                    rated_movies = {movie: ratings_data.get(movie, "Not rated") for movie in watched}
                    
                    rated_df = pd.DataFrame({
                        'Movie': rated_movies.keys(),
                        'Your Rating': rated_movies.values()
                    })
                    
                    st.dataframe(rated_df, use_container_width=True)
                    
                    if st.button("Clear Watch History"):
                        clear_watch_history(watchlist_file)
                else:
                    st.info("You haven't marked any movies as watched yet.")
            
            with watchlist_tabs[2]:
                st.subheader("Add Movies to Watchlist")
                
                # Show top-rated movies for easy addition
                st.markdown("**Top-Rated Movies**")
                #top_rated = filtered_df.sort_values(by='vote_average', ascending=False).head(10)
                
                # Multi-select interface for adding movies
                movies_to_add = st.multiselect(
                    "Select movies to add to your watchlist:",
                    df['title'].tolist()
                )
                
                if st.button("Add Selected Movies") and movies_to_add:
                    for movie in movies_to_add:
                        add_to_watchlist(movie, watchlist_file)
                
                # Manual movie addition
                st.markdown("---")
                st.markdown("**Add Movie Manually**")
                manual_movie = st.text_input("Movie Title:")
                
                if st.button("Add to Watchlist", key="add_manual_movie") and manual_movie:
                    add_to_watchlist(manual_movie, watchlist_file)
        
        # Insights tab
        with tabs[4]:
            st.subheader("📊 Insights & Analytics")
            
            insight_tabs = st.tabs(["Your Profile", "Genre Analysis", "Rating Analysis", "Viewing Trends"])
            
            with insight_tabs[0]:
                st.subheader("Your Movie Profile")
                
                # User watch stats
                st.markdown("### Watch Statistics")
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                
                with stats_col1:
                    watched = watchlist_data.get("watched", [])
                    st.metric("Movies Watched", len(watched))
                
                with stats_col2:
                    to_watch = watchlist_data.get("to_watch", [])
                    st.metric("On Watchlist", len(to_watch))
                
                with stats_col3:
                    # Calculate average rating
                    ratings_file = f"feedback/{username}_ratings.json"
                    ratings_data = load_user_data(ratings_file, {})
                    if ratings_data:
                        avg_rating = sum(ratings_data.values()) / len(ratings_data)
                        st.metric("Average Rating Given", f"{avg_rating:.1f} ⭐")
                
                # Genre preference visualization
                if watched:
                    st.markdown("### Your Genre Preferences")
                    
                    # Extract genres from watched movies
                    genre_counts = {}
                    for movie in watched:
                        movie_data = df[df['title'] == movie]
                        if not movie_data.empty and 'genres_list' in movie_data.columns:
                            genres = movie_data['genres_list'].iloc[0]
                            if isinstance(genres, list):
                                for genre in genres:
                                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
                    
                    if genre_counts:
                        # Create genre distribution chart
                        genre_df = pd.DataFrame({
                            'Genre': list(genre_counts.keys()),
                            'Count': list(genre_counts.values())
                        }).sort_values('Count', ascending=False)
                        
                        fig = px.pie(
                            genre_df, 
                            values='Count', 
                            names='Genre',
                            title="Genres You Watch",
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Watch more movies to see your genre preferences!")
                
                # Rating distribution
                if ratings_data:
                    st.markdown("### Your Rating Distribution")
                    
                    # Group ratings into bins
                    rating_counts = {}
                    for rating in ratings_data.values():
                        rating_counts[rating] = rating_counts.get(rating, 0) + 1
                    
                    rating_df = pd.DataFrame({
                        'Rating': list(rating_counts.keys()),
                        'Count': list(rating_counts.values())
                    }).sort_values('Rating')
                    
                    fig = px.bar(
                        rating_df,
                        x='Rating',
                        y='Count',
                        title="How You Rate Movies",
                        color='Rating',
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with insight_tabs[1]:
                st.subheader("Genre Analysis")
                
                # Create genre distribution chart
                all_genres = []
                for genres in filtered_df['genres_list']:
                    if isinstance(genres, list):
                        all_genres.extend(genres)
                
                genre_counts = {}
                for genre in all_genres:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
                
                # Sort and take top genres
                genre_df = pd.DataFrame({
                    'Genre': list(genre_counts.keys()),
                    'Count': list(genre_counts.values())
                }).sort_values('Count', ascending=False)
                
                # Bar chart of genre distribution
                fig = px.bar(
                    genre_df.head(15), 
                    x='Count', 
                    y='Genre',
                    orientation='h',
                    title="Most Common Genres in Database",
                    color='Count',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Genre correlation heatmap
                st.markdown("### Genre Co-occurrence")
                st.write("Which genres frequently appear together?")
                
                # Create co-occurrence matrix
                top_genres = genre_df.head(10)['Genre'].tolist()
                co_matrix = np.zeros((len(top_genres), len(top_genres)))
                
                # Fill co-occurrence matrix
                for genres in filtered_df['genres_list']:
                    if isinstance(genres, list):
                        indices = [i for i, g in enumerate(top_genres) if g in genres]
                        for i in indices:
                            for j in indices:
                                co_matrix[i, j] += 1
                
                # Create heatmap
                fig = px.imshow(
                    co_matrix,
                    labels=dict(x="Genre", y="Genre", color="Co-occurrence"),
                    x=top_genres,
                    y=top_genres,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with insight_tabs[2]:
                st.subheader("Rating Analysis")
                
                # Rating distribution histogram
                fig = px.histogram(
                    filtered_df,
                    x="vote_average",
                    nbins=20,
                    title="Rating Distribution",
                    color_discrete_sequence=['#3366CC'],
                    opacity=0.8
                )
                fig.update_layout(
                    xaxis_title="Rating",
                    yaxis_title="Count",
                    bargap=0.05
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Average rating by genre
                st.markdown("### Average Rating by Genre")
                
                # Calculate average rating per genre
                genre_ratings = {}
                genre_counts = {}
                
                for _, row in filtered_df.iterrows():
                    if isinstance(row.get('genres_list'), list) and not np.isnan(row.get('vote_average', 0)):
                        for genre in row['genres_list']:
                            if genre not in genre_ratings:
                                genre_ratings[genre] = 0
                                genre_counts[genre] = 0
                            genre_ratings[genre] += row['vote_average']
                            genre_counts[genre] += 1
                
                # Calculate averages
                for genre in genre_ratings:
                    if genre_counts[genre] > 0:
                        genre_ratings[genre] /= genre_counts[genre]
                
                # Create dataframe
                genre_rating_df = pd.DataFrame({
                    'Genre': list(genre_ratings.keys()),
                    'Average Rating': list(genre_ratings.values()),
                    'Movie Count': list(genre_counts.values())
                }).sort_values('Average Rating', ascending=False)
                
                # Filter to genres with enough movies
                genre_rating_df = genre_rating_df[genre_rating_df['Movie Count'] >= 10]
                
                # Create bar chart
                fig = px.bar(
                    genre_rating_df.head(15),
                    x='Genre',
                    y='Average Rating',
                    color='Movie Count',
                    title="Average Rating by Genre",
                    labels={'Average Rating': 'Average Rating (⭐)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with insight_tabs[3]:
                st.subheader("Viewing Trends")
                
                # Movies by decade
                st.markdown("### Movie Release Trends")
                
                # Create decade bins
                filtered_df['decade'] = (filtered_df['year'] // 10) * 10
                decade_counts = filtered_df['decade'].value_counts().reset_index()
                decade_counts.columns = ['Decade', 'Count']
                decade_counts = decade_counts.sort_values('Decade')
                
                # Bar chart of movies by decade
                fig = px.bar(
                    decade_counts,
                    x='Decade',
                    y='Count',
                    title="Movies by Decade",
                    color='Count',
                    labels={'decade': 'Decade', 'Count': 'Number of Movies'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Feedback & Evaluation tab
        with tabs[5]:
            st.subheader("🔄 Feedback & Model Training")
            
            feedback_tabs = st.tabs(["Rate the App", "Movie Preferences", "Model Training Stats", "Model Evaluation"])
            
            with feedback_tabs[0]:
                st.markdown("### Help Us Improve Nova")
                st.write("Your feedback helps us enhance the recommendation engine and user experience.")
                
                # User experience rating
                feedback_score = st.slider("Rate your experience with Nova:", 1.0, 5.0, 4.0, 0.5)
                
                # More specific feedback categories
                st.markdown("### Rate Specific Features")
                col1, col2 = st.columns(2)
                
                with col1:
                    recommendation_quality = st.slider("Recommendation Quality:", 1.0, 5.0, 4.0, 0.5)
                    app_usability = st.slider("App Usability:", 1.0, 5.0, 4.0, 0.5)
                
                with col2:
                    search_functionality = st.slider("Search Functionality:", 1.0, 5.0, 4.0, 0.5)
                    movie_database = st.slider("Movie Database Quality:", 1.0, 5.0, 4.0, 0.5)
                
                # Open feedback
                feedback_text = st.text_area("Additional suggestions or comments:")
                
                if st.button("Submit Feedback"):
                    feedback_data = {
                        "username": username,
                        "overall_score": feedback_score,
                        "recommendation_quality": recommendation_quality,
                        "app_usability": app_usability,
                        "search_functionality": search_functionality,
                        "movie_database": movie_database,
                        "feedback_text": feedback_text,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    feedback_dir = os.path.join("feedback", "system")
                    os.makedirs(feedback_dir, exist_ok=True)
                    
                    feedback_file_path = os.path.join(feedback_dir, f"{username}_feedback.json")
                    
                    # Save feedback
                    existing_feedback = load_user_data(feedback_file_path, [])
                    if not isinstance(existing_feedback, list):
                        existing_feedback = [existing_feedback]
                    
                    existing_feedback.append(feedback_data)
                    save_success, _ = save_user_data(feedback_file_path, existing_feedback)
                    
                    if save_success:
                        st.success("Thank you for your feedback!")
            
            with feedback_tabs[1]:
                st.markdown("### Your Movie Preferences")
                st.write("This information helps our recommendation engine understand your tastes.")
                
                # Movie attribute preferences
                st.markdown("#### What matters most to you in movies?")
                
                preferences = {
                    "Story": st.slider("Story Importance:", 1, 10, 7),
                    "Visuals": st.slider("Visual Effects/Cinematography:", 1, 10, 6),
                    "Acting": st.slider("Acting Quality:", 1, 10, 8),
                    "Originality": st.slider("Originality/Uniqueness:", 1, 10, 7),
                    "Emotional Impact": st.slider("Emotional Impact:", 1, 10, 6)
                }
                
                # Movie mood preferences
                st.markdown("#### What kinds of movies do you enjoy?")
                moods_col1, moods_col2 = st.columns(2)
                
                with moods_col1:
                    mood_prefs = {
                        "Action-packed": st.checkbox("Action-packed", value=True),
                        "Thought-provoking": st.checkbox("Thought-provoking"),
                        "Feel-good": st.checkbox("Feel-good"),
                        "Suspenseful": st.checkbox("Suspenseful"),
                        "Funny": st.checkbox("Funny")
                    }
                    
                with moods_col2:
                    mood_prefs.update({
                        "Emotional": st.checkbox("Emotional"),
                        "Scary": st.checkbox("Scary"),
                        "Inspiring": st.checkbox("Inspiring"),
                        "Relaxing": st.checkbox("Relaxing"),
                        "Educational": st.checkbox("Educational")
                    })
                
                # Save preferences
                if st.button("Save Preferences"):
                    prefs_data = {
                        "attribute_preferences": preferences,
                        "mood_preferences": mood_prefs,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    prefs_file = f"profiles/{username}_preferences.json"
                    save_success, error_msg = save_user_data(prefs_file, prefs_data)
                    
                    if save_success:
                        st.success("Your preferences have been saved and will improve your recommendations!")
                    else:
                        st.error(f"Error saving preferences: {error_msg}")
            
            with feedback_tabs[2]:
                st.markdown("### Model Training Statistics")
                
                # Count feedback and ratings data
                feedback_dir = "feedback"
                
                rating_files = 0
                total_ratings = 0
                total_likes = 0
                total_dislikes = 0
                
                if os.path.exists(feedback_dir):
                    # Count rating files and total ratings
                    for file in os.listdir(feedback_dir):
                        if file.endswith("_ratings.json"):
                            rating_files += 1
                            ratings_data = load_user_data(os.path.join(feedback_dir, file), {})
                            total_ratings += len(ratings_data)
                            
                        # Count feedback data
                        if file.endswith("_movie_feedback.json"):
                            feedback_data = load_user_data(os.path.join(feedback_dir, file), {"liked": [], "disliked": []})
                            total_likes += len(feedback_data.get("liked", []))
                            total_dislikes += len(feedback_data.get("disliked", []))
                
                # Display metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Users With Ratings", rating_files)
                with metric_col2:
                    st.metric("Total Ratings", total_ratings)
                with metric_col3:
                    feedback_signals = total_likes + total_dislikes
                    st.metric("Feedback Signals", feedback_signals)
                
                # Model quality estimate based on data quantity
                if total_ratings > 0:
                    # Simple formula to estimate quality (just for visual effect)
                    # This would be replaced by actual model metrics in a real system
                    quality_score = min(5.0, max(1.0, 2 + (total_ratings / 100) * 0.5 + (feedback_signals / 50) * 0.5))
                    
                    st.markdown("### Model Quality Estimate")
                    st.progress(quality_score / 5)
                    st.caption(f"Estimated quality score: {quality_score:.1f}/5")
                    
                    st.info("""
                    **How to improve the model:**
                    1. Rate more movies you've watched
                    2. Mark recommendations as 'Not Interested' when appropriate
                    3. Add movies to your watchlist
                    
                    The more data you provide, the better your recommendations will become!
                    """)
                else:
                    st.warning("Not enough rating data to estimate model quality.")
            
            with feedback_tabs[3]:
                st.markdown("### Recommendation Model Evaluation")
                
                # Check if evaluation results exist
                eval_file = os.path.join("evaluation", "recommender_evaluation.json")
                eval_image = os.path.join("evaluation", "recommender_comparison.png")
                
                if os.path.exists(eval_file):
                    # Load and display results
                    with open(eval_file, 'r') as f:
                        evaluation_results = json.load(f)
                    
                    # Display results table
                    results_df = pd.DataFrame(evaluation_results)
                    st.dataframe(results_df.T, use_container_width=True)
                    
                    # Display metrics explanation
                    st.markdown("""
                    **Metrics Explained:**
                    - **Precision**: Proportion of recommended items that were relevant
                    - **Recall**: Proportion of relevant items that were recommended
                    - **F1 Score**: Harmonic mean of precision and recall
                    - **Hit Rate**: Proportion of users for whom at least one recommendation was relevant
                    - **Coverage**: Proportion of all items that the recommender can recommend
                    - **Diversity**: Average number of unique genres in recommendations
                    - **Runtime**: Average time to generate recommendations (seconds)
                    """)
                    
                    # Show comparison plot if available
                    if os.path.exists(eval_image):
                        st.image(eval_image, caption="Recommender Performance Comparison")
                else:
                    st.info("No evaluation results available yet. Run the evaluation script to compare models.")
                    
                    if st.button("Run Evaluation (This may take several minutes)"):
                        with st.spinner("Evaluating recommendation models..."):
                            try:
                                # Import the evaluation module here to avoid circular imports
                                from evaluate_recommenders import evaluate_all_recommenders
                                
                                evaluator = evaluate_all_recommenders()
                                st.success("Evaluation completed! Refresh the page to see results.")
                            except Exception as e:
                                st.error(f"Evaluation failed: {str(e)}")
    else:
        # Not logged in state
        st.warning("👤 Please enter a username in the sidebar to get personalized recommendations.")
        
        # Show featured movies that are truly popular with sufficient ratings
        st.subheader("🌟 Featured Popular Movies")
        
        if not df.empty:
            # Find popular and well-rated movies with sufficient vote counts
            featured_df = df[
                (df['vote_average'] >= 7.5) &   # High rating
                (df['vote_count'] >= 500) &     # Enough votes to be reliable
                (df['popularity'] >= 50)        # Has some popularity
            ].sort_values(by=['vote_average', 'popularity'], ascending=False).head(5)
            
            # Fallback if no movies match strict criteria
            if featured_df.empty:
                featured_df = df[df['vote_count'] >= 200].sort_values(by='vote_average', ascending=False).head(5)
            
            # Display featured movies in cards
            for i, (_, movie) in enumerate(featured_df.iterrows()):
                with st.expander(f"{movie['title']} ({movie.get('year', 'N/A')}) - {movie.get('vote_average', 'N/A')}⭐", expanded=(i==0)):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"**Rating:** {movie.get('vote_average', 'N/A')}⭐ ({movie.get('vote_count', 'N/A')} votes)")
                        st.markdown(f"**Genres:** {movie.get('genres_clean', 'N/A')}")
                        st.markdown(f"**Director:** {movie.get('director', 'N/A')}")
                        st.markdown(f"**Cast:** {movie.get('cast_clean', 'N/A')}")
                    
                    with col2:
                        st.markdown("**Overview:**")
                        st.markdown(f"*{movie.get('overview', 'No overview available')}*")
            
            # Some basic app stats
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Movies in Database", f"{len(df):,}")
            with col2:
                years_range = f"{int(df['year'].min())} - {int(df['year'].max())}"
                st.metric("Year Range", years_range)
            with col3:
                genres_count = len(set().union(*df['genres_list'].dropna()))
                st.metric("Unique Genres", genres_count)

# Add this function to handle new user onboarding

def onboard_new_user(username, df):
    """Onboard a new user with initial movie ratings
    
    This function presents 25 movies (a mix of popular and diverse films) for the
    user to rate, helping to build an initial profile for better recommendations.
    """
    st.title("🎬 Welcome to Nova!")
    st.markdown("""
    ### Let's get to know your movie taste
    
    Rate these 25 movies to get started with personalized recommendations.
    If you haven't seen a movie, just click "Haven't seen it" to move on.
    """)
    
    # Load or create ratings file
    ratings_file = f"feedback/{username}_ratings.json"
    ratings_data = load_user_data(ratings_file, {})
    
    # Load or create watchlist
    watchlist_file = f"watchlists/{username}_watchlist.json"
    watchlist_data = load_user_data(watchlist_file, {"to_watch": [], "watched": []})
    
    # Create a more balanced set of movies (popular + some diverse picks)
    if 'onboarding_movies' not in st.session_state:
        diverse_movies = []
        
        # 1. Start with highly popular mainstream movies (10 movies)
        blockbusters = df[(df['vote_average'] >= 7.0) & 
                           (df['vote_count'] >= 2500) & 
                           (df['popularity'] >= 80)]
        if not blockbusters.empty:
            diverse_movies.extend(blockbusters.sample(min(10, len(blockbusters)))['title'].tolist())
        
        # 2. Add some critically acclaimed films that are still well-known (5 movies)
        acclaimed = df[(df['vote_average'] >= 8.0) & 
                        (df['vote_count'] >= 500) & 
                        (df['vote_count'] <= 2500)]
        if not acclaimed.empty:
            diverse_movies.extend(acclaimed.sample(min(5, len(acclaimed)))['title'].tolist())
        
        # 3. Add a few international films that gained mainstream attention (3 movies)
        international = df[(df['original_language'] != 'en') & 
                            (df['vote_average'] >= 7.5) &
                            (df['vote_count'] >= 300)]
        if not international.empty:
            diverse_movies.extend(international.sample(min(3, len(international)))['title'].tolist())
        
        # 4. Add classics from different decades (4 movies)
        classics_decades = [(1960, 1979), (1980, 1999)]
        for start, end in classics_decades:
            classics = df[(df['year'] >= start) & (df['year'] <= end) & 
                           (df['vote_average'] >= 7.5) &
                           (df['vote_count'] >= 500)]
            if not classics.empty:
                diverse_movies.extend(classics.sample(min(2, len(classics)))['title'].tolist())
        
        # 5. Add a small selection of hidden gems (3 movies)
        hidden_gems = df[(df['vote_average'] >= 7.5) & 
                           (df['vote_count'] >= 100) & 
                           (df['vote_count'] <= 300)]
        if not hidden_gems.empty:
            diverse_movies.extend(hidden_gems.sample(min(3, len(hidden_gems)))['title'].tolist())
        
        # Remove duplicates and limit to 25
        diverse_movies = list(dict.fromkeys(diverse_movies))[:25]
        
        # If we still don't have 25, fill with more popular movies
        if len(diverse_movies) < 25:
            more_popular = df[(df['vote_average'] >= 6.5) & (df['vote_count'] >= 500)]
            # Exclude movies already in our list
            more_popular = more_popular[~more_popular['title'].isin(diverse_movies)]
            # Add enough to reach 25
            if not more_popular.empty:
                diverse_movies.extend(more_popular.sample(min(25 - len(diverse_movies), len(more_popular)))['title'].tolist())
        
        # Shuffle the list for better user experience
        import random
        random.seed(42)  # For reproducibility
        random.shuffle(diverse_movies)
        
        # Store the movie list in session state so it doesn't change between interactions
        st.session_state.onboarding_movies = diverse_movies
    
    # Use the stored movie list
    diverse_movies = st.session_state.onboarding_movies
    
    # Initialize session state to track progress if not already present
    if 'onboarding_index' not in st.session_state:
        st.session_state.onboarding_index = 0
        st.session_state.onboarded_count = 0
    
    # Calculate progress
    total_movies = len(diverse_movies)
    current_index = st.session_state.onboarding_index
    
    # Check if we've reached the end
    if current_index >= total_movies:
        st.success("🎉 You've completed the onboarding process!")
        st.markdown(f"You've rated {st.session_state.onboarded_count} movies. Now you can explore personalized recommendations!")
        
        # Reset onboarding state and mark as completed
        del st.session_state.onboarding_index
        del st.session_state.onboarded_count
        del st.session_state.onboarding_movies
        
        # Create a profile file to mark onboarding as completed
        profile_file = f"profiles/{username}.json"
        profile_data = load_user_data(profile_file, {"liked_movies": [], "favorite_genres": []})
        profile_data['onboarding_complete'] = True
        save_user_data(profile_file, profile_data)
        
        # Add a button to continue to the main app
        if st.button("Continue to Nova"):
            st.rerun()
        
        return True
    
    # Show progress bar
    progress = current_index / total_movies
    st.progress(progress)
    st.caption(f"Movie {current_index + 1} of {total_movies}")
    
    # Get current movie to rate
    current_movie = diverse_movies[current_index]
    
    # Display movie details for rating
    movie_data = df[df['title'] == current_movie]
    
    if not movie_data.empty:
        movie = movie_data.iloc[0]
        
        # Create two columns
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader(movie['title'])
            st.write(f"**Year:** {movie.get('year', 'N/A')}")
            st.write(f"**Rating:** {movie.get('vote_average', 'N/A')}⭐")
            st.write(f"**Genres:** {movie.get('genres_clean', 'N/A')}")
            st.write(f"**Director:** {movie.get('director', 'N/A')}")
            if movie.get('original_language') != 'en':
                st.write(f"**Language:** {movie.get('original_language', 'N/A')}")
        
        with col2:
            st.write("**Overview:**")
            st.write(movie.get('overview', 'No overview available'))
            
            # Add popularity indicator so users can see if it's mainstream or not
            popularity_level = "Blockbuster" if movie.get('popularity', 0) > 80 else (
                              "Popular" if movie.get('popularity', 0) > 40 else 
                              "Indie/Lesser Known")
            vote_count = movie.get('vote_count', 0)
            st.caption(f"{popularity_level} • {int(vote_count):,} ratings")
        
        # Rating options with improved UI
        st.markdown("### Have you seen this movie?")
        
        # Display rating slider 
        rating = st.slider("Your rating:", 1.0, 5.0, 3.0, 0.5, key=f"rating_{current_index}")
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        # Haven't seen it button
        with col1:
            if st.button("Haven't seen it", key=f"not_seen_{current_index}"):
                # Move to next movie without saving a rating
                st.session_state.onboarding_index += 1
                st.rerun()
        
        # Previous button - in middle to avoid accidental clicks
        with col2:
            if current_index > 0:
                if st.button("Previous", key=f"prev_{current_index}"):
                    st.session_state.onboarding_index -= 1
                    st.rerun()
        
        # Rate and next button
        with col3:
            if st.button("Rate & Next", key=f"rate_{current_index}"):
                # Save rating
                ratings_data[current_movie] = rating
                save_user_data(ratings_file, ratings_data)
                
                # Add to watched list
                if current_movie not in watchlist_data["watched"]:
                    watchlist_data["watched"].append(current_movie)
                    save_user_data(watchlist_file, watchlist_data)
                
                # Increment count of rated movies
                st.session_state.onboarded_count += 1
                
                # Move to next movie
                st.session_state.onboarding_index += 1
                st.rerun()
    
    return False

def add_to_watchlist(movie_title, watchlist_file):
    """Add a movie to the watchlist"""
    watchlist = load_user_data(watchlist_file, {"to_watch": [], "watched": []})
    
    if movie_title not in watchlist["to_watch"]:
        watchlist["to_watch"].append(movie_title)
        save_success, error_msg = save_user_data(watchlist_file, watchlist)
        
        if save_success:
            st.success(f"Added '{movie_title}' to your watchlist!")
        else:
            st.error(f"Error updating watchlist: {error_msg}")
    else:
        st.info(f"'{movie_title}' is already in your watchlist.")

def mark_as_watched(movie_title, watchlist_file, username=None, rating=3.0, review=""):
    """Mark a movie as watched with rating and review"""
    # Only proceed if username is provided (should always be the case)
    if not username:
        st.error("Username required to mark movies as watched")
        return
        
    watchlist = load_user_data(watchlist_file, {"to_watch": [], "watched": []})
    
    if movie_title in watchlist["to_watch"]:
        # Remove from to_watch
        watchlist["to_watch"].remove(movie_title)
        
        # Add to watched
        if movie_title not in watchlist["watched"]:
            watchlist["watched"].append(movie_title)
        
        # Update watchlist
        save_user_data(watchlist_file, watchlist)
        
        # Save rating
        ratings_file = f"feedback/{username}_ratings.json"
        ratings = load_user_data(ratings_file, {})
        ratings[movie_title] = rating
        save_user_data(ratings_file, ratings)
        
        # Save review if provided
        if review.strip():
            reviews_file = f"feedback/{username}_reviews.json"
            reviews = load_user_data(reviews_file, {})
            reviews[movie_title] = {
                "text": review,
                "rating": rating,
                "date": datetime.now().strftime("%Y-%m-%d")
            }
            save_user_data(reviews_file, reviews)
        
        st.success(f"Marked '{movie_title}' as watched with a rating of {rating}!")
        st.rerun()

def remove_from_watchlist(movie_title, watchlist_file):
    """Remove a movie from the watchlist"""
    watchlist = load_user_data(watchlist_file, {"to_watch": [], "watched": []})
    
    if movie_title in watchlist["to_watch"]:
        watchlist["to_watch"].remove(movie_title)
        
        save_success, error_msg = save_user_data(watchlist_file, watchlist)
        if save_success:
            st.success(f"Removed '{movie_title}' from watchlist!")
            st.rerun()
        else:
            st.error(f"Error updating watchlist: {error_msg}")

def clear_watch_history(watchlist_file):
    """Clear the watch history"""
    watchlist = load_user_data(watchlist_file, {"to_watch": [], "watched": []})
    
    watchlist["watched"] = []
    
    save_success, error_msg = save_user_data(watchlist_file, watchlist)
    if save_success:
        st.success("Watch history cleared!")
        st.rerun()
    else:
        st.error(f"Error clearing watch history: {error_msg}")

def display_watchlist(watchlist, mark_watched_callback=None, remove_callback=None):
    """Display user's watchlist"""
    st.subheader("Movies To Watch")
    
    if not watchlist.get("to_watch"):
        st.info("Your watchlist is empty. Add movies from search or recommendations!")
        return
    
    # Display movies to watch with expand/collapse for review
    for i, movie in enumerate(watchlist["to_watch"]):
         with st.expander(f"{i+1}. {movie}"):
            # Rating slider
            rating = st.slider("Your rating:", 1.0, 5.0, 3.0, 0.5, key=f"rate_watchlist_{i}")
            review_text = st.text_area("Add your review (optional):", key=f"review_watchlist_{i}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Review & Mark as Watched", key=f"save_review_{i}") and mark_watched_callback:
                    # Pass both the movie and the rating to the callback
                    mark_watched_callback(movie, rating=rating, review=review_text)
            with col2:
                if st.button("✖ Remove from Watchlist", key=f"remove_{i}") and remove_callback:
                    remove_callback(movie)

if __name__ == "__main__":
    main()