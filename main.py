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
from utils import create_directories, sanitize_username, load_user_data, save_user_data, PerformanceTracker, get_date_based_seed
import ui

# Import recommenders
from recommenders.content_based import ContentBasedRecommender, EnhancedContentRecommender, get_recommender as get_content_recommender
from recommenders.collaborative import CollaborativeFilteringRecommender, TrueCollaborativeRecommender, get_recommender as get_collab_recommender
from recommenders.knn import KNNRecommender, get_recommender as get_knn_recommender
from recommenders.matrix_factorization import MatrixFactorizationRecommender, get_recommender as get_mf_recommender
from recommenders.neural import NeuralRecommender, get_recommender as get_neural_recommender
from recommenders.association_rules import AssociationRulesRecommender, get_recommender as get_assoc_recommender
from recommenders.base import combine_recommendations

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
    
    # Set file paths
    username = sanitize_username(username_input)
    profile_file = f"profiles/{username}.json"
    feedback_file = f"feedback/{username}_feedback.json"
    watchlist_file = f"watchlists/{username}_watchlist.json"
    
    # Process user data if username is provided
    if username:
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
            # Search functionality
            st.subheader("🔍 Advanced Search")
            col1, col2 = st.columns([3, 1])
            with col1:
                search_query = st.text_input("Search for a movie by title, director, actor, or keywords")
            with col2:
                search_type = st.selectbox("Search by", ["Title", "Director", "Actor", "Keywords"])
            
            if search_query:
                performance_tracker.start('search_time')
                if search_type == "Title":
                    results = filtered_df[filtered_df['title'].str.contains(search_query, case=False, na=False)]
                elif search_type == "Director":
                    results = filtered_df[filtered_df['director'].str.contains(search_query, case=False, na=False)]
                elif search_type == "Actor":
                    results = filtered_df[filtered_df['cast_clean'].str.contains(search_query, case=False, na=False)]
                elif search_type == "Keywords":
                    results = filtered_df[filtered_df['overview'].str.contains(search_query, case=False, na=False)]
                
                search_time = performance_tracker.stop('search_time')
                
                if not results.empty:
                    display_cols = ['title', 'genres_clean', 'year', 'vote_average', 'director', 'cast_clean']
                    safe_cols = [col for col in display_cols if col in results.columns]
                    
                    st.subheader(f"Search Results for '{search_query}' ({len(results)} results):")
                    st.dataframe(results[safe_cols], use_container_width=True)
                    st.caption(f"Search completed in {search_time:.4f} seconds")
                    
                    # Preview first result
                    if len(results) > 0:
                        with st.expander("Preview First Result", expanded=False):
                            first_result = results.iloc[0]
                            movie_details = get_movie_details(df, first_result['title'])
                            
                            # Display movie details
                            ui.display_movie_details(
                                movie_details,
                                lambda title: add_to_watchlist(title, watchlist_file)
                            )
                else:
                    st.warning(f"No results found for '{search_query}' in {search_type}.")
            
            # Dataset exploration
            st.subheader("📚 Explore the TMDB Dataset")
            display_cols = ['title', 'genres_clean', 'year', 'original_language', 'vote_average', 'vote_count']
            safe_cols = [col for col in display_cols if col in filtered_df.columns]
            st.dataframe(filtered_df[safe_cols].head(1000), use_container_width=True)
        
        # Recommendations tab
        with tabs[1]:
            st.subheader("🤖 Personalized Recommendations")
            
            # Let user select recommendation algorithm
            algo_options = [
                "Content-Based", 
                "Enhanced Content-Based", 
                "Collaborative Filtering",
                "K-Nearest Neighbors",
                "Matrix Factorization",
                "Neural Network",
                "Association Rules",
                "Hybrid Recommendations"
            ]
            
            recommender_type = st.selectbox(
                "Choose a recommendation algorithm:",
                algo_options
            )
            
            # Generate recommendations if user has selected movies
            if selected_movies:
                performance_tracker.start('rec_time')
                recommendations = []
                
                try:
                    if recommender_type == "Content-Based":
                        recommender = get_content_recommender(df, "content", title_to_index)
                        recommendations = recommender.recommend(selected_movies, n=10)
                        
                    elif recommender_type == "Enhanced Content-Based":
                        recommender = get_content_recommender(df, "enhanced", title_to_index)
                        recommendations = recommender.recommend(selected_movies, n=10, favorite_genres=selected_genres)
                        
                    elif recommender_type == "Collaborative Filtering":
                        recommender = get_collab_recommender(df, "hybrid", title_to_index)
                        recommendations = recommender.recommend(selected_movies, n=10, method='hybrid')
                        
                    elif recommender_type == "K-Nearest Neighbors":
                        recommender = KNNRecommender(df, title_to_index)
                        recommender.fit()
                        recommendations = recommender.recommend(selected_movies, n=10)
                        
                    elif recommender_type == "Matrix Factorization":
                        recommender = MatrixFactorizationRecommender(df, title_to_index)
                        recommendations = recommender.recommend(selected_movies, n=10)
                        
                    elif recommender_type == "Neural Network":
                        recommender = NeuralRecommender(df, title_to_index)
                        recommendations = recommender.recommend(selected_movies, n=10)
                        
                    elif recommender_type == "Association Rules":
                        recommender = AssociationRulesRecommender(df, title_to_index)
                        recommendations = recommender.recommend(selected_movies, n=10)
                        
                    elif recommender_type == "Hybrid Recommendations":
                        # Create multiple recommenders
                        content_rec = get_content_recommender(df, "content", title_to_index)
                        collab_rec = get_collab_recommender(df, "hybrid", title_to_index)
                        knn_rec = KNNRecommender(df, title_to_index)
                        knn_rec.fit()
                        
                        # Get recommendations from each
                        content_results = content_rec.recommend(selected_movies, n=20)
                        collab_results = collab_rec.recommend(selected_movies, n=20, method='hybrid')
                        knn_results = knn_rec.recommend(selected_movies, n=20)
                        
                        # Combine with weights (content gets more weight)
                        recommendations = combine_recommendations(
                            [content_results, collab_results, knn_results],
                            weights=[0.5, 0.3, 0.2],
                            n=10
                        )
                    
                    rec_time = performance_tracker.stop('rec_time')
                    
                    # Display recommendations
                    if recommendations:
                        st.dataframe(pd.DataFrame(recommendations), use_container_width=True)
                        st.caption(f"Recommendations generated in {rec_time:.4f} seconds")
                        
                        # Allow rating of recommendations
                        st.subheader("⭐ Rate These Recommendations")
                        
                        # Load existing ratings
                        ratings_feedback = load_user_data(feedback_file, {})
                        
                        # Display rating UI for top 5 recommendations
                        top_recs = recommendations[:5]
                        cols = st.columns(3)
                        
                        for idx, rec in enumerate(top_recs):
                            with cols[idx % 3]:
                                st.write(f"**{rec['Title']}** ({rec.get('Year', 'N/A')})")
                                st.write(f"_{rec.get('Genres', 'N/A')}_")
                                st.write(f"Rating: {rec.get('Rating', 'N/A')} ⭐")
                                user_rating = st.slider(
                                    "Your Rating", 
                                    1.0, 5.0, 
                                    ratings_feedback.get(rec['Title'], 3.0),
                                    step=0.5, 
                                    key=f"rating_{rec['Title']}"
                                )
                                ratings_feedback[rec['Title']] = user_rating
                        
                        if st.button("Save Ratings"):
                            save_success, error_msg = save_user_data(feedback_file, ratings_feedback)
                            if save_success:
                                st.success("Ratings saved!")
                            else:
                                st.error(f"Couldn't save ratings: {error_msg}")
                    else:
                        st.info("No recommendations generated. Try selecting different movies.")
                
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")
                    performance_tracker.stop('rec_time')
                    
            else:
                st.info("Please select movies you like to get personalized recommendations.")
            
            # Trending movies section
            st.subheader("🔥 Trending Now")
            if 'popularity' in filtered_df.columns:
                trending = filtered_df.sort_values(by='popularity', ascending=False).head(10)
                display_cols = ['title', 'genres_clean', 'vote_average', 'popularity']
                safe_cols = [col for col in display_cols if col in trending.columns]
                st.dataframe(trending[safe_cols], use_container_width=True)
        
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
            st.subheader("📋 Your Movie Watchlist")
            
            # Load watchlist data
            watchlist_data = load_user_data(
                watchlist_file, 
                {"to_watch": [], "watched": []}
            )
            
            watchlist_tabs = st.tabs(["To Watch", "Watched", "Add Movies"])
            
            with watchlist_tabs[0]:
                ui.display_watchlist(
                    watchlist_data,
                    mark_watched_callback=lambda movie: mark_as_watched(movie, watchlist_file),
                    remove_callback=lambda movie: remove_from_watchlist(movie, watchlist_file)
                )
            
            with watchlist_tabs[1]:
                ui.display_watch_history(
                    watchlist_data.get("watched", []),
                    lambda: clear_watch_history(watchlist_file)
                )
            
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
                    for movie in movies_to_add:
                        add_to_watchlist(movie, watchlist_file)
                
                # Manual movie addition
                st.markdown("---")
                st.markdown("**Add Movie Manually**")
                manual_movie = st.text_input("Movie Title:")
                
                if st.button("Add to Watchlist") and manual_movie:
                    add_to_watchlist(manual_movie, watchlist_file)
        
        # Insights tab
        with tabs[4]:
            st.subheader("📊 Insights & Analytics")
            
            if not filtered_df.empty:
                # Show genre distribution
                st.markdown("### Rating & Genre Distribution")
                ui.show_genre_distribution(filtered_df)
                
                # Movie preferences section
                if selected_movies or selected_genres:
                    st.markdown("### Your Movie Preferences")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Favorite Genres**")
                        if selected_genres:
                            for genre in selected_genres:
                                st.markdown(f"- {genre}")
                        else:
                            st.info("No favorite genres selected yet.")
                    
                    with col2:
                        st.markdown("**Your Rated Movies**")
                        ratings_file = f"feedback/{username}_ratings.json"
                        ratings_data = load_user_data(ratings_file, {})
                        
                        if ratings_data:
                            ratings_df = pd.DataFrame({
                                'Movie': list(ratings_data.keys()),
                                'Rating': list(ratings_data.values())
                            })
                            
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
                            
                # Create wordcloud from liked movies overview text
                st.subheader("🔤 Word Cloud of Liked Movies")
                if selected_movies:
                    liked_overviews = " ".join(
                        filtered_df[filtered_df['title'].isin(selected_movies)]['overview'].dropna().tolist()
                    )
                    ui.create_wordcloud(liked_overviews, dark_mode)
                else:
                    st.info("Select movies you like to generate a word cloud.")
            
        # Feedback & Evaluation tab
        with tabs[5]:
            st.subheader("🔄 Feedback & Evaluation")
            
            # Display performance metrics
            ui.display_metrics(performance_tracker.get_metrics())
            
            # User feedback
            st.markdown("### Your Feedback")
            feedback_score = st.slider("Rate your experience with Nova (1-5)", 1.0, 5.0, 3.0, 0.5)
            feedback_text = st.text_area("Tell us how we can improve:")
            
            if st.button("Submit Feedback"):
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
                existing_feedback = load_user_data(feedback_file_path, [])
                if not isinstance(existing_feedback, list):
                    existing_feedback = [existing_feedback]
                
                existing_feedback.append(feedback_data)
                save_success, _ = save_user_data(feedback_file_path, existing_feedback)
                
                if save_success:
                    st.success("Thank you for your feedback!")
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

def mark_as_watched(movie_title, watchlist_file):
    """Mark a movie as watched"""
    watchlist = load_user_data(watchlist_file, {"to_watch": [], "watched": []})
    
    if movie_title in watchlist["to_watch"]:
        watchlist["to_watch"].remove(movie_title)
        if movie_title not in watchlist["watched"]:
            watchlist["watched"].append(movie_title)
        
        save_success, error_msg = save_user_data(watchlist_file, watchlist)
        if save_success:
            st.success(f"Marked '{movie_title}' as watched!")
            st.experimental_rerun()
        else:
            st.error(f"Error updating watchlist: {error_msg}")

def remove_from_watchlist(movie_title, watchlist_file):
    """Remove a movie from the watchlist"""
    watchlist = load_user_data(watchlist_file, {"to_watch": [], "watched": []})
    
    if movie_title in watchlist["to_watch"]:
        watchlist["to_watch"].remove(movie_title)
        
        save_success, error_msg = save_user_data(watchlist_file, watchlist)
        if save_success:
            st.success(f"Removed '{movie_title}' from watchlist!")
            st.experimental_rerun()
        else:
            st.error(f"Error updating watchlist: {error_msg}")

def clear_watch_history(watchlist_file):
    """Clear the watch history"""
    watchlist = load_user_data(watchlist_file, {"to_watch": [], "watched": []})
    
    watchlist["watched"] = []
    
    save_success, error_msg = save_user_data(watchlist_file, watchlist)
    if save_success:
        st.success("Watch history cleared!")
        st.experimental_rerun()
    else:
        st.error(f"Error clearing watch history: {error_msg}")

if __name__ == "__main__":
    main()