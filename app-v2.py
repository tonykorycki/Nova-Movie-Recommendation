import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import numpy as np
import seaborn as sns
from wordcloud import WordCloud
from collections import defaultdict
import plotly.express as px

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



# === LOAD DATA ===
@st.cache_data

def load_data():
    movies_df = pd.read_csv("tmdb_5000_movies.csv")
    credits_df = pd.read_csv("tmdb_5000_credits.csv")

    credits_df.rename(columns={'movie_id': 'id'}, inplace=True)
    merged_df = movies_df.merge(credits_df, on='id')

    merged_df.rename(columns={'title_x': 'title'}, inplace=True)

    merged_df['overview'] = merged_df['overview'].fillna('unknown')
    merged_df['homepage'] = merged_df['homepage'].fillna('none')
    merged_df['tagline'] = merged_df['tagline'].fillna('none')
    merged_df.dropna(axis=0, how='any', inplace=True)

    def extract_names(json_str, key="name", max_items=3):
        try:
            data = json.loads(json_str.replace("'", '"'))
            return ", ".join([item[key] for item in data[:max_items]])
        except:
            return ""
        
    def extract_list(json_str, key="name"):  # for filtering
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
        except json.JSONDecodeError:
            return ""
        return ""

    merged_df['genres_clean'] = merged_df['genres'].apply(lambda x: extract_names(x))
    merged_df['cast_clean'] = merged_df['cast'].apply(lambda x: extract_names(x))
    merged_df['cast_list'] = merged_df['cast'].apply(lambda x: extract_list(x))
    merged_df['director'] = merged_df['crew'].apply(extract_director)
    merged_df['year'] = pd.to_datetime(merged_df['release_date'], errors='coerce').dt.year

    return merged_df

df = load_data()

# Apply filters to dataframe

for folder in ["profiles", "feedback", "watchlists"]:
    os.makedirs(folder, exist_ok=True)

st.sidebar.header("👤 User Profile")
username = st.sidebar.text_input("Enter your username:")
profile_file = f"profiles/{username}.json"
feedback_file = f"feedback/{username}_feedback.json"
watchlist_file = f"watchlists/{username}_watchlist.json"

if username:
    if os.path.exists(profile_file):
        with open(profile_file, 'r') as f:
            profile_data = json.load(f)
    else:
        profile_data = {"liked_movies": [], "favorite_genres": []}

    genres = sorted(set(", ".join(df['genres_clean']).split(", ")))
    selected_genres = st.sidebar.multiselect("Favorite genres", genres, default=profile_data['favorite_genres'])
    movie_options = df['title'].sort_values().tolist()
    selected_movies = st.sidebar.multiselect("Liked movies", movie_options, default=profile_data['liked_movies'])

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔎 Filters")
    min_year, max_year = int(df['year'].min()), int(df['year'].max())
    year_range = st.sidebar.slider("Year Range", min_year, max_year, (2000, max_year))
    min_runtime, max_runtime = int(df['runtime'].min()), int(df['runtime'].max())
    runtime_range = st.sidebar.slider("Runtime (minutes)", min_runtime, max_runtime, (60, 180))
    languages = sorted(df['original_language'].unique())
    selected_languages = st.sidebar.multiselect("Languages", languages, default=['en'])
    st.sidebar.subheader("🎭 Filter by Actor")
    all_actors = sorted({actor for cast in df['cast_list'] for actor in cast})
    selected_actors = st.sidebar.multiselect("Select actors:", all_actors)

    filtered_df = df[
    (df['year'] >= year_range[0]) & (df['year'] <= year_range[1]) &
    (df['runtime'] >= runtime_range[0]) & (df['runtime'] <= runtime_range[1]) &
    (df['original_language'].isin(selected_languages)) &
    (df['cast_list'].apply(lambda cast: any(actor in cast for actor in selected_actors)) if selected_actors else True)
]
    
    tfidf = TfidfVectorizer(stop_words='english')
    filtered_df['overview'] = filtered_df['overview'].fillna('')

    if filtered_df.empty or filtered_df['overview'].str.strip().eq('').all():
        st.error("No movies match your filters.")
        st.stop()

    tfidf_matrix = tfidf.fit_transform(df['overview'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    title_to_index = pd.Series(df.index, index=df['title']).drop_duplicates()

    profile_data['liked_movies'] = selected_movies
    profile_data['favorite_genres'] = selected_genres

    with open(profile_file, 'w') as f:
        json.dump(profile_data, f)

    st.session_state["genres"] = selected_genres
    st.session_state["liked_movies"] = selected_movies

        # === 🎲 Surprise Me Button ===
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎲 Surprise Me")
    if st.sidebar.button("Give me a random movie!"):
        surprise = df.sample(1).iloc[0]
        st.sidebar.markdown(f"**🎬 {surprise['title']} ({surprise['year']})**")
        st.sidebar.markdown(f"Genres: _{surprise['genres_clean']}_")
        st.sidebar.markdown(f"⭐ Rating: {surprise['vote_average']}")
        st.sidebar.markdown(f"🧑‍🎨 Director: {surprise['director']}")
        if surprise['homepage'] != 'none':
            st.sidebar.markdown(f"[🌐 Trailer/More Info]({surprise['homepage']})")
        if surprise['tagline'] != 'none':
            st.sidebar.markdown(f"_\"{surprise['tagline']}\"_")


    tabs = st.tabs(["Explore Movies", "Recommendations", "Movie of the Day", "Watchlist & History", "Insights", "Feedback & Evaluation"])

    with tabs[0]:

        search_query = st.text_input("🔍 Search for a movie by title")
        if search_query:
            results = df[df['title'].str.contains(search_query, case=False, na=False)]
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
        st.dataframe(filtered_df[['title', 'genres_clean', 'year', 'original_language', 'vote_average', 'vote_count']], use_container_width=True)

            

    with tabs[1]:
        st.subheader("🤖 Personalized Recommendations")
        recommendations = []
        top_recs = []

        if selected_movies:
            indices = [title_to_index[movie] for movie in selected_movies if movie in title_to_index]
            sim_scores = np.sum(cosine_sim[indices], axis=0)
            sim_scores = list(enumerate(sim_scores))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = [i for i in sim_scores if i[0] not in indices]

            for i, score in sim_scores[:50]:
                movie = df.iloc[i]
                genre_match = sum(1 for genre in selected_genres if genre.strip() in movie['genres_clean'].split(", "))
                weight = 0.5 * score + 0.3 * (genre_match / len(selected_genres) if selected_genres else 0) + 0.2 * (movie['vote_average'] / 10)
                recommendations.append((movie['title'], movie['genres_clean'], movie['year'], movie['vote_average'], round(score, 3), round(weight, 3)))

            recommendations.sort(key=lambda x: x[-1], reverse=True)
            top_recs = recommendations[:5]
    
            rec_df = pd.DataFrame(recommendations, columns=['Title', 'Genres', 'Year', 'Rating', 'Similarity', 'Score'])
            st.dataframe(rec_df.head(10), use_container_width=True)

            st.subheader("⭐ Rate These Recommendations")
            rating_feedback = {}
            for title, genre, year, rating, sim, score in top_recs:
                with st.expander(f"{title} ({year}) — {genre}"):
                    st.write(f"**Vote Avg:** {rating} ⭐ | **Similarity:** {sim}")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        user_rating = round(st.slider("Rating", 1.0, 5.0, 3.0, step=0.5, key=title))
                    rating_feedback[title] = user_rating

            if st.button("Save Ratings"):
                with open(feedback_file, 'w') as f:
                    json.dump(rating_feedback, f)
                st.success("Ratings saved!")

        if selected_genres:
            st.markdown("Based on your favorite genres:")
            def genre_score(movie_genres):
                return sum(1 for genre in selected_genres if genre.strip() in movie_genres.split(", "))

            filtered_df['genre_score'] = df['genres_clean'].apply(genre_score)
            filtered = filtered_df[filtered_df['genre_score'] > 0].sort_values(by=['genre_score', 'vote_average'], ascending=False).head(10)
            st.dataframe(filtered[['title', 'genres_clean', 'year', 'vote_average']], use_container_width=True)

        if not selected_movies and not selected_genres:
            st.info("Add liked movies or genres to get personalized recommendations.")
            

        st.subheader("🔥 Trending Now")
        trending = filtered_df.sort_values(by='popularity', ascending=False).head(10)
        st.dataframe(trending[['title', 'genres_clean', 'vote_average', 'popularity']], use_container_width=True)

        st.subheader("👥 Users Like You Also Liked")
        dummy_users = {f'user{i}': df['title'].sample(5).tolist() for i in range(1, 6)}
        dummy_users[username] = selected_movies

        all_titles = sorted(set(filtered_df['title']))
        user_movie_matrix = pd.DataFrame(0, index=dummy_users.keys(), columns=all_titles)
        for user, titles in dummy_users.items():
            for title in titles:
                if title in user_movie_matrix.columns:
                    user_movie_matrix.loc[user, title] = 1

        sim_matrix = cosine_similarity(user_movie_matrix)
        sim_df = pd.DataFrame(sim_matrix, index=user_movie_matrix.index, columns=user_movie_matrix.index)
        similar_users = sim_df[username].drop(username).sort_values(ascending=False)
        top_user = similar_users.idxmax()
        liked_by_top_user = user_movie_matrix.loc[top_user]
        liked_titles = liked_by_top_user[liked_by_top_user == 1].index.tolist()
        suggested = [title for title in liked_titles if title not in selected_movies]
        st.dataframe(filtered_df[filtered_df['title'].isin(suggested)][['title', 'genres_clean', 'vote_average']].head(5))

    with tabs[5]:
        st.subheader("📈 Evaluation Dashboard")
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                user_feedback = json.load(f)
            feedback_summary = pd.Series(user_feedback).value_counts()
            st.bar_chart(feedback_summary)
        else:
            st.info("No ratings submitted yet.")

    with tabs[2]:
        mx = df['popularity'].quantile(0.90)
        st.subheader("🌟 Movie of the Day")
        lesser_known = df[(df['vote_average'] > 7) & (df['popularity'] < mx)]
        pick = lesser_known.sample(1).iloc[0]
        st.markdown(f"### **{pick['title']}** ({pick['year']})")
        st.markdown(f"**Genres:** {pick['genres_clean']}")
        st.markdown(f"**Overview:** {pick['overview']}")
        st.markdown(f"**Rating:** {pick['vote_average']} ⭐")
        if pick['homepage'] != 'none':
            st.markdown(f"[**Watch Trailer or More Info**]({pick['homepage']})")

    with tabs[4]:
        st.subheader("📊 Visual Insights")

        genre_counts = pd.Series(", ".join(df['genres_clean']).split(", ")).value_counts().head(10)
        fig1, ax1 = plt.subplots()
        genre_counts.plot(kind='barh', ax=ax1)
        ax1.set_title("Top Genres")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.scatter(df['vote_average'], df['popularity'], alpha=0.3)
        ax2.set_xlabel("Vote Average")
        ax2.set_ylabel("Popularity")
        ax2.set_title("Rating vs Popularity")
        st.pyplot(fig2)

        year_counts = df['year'].value_counts().sort_index()
        fig3, ax3 = plt.subplots()
        year_counts.plot(kind='line', ax=ax3)
        ax3.set_title("Movies Released per Year")
        ax3.set_xlabel("Year")
        ax3.set_ylabel("Number of Movies")
        st.pyplot(fig3)

        top_directors = df.groupby('director')['vote_average'].mean().sort_values(ascending=False).head(10)
        fig4, ax4 = plt.subplots()
        top_directors.plot(kind='bar', ax=ax4)
        ax4.set_title("Top Directors by Average Rating")
        st.pyplot(fig4)

        lang_counts = df['original_language'].value_counts().head(10)
        fig5, ax5 = plt.subplots()
        lang_counts.plot(kind='pie', ax=ax5, autopct='%1.1f%%')
        ax5.set_ylabel('')
        ax5.set_title("Most Common Languages")
        st.pyplot(fig5)

        fig6, ax6 = plt.subplots()
        df['runtime'].dropna().plot(kind='hist', bins=30, ax=ax6)
        ax6.set_title("Movie Runtime Distribution")
        ax6.set_xlabel("Runtime (minutes)")
        st.pyplot(fig6)

        st.subheader("🔤 Word Cloud of Liked Movies")
        liked_overviews = df[df['title'].isin(selected_movies)]['overview'].str.cat(sep=' ')
        if liked_overviews:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(liked_overviews)
            fig7, ax7 = plt.subplots(figsize=(10, 5))
            ax7.imshow(wordcloud, interpolation='bilinear')
            ax7.axis("off")
            st.pyplot(fig7)

    with tabs[3]:
        st.subheader("🎯 Your Watchlist")
        watchlist = []
        if os.path.exists(watchlist_file):
            with open(watchlist_file, 'r') as f:
                watchlist = json.load(f)
        selected_watchlist = st.multiselect("Add movies to your watchlist:", movie_options, default=watchlist, placeholder="Search for movies")
        if st.button("Add to Watchlist"):
            with open(watchlist_file, 'w') as f:
                json.dump(selected_watchlist, f)
            st.success("Added to Watchlist!")
        if selected_watchlist:
            st.write("📽 Movies in your watchlist:")
            st.dataframe(df[df['title'].isin(selected_watchlist)][['title', 'genres_clean', 'year', 'vote_average']])

        st.subheader("📌 Interaction History")
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                history_data = json.load(f)
            st.write("Movies you rated:")
            history_df = pd.DataFrame(history_data.items(), columns=['Title', 'Your Rating'])
            st.dataframe(history_df)
        else:
            st.info("No rating history yet.")

        def plot_genre_radar(user_movies):
            genre_pool = [g for m in user_movies for g in df[df['title'] == m]['genres_clean'].values[0].split(", ")]
            genre_counts = pd.Series(genre_pool).value_counts()
            radar_df = pd.DataFrame({
                'Genre': genre_counts.index,
                'Count': genre_counts.values
            })
            fig = px.line_polar(radar_df, r='Count', theta='Genre', line_close=True, title="Your Genre Profile",
                                height=500)
            st.plotly_chart(fig, use_container_width=True)

        if username and os.path.exists(profile_file):
            with open(profile_file, 'r') as f:
                data = json.load(f)
            if data['liked_movies']:
                st.subheader("📊 Your Genre Radar")
                plot_genre_radar(data['liked_movies'])

else:
    st.warning("Please enter a username to use Nova.")
