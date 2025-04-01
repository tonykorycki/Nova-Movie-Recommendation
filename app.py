import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# === PAGE SETUP ===
st.set_page_config(page_title="Nova Movie Recs", layout="wide")
st.title("🎬 Nova: Your Personal Movie Recommender")

# === LOAD DATA ===
@st.cache_data

def load_data():
    movies_df = pd.read_csv("tmdb_5000_movies.csv")
    credits_df = pd.read_csv("tmdb_5000_credits.csv")

    # Merge credits with movies on ID
    credits_df.rename(columns={'movie_id': 'id'}, inplace=True)
    merged_df = movies_df.merge(credits_df, on='id')

    merged_df.rename(columns={'title_x': 'title'}, inplace=True)

    merged_df['overview'] = merged_df['overview'].fillna('unknown')
    merged_df['homepage'] = merged_df['homepage'].fillna('none')
    merged_df['tagline'] = merged_df['tagline'].fillna('none')



    merged_df.dropna(axis=0, how='any', inplace=True)
    #print(merged_df.isnull().sum())


    # Parse genres and cast from JSON strings to usable lists
    def extract_names(json_str, key="name", max_items=3):
        try:
            data = json.loads(json_str.replace("'", '"'))
            return ", ".join([item[key] for item in data[:max_items]])
        except:
            return ""

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
    merged_df['director'] = merged_df['crew'].apply(extract_director)

    # Extract year
    merged_df['year'] = pd.to_datetime(merged_df['release_date'], errors='coerce').dt.year

    return merged_df

df = load_data()

# === TF-IDF DESCRIPTION VECTORIZER ===
tfidf = TfidfVectorizer(stop_words='english')
df['overview'] = df['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df['overview'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
title_to_index = pd.Series(df.index, index=df['title']).drop_duplicates()

# === USER PROFILE MANAGEMENT ===
st.sidebar.header("👤 User Profile")
username = st.sidebar.text_input("Enter your username:")
profile_file = f"profiles/{username}.json"

if not os.path.exists("profiles"):
    os.makedirs("profiles")

if username:
    if os.path.exists(profile_file):
        with open(profile_file, 'r') as f:
            profile_data = json.load(f)
    else:
        profile_data = {"liked_movies": [], "favorite_genres": []}

    # === PROFILE INPUT ===
    genres = sorted(set(", ".join(df['genres_clean']).split(", ")))
    selected_genres = st.sidebar.multiselect("Favorite genres", genres, default=profile_data['favorite_genres'])
    movie_options = df['title'].sort_values().tolist()
    selected_movies = st.sidebar.multiselect("Liked movies", movie_options, default=profile_data['liked_movies'])

    # Update profile data
    profile_data['liked_movies'] = selected_movies
    profile_data['favorite_genres'] = selected_genres

    with open(profile_file, 'w') as f:
        json.dump(profile_data, f)

    st.session_state["genres"] = selected_genres
    st.session_state["liked_movies"] = selected_movies

    # === PAGE NAVIGATION ===
    tabs = st.tabs(["Explore Movies", "Recommendations", "Movie of the Day", "Insights"])

    # === Explore Movies ===
    with tabs[0]:
        st.subheader("📚 Explore the TMDB Dataset")
        st.dataframe(df[['title', 'genres_clean', 'year', 'original_language', 'vote_average', 'vote_count']], use_container_width=True)

    # === Recommendations ===
    with tabs[1]:
        st.subheader("🤖 Personalized Recommendations")
        if selected_movies:
            indices = [title_to_index[movie] for movie in selected_movies if movie in title_to_index]
            sim_scores = sum(cosine_sim[i] for i in indices)
            sim_scores = list(enumerate(sim_scores))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = [i for i in sim_scores if i[0] not in indices]

            recommendations = []
            for i, score in sim_scores[:50]:
                movie = df.iloc[i]
                genre_match = sum(1 for genre in selected_genres if genre.strip() in movie['genres_clean'].split(", "))
                weight = 0.5 * score + 0.3 * (genre_match / len(selected_genres) if selected_genres else 0) + 0.2 * (movie['vote_average'] / 10)
                recommendations.append((movie['title'], movie['genres_clean'], movie['year'], movie['vote_average'], round(score, 3), round(weight, 3)))

            recommendations.sort(key=lambda x: x[-1], reverse=True)
            rec_df = pd.DataFrame(recommendations, columns=['Title', 'Genres', 'Year', 'Rating', 'Similarity', 'Score'])
            st.dataframe(rec_df.head(10), use_container_width=True)
        if selected_genres:
            st.markdown("Based on your favorite genres:")
            def genre_score(movie_genres):
                return sum(1 for genre in selected_genres if genre.strip() in movie_genres.split(", "))

            df['genre_score'] = df['genres_clean'].apply(genre_score)
            filtered = df[df['genre_score'] > 0].sort_values(by=['genre_score', 'vote_average'], ascending=False).head(10)
            st.dataframe(filtered[['title', 'genres_clean', 'year', 'vote_average']], use_container_width=True)

        if not recommendations and not selected_genres:
            st.info("Add liked movies or genres to get personalized recommendations.")


    # === Movie of the Day ===
    with tabs[2]:
        mx = df['popularity'].quantile(0.90)
        st.subheader("🌟 Movie of the Day")
        lesser_known = df[(df['vote_average'] > 7) & (df['popularity'] < mx)]
        pick = lesser_known.sample(1).iloc[0]
        st.markdown(f"### **{pick['title']}** ({pick['year']})")
        st.markdown(f"**Genres:** {pick['genres_clean']}")
        st.markdown(f"**Overview:** {pick['overview']}")
        st.markdown(f"**Rating:** {pick['vote_average']} ⭐")

     # === Visual Insights ===
    with tabs[3]:
        st.subheader("📊 Visual Insights")

        # Genre popularity
        genre_counts = pd.Series(", ".join(df['genres_clean']).split(", ")).value_counts().head(10)
        fig1, ax1 = plt.subplots()
        genre_counts.plot(kind='barh', ax=ax1)
        ax1.set_title("Top Genres")
        st.pyplot(fig1)

        # Ratings vs Popularity
        fig2, ax2 = plt.subplots()
        ax2.scatter(df['vote_average'], df['popularity'], alpha=0.3)
        ax2.set_xlabel("Vote Average")
        ax2.set_ylabel("Popularity")
        ax2.set_title("Rating vs Popularity")
        st.pyplot(fig2)

        # Movie count by year
        year_counts = df['year'].value_counts().sort_index()
        fig3, ax3 = plt.subplots()
        year_counts.plot(kind='line', ax=ax3)
        ax3.set_title("Movies Released per Year")
        ax3.set_xlabel("Year")
        ax3.set_ylabel("Number of Movies")
        st.pyplot(fig3)

        # Top directors by average rating
        top_directors = df.groupby('director')['vote_average'].mean().sort_values(ascending=False).head(10)
        fig4, ax4 = plt.subplots()
        top_directors.plot(kind='bar', ax=ax4)
        ax4.set_title("Top Directors by Average Rating")
        st.pyplot(fig4)

        # Language distribution
        lang_counts = df['original_language'].value_counts().head(10)
        fig5, ax5 = plt.subplots()
        lang_counts.plot(kind='pie', ax=ax5, autopct='%1.1f%%')
        ax5.set_ylabel('')
        ax5.set_title("Most Common Languages")
        st.pyplot(fig5)

        # Runtime distribution
        fig6, ax6 = plt.subplots()
        df['runtime'].dropna().plot(kind='hist', bins=30, ax=ax6)
        ax6.set_title("Movie Runtime Distribution")
        ax6.set_xlabel("Runtime (minutes)")
        st.pyplot(fig6)
else:
    st.warning("Please enter a username to use MovieMate.")
