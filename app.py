import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD, KNNBasic

st.title("🎬 Movie Recommendation System")


# Step 1: Load Data

@st.cache_data
def load_ratings():
    return pd.read_csv(
        "u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

@st.cache_data
def load_movies():
    movie_cols = [
        "movie_id", "title", "release_date", "video_release_date", "IMDb_URL",
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]
    return pd.read_csv(
        "u.item",
        sep="|",
        names=movie_cols,
        encoding="latin-1"
    )

ratings = load_ratings()
movies = load_movies()

# Merge ratings with titles
data = pd.merge(ratings, movies[["movie_id", "title"]], on="movie_id")


# Step 2: Build Matrices

# Movie-user rating matrix for item-based CF
movie_user_matrix = data.pivot_table(index="title", columns="user_id", values="rating").fillna(0)
similarity_matrix = cosine_similarity(movie_user_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=movie_user_matrix.index, columns=movie_user_matrix.index)

# Genre-based features for CBF
genre_cols = movies.columns[5:]
genre_features = movies[genre_cols].values
cosine_sim = cosine_similarity(genre_features, genre_features)


# Step 3: Train Collaborative Filtering Model (Surprise SVD)

@st.cache_resource
def train_cf_model():
    reader = Reader(line_format="user item rating timestamp", sep="\t", rating_scale=(1, 5))
    dataset = Dataset.load_from_df(ratings[["user_id", "movie_id", "rating"]], reader)
    trainset = dataset.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)
    return algo

cf_model = train_cf_model()


# Step 4: Recommendation Functions

# Item-based Collaborative Filtering
def recommend_movies_cf(movie_name, n=5):
    if movie_name not in similarity_df.index:
        return ["Movie not found"]
    sim_scores = similarity_df[movie_name].sort_values(ascending=False)
    return sim_scores.iloc[1:n+1].index.tolist()

# Content-Based Filtering
def recommend_movies_cbf(movie_title, n=5):
    idx = movies[movies["title"] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]["title"].tolist()

# Hybrid (CF + CBF)
def hybrid_recommend(user_id, movie_title, n=5, alpha=0.7):
    idx = movies[movies["title"] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:50]  # top 50 similar
    movie_indices = [i[0] for i in sim_scores]
    cbf_movies = movies.iloc[movie_indices][["movie_id", "title"]]
    cbf_scores = np.array([s[1] for s in sim_scores])

    cf_scores = []
    for mid in cbf_movies["movie_id"]:
        try:
            pred = cf_model.predict(user_id, mid).est
        except:
            pred = 0
        cf_scores.append(pred)
    cf_scores = np.array(cf_scores)

    final_scores = alpha * cf_scores + (1 - alpha) * cbf_scores
    cbf_movies["score"] = final_scores
    cbf_movies = cbf_movies.sort_values("score", ascending=False)
    return cbf_movies.head(n)["title"].tolist()


# Step 5: Streamlit UI

st.subheader("Sample of MovieLens 100k Data")
st.dataframe(data.head())

movie_list = movies["title"].tolist()
selected_movie = st.selectbox("🎥 Select a Movie:", movie_list)

method = st.radio(
    "Choose Recommendation Method:",
    ("Collaborative Filtering (Item-based)", "Content-Based Filtering", "Hybrid (CF + CBF)")
)

if selected_movie:
    if method == "Collaborative Filtering (Item-based)":
        st.subheader(f"🔎 Top 5 similar movies to '{selected_movie}' (CF):")
        recs = recommend_movies_cf(selected_movie)
        for i, m in enumerate(recs, 1):
            st.write(f"{i}. {m}")

    elif method == "Content-Based Filtering":
        st.subheader(f"🎭 Top 5 similar movies to '{selected_movie}' (CBF):")
        recs = recommend_movies_cbf(selected_movie)
        for i, m in enumerate(recs, 1):
            st.write(f"{i}. {m}")

    else:  # Hybrid
        user_id = st.number_input("Enter User ID for Hybrid (1–943)", min_value=1, max_value=943, value=1)
        alpha = st.slider("Weight for Collaborative Filtering", 0.0, 1.0, 0.7)
        st.subheader(f"⚡ Hybrid Recommendations for User {user_id} and '{selected_movie}':")
        recs = hybrid_recommend(user_id, selected_movie, alpha=alpha)
        for i, m in enumerate(recs, 1):
            st.write(f"{i}. {m}")
