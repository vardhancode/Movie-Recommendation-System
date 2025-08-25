# Movie-Recommendation-System
This project is a Movie Recommender System built using the MovieLens 100k dataset.
It implements three types of recommenders:

-Collaborative Filtering (CF)

  Uses user ratings to recommend movies.
  Example: “People who liked Movie A also liked Movie B.”

-Content-Based Filtering (CBF)

  Uses movie metadata (like genres) to recommend similar movies.
  Example: “You liked Toy Story → you may also like other animated/family movies.”

-Hybrid (CF + CBF)

  Combines both CF and CBF with a weight factor.
  Example: “You liked Toy Story, and people with similar taste also liked Aladdin → recommendation blends both.”
  

 !!Installation:
  -Clone the repo / create a folder
  git clone <repo-url>
  cd Movie-Recommender


  -Install required packages
  pip install streamlit pandas scikit-learn scikit-surprise


!!How It Works:

1. Collaborative Filtering (CF)

  Uses Surprise library (SVD, KNNBasic).

  Learns from user-item rating patterns.

  Example:
  User 1 rated Star Wars and Empire Strikes Back highly.
  The system predicts User 1 might also like Return of the Jedi.

2. Content-Based Filtering (CBF)

  Uses movie genre vectors.
  Calculates cosine similarity between movies.
  Example:
  If you liked Toy Story (Animation, Children’s, Comedy), you’ll get other animated/family movies.

3. Hybrid Recommender

  Combines CF and CBF.

  Formula:
  final_score = α * CF_score + (1 - α) * CBF_score
  α (alpha) = weight for CF.
  α close to 1 → more CF.
  α close to 0 → more CBF.

Fixes weaknesses:

  CF suffers from cold start (new movies).
  CBF ignores crowd wisdom.
  Hybrid balances both.
  

!!Running the App:

  Run the app:
  streamlit run app.py


  In the browser:
  You’ll see dataset preview. Select a movie from dropdown. Choose CF, CBF, or Hybrid.

  If Hybrid:
  Enter User ID (1–943).
  Adjust α (weight for CF) with the slider.
  Example:
  Select movie: Toy Story (1995)

Method:

  CF → recommends movies that similar users rated highly.
  CBF → recommends other family/animation movies.
  Hybrid (α = 0.7, User ID = 196) → blends both.

