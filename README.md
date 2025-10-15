Movie Recommender System
Overview

The Movie Recommender System is a content-based recommendation application that suggests movies similar to a user-selected title. It uses Natural Language Processing (NLP) techniques to analyze movie plot summaries and identify related films based on textual similarity. The system is built with Python and Streamlit for an interactive user experience.

Features

Recommends the top five movies similar to a given title.

Uses TF-IDF vectorization and cosine similarity for NLP-based analysis.

Includes a “Surprise Me” feature that recommends random movies for exploration.

Precomputes similarity values for efficient performance.

Simple and intuitive Streamlit interface.

Dataset

The project uses the TMDB 5000 Movie Dataset from Kaggle.
Primary file: tmdb_5000_movies.csv
Optional file: tmdb_5000_credits.csv for additional cast and crew data.

Key attributes include:

title: Movie title used for recommendation input.

overview: Movie plot used for similarity calculation.

genres and keywords: Optional metadata for richer results.

How It Works

The dataset is loaded and cleaned using pandas.

TF-IDF vectorization converts each movie’s overview into numerical vectors.

Cosine similarity calculates how closely related movies are based on their plot content.

Users can input a movie title to receive top recommendations, or select “Surprise Me” to get a random movie suggestion.

Tech Stack

Python

pandas, numpy, scikit-learn

Streamlit (for web interface)

Usage

Install dependencies from requirements.txt.

Run the application using: streamlit run app.py
