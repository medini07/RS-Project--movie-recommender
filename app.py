import os
import json
import random
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from difflib import get_close_matches


APP_TITLE = "🎬 Movie Recommender System"
DATA_FILE_CANDIDATES = [
    "tmdb_5000_movies.csv",
    os.path.join("data", "tmdb_5000_movies.csv"),
    os.path.join("archive", "tmdb_5000_movies.csv"),
]
HISTORY_CSV = "search_history.csv"


@st.cache_data(show_spinner=False)
def load_movies_dataframe() -> pd.DataFrame:
    """Load movies CSV from known locations, clean, and return DataFrame.

    Expects at least `title` and `overview`. Keeps optional `genres`, `keywords` when present.
    """
    last_error: Optional[str] = None
    for candidate in DATA_FILE_CANDIDATES:
        if os.path.exists(candidate):
            try:
                df = pd.read_csv(candidate)
                break
            except Exception as exc:  # noqa: BLE001 - show useful message in the UI
                last_error = f"Failed reading '{candidate}': {exc}"
                continue
    else:
        message = (
            "Dataset not found. Please place 'tmdb_5000_movies.csv' in the project root "
            "or in a 'data/' folder."
        )
        if last_error:
            message += f"\nLast read error: {last_error}"
        raise FileNotFoundError(message)

    # Keep only relevant columns if present
    keep_cols = [col for col in ["id", "title", "overview", "genres", "keywords", "vote_average"] if col in df.columns]
    df = df[keep_cols].copy()

    # Drop rows without an overview
    if "overview" in df.columns:
        df = df.dropna(subset=["overview"]).reset_index(drop=True)
    else:
        raise ValueError("The dataset must include an 'overview' column.")

    # Ensure there is a title column
    if "title" not in df.columns:
        raise ValueError("The dataset must include a 'title' column.")

    # Normalize title whitespace
    df["title"] = df["title"].astype(str).str.strip()

    return df


def _parse_genres_cell(genres_cell: str) -> List[str]:
    """Parse the TMDB 'genres' cell which is often a JSON-like string.

    Returns a list of genre names. If parsing fails, returns an empty list.
    """
    if not isinstance(genres_cell, str) or not genres_cell.strip():
        return []

    # Try JSON load first (many datasets have proper JSON)
    try:
        parsed = json.loads(genres_cell)
        if isinstance(parsed, list):
            names = [g.get("name") for g in parsed if isinstance(g, dict) and g.get("name")]
            return [str(name) for name in names]
    except Exception:
        pass

    # Fallback: very rough parsing for stringified lists of dicts
    try:
        # Example: "[{\"id\": 28, \"name\": \"Action\"}, ...]"
        # Remove braces and split by 'name'
        names: List[str] = []
        pieces = genres_cell.split("'name': ") if "'name': " in genres_cell else genres_cell.split('"name": ')
        for piece in pieces[1:]:
            # piece starts with '"Action"}, {...' or "'Action'}, {..."
            if piece:
                name = piece.split(",")[0].strip().strip("'\"")
                if name:
                    names.append(name)
        return names
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def extract_all_genres(movies_df: pd.DataFrame) -> List[str]:
    if "genres" not in movies_df.columns:
        return []
    all_genres: List[str] = []
    for cell in movies_df["genres"].fillna(""):
        all_genres.extend(_parse_genres_cell(cell))
    return sorted(sorted(set(g for g in all_genres if g)))


@st.cache_resource(show_spinner=False)
def build_tfidf_and_similarity(overview_series: pd.Series) -> Tuple[TfidfVectorizer, np.ndarray]:
    """Create TF-IDF over overviews and compute cosine similarity matrix.

    Returns the vectorizer and the dense similarity matrix for fast lookup.
    """
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(overview_series.astype(str))
    # Use linear kernel which is equivalent to cosine similarity on L2-normalized TF-IDF vectors
    similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix).astype(np.float32)
    return vectorizer, similarity_matrix


@st.cache_data(show_spinner=False)
def build_title_index_map(titles: pd.Series) -> dict:
    """Map lowercase titles to their row indices. If duplicates, keep first occurrence."""
    title_to_index: dict = {}
    for idx, title in enumerate(titles.astype(str)):
        key = title.strip().lower()
        if key not in title_to_index:
            title_to_index[key] = idx
    return title_to_index


def find_best_title_match(query: str, choices: List[str]) -> Optional[str]:
    """Return the closest matching title using fuzzy matching; exact-insensitive first."""
    if not query:
        return None
    query_norm = query.strip().lower()
    if query_norm in choices:
        return query_norm
    matches = get_close_matches(query_norm, choices, n=1, cutoff=0.6)
    return matches[0] if matches else None


def filter_by_genre(indices: List[int], movies_df: pd.DataFrame, selected_genre: Optional[str]) -> List[int]:
    if not selected_genre or "genres" not in movies_df.columns:
        return indices
    filtered: List[int] = []
    for idx in indices:
        genres_here = _parse_genres_cell(movies_df.at[idx, "genres"]) if idx < len(movies_df) else []
        if selected_genre in genres_here:
            filtered.append(idx)
    return filtered


def recommend(
    movie_title: str,
    movies_df: pd.DataFrame,
    similarity_matrix: np.ndarray,
    title_to_index: dict,
    genre_filter: Optional[str] = None,
    top_k: Optional[int] = None,
) -> List[Tuple[str, Optional[float]]]:
    """Recommend similar movies by overview similarity.

    Returns a list of (title, score) pairs.
    """
    if not movie_title:
        return []

    all_keys = list(title_to_index.keys())
    best_key = find_best_title_match(movie_title, all_keys)
    if best_key is None:
        return []

    idx = title_to_index[best_key]
    scores = list(enumerate(similarity_matrix[idx]))
    # Exclude the movie itself (index match)
    scores = [pair for pair in scores if pair[0] != idx]
    # Sort by descending similarity
    scores.sort(key=lambda x: x[1], reverse=True)

    # Optionally filter by genre
    candidate_indices = [i for i, _ in scores]
    candidate_indices = filter_by_genre(candidate_indices, movies_df, genre_filter)

    # Rebuild list with scores in the preserved order
    top_scored: List[Tuple[int, float]] = []
    for i, s in scores:
        if i in candidate_indices:
            top_scored.append((i, float(s)))
            if top_k is not None and len(top_scored) >= max(top_k, 0):
                break

    results: List[Tuple[str, Optional[float]]] = []
    for i, s in top_scored:
        title = str(movies_df.at[i, "title"]) if i < len(movies_df) else ""
        results.append((title, s))
    return results


def save_search_history(query: str, results: List[Tuple[str, Optional[float]]]) -> None:
    try:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "query": query,
            "results": "; ".join([title for title, _ in results]),
        }
        exists = os.path.exists(HISTORY_CSV)
        pd.DataFrame([row]).to_csv(HISTORY_CSV, mode="a", index=False, header=not exists)
    except Exception:
        # Silently ignore history write failures to not break the UX
        pass


def get_tmdb_api_key() -> Optional[str]:
    # Prefer Streamlit secrets if available, fallback to environment variable
    try:
        secret_key = st.secrets.get("TMDB_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        secret_key = None
    return secret_key or os.environ.get("TMDB_API_KEY")


def fetch_poster_urls(titles: List[str]) -> List[Optional[str]]:
    api_key = get_tmdb_api_key()
    if not api_key:
        return [None for _ in titles]

    base_url = "https://api.themoviedb.org/3"
    image_base = "https://image.tmdb.org/t/p/w342"
    urls: List[Optional[str]] = []
    for title in titles:
        try:
            resp = requests.get(
                f"{base_url}/search/movie",
                params={
                    "api_key": api_key,
                    "query": title,
                    "language": "en-US",
                    "include_adult": "false",
                },
                timeout=8,
            )
            if resp.status_code == 401 or resp.status_code == 403:
                # Invalid or unauthorized key – stop further calls
                return [None for _ in titles]
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results") or []
                if results:
                    poster_path = results[0].get("poster_path")
                    urls.append(image_base + poster_path if poster_path else None)
                else:
                    urls.append(None)
            else:
                urls.append(None)
        except Exception:
            urls.append(None)
    return urls


def pick_random_title(movies_df: pd.DataFrame) -> Optional[str]:
    if movies_df.empty:
        return None
    return str(movies_df.sample(1, random_state=random.randint(0, 1_000_000)).iloc[0]["title"]).strip()


def main() -> None:
    st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

    # Optional local styling
    if os.path.exists("style.css"):
        try:
            with open("style.css", "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        except Exception:
            pass

    st.title(APP_TITLE)
    st.caption("Enter a movie title to get similar recommendations based on plot summaries.")

    # Load data and build model pieces
    try:
        movies_df = load_movies_dataframe()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    all_genres = extract_all_genres(movies_df)
    _, similarity_matrix = build_tfidf_and_similarity(movies_df["overview"])
    title_to_index = build_title_index_map(movies_df["title"])

    # Sidebar controls
    with st.sidebar:
        st.subheader("Options")
        selected_genre = None
        if all_genres:
            selected_genre = st.selectbox("Filter by genre (optional)", ["(none)"] + all_genres)
            if selected_genre == "(none)":
                selected_genre = None

        min_similarity = st.slider(
            "Minimum similarity",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Only show recommendations with similarity at or above this threshold.",
        )

        st.markdown("---")
        if st.button("Surprise Me 🎲"):
            random_title = pick_random_title(movies_df)
            if random_title:
                st.session_state["last_query"] = random_title
                st.toast(f"Surprise pick: {random_title}")

        tmdb_key_present = bool(get_tmdb_api_key())
        if tmdb_key_present:
            st.caption("Posters enabled via TMDB API Key.")
        else:
            st.caption("Set TMDB_API_KEY env var or Streamlit secret to show posters.")

        st.markdown("---")
        with st.expander("About"):
            st.write(
                "This demo uses TF-IDF + cosine similarity on movie overviews from the TMDB 5000 dataset."
            )

    # Main interaction
    default_query = st.session_state.get("last_query", "")
    movie_query = st.text_input("Movie title", value=default_query, placeholder="e.g., Avatar")
    recommend_clicked = st.button("Recommend")

    if recommend_clicked:
        with st.spinner("Finding similar movies..."):
            results = recommend(
                movie_query,
                movies_df,
                similarity_matrix,
                title_to_index,
                genre_filter=selected_genre,
                top_k=None,
            )

        # Keep only results with similarity >= selected threshold, sorted descending by score
        filtered_results = sorted(
            [(title, score) for title, score in results if (score or 0.0) >= min_similarity],
            key=lambda x: (x[1] or 0.0),
            reverse=True,
        )

        if not filtered_results:
            st.info(
                f"No results at or above {int(min_similarity * 100)}% similarity. Showing top 5 closest instead."
            )
            # Fallback: show top-5 overall (already sorted by similarity)
            filtered_results = results[:5]

        save_search_history(movie_query, results)
        st.session_state["last_query"] = movie_query

        titles_only = [title for title, _ in filtered_results]
        poster_urls = fetch_poster_urls(titles_only)
        if all(url is None for url in poster_urls):
            if tmdb_key_present:
                st.warning(
                    "Could not load posters from TMDB even though an API key is set. "
                    "Please verify the key and network connectivity."
                )

        st.subheader("Recommended movies:")
        cols = st.columns(5, gap="small") if titles_only else []
        for i, (title, score) in enumerate(filtered_results):
            col = cols[i % len(cols)] if cols else None
            score_pct = int(round((score or 0.0) * 100))
            if col is not None:
                with col:
                    if poster_urls[i]:
                        st.image(poster_urls[i], use_container_width=True)
                    st.markdown(f"**{title}**")
                    st.caption(f"Similarity: {score_pct}%")
            else:
                st.write(f"{i+1}. {title} ({score_pct}%)")


if __name__ == "__main__":
    main()


