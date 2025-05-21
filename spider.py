import os
import requests
import pandas as pd
from tqdm import tqdm
import time
import re

# Configuration
TMDB_API_KEY = ""  # Replace with your actual API key
ML_DATA_DIR = "./ml-1m/"
OUTPUT_DIR = "./poster/"
DELAY = 0.001  # Delay between API requests to avoid rate limiting

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_movie_title(title):
    """Extract the movie title without the year."""
    match = re.search(r"(.+)(\(\d{4}\))", title)
    if match:
        return match.group(1).strip()
    return title


def search_movie(title, year=None):
    """Search for a movie on TMDB API."""
    search_url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "include_adult": "false",
    }

    if year:
        params["year"] = year

    response = requests.get(search_url, params=params)

    if response.status_code == 200:
        results = response.json().get("results", [])
        if results:
            return results[0]  # Return the first (most relevant) result

    return None


def get_movie_year(title):
    """Extract year from movie title if present."""
    match = re.search(r"\((\d{4})\)", title)
    if match:
        return match.group(1)
    return None


def download_poster(movie_id, poster_path, filename):
    """Download the movie poster image."""
    if not poster_path:
        return False

    base_url = "https://image.tmdb.org/t/p/w154"  # Use w500 size
    poster_url = f"{base_url}{poster_path}"

    response = requests.get(poster_url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        return True
    return False


def main():
    # Load movies data
    last_processed_id = 3098
    movies_file = os.path.join(ML_DATA_DIR, "movies.dat")

    # MovieLens 1M dataset uses :: as delimiter and has no header
    # Define column names based on the dataset description
    movies_columns = ["MovieID", "Title", "Genres"]

    # Read with proper encoding
    movies_df = pd.read_csv(
        movies_file,
        sep="::",
        header=None,
        names=movies_columns,
        encoding="latin-1",
        engine="python",  # Use python engine for custom delimiter
    )

    # Keep track of successful downloads
    successful_downloads = 0

    # Process each movie
    for _, movie in tqdm(
        movies_df.iterrows(), total=len(movies_df), desc="Downloading posters"
    ):
        movie_id = movie["MovieID"]
        if movie_id <= last_processed_id:
            continue
        title = movie["Title"]

        # Clean title and extract year
        clean_title = clean_movie_title(title)
        year = get_movie_year(title)

        # Search for the movie
        tmdb_movie = search_movie(clean_title, year)

        if tmdb_movie and tmdb_movie.get("poster_path"):
            poster_path = tmdb_movie["poster_path"]
            poster_filename = os.path.join(OUTPUT_DIR, f"{movie_id}.jpg")

            # Download the poster
            success = download_poster(movie_id, poster_path, poster_filename)

            if success:
                successful_downloads += 1

        # Add delay to avoid rate limiting
        time.sleep(DELAY)

    print(f"Downloaded {successful_downloads} posters out of {len(movies_df)} movies")


if __name__ == "__main__":
    main()
