import requests
import zipfile
import io
import pandas as pd


def fetch_ml_100k():
    """Fetches the MovieLens 100k Dataset.

    Returns
    -------
    pandas.dataframe
        Of shape `m by 3`, Columns - ``userId``, ``movieId``,``rating``
    """
    r = requests.get(
        'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip')
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()

    ratings_csv_path = 'ml-latest-small/ratings.csv'
    fields = ['userId', 'movieId', 'rating']
    df_ratings = pd.read_csv(ratings_csv_path)[fields]

    return df_ratings
