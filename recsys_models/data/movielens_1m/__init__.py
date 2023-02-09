import pandas as pd
import numpy as np
from pathlib import Path 
import math

########################
# MovieLens - 1 M data #
########################

def read_data(path="/Users/tien/Documents/PythonEnvs/pytorch/jup/recsys_models/data/movielens_1m/ml-1m/"):
    
    path = Path(path)
    
    files = {}
    for filename in path.glob('*'):
        if filename.suffix == '.csv':
            files[filename.stem] = pd.read_csv(filename)
        elif filename.suffix == '.dat':
            if filename.stem == 'ratings':
                columns = ['userId', 'movieId', 'rating', 'timestamp']
            else:
                columns = ['movieId', 'title', 'genres']
            data = pd.read_csv(filename, sep='::', names=columns, engine='python', encoding='latin-1')
            files[filename.stem] = data
    return files['ratings'], files['movies']


def tabular_preview(ratings, n=15):
    """Creates a cross-tabular view of users vs movies."""
    
    user_groups = ratings.groupby('userId')['rating'].count()
    top_users = user_groups.sort_values(ascending=False)[:15]

    movie_groups = ratings.groupby('movieId')['rating'].count()
    top_movies = movie_groups.sort_values(ascending=False)[:15]

    top = (
        ratings.
        join(top_users, rsuffix='_r', how='inner', on='userId').
        join(top_movies, rsuffix='_r', how='inner', on='movieId'))

    return pd.crosstab(top.userId, top.movieId, top.rating, aggfunc=np.sum)


def create_dataset(ratings, top=None):
    """
        Not entirely sure what this does. The returned dataframes are just
        the splitted of the ratings.
    """
    if top is not None:
        ratings.groupby('userId')['rating'].count()
    
    unique_users = ratings.userId.unique()
    user_to_index = {user_id: index for index, user_id in enumerate(unique_users)}
    new_users = ratings.userId.map(user_to_index)
    
    unique_movies = ratings.movieId.unique()
    movie_to_index = {movie_id: index for index, movie_id in enumerate(unique_movies)}
    new_movies = ratings.movieId.map(movie_to_index)
    
    n_users = unique_users.shape[0]
    n_movies = unique_movies.shape[0]
    
    X = pd.DataFrame({'user_id': new_users, 'movie_id': new_movies})
    y = ratings['rating'].astype(np.float32)
    return (n_users, n_movies), (X, y), (user_to_index, movie_to_index)

# Iterator
class RatingsIterator:
    
    def __init__(self, user_movie_matrix, ratings, batch_size=32, shuffle=True):
        
        user_movie_matrix, ratings = np.asarray(user_movie_matrix), np.asarray(ratings)
        
        if shuffle:
            # X.shape[0] is an interger, so we just want to return a list of indices here
            index = np.random.permutation(user_movie_matrix.shape[0])
            
            user_movie_matrix = user_movie_matrix[index]
            ratings = ratings[index]
            
        self.user_movie_matrix = user_movie_matrix
        self.ratings = ratings
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_batches = int(math.ceil(user_movie_matrix.shape[0] // batch_size))
        self._current = 0
        
    def __iter__(self):
        return self
        
    def __next__(self):
        return self.next()
    
    def next(self):
        
        if self._current >= self.n_batches:
            raise StopIteration()
    
        k = self._current
        self._current += 1
        bs = self.batch_size
        
        start_index, end_index = k*bs, (k+1)*bs
        
        return self.user_movie_matrix[start_index:end_index], self.ratings[start_index:end_index]