
# Numpy and Pandas
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn

# OS things
from pathlib import Path

# Typing
from typing import Tuple, Dict, Union

from jup.examples.recsys_models.neutral_cf.neutral_cf import NeutralCF
from jup.examples.recsys_models.neutral_cf.neutral_cf import InnerLayer

# 
from tqdm import tqdm
import copy

DATA_PATH = "/Users/tien/Documents/PythonEnvs/pytorch/jup/examples/recsys_models/neutral_cf/data/movielens/ml-1m"
path_object = Path(DATA_PATH)

__all__ = ['run']

def read_data(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    
    print("files")
    print(files)
    
    return files['ratings'], files['movies'] 

def create_dataset(ratings) -> Dict[str, Union[int, pd.DataFrame, Dict]]:
    """
    Because of a high cardinality, we want to reduce the size of userId, and movieIds
    """
    
    unique_users = ratings['userId'].unique()
    user_to_index = {u: i for i, u in enumerate(unique_users)}
    new_users = ratings.userId.map(user_to_index)
    new_users = new_users.astype(int)
    
    unique_movies = ratings['movieId'].unique()
    movie_to_index = {m: i for i, m in enumerate(unique_movies)}
    new_movies = ratings.movieId.map(movie_to_index)
    
    n_users = int(unique_users.shape[0])
    m_movies = int(unique_movies.shape[0])
    
    # create a new data frame
    X = pd.DataFrame(
        {
            'user_id': new_users,
            'movie_id': new_movies  
        }
    )   
    
    y = ratings['rating'].astype(np.float32)
    
    return { 
            'X': X,
            'y': y,
            'n_users': n_users,
            'm_movies': m_movies,
            'user_to_index': user_to_index,
            'movie_to_index': movie_to_index
        }
    
class RatingsIterator:
    """
        We return the iterator of (user_id, movie_id, rating)
    """
    
    def __init__(self, 
                 user_movie_pairs: pd.DataFrame,
                 ratings: pd.DataFrame,
                 batch_size=32,
                 shuffle=True):
        
        # need to use np.asarray here to set back the indices
        self.user_movie_pairs = np.asarray(user_movie_pairs)
        self.ratings = np.asarray(ratings)
        
        if shuffle:
            # X.shape[0] is an interger, so we just want to return a list of indices here
            indices = np.random.permutation(user_movie_pairs.shape[0])
            
            self.user_movie_pairs = self.user_movie_pairs[indices]
            self.ratings = self.ratings[indices]
            
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_batches = int(np.ceil(ratings.shape[0] / self.batch_size))
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
        
        start_index, end_index = k*self.batch_size, (k+1)*self.batch_size
        
        return self.user_movie_pairs[start_index:end_index], self.ratings[start_index:end_index]
        

def batches(user_rating_matrix: pd.DataFrame, ratings: pd.DataFrame, batch_size=32, shuffle=True) -> Tuple[torch.Tensor, torch.Tensor]:
    iterator = RatingsIterator(
        user_movie_pairs=user_rating_matrix,
        ratings=ratings,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    for xb, yb in iterator:
        xb = torch.LongTensor(xb)
        yb = torch.FloatTensor(yb)
        
        '''
        we got this issue:
        UserWarning: Using a target size (torch.Size([2000])) that is different to the input size (torch.Size([2000, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
        return F.mse_loss(input, target, reduction=self.reduction)
        
        2000 is batch_size, below is the fix
        https://stackoverflow.com/questions/65219569/pytorch-gives-incorrect-results-due-to-broadcasting
        '''
        
        # torch.tensor.view is to return with a different shape
        new_shape = (len(yb), 1)
        yb = yb.view(new_shape)
        
        yield xb, yb
        
        
def run():
    
    ratings, movies = read_data(path_object)
    dataset = create_dataset(ratings)
    
    num_users = int(dataset['n_users'])
    num_movies = int(dataset['m_movies'])
    user_movie_pairs = dataset['X']
    
    # print("user movie pairs")
    # print(user_movie_pairs)
    
    # prepare to train model
    x_train, x_valid, y_train, y_valid = train_test_split(user_movie_pairs, dataset['y'], test_size=0.2)
    
    data = dict()
    data['train'] = (x_train, y_train)
    data['val'] = (x_valid, y_valid)
    dataset_sizes = {
        'train': len(x_train),
        'val': len(x_valid)
    }
    
    # model and training
    hidden_layers = []

    for num_in, dropout in zip(
        [100]*3,
        [0.5]*3
    ):
        hidden_layers.append(
            InnerLayer(
                in_features=num_in,
                drop_out=dropout
            )
        )
    
    # prepare to train models
    model = NeutralCF(
        n_users=num_users,
        n_items=num_movies,
        embedding_dim=150,
        inner_layers=hidden_layers,
        dropout_prob=0.05,
    )
    
    lr = 1e-3
    wd = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss(reduction='sum')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    patience = 10
    
    # to capture the loss
    lossses =   [] # capture the losses for each of epoch
    losses_test = [] # capture the loses for teh test data
    
    iterations = []
    iter = 0
    N_EPOCHS = 1000
    no_improvments = 0
    
    best_loss = np.inf
    
    
    # starting to train the model
    for epoch in tqdm(range(int(N_EPOCHS)), desc='Training Epochs'):
        stats = {'epoch': epoch + 1, 'total': N_EPOCHS}
        
        for phase in ['train', 'val']:
            training = (phase == 'train')
            
            running_loss = 0
            n_batches = 0
            
            x_data, y_data = data[phase]
            
            for batch in batches(x_data, y_data, batch_size=2000, shuffle=True):
                x_batch, y_batch = [b.to(device) for b in batch]
                optimizer.zero_grad()

                # only compute gradients during 'train' phase
                with torch.set_grad_enabled(training):
                    # x_batch[:,0]: take everything in the first (0-index row): users
                    user_ids = x_batch[:, 0] 
                    item_ids = x_batch[:, 1]
                            
                    outputs = model(user_ids, item_ids)
                
                    loss = criterion(outputs, y_batch)
                    
                    # dont' update weights and rates when in 'val' phase
                    if training:
                        # loss.backward() computes dloss/dx for every parameter 
                        # x which has requires_grad=True. 
                        # These are accumulated into x.grad for every parameter x. In pseudo-code:
                        # x.grad += dloss/dx
                        # source: https://discuss.pytorch.org/t/what-does-the-backward-function-do/9944
                        loss.backward()
                
                        # optimizer.step updates the value of x using the gradient x.grad. 
                        # For example, the SGD optimizer performs:
                        # x += -lr * x.grad
                        # optimizer.zero_grad() -> should not put it here as the eval value is really bad
                        optimizer.step()
                    
                running_loss += loss.item()
        
            epoch_loss = running_loss / dataset_sizes[phase]
            stats[phase] = epoch_loss

            # early stopping: save weights of the best weights so far.
            if phase == "val":
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), "/Users/tien/Documents/PythonEnvs/pytorch/jup/examples/recsys_models/neutral_cf/best_model.pth")

                    best_weights = copy.deepcopy(model.state_dict())
                    no_improvments = 0
                else:
                    no_improvments += 1
                    
        if no_improvments >= patience:
            print('early stopping after epoch {epoch:03d}'.format(**stats))
            break
        print(f"epoch: {epoch} is done!")
                    
    
if __name__ == '__main__':
    # import doctest
    # doctest.testmod()
    run()
    