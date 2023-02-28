# nn.Module is the base class for all the network

from itertools import zip_longest

import torch
from torch import nn


def get_list(n):
    """
        Give an integer or a list of integers, encapsulate it in a list
    """
    if isinstance(n, (int, float)):
        # just convert this to a list containing the value
        return [n]
    elif hasattr(n, '__iter__'): # this is a list
        return list(n) # encapsulat this as a list
    
    raise TypeError("layers configuration should be a a single number or a list of numbers")

class EmbeddingNetwork(nn.Module):
    """
    Creates a dense network with embedding layers.
    
    Args:
    
        n_users:            
            Number of unique users in the dataset.

        n_movies: 
            Number of unique movies in the dataset.

        n_factors: 
            Number of columns in the embeddings matrix.

        embedding_dropout: 
            Dropout rate to apply right after embeddings layer.

        hidden:
            A single integer or a list of integers defining the number of 
            units in hidden layer(s).

        dropouts: 
            A single integer or a list of integers defining the dropout 
            layers rates applied right after each of hidden layers.
            
    """
    
    def __init__(self, n_users, n_movies, n_factors=50, embedding_droppout_prob=0.02,
                 hidden=10, dropouts=0.2):
        
        super().__init__() # because we subclass nn.Module
        hidden = get_list(hidden) # encapsulate as a list of hidden layer
        dropouts = get_list(dropouts) # encapsulate as a list of dropouts
        
        n_last = hidden[-1] # getting the last layer
        
        # TODO (@tien): will need to move this one out as 
        # this is not preferable.
        def gen_layers(n_in):
            """
            A generator that yields a sequence of hidden layers and 
            their activations/dropouts.
            
            Note that the function captures `hidden` and `dropouts` 
            values from the outer scope.
            """
            nonlocal hidden, dropouts
            assert len(dropouts) <= len(hidden)
            
            for n_out, rate in zip_longest(hidden, dropouts):
                # Applies a linear transformation to the incoming data: y=xA^T + b
                # we do learn bout the additive bias
                # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

                yield nn.Linear(n_in, n_out)
                yield nn.ReLU()  # rectified linear unit function element-wise
                if rate is not None and rate > 0.:
                    yield nn.Dropout(rate)
                n_in = n_out
        
        # Build the matrix to store the embedding for n_users
        # Embedding: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        # A simple look up table that stores embeddingof a fixed dictionary and size.
        # n_users: n number of embeddings
        self.u = nn.Embedding(n_users, n_factors)
        
        # Build the matrix to store the embedding for m_movies
        # A  look up table that stores embeddings for n_movies
        self.m = nn.Embedding(n_movies, n_factors)
        
        # During training, randomly zeroes some of the elements of the input tensor with 
        # probability p using samples from a Bernoulli distribution. 
        # Each channel will be zeroed out independently on every forward call.
        self.drop = nn.Dropout(embedding_droppout_prob)
        
        # A Sequential container. Modules will be added to it in the order they are passed
        # in the constructor.
        # For the code below, we'll have somethings like
        # hiden = nn.Sequential(
        #  nn.Linear(...),
        #  nn.ReLU(),
        #  ...
        #  nn.Linear(...),
        #  nn.ReLU()
        # )
        self.hidden = nn.Sequential(*list(gen_layers(n_factors * 2)))
        
        # This is just the final layer - where the last output is just one number
        self.fc = nn.Linear(n_last, 1)
        self._init()
        
    def forward(self, users, movies, minmax=None):
        # Concatenates the given sequence of seq tensors in the given dimension. 
        # All tensors must either have the same shape (except in the concatenating dimension) or be empty.
        # dim=1: concat all the tensors horizontally
        # This is very simimilar to Neural Collaborative Filter (https://www.kaggle.com/code/jamesloy/deep-learning-based-recommender-systems)
        features = torch.cat([self.u(users), self.m(movies)], dim=1)
        
        # we still drop out at the probability of embedding_droppout_prob
        x = self.drop(features)
        
        # x is just a final layer
        x = self.hidden(x)
        
        # compute the sigmoid on the last layers
        out = torch.sigmoid(self.fc(x))
        if minmax is not None:
            min_rating, max_rating = minmax
            out = out*(max_rating - min_rating + 1) + min_rating - 0.5
        return out
    
    def _init(self):
        """
        Setup embeddings and hidden layers with reasonable initial values.
        """
        
        def init(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
                
        self.u.weight.data.uniform_(-0.05, 0.05)
        self.m.weight.data.uniform_(-0.05, 0.05)
        
        # Apply on all of hidden layers as well as the final layer
        self.hidden.apply(init)
        init(self.fc)