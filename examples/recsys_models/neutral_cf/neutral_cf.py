import torch 
from torch import nn

from collections import namedtuple
from typing import List


InnerLayer = namedtuple("InnerLayer", ["in_features", "drop_out"])
    

class NeutralCF(nn.Module):
    
    def __init__(self,
                 n_users: int,
                 n_items: int,
                 embedding_dim: int,
                 device="cpu",
                 dropout_prob: float = 0.02,
                 inner_layers: List[InnerLayer] = [],
                 ):
        
        super(NeutralCF, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.device = device 
        self.dropout_prob = dropout_prob
        self.inner_layers = inner_layers
        
        # embedding
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # dropout
        self.dropout = nn.Dropout(p=self.dropout_prob)
        
        # build a hidden layer
        # the reason that we have embedding_dim*2 is because 
        # this is for both user and item embeddings
        self.hidden = nn.Sequential(
            *self.generate_layers(self.embedding_dim*2)
        )

        # final layer - where the last output is just one number
        self.final_layer = nn.Linear(
            self.inner_layers[-1].in_features
            , 1)
        
        # initiailze weights
        self.hidden.apply(self.setup_initial_values)
        self.setup_initial_values(self.final_layer)
        
    def setup_initial_values(self, layer: nn.Module):
        if type(layer) == nn.Linear:
            torch.nn.init.xavier_uniform(layer.weight)
            
            # need to check what this means
            layer.bias.data.fill_(0.0)
        
    def generate_layers(self, num_input: int) -> List[nn.Module]:
        """
        Generate a list of layers that are used in the Sequential Module.
        Args:
            num_input (int): number of input,

        Returns:
            List[nn.Module]: list of nn.Module
        """
        layers = []
        for n_output, drop_out_rate in self.inner_layers:
            layers.append(nn.Linear(num_input, n_output))
            layers.append(nn.ReLU())
            if drop_out_rate > 0:
                layers.append(nn.Dropout(p=drop_out_rate))
            num_input = n_output
        return layers
    
    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        # Concatenates the given sequence of seq tensors in the given dimension. 
        # All tensors must either have the same shape (except in the concatenating dimension) or be empty.
        # dim=1: concat all the tensors horizontally
        # This is very simimilar to Neural Collaborative Filter (https://www.kaggle.com/code/jamesloy/deep-learning-based-recommender-systems)
        # embedding
        user_embeddings = self.user_embedding(users)
        item_embeddings = self.item_embedding(items)
        
        # concatenate
        features = torch.cat([user_embeddings, item_embeddings], dim=-1)
        
        # we still drop out at the probability of the embedding droppout prob
        x = self.dropout(features)
        
        # x is just a final layer
        x = self.hidden(x)
        
        # compute the sigmoid on the last layers
        return torch.sigmoid(self.final_layer(x))
        
        