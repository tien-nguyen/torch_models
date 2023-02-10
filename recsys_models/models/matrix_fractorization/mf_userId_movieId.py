from torch import nn

# nn.Module is the base class for all the network

class MF(nn.Module):
    """
    Creates a dense network with embedding layers.
    
    Args:
    
        n_users:            
            Number of unique users in the dataset.

        n_movies: 
            Number of unique movies in the dataset.

        n_factors: 
            Number of columns in the embeddings matrix.     
    """
    
    def __init__(self, n_users, n_movies, n_factors=50):
        
        super().__init__() # because we subclass nn.Module
        
        # Build the matrix to store the embedding for n_users
        # Embedding: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        # A simple look up table that stores embeddingof a fixed dictionary and size.
        # n_users: n number of embeddings
        self.u = nn.Embedding(n_users, n_factors)
        
        # Build the matrix to store the embedding for m_movies
        # A  look up table that stores embeddings for n_movies
        self.m = nn.Embedding(n_movies, n_factors)
        
        # Initialize the embedding vector
        nn.init.normal_(self.u.weight, 0, 0.1)
        nn.init.normal_(self.m.weight, 0, 0.1)
        

        
    def forward(self, users, movies): 
        users_latent = self.u(users)
        movies_latent = self.m(movies)
        
        # Need to understand what dim = 1 here means (TODO)
        # Need to understand what this does here
        # Note:
        # The passed in information is the pair (user, movie).
        # Meaning: 
        #   users is a list of users
        #   movies is a list of movies
        #   The two lists have the same size
        #   At a given index i, we compute the rating 
        #   of a user @ users[i] with regards to a movie @ movies[i]
        # we do not do a matrix multiplication here as this is pair-wise computation
        # For more understanding, see the note here:
        # https://developers.google.com/machine-learning/recommendation/collaborative/matrix
        # Note: Observe that the (i,j) entry of U . V (transpose) of the embeddings of
        # a user i and item j
        # 
        # Below is the dot product of the two embeddinds with the same len
        # we sum() so that each user & movie pair only have one value (ratin)
        return (users_latent * movies_latent).sum(dim=1) 