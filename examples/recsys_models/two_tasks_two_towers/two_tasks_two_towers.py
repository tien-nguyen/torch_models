"""
In this work, we are buiding a two-tasks-two-towers model.
These tasks are shared the same bottom DNN.

The two tasks are:
    rating: regression
    sharing: share or not share
"""

# import torch
import torch.nn as nn
import torch

#
from typing import List, Union, Dict

from jup.examples.recsys_models.two_tasks_two_towers.features import SparseFeature, DenseFeature
from jup.examples.recsys_models.two_tasks_two_towers.features import build_input_feature_column_index


from dataclasses import dataclass


@dataclass
class MLPConfig:
    input_dims: int
    hidden_dims: List[int]
    activation='relu'
    use_bn=False
    dropout_rate=0.1
    init_std=0.0001
    seed=1024
    
    
class Identity(nn.Module):

    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs
    
def activation_layer(name: str, hidden_size: int) -> nn.Module:
    """Construct activation layers
    Args:
        name: str or nn.Module, name of activation function
        hidden_size: int, used for Dice activation // Data Adaptive Activation Function
    Return:
        act_layer: activation_layer
    """
    layers = {
        'sigmoid': nn.Sigmoid(),
        'linear': Identity(),
        'relu': nn.ReLU(inplace=True),
        'prelu': nn.PReLU(),
    }
        
    if isinstance(name, str):
        name = name.lower()
        return layers[name]
    elif isinstance(name, nn.Module):
        return name()
    else:
        raise NotImplementedError
    
    
def create_embedding_matrix(
    sparse_features: List[SparseFeature],
    init_stds=0.001,
    device="cpu"
) -> nn.ModuleDict:
    """
    It is a common practice that for sparse features, we create
    embedding dictionary for each sparse feature.
    
    Return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}
    """
    # reading:
    # https://medium.com/@gautam.e/what-is-nn-embedding-really-de038baadd24
    embedding_dict = nn.ModuleDict(
        {
            feature.embedding_name: nn.Embedding(
                feature.vocabulary_size, feature.embedding_dim
            ) for feature in sparse_features
        }
    )
    
    # we need to initialize the embeddings
    for tensor in embedding_dict.values():
        tensor: torch.Tensor
        nn.init.normal_(tensor.weight, mean=0, std=init_stds)
    
    return embedding_dict.to(device)
                

def compute_input_dim(dense_features: List[DenseFeature],
                      sparse_features: List[SparseFeature]) -> int:
    '''
        To compute input dimensions
    '''

    dense_feature_dim = sum(
        map(lambda x : x.dimension, dense_features)
    )
        
    sparse_feature_dim = sum(
        feat.embedding_dim for feat in sparse_features
    )
    return dense_feature_dim + sparse_feature_dim


class MLP(nn.Module):
    """The Multi Layer Perception
    Args:
        inputs_dim: input feature dimension

        hidden_units: list of positive integer, the layer number and units in each layer

        activation: activation function to use

        l2_reg: float between 0 and 1.
            L2 regularizer strength applied to the kernel weights matrix.

        dropout_rate:
            float in [0,1). Fraction of the units to drop. We cannot drop all of it

        use_bn: bool
            whether use BatchNormalization before activation or not

        seed: A python integer to use as random seed.
    """
    
    def __init__(self, mlp_config: MLPConfig, **kwargs):
        
        super(MLP, self).__init__()
        
        self.mlp_config = mlp_config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dropout = nn.Dropout(self.mlp_config.dropout_rate)
        
        if len(self.mlp_config.hidden_dims) == 0:
            raise ValueError("hidden_dims is empty!")
        
        # this is the DNN layer, everything is hidden
        self.hidden_units = [self.mlp_config.input_dims] + self.mlp_config.hidden_dims
        
        # https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
        # can be indexed like a regular Python list, but modules it contains
        # are properly registerd and will be visibile by all Module methods
        
        # say we have 3 inputs, and 2 output,
        # then we will declare this as nn.Linear(3, 2)
        self.linears = nn.ModuleList(
            [
                nn.Linear(self.hidden_units[i], self.hidden_units[i+1])
                for i in range(len(self.hidden_units) - 1)
            ]
        )
        
        # note, we just dont do the Batch Norm for the first and the last layers
        if self.use_bn:
            self.bn = nn.ModuleList(
                [
                    nn.BatchNorm1d(self.hidden_units[i + 1])
                    for i in range(len(self.hidden_units) - 1)
                ]
            )

        # we also dont do the activation layers for the first and the last layers
        self.activation_layers = nn.ModuleList(
            [
                activation_layer(self.mlp_config.activation,
                                 self.hidden_units[i + 1])
                for i in range(len(self.hidden_units) - 1)
            ]
        )
        
        # initialize the tensor
        for name, tensor in self.linears.named_parameters():
            if "weight" in name:
                nn.init.normal_(tensor, mean=0, std=self.mlp_config.init_std)
                
        # assign it to a device
        self.to(self.device)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        deep_inputs = inputs
        
        for i in range(len(self.linears) - 1):
            
            # fc = fully connected
            fc = self.linears[i](deep_inputs)
            
            # if batch normalized is used
            if self.use_bn:
                fc = self.bn[i](fc)
                
            fc = self.activation_layers[i](fc)
            fc = self.dropout(fc)
            
            deep_inputs = fc
        
        return deep_inputs
        
        
class TwoTaskTwoTowerSharedBottom(torch.Module):
    
    def __init__(self,
                 features: List[Union[SparseFeature, DenseFeature]],
                 shared_mlp_config: MLPConfig,
                 mlp_tower_configs: List[MLPConfig],
                 tower_dnn_hidden_units: List[int] = [64,],
                 embedding_init_std: float = 0.001,
                 device: str = "cpu"):
        
        super(TwoTaskTwoTowerSharedBottom, self).__init__()
        
        self.features = features
        self.embedding_init_std = embedding_init_std
        self.device = device
        self.shared_mlp_config = shared_mlp_config
        self.mlp_tower_configs = mlp_tower_configs
        
        self.dense_features = list(
            filter(
                lambda x : isinstance(x, DenseFeature),
                self.features
            )
        )
        
        self.sparse_features = list(
            filter(
                lambda x: isinstance(x, SparseFeature),
                self.features
            ) 
        )
                    
        
        self.bottom_dnn_hidden_units = bottom_dnn_hidden_units
        self.tower_dnn_hidden_units = tower_dnn_hidden_units
        
        
        # First, we create a feature column index
        feature_col_index = build_input_feature_column_index(
            self.features
        )
        
        # We first create embeddings for sparse features
        self.embedding_dict = create_embedding_matrix(self.sparse_features, 
                                                      self.embedding_init_std, 
                                                      self.device)
        
        self.input_dim = compute_input_dim(
            dense_features=self.dense_features,
            sparse_features=self.sparse_features
        )
        
        # we build on the DNN block
        self.bottom_mlp = MLP(
            mlp_config=self.shared_mlp_config,
        )
        
        # there are two towers
        self.mlp_tower = nn.ModuleList(
            [
                MLP(
                    self.mlp_tower_configs[i],
                )
                for i in range(len(self.mlp_tower_configs))
            ]
        )
        
        # TODO (needs to add regularization)
        
        # the final layer of the two towers dnn layers
        self.tower_mlp_final_layer = nn.ModuleList(
            [
                nn.Linear(
                    mlp_tower_configs.hidden_dims[-1], 
                    1,
                    bias = False
                )
                for mlp_tower_config in self.mlp_tower_configs
            ]
        )
        
        
        # output nodes
        
        