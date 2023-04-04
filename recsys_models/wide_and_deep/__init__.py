from jup.recsys_models.models.base import DNNBaseModel
from jup.recsys_models.models.base import Linear
from jup.recsys_models.core.inputs import combine_dnn_input

from jup.recsys_models.features import DenseFeature, SparseFeature
from jup.recsys_models.core.mlp import DNN
from jup.recsys_models.core.prediction_layer import PredictionLayer

import torch.nn as nn

from typing import List, Union

class WDL(DNNBaseModel):
    """
        This is for wide and deep learning model architectures.
        We need to have another one that is used for a linear feature column here.
    """
    
    def __init__(self, 
                 features : List[Union[SparseFeature, DenseFeature]],
                 linear_features: List[Union[SparseFeature, DenseFeature]],  # to use for the linear feature part
                 dnn_hidden_units = (256, 128),
                 l2_reg_linear=1e-5,
                 l2_reg_embedding=1e-5,
                 l2_reg_dnn=0,
                 init_std=0.0001,
                 seed=1024,
                 dnn_dropout=0,
                 dnn_activation='relu',
                 dnn_use_bn=False,
                 task='binary',
                 device='cpu', 
                 gpus=None
                 ):
        
        super(WDL, self).__init__(
            features=features,
            embedding_init_std=init_std,
            device=device,
            seed=seed,
            tensorboard_path=None  
        )
        
        # why we need this?
        # self.use_dnn = len(self.dense_features) > 0 and len(dnn_hidden_units) > 0
        self.dnn_activation = dnn_activation
        self.dnn_hidden_units = dnn_hidden_units
        self.linear_features = linear_features # used for the linear features only
        
        # DNN Components
        self.dnn = DNN(
            input_dims=self.input_dim,
            hidden_units=self.dnn_hidden_units,
            activation=dnn_activation,
            l2_reg=l2_reg_dnn,
            dropout_rate=dnn_dropout,
            use_bn=dnn_use_bn)
        
    
        self.dnn_linear = nn.Linear(
            dnn_hidden_units[-1],
            1,
            bias=False
        ).to(device=device)
        
        # need to double check this.
        # what is named parameter here?
        self.add_regularization_weight(
            filter(
                lambda x: 'weight' in x[0] and 'bn' not in x[0],
                self.dnn.named_parameters()),
                l2=l2_reg_dnn
        )
        
        self.add_regularization_weight(
            self.dnn_linear.weight, 
            l2=l2_reg_dnn
        )
        
        # Linear components
        self.linear_model = Linear(
            linear_features=linear_features
        )
            
        self.add_regularization_weight(
            self.linear_model.parameters(),
            l2=l2_reg_linear
        )
        
        self.to(device)
        
        # Prediction Layer
        self.out = PredictionLayer(task,)
        
    def forward(self, data):
        
        sparse_embedding_list, dense_value_list = self.inputs_from_feature_columns(data)
        
        logit = self.linear_model(data)
        
        # need to build out this for the linear features.
        dnn_input = combine_dnn_input(
            sparse_embedding_list=sparse_embedding_list,
            dense_value_list=dense_value_list,
        )
        
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)
        
        # combine the two logit together
        logit += dnn_logit
        
        y_pred = self.out(logit)
        
        return y_pred
        
        