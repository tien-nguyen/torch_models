import torch
import torch.nn as nn

'''
Note:
    this implementation is different from what we have 
    when we build out multi-task perceptions
'''

class Regression(nn.Module):
    
    """
        Initializes a new instance of the LogisticRegression class.

    Args:
        self (LogisticRegression): the instance that the method operates on.
    """
    def __init__(self, input_dim: int, output_dim: int, task='binary', use_bias=True, **kwargs):
        super(Regression, self).__init__()
        
        assert task in ['binary',  "regression"], "task must be either binary or regresion"
        
        self.input_dim = input_dim
        self.output_dim = output_dim 
        
        super(Regression, self).__init__()
        
        self.use_bias = use_bias
        self.task = task
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1, ))) 
            self.input_dim += 1
        
        self.task = task
        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor: # need to have the notation for this inputs here
        
        # the code below is for when we build out multi layer perception
        # we do not need to do it here
        if self.use_bias:
            inputs = inputs + self.bias
        
        outputs = self.linear(inputs)
        
        if self.task == "binary":  # if this is a binary classification
            outputs = torch.sigmoid(inputs)

        return outputs 
        '''
        outputs = inputs
        
        if self.use_bias:
            outputs += self.bias
            
        if self.task == "binary":  # if this is a binary classification
            outputs = torch.sigmoid(outputs)
            
        return outputs
        '''
        
            