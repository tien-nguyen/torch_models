import torch
import torch.nn as nn
 
from sklearn.datasets import make_classification
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List

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
    def __init__(self, input_dim: int, output_dim: int, use_bias=False, **kwargs):
        super(Regression, self).__init__()
              
        self.input_dim = input_dim
        self.output_dim = output_dim 
        
        super(Regression, self).__init__()
        
        self.use_bias = use_bias

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1, ))) 
            self.input_dim += 1
        
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 28),
            torch.nn.ReLU(),
            torch.nn.Linear(28, 28),
            torch.nn.ReLU(),
            torch.nn.Linear(28, self.output_dim)
        )
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor: # need to have the notation for this inputs here
        
        # the code below is for when we build out multi layer perception
        # we do not need to do it here
        if self.use_bias:
            inputs = inputs + self.bias
        
        outputs = self.layer(inputs)
        
        return torch.sigmoid(outputs)
    
    
def prepare_dummy_data_for_two_classes(n_samples: int, n_features: int) -> List[np.ndarray]:
    
    separable = False
    red = blue = []
    
    while not separable:
        samples = make_classification(n_samples=n_samples, 
                                      n_features=n_features, n_redundant=0, 
                                      n_informative=1,
                                      n_clusters_per_class=1, flip_y=-1)
        red = samples[0][samples[1] == 0]
        blue = samples[0][samples[1] == 1]
        separable = any([red[:, k].max() < blue[:, k].min() or red[:, k].min() > blue[:, k].max() for k in range(2)])

    red_labels = np.zeros(len(red)) # marking this as zeros
    blue_labels = np.ones(len(blue)) # marking this as ones
    
    labels = np.append(red_labels, blue_labels)
    inputs = np.concatenate((red, blue), axis=0) # vertically joined red and blue
    
    return inputs, labels


def run():
    
    x, y = prepare_dummy_data_for_two_classes(n_samples=500, n_features=3)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    
    model = Regression(input_dim=3, output_dim=1)
    
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    
    n_epochs = 100   # number of epochs to run
    batch_size = 10  # size of each batch
    batch_start = torch.arange(0, len(x_train), batch_size).tolist()
    
    # hold the best
    best_loss = np.inf   # init to infinity
    loss = np.inf
    
    best_weights = None

    for epoch in range(n_epochs):
        model.train()
        for start in batch_start:
            # take a batch
            x_batch = x_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            # forward pass
            y_pred = model(x_batch)
            
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()

        # evaluate accuracy at end of each epoch
        if epoch %10 == 0:
            model.eval()
            y_pred = model(x_test)
            # y_pred = torch.squeeze(y_pred)
            loss = loss_fn(y_pred, y_test)
            print(f'Epoch: {epoch}, Loss: {loss.item()}')
            
        if loss < best_loss:
            best_loss = loss
            best_weights = model.state_dict()
    
    PATH = "/Users/tien/Documents/PythonEnvs/pytorch/jup/examples/recsys_models/logistic_regression/logistic_regression_simple.pt"
    torch.save(model.state_dict(), PATH)
    
    test_model = Regression(input_dim=3, output_dim=1)
    test_model.load_state_dict(torch.load(PATH))
    
if __name__ == '__main__':
    run()
    