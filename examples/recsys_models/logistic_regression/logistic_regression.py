from sklearn.datasets import make_classification
import numpy as np

import torch.nn as nn
import torch 

from sklearn.model_selection import train_test_split
from typing import List

# other imports
from tqdm import tqdm

# from typing import tuple

class Regression(nn.Module):
    
    def __init__(self, input_dim: int, output_dim: int, task='binary', use_bias=True, **kwargs):
        
        super(Regression, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task = task
        self.use_bias = use_bias
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1, )))

        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        
        outputs = self.linear(inputs)
        
        if self.use_bias:
            outputs += self.bias

        if self.task == "binary":
            outputs = torch.sigmoid(outputs)
            
        return outputs
    
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
    
    return [inputs, labels]


def run():
    
    # overall
    N_FEATURES = 2
    N_EPOCHS = 1000
    
    # prepare data
    inputs, labels = prepare_dummy_data_for_two_classes(n_samples=1000, n_features=N_FEATURES)
    x_train, x_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2)
    
    x_train, x_test = torch.Tensor(x_train), torch.Tensor(x_test)
    y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)
    
    # create model, set criterion, and optimizer
    model = Regression(input_dim=N_FEATURES, output_dim=1, task='binary', use_bias=True)
    creterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # prepare for training
    losses = [] # capture the losses for each of epoch
    losses_test = [] # capture the loses for test data
    
    iterations = []
    iter = 0
    
    for epoch in tqdm(range(int(N_EPOCHS)), desc="Training Epochs"):
        x = x_train
        labels = y_train
        
        # setting the stored gradients equal to zero
        optimizer.zero_grad()
        
        outputs = model(x_train)
        
        loss = creterion(torch.squeeze(outputs), labels)
        
        # compute the gradient of the given tensor w.r.t. graph leaves
        loss.backward()
        
        # updates the weights and biases with the optimzer (SGD)
        optimizer.step()
        
        iter += 1
        
        if iter % N_EPOCHS == 0:
            
            # why do we need to do this?
            # because during the evaluation, the model is not trained
            with torch.no_grad(): 
                
                # Calculate the loss and accuracy for the test dataset
                correct_test = 0
                total_test = 0
                
                output_test = model(x_test)
                output_test = torch.squeeze(output_test)
                loss_test = creterion(output_test, y_test)
                
                # tensor.round() means that the threshold is at 0.5
                
                predicted_test = output_test.round().detach().numpy()
                total_test = y_test.size(0)
                correct_test = np.sum(
                    predicted_test == y_test.detach().numpy()
                )
                
                accuracy_test = 100 * correct_test / total_test
                losses_test.append(loss_test.item())
                
                # what does round().detach() do?
                # https://stackoverflow.com/questions/55466298/pytorch-cant-call-numpy-on-variable-that-requires-grad-use-var-detach-num
                # https://stackoverflow.com/questions/63582590/why-do-we-call-detach-before-calling-numpy-on-a-pytorch-tensor
                '''
                    I think the most crucial point to understand here is the difference between a torch.tensor and np.ndarray:
                    While both objects are used to store n-dimensional matrices (aka "Tensors"), torch.tensors has an additional "layer" - which is storing the computational graph leading to the associated n-dimensional matrix.

                    So, if you are only interested in efficient and easy way to perform mathematical operations on matrices np.ndarray or torch.tensor can be used interchangeably.

                    However, torch.tensors are designed to be used in the context of gradient descent optimization, and therefore they hold not only a tensor with numeric values, but (and more importantly) the computational graph leading to these values. This computational graph is then used (using the chain rule of derivatives) to compute the derivative of the loss function w.r.t each of the independent variables used to compute the loss.

                    As mentioned before, np.ndarray object does not have this extra "computational graph" layer and therefore, when converting a torch.tensor to np.ndarray you must explicitly remove the computational graph of the tensor using the detach() command.
                '''
                predicted_test = output_test.round().detach().numpy()
                total_test = y_test.size(0)
                correct_test = np.sum(
                    predicted_test == y_test.detach().numpy()
                )
                
                accuracy_test = 100.0 * correct_test / total_test
                losses_test.append(loss_test.item())
                
                
                total_test += y_test.size(0) # get the first dimension
                
                # calculcate the loss and accuracy for the train dataset
                total = y_train.size(0)
                
                _output = torch.squeeze(
                    outputs
                ).round() # not sure what .round() means for
                
                correct = np.sum(
                    _output.detach().numpy() == y_train.detach().numpy()
                )
                
                accuracy = 100.0 * correct / total
                losses.append(loss.item())
                
                iterations.append(iter)
                
                print(f"Iteration: {iter}. \nTest - Loss: {loss_test.item()}. Accuracy: {accuracy_test}")
                print(f"Train -  Loss: {loss.item()}. Accuracy: {accuracy}\n")

    PATH = "/Users/tien/Documents/PythonEnvs/pytorch/jup/examples/recsys_models/simple_regression/logistic_regression.pt"
    torch.save(model.state_dict(), PATH)
    
    test_model = Regression(input_dim=N_FEATURES, output_dim=1, task='binary', use_bias=True)
    test_model.load_state_dict(torch.load(PATH))
    
    test_data = torch.Tensor([0.2, 0.5])
    y_pred = round(model(test_data).round().detach().numpy()[0])
    
    print("making prediction")
    print(y_pred)
    
    test_data = torch.Tensor([0.9, 0.9])
    y_pred = round(model(test_data).round().detach().numpy()[0])
    
    print("making prediction 2")
    print(y_pred)
    
    
    
if __name__ == "__main__":
    run()
