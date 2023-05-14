import copy

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

def read_data():
    print("Reading data...")
    data = fetch_california_housing()
    
    print("Done reading data")
    
    return data.data, data.target

def run():
    
    X, y = read_data()
    print("------")
    print(X)
    
    # train-test split for model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

    # Convert to 2D PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    
    print("---- ytest")
    print(y_test)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    print("---- ytest2")
    print(y_test)
    
    # Define the model
    # model = nn.Sequential(
    #     nn.Linear(8, 24),
    #     nn.ReLU(),
    #     # nn.Linear(24, 12),
    #     # nn.ReLU(),
    #     # nn.Linear(12, 6),
    #     # nn.ReLU(),
    #     nn.Linear(24, 1)
    # )

    model = nn.Sequential(
        nn.Linear(8, 1),
        # nn.ReLU(),
        # # nn.Linear(24, 12),
        # # nn.ReLU(),
        # # nn.Linear(12, 6),
        # # nn.ReLU(),
        # nn.Linear(24, 1)
    )
        
    # loss function and optimizer
    loss_fn = nn.MSELoss()  # mean square error
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 100   # number of epochs to run
    batch_size = 10  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size).tolist()

    # Hold the best model
    best_mse = np.inf   # init to infinity
    best_weights = None
    history = []

    for epoch in range(n_epochs):
        model.train()
        for start in batch_start:
            # take a batch
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()

        # evaluate accuracy at end of each epoch
        if epoch %10 == 0:
            model.eval()
            y_pred = model(X_test)
            mse = loss_fn(y_pred, y_test)
            mse = float(mse)
            history.append(mse)
            print(f"epoch: {epoch} - mse: {mse}")
        
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())


    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    print("MSE: %.2f" % best_mse)
    print("RMSE: %.2f" % np.sqrt(best_mse))
    # plt.plot(history)
    # plt.show()
    
if __name__ == "__main__":
    run()