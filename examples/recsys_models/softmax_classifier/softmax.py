import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

def generate_dataset(size, classes=2, noise=0.5):
    # Generate random datapoints
    labels = np.random.randint(0, classes, size)
    x = (np.random.rand(size) + labels) / classes
    y = x + np.random.rand(size) * noise
    # Reshape data in order to merge them
    x = x.reshape(size, 1)
    y = y.reshape(size, 1)
    labels = labels.reshape(size, 1)
    # Merge the data
    data = np.hstack((x, y, labels))
    return data


class toy_data(Dataset):
    "The data for multi-class classification"
    def __init__(self):
        # single input
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        # multi-class output
        self.y = torch.zeros(self.x.shape[0])
        self.y[(self.x > -2.0)[:, 0] * (self.x < 0.0)[:, 0]] = 1 
        self.y[(self.x >= 0.0)[:, 0] * (self.x < 2.0)[:, 0]] = 2 
        self.y[(self.x >= 2.0)[:, 0]] = 3
        self.y = self.y.type(torch.LongTensor)
        self.len = self.x.shape[0]
 
    def __getitem__(self, idx):
        "accessing one element in the dataset by index"
        return self.x[idx], self.y[idx] 
 
    def __len__(self):
        "size of the entire dataset"
        return self.len
    
    

data = toy_data()
print("first ten data samples: ", data.x[0:10])
print("first ten data labels: ", data.y[0:10])
print(len(data))


class Softmax(torch.nn.Module):
    "custom softmax module"
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.linear(x)
        return pred
 
def run():
    # call Softmax Classifier
    model_softmax = Softmax(1, 4)
    model_softmax.state_dict()
 
    # define loss, optimizier, and dataloader
    optimizer = torch.optim.SGD(model_softmax.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(dataset=data, batch_size=2)
 
    # Train the model
    Loss = []
    epochs = 100
    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = model_softmax(x)
            loss = criterion(y_pred, y)
            Loss.append(loss)
            loss.backward()
            optimizer.step()
    print("Done!")
 
    # Make predictions on test data
    pred_model =  model_softmax(data.x)
    _, y_pred = pred_model.max(1)
    print("model predictions on test data:", y_pred)
 
    # check model accuracy
    correct = (data.y == y_pred).sum().item()
    acc = correct / len(data)
    print("model accuracy: ", acc)
    

if __name__ == "__main__":
    run()