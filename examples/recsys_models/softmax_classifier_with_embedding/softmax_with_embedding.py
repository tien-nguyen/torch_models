import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from torch import nn

from sklearn.model_selection import train_test_split


def generate_dataset(size, classes=2, noise=0.5, number_of_users=3):
    # Generate random datapoints
    labels = np.random.randint(0, classes, size)
    user_ids = np.random.randint(0, number_of_users, size)
    x = (np.random.rand(size) + labels) / classes
    y = x + np.random.rand(size) * noise
    # Reshape data in order to merge them
    x = x.reshape(size, 1)
    y = y.reshape(size, 1)
    labels = labels.reshape(size, 1)
    user_ids = user_ids.reshape(size, 1) 
    # Merge the data
    data = np.hstack((user_ids, x, y, labels))
    return data


class TrainData(Dataset):
    "The data for multi-class classification"
    def __init__(self, inputs: np.ndarray, labels: np.ndarray):
        
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels)
        self.labels = self.labels.type(torch.LongTensor)
        
        self.len = inputs.shape[0]
 
    def __getitem__(self, idx):
        "accessing one element in the dataset by index"
        return self.inputs[idx], self.labels[idx] 
 
    def __len__(self):
        "size of the entire dataset"
        return self.len


class Softmax(torch.nn.Module):
    "custom softmax module"
    def __init__(self, n_inputs, n_outputs, n_users=3, embedding_dim=10, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_users = n_users
        
        # dropout
        self.dropout = nn.Dropout(p=dropout)
        
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.linear = torch.nn.Linear(n_inputs+embedding_dim, n_outputs)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        user_ids = x[:, 0]
        inputs = x[:, 1:]
        
        user_ids = user_ids.type(torch.LongTensor)
        
        user_embeddings = self.user_embedding(user_ids)
        
        features = torch.cat((user_embeddings, inputs), dim=-1)
        
        features = self.dropout(features)
        
        pred = self.linear(features)
        
        return pred
 
def run():
    
    data = generate_dataset(size=1000, classes=3, noise=0.5)

    labels = data[:, -1]
    inputs = data[:, :-1]
    
    x_train, x_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2)
    
    train_data = TrainData(x_train, y_train)
    test_data = TrainData(x_test, y_test)
    
    # call Softmax Classifier
    # two features, 3 classes
    model_softmax = Softmax(2, 3)
    model_softmax.state_dict()
 
    # define loss, optimizier, and dataloader
    #  for L2 regularization, we use weight decay
    # https://pytorch.org/docs/stable/optim.html
    optimizer = torch.optim.SGD(model_softmax.parameters(), lr=0.01, weight_decay=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(dataset=train_data, batch_size=2)
 
    # Train the model
    Loss = []
    epochs = 1000
    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = model_softmax(x)
            loss = criterion(y_pred, y)
            Loss.append(loss)
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            # test out
            with torch.no_grad():
                pred_model = model_softmax(test_data.inputs)
                _, y_pred = pred_model.max(1)
                    
                correct = (test_data.labels == y_pred).sum().item()
                acc = correct / len(data)
                print("model accuracy: ", acc)
    print("Done!")
 
    # Make predictions on test data
    pred_model =  model_softmax(test_data.inputs)
    _, y_pred = pred_model.max(1)
    # print("model predictions on test data:", y_pred)
 
    # check model accuracy
    correct = (test_data.labels == y_pred).sum().item()
    acc = correct / len(data)
    print("model accuracy: ", acc)
    

if __name__ == "__main__":
    run()