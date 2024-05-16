import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from DNN_classifier import Batch_Net

if __name__ == '__main__':
    dataset = pd.read_csv('transferdata_4_fix.csv', header=None)
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 5000].values

    y[y == 4] = 0  # The output of DNN includes "0, 1, 2, 3", so we need to replace "4" with "0" temporarily.

    from sklearn.model_selection import train_test_split

    x_data, x_test, y_data, y_test = train_test_split(x, y, test_size=0.9, random_state=0)
    x_data = torch.tensor(x_data, dtype=torch.float32)
    y_data = torch.tensor(y_data, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    my_net = torch.load('DNN-classify-total.pkl')

    batch_size = 32
    dataloader_x = DataLoader(dataset=x_data, batch_size=batch_size)
    dataloader_y = DataLoader(dataset=y_data, batch_size=batch_size)

    lr = 0.001

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(my_net.parameters(), lr)

    for epoch in range(20):
        my_net.train()
        lossvalue = []
        for x, y in zip(dataloader_x, dataloader_y):
            pred = my_net(x)
            y = y.reshape(1, -1).squeeze(0).byte()
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lossvalue.append(loss.item())
        loss_mean = sum(lossvalue) / len(lossvalue)
        print('epoch:', epoch, '  loss:', loss_mean)

    correct = 0
    total = 0
    my_net.eval()
    outputs = my_net(x_test)
    _, predict = torch.max(outputs.data, dim=1)
    total += y_test.size(0)
    correct += (predict == y_test).sum().item()
    predict[predict == 0] = 4
    y_test[y_test == 0] = 4
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    result = confusion_matrix(y_test, predict)
    print("Confusion Matrix:")
    print(result)
    result2 = accuracy_score(y_test, predict)
    print("Accuracy:", result2)
