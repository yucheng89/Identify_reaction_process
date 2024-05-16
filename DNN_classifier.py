import os
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Batch_Net(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, n_hidden_5, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.BatchNorm1d(n_hidden_3), nn.Dropout(p=0.5))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4), nn.BatchNorm1d(n_hidden_4), nn.ReLU(True))
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5), nn.BatchNorm1d(n_hidden_5), nn.ReLU(True))
        self.layer6 = nn.Sequential(nn.Linear(n_hidden_5, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x


if __name__ == '__main__':

    dataset = pd.read_csv('originaldata_4_fix.csv',header=None)

    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 5000].values

    y[y==4] = 0   # The output of DNN includes "0, 1, 2, 3", so we need to replace "4" with "0" temporarily.

    from sklearn.model_selection import train_test_split
    x_data, x_test, y_data, y_test = train_test_split(x, y, test_size=0.10, random_state=0 )
    x_data = torch.tensor(x_data,dtype=torch.float32)
    y_data = torch.tensor(y_data,dtype=torch.float32)
    x_test = torch.tensor(x_test,dtype=torch.float32)
    y_test = torch.tensor(y_test,dtype=torch.float32)

    batch_size = 128
    dataloader_x = DataLoader(dataset=x_data, batch_size=batch_size)
    dataloader_y = DataLoader(dataset=y_data, batch_size=batch_size)

    lr = 0.01

    my_net = Batch_Net(5000, 4500, 4000, 3000, 1000, 500, 4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(my_net.parameters(), lr)

    for epoch in range(15):
        my_net.train()
        lossvalue=[]
        for x,y in zip(dataloader_x,dataloader_y):
            pred = my_net(x)
            y = y.reshape(1,-1).squeeze(0).byte()
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lossvalue.append(loss.item())
        loss_mean = sum(lossvalue) / len(lossvalue)
        print('epoch:',epoch,'  loss:',loss_mean)

    correct = 0
    total = 0
    my_net.eval()
    outputs = my_net(x_test)
    _,predict = torch.max(outputs.data,dim=1)
    total += y_test.size(0)
    correct += (predict == y_test).sum().item()
    print('Accuracy on testdata:%d %%'%(100*correct/total))


    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    y_test[y_test == 0] = 4
    predict[predict == 0] = 4
    result = confusion_matrix(y_test, predict)
    print("Confusion Matrix:")
    print(result)
    result2 = accuracy_score(y_test, predict)
    print("Accuracy:", result2)

    #torch.save(my_net, 'DNN-classify-total_response_4_unfix.pkl')
    #torch.save(my_net.state_dict(), 'DNN-classify_4_fix_para.pkl')


