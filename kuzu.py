# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.A = nn.Linear(784, 10)

    def forward(self, x):
        # view() Flattens the image
        x = x.view(-1, 784)
        x = self.A(x)
        x = F.log_softmax(x, dim=1)
        return x

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        self.in_to_hid = nn.Linear(784, 530)
        self.hid_to_hid = nn.Linear(530, 100)
        self.hid_to_out = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.in_to_hid(x)
        x = x.tanh()
        x = self.hid_to_hid(x)
        x = x.tanh()
        x = self.hid_to_out(x)
        x = F.log_softmax(x, dim=1)
        return x

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        # Convolution Layer Size (28 + 1 - 3) = 26 
        self.mpool1 = nn.MaxPool2d(2)
        # Next Layer Size (1 + (26 - 2)/2) * (1 + (26 - 2)/2) = 13 * 13
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        # Convolution Layer Size (13 + 1 - 3) = 11
        self.mpool2 = nn.MaxPool2d(2)
        # Next Layer Size (1 + (11 - 2)/2) * (1 + (11 - 2)/2) = 5 * 5
        self.fc1 = nn.Linear(32*5*5, 540)
        self.fc2 = nn.Linear(540, 10)

    def forward(self, x):   
        x = self.conv1(x)
        x = F.relu(x)
        x = self.mpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.mpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

class NetConvFull(nn.Module):
    # two convolutional layers and one fully connected layer
    # all using relu, followed by log_softmax
    # no max-pooling

    def __init__(self):
            super(NetConvFull, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
            # Convolution Layer Size (28 + 1 - 5) = 24
            self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
            # Convolution Layer Size (24 + 1 - 5) = 20
            self.fc1 = nn.Linear(32*20*20, 540)
            self.fc2 = nn.Linear(540, 10)

    def forward(self, x):   
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

class NetDropout(nn.Module):
    # two convolutional layers and one fully connected layer
    # all using relu, followed by log_softmax
    # with max-pooling and dropout

    def __init__(self):
        super(NetDropout, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        # Convolution Layer Size (28 + 1 - 3) = 26 
        self.mpool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(p=0.2)
        # Next Layer Size (1 + (26 - 2)/2) * (1 + (26 - 2)/2) = 13 * 13
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        # Convolution Layer Size (13 + 1 - 3) = 11
        self.mpool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(p=0.2)
        # Next Layer Size (1 + (11 - 2)/2) * (1 + (11 - 2)/2) = 5 * 5
        self.fc1 = nn.Linear(32*5*5, 540)
        self.fc2 = nn.Linear(540, 10)

    def forward(self, x):   
        x = self.conv1(x)
        x = F.relu(x)
        x = self.mpool1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.mpool2(x)
        x = self.drop2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

class NetFullDrop(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFullDrop, self).__init__()
        self.in_to_hid = nn.Linear(784, 530)
        self.drop1 = nn.Dropout(p=0.1)
        self.hid_to_hid = nn.Linear(530, 100)
        self.drop2 = nn.Dropout(p=0.1)
        self.hid_to_out = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.in_to_hid(x)
        x = self.drop1(x)
        x = x.tanh()
        x = self.hid_to_hid(x)
        x = self.drop2(x)
        x = x.tanh()
        x = self.hid_to_out(x)
        x = F.log_softmax(x, dim=1)
        return x