# Import necessary libraries
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.metrics import accuracy_score
import random
from torch.utils import data
from tqdm import tqdm

import matplotlib.pyplot as plt
import time

# random seed
np.random.seed(11785)
torch.manual_seed(11785)


cuda = torch.cuda.is_available()
num_workers = 8 if cuda else 0
print("Cuda = "+str(cuda)+" with num_workers = "+str(num_workers))


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


transforms_train = torchvision.transforms.Compose([
    torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(20),
    # torchvision.transforms.CenterCrop((60,60)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transforms_test = torchvision.transforms.Compose([
    # torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_dataset = torchvision.datasets.ImageFolder(root='./train_data',
                                                 transform=transforms_train)
print('train_data',train_dataset.__len__(), len(train_dataset.classes))

dev_dataset = torchvision.datasets.ImageFolder(root='./val_data',
                                               transform=transforms_test)
print('val data',dev_dataset.__len__(), len(dev_dataset.classes))

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                               shuffle=True, num_workers=8)
dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=128,
                                             shuffle=False, num_workers=8)





# This is the simplest possible residual block, with only one CNN layer.
# Looking at the paper, you can extend this block to have more layers, bottleneck, grouped convs (from shufflenet), etc.
# Or even look at more recent papers like resnext, regnet, resnest, senet, etc.
class SimpleResidualBlock(nn.Module):
    def __init__(self, in_channel,out_channel, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True)

        if stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        # print(np.shape(x))
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)

        shortcut = self.shortcut(x)
        # print(np.shape(out),np.shape(shortcut))
        out = self.relu(out + shortcut)

        return out





# This has hard-coded hidden feature sizes.
# You can extend this to take in a list of hidden sizes as argument if you want.
class ClassificationNetwork(nn.Module):
    def __init__(self, in_features, num_classes,feat_dim = 2):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_features, 64, kernel_size=3, stride=1, padding=3, bias=False), # kernal 3 stride 1 & maxpool
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            torch.nn.Dropout(p=0.2),
            SimpleResidualBlock(64,64), # *3
            SimpleResidualBlock(64,64),
            SimpleResidualBlock(64,64),

            torch.nn.Dropout(p=0.2),
            SimpleResidualBlock(64,128,stride=2), # *4
            SimpleResidualBlock(128,128),
            SimpleResidualBlock(128,128),
            SimpleResidualBlock(128,128),

            torch.nn.Dropout(p=0.2),
            SimpleResidualBlock(128,256,stride=2),   # *6
            SimpleResidualBlock(256,256),
            SimpleResidualBlock(256,256),
            SimpleResidualBlock(256,256),
            SimpleResidualBlock(256,256),
            SimpleResidualBlock(256,256),

            torch.nn.Dropout(p=0.2),
            SimpleResidualBlock(256,512,stride=2), # *3
            SimpleResidualBlock(512,512),
            SimpleResidualBlock(512,512),


            nn.AdaptiveAvgPool2d((1, 1)), # For each channel, collapses (averages) the entire feature map (height & width) to 1x1
            nn.Flatten(), # the above ends up with batch_size x 64 x 1 x 1, flatten to batch_size x 64
        )
        self.linear = nn.Linear(512, feat_dim)
        self.relu = nn.ReLU(inplace=True)
        self.linear_output = nn.Linear(512,num_classes)
    def forward(self, x, return_embedding=False):
        embedding = self.layers(x)
        embedding_out = self.relu(self.linear(embedding))
        output = self.linear_output(embedding)
        if return_embedding:
            return embedding_out,output
        else:
            return output



numEpochs = 40
in_features = 3 # RGB channels

learningRate = 0.0001
weightDecay = 5e-5

num_classes = len(train_dataset.classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

network = ClassificationNetwork(in_features, num_classes)
network = network.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.9)# Sheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=2)



# Train 30 epoch
best_accuracy = -1
accuracy_val = []
network = torch.load('./best_model.pt')

# Validate
network.eval()
num_correct = 0
avg_loss_val = 0
with torch.no_grad():
    for batch_num, (x, y) in tqdm(enumerate(dev_dataloader),position=0, leave=True):
        x, y = x.to(device), y.to(device)
        outputs = network(x)

        loss = criterion(outputs, y.long())
        avg_loss_val += loss.item()

        num_correct += (torch.argmax(outputs, axis=1) == y).sum().item()
    best_accuracy = num_correct / len(dev_dataset)
    avg_loss_validation = avg_loss_val / len(dev_dataloader)
    print('accuracy',best_accuracy,'loss',avg_loss_validation)


for epoch in range(numEpochs):

    # Train
    network.train()
    avg_loss = 0.0
    for batch_num, (x, y) in tqdm(enumerate(train_dataloader),position=0, leave=True):
        optimizer.zero_grad()

        x, y = x.to(device), y.to(device)

        outputs = network(x)

        loss = criterion(outputs, y.long())
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()

        if batch_num % 100 == 99:
            print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch, batch_num+1, avg_loss/100))
            avg_loss = 0.0

    # Validate
    network.eval()
    num_correct = 0
    avg_loss_val = 0
    with torch.no_grad():
        for batch_num, (x, y) in tqdm(enumerate(dev_dataloader),position=0, leave=True):
            x, y = x.to(device), y.to(device)
            outputs = network(x)

            loss = criterion(outputs, y.long())
            avg_loss_val += loss.item()

            num_correct += (torch.argmax(outputs, axis=1) == y).sum().item()
        accuracy = num_correct / len(dev_dataset)
        accuracy_val.append(accuracy)
        avg_loss_validation = avg_loss_val / len(dev_dataloader)

        print('Epoch: {}, Validation Accuracy: {:.2f}'.format(epoch, accuracy))
        print('Epoch: {}, Validation loss: {:.2f}'.format(epoch, avg_loss_validation))
        scheduler.step(avg_loss_validation)
        print('Epoch {}, lr {}'.format(
              epoch, optimizer.param_groups[0]['lr']))

    if accuracy > best_accuracy:
      best_accuracy = accuracy
      print('val accuracy',accuracy_val)
      torch.save(network, 'model_classification_34_dropout_wth_resize.pt')
