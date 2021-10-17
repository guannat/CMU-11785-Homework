# Import necessary libraries
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.metrics import accuracy_score
import random


from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# random seed
np.random.seed(11785)
torch.manual_seed(11785)

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
sys.version
print(cuda, sys.version)





# load training data
raw_train = np.load('./train.npy', allow_pickle=True)
raw_train_labels = np.load('./train_labels.npy', allow_pickle=True)
print("Raw training data shape:", raw_train.shape)
print("Raw training data label shape:", raw_train_labels.shape)

# load dev data
raw_dev = np.load('./dev.npy', allow_pickle=True)
raw_dev_labels = np.load('./dev_labels.npy', allow_pickle=True)
print("Raw dev data shape", raw_dev.shape)
print("Raw dev data label shape", raw_dev_labels.shape)


# Stacked training dataset
stack_train = np.vstack(raw_train)
stack_train_labels = np.hstack(raw_train_labels)
print('stack_train',stack_train.shape,'stack_train_labels',stack_train_labels.shape)
del raw_train,raw_train_labels

# Stacked val dataset
stack_dev = np.vstack(raw_dev)
stack_dev_labels = np.hstack(raw_dev_labels)
print('stack_dev',stack_dev.shape,'stack_dev_labels',stack_dev_labels.shape)
del raw_dev,raw_dev_labels

# convert data from numpy to tensor
train_data = torch.tensor(stack_train)
train_labels = torch.tensor(stack_train_labels,dtype=torch.long)
print('train_data',train_data.shape,'train_labels',train_labels.shape)
dev_data = torch.tensor(stack_dev)
dev_labels = torch.tensor(stack_dev_labels,dtype=torch.long)
print('dev_data',dev_data.shape,'dev_labels',dev_labels.shape)
del stack_train,stack_train_labels,stack_dev,stack_dev_labels


# define dataset & padding
class MyDataset(data.Dataset):
  def __init__(self, X, Y, padding_size):
    self.X = X
    self.Y = Y
    self.length = len(self.X)
    self.dim = self.X.shape[1]
    self.padding_size = padding_size

  def __len__(self):
    return self.length

  def __getitem__(self, index):
    x = self.X[index].float()
    y = self.Y[index]

    # padding
    if index < self.padding_size:
        x_f = self.X[:index,:].float()
        x_b = self.X[(index+1):(index+self.padding_size+1)].float()

        res = torch.cat([x_f, x.unsqueeze(0), x_b],dim=0)
        res = F.pad( res, (0, 0, (self.padding_size-index), 0), mode = 'constant' ) # ordering of padding widths is different

    elif  (index+self.padding_size+1) > self.length :
        x_f = self.X[(index-self.padding_size):(index),:].float()
        x_b_length = (self.length - 1 -index)
        if  x_b_length == 0:
          x_b = self.X[:0,:].float()
        else:
          x_b = self.X[-x_b_length:,:].float()

        res = torch.cat([x_f, x.unsqueeze(0), x_b ],dim=0)
        res = F.pad( res, (0, 0, 0, (self.padding_size - x_b_length)), mode = 'constant' ) # ordering of padding widths is different

    else:
        # x_f = self.X[(index-self.padding_size):(index),:].float()
        # x_b = self.X[(index+1):(index+self.padding_size+1)].float()
        # res = torch.cat([x_f, x.unsqueeze(0), x_b ],dim=0)
        res = self.X[(index-self.padding_size):(index+self.padding_size+1),:]

    return res.flatten().float(),y


# batch size and context size
context_size = 30
# training set
train_dataset = MyDataset(train_data, train_labels, context_size)
train_loader = data.DataLoader(
           dataset=train_dataset,
           batch_size=500,
           num_workers=8,
           shuffle=True,
           drop_last=True,
           pin_memory=True)

# dev set
dev_dataset = MyDataset(dev_data, dev_labels, context_size)
dev_loader = data.DataLoader(
          dataset=dev_dataset,
          batch_size=500,
          num_workers=8,
          shuffle=True,
          drop_last=True,
          pin_memory=True)



# model setting
class Model(nn.Module):
  def __init__(self, size):
    super().__init__()

    self.model = nn.Sequential(nn.Linear(size[0], size[1]) ,nn.BatchNorm1d(size[1]), nn.LeakyReLU(0.1),nn.Dropout(p=0.2),
                               nn.Linear(size[1], size[2]) ,nn.BatchNorm1d(size[2]), nn.LeakyReLU(0.1),nn.Dropout(p=0.2),
                               nn.Linear(size[2], size[3]) ,nn.LeakyReLU(0.1),
                               nn.Linear(size[3], size[4]) ,nn.LeakyReLU(0.1),
                               nn.Linear(size[4], size[5]))


  def forward(self, x):

    # return output
    return self.model(x)



# initialize model
model = Model([(40*(2*context_size+1)), 3000, 2000, 1024, 512, 71])
model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=2)

criterion = nn.CrossEntropyLoss()

n_epochs = 15
best_accuracy = -1



# define training
def train(model, train_loader, optimizer, criterion):
  predictions = []
  actuals = []

  model.train()

  running_loss = 0.0
  total_correct = 0

  # x -> a batch of xs
  for id, (x, y) in tqdm(enumerate(train_loader),position=0, leave=True):
    # clear gradient
    optimizer.zero_grad()

    x, y = x.to(device), y.to(device)

    out = model(x)

    loss_value = criterion(out , y.long()) # real value 0.569
    running_loss += loss_value.item()

    out = out.cpu().detach().numpy()
    actual = y.cpu().detach().numpy()

    # convert to class labels
    out = np.argmax(out, axis=1)

    # reshape for stacking
    actual = actual.reshape((len(actual), 1))
    out = out.reshape((len(out), 1))

    # store
    predictions.append(out)
    actuals.append(actual)

    # backpro
    loss_value.backward()
    optimizer.step()

  predictions, actuals = np.vstack(predictions), np.vstack(actuals)
  avg_loss = running_loss / len(train_loader)
  acc = accuracy_score(actuals, predictions)
  return avg_loss, acc


# define Evaluation
def eval(model, dev_loader, optimizer, criterion):
  predictions = []
  actuals = []

  model.eval()

  running_loss = 0.0
  total_correct = 0

  with torch.no_grad():
      for id, (x, y) in tqdm(enumerate(dev_loader),position=0, leave=True):
          x, y = x.to(device), y.to(device)

          # prediction
          out = model(x)

          loss_value = criterion(out , y.long()) # real value 0.569
          running_loss += loss_value.item()

          # retrieve numpy array
          out = out.cpu().detach().numpy()
          actual = y.cpu().detach().numpy()

          # convert to class labels
          out = np.argmax(out, axis=1)

          # reshape for stacking
          actual = actual.reshape((len(actual), 1))
          out = out.reshape((len(out), 1))
          # store
          predictions.append(out)
          actuals.append(actual)


  predictions, actuals = np.vstack(predictions), np.vstack(actuals)
  avg_loss = running_loss / len(dev_loader)
  acc = accuracy_score(actuals, predictions)
  return avg_loss, acc



# Epoch

loss_train = []
loss_val = []
accuracy_train = []
accuracy_val = []

for epoch in range(n_epochs):
  # Training
  start_time = time.time()
  avg_loss, accuracy = train(model, train_loader, optimizer, criterion)
  end_time = time.time()

  loss_train.append(avg_loss)
  accuracy_train.append(accuracy)

  print("Epoch: "+str(epoch)+", training avg_loss: "+str(avg_loss)+", training accuracy:"+str(accuracy))

  # Evaluation
  start_time = time.time()
  avg_loss, accuracy = eval(model, dev_loader, optimizer, criterion)
  end_time = time.time()

  loss_val.append(avg_loss)
  accuracy_val.append(accuracy)
  print("Epoch: "+str(epoch)+", validation avg_loss: "+str(avg_loss)+", validation accuracy:"+str(accuracy))


  if accuracy > best_accuracy:
    best_accuracy = accuracy
    torch.save(model, 'model_3.pt') # linear + batch

  scheduler.step(avg_loss)
  print('Epoch {}, lr {}'.format(
        epoch, optimizer.param_groups[0]['lr']))

# Epoch 13 best evaluation accuracy

# load test data
test_data = np.load('test.npy', allow_pickle=True)
stack_test = np.vstack(test_data)
test =  torch.tensor(stack_test)

print("Raw test shape: ", test_data.shape)
print("Stacked test shape: ", stack_test.shape)
print("Test shape: ",test.shape)
# del test_data,stack_test

context_size = 30
# Create test dataset and data loader
test_dataset = MyDataset(X=test, padding_size=context_size)
test_loader = data.DataLoader(
          dataset=test_dataset,
          shuffle=False,
          batch_size=500,
          num_workers=8,
          pin_memory=True,
          drop_last=False)


# load model (epoch 13)


model = torch.load('model_3.pt')
predictions = []
model.eval()

with torch.no_grad():
  start_time = time.time()
  for id, x in enumerate(test_loader):
    x = x.to(device)

    out = model(x)

    # retrieve numpy array
    out = out.cpu().detach().numpy()
    # convert to class labels
    out = np.argmax(out, axis=1)
    # reshape for stacking
    out = out.reshape((len(out), 1))
    # store
    predictions.append(out)

predictions = np.vstack(predictions)
print(len(predictions))

# remove brackets
new_list = [str(elem) for elem in predictions.tolist()]
new_list = [elem.replace("[","").replace("]","") for elem in new_list]

# write csv file
with open("trial.csv", 'w') as fh:
  fh.write('id,label\n')
  for i in range(len(new_list)):
    fh.write(str(i)+ ',' + new_list[i] + "\n")
