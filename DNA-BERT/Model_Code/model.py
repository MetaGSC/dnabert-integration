import os
import sys
import re
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from Bio import SeqIO
from datetime import datetime
from numpy.random import randint
import torch
import torch.utils.data
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

# from HDF5dataset import HDF5Dataset
# from WeightsCreator import make_weights_for_balanced_classes

import matplotlib
import matplotlib.pyplot as plt

#==================================================================================================================================================================================================

# Hyper Parameters of the Model

inputFeatures = 96
# layer_array = [90, 80, 70, 60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 40, 30, 20, 10]

# layer_array = [256, 256, 256, 256]
layer_array = [49, 49]
outputSize = 2
momentum = 0.1
dropoutProb = 0.1
batchSize = 200
num_epochs = 20
opt_func = torch.optim.Adam
lr = 0.05
num_workers = 2

#==================================================================================================================================================================================================

global correct_test_results
correct_test_results = 0

global tp_p, tn_p, fp_p, fn_p
global tp_c, tn_c, fp_c, fn_c

tp_p = 0
tn_p = 0
fp_p = 0
fn_p = 0
tp_c = 0
tn_c = 0
fp_c = 0
fn_c = 0

plasmid = 0
chromosome = 1

# Setup the Cude Device

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_default_device()
print(f'{device} set as the default device')

#====================================================================================================================================================================================================    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.savefig('Figures/accuracies.png')
    plt.show()
    plt.clf()

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.savefig('Figures/losses.png')
    plt.show()
    plt.clf()

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

class Model(nn.Module):
    def __init__(self, in_size, layer_array = [512, 512, 256, 256], out_size = 28):
        super().__init__()
        self.network = nn.Sequential(
          nn.Linear(in_size, layer_array[0]),
          nn.ReLU(),
          nn.Dropout(dropoutProb),
          nn.Linear(layer_array[0], layer_array[1]),
          nn.ReLU(),
          nn.Dropout(dropoutProb),
        #   nn.Linear(layer_array[1], layer_array[2]),
        #   nn.ReLU(),
        #   nn.Dropout(dropoutProb),
        #   nn.Linear(layer_array[2], layer_array[3]),
        #   nn.ReLU(),
        #   nn.Dropout(dropoutProb),
        #   nn.Linear(layer_array[3], layer_array[4]),
        #   nn.ReLU(),
        #   nn.Dropout(dropoutProb),
        #   nn.Linear(layer_array[4], layer_array[5]),
        #   nn.ReLU(),
        #   nn.Dropout(dropoutProb),
        #   nn.Linear(layer_array[5], layer_array[6]),
        #   nn.ReLU(),
        #   nn.Dropout(dropoutProb),
        #   nn.Linear(layer_array[6], layer_array[7]),
        #   nn.ReLU(),
        #   nn.Dropout(dropoutProb),
        #   nn.Linear(layer_array[7], layer_array[8]),
        #   nn.ReLU(),
        #   nn.Dropout(dropoutProb),
        #   nn.Linear(layer_array[8], layer_array[9]),
        #   nn.ReLU(),
        #   nn.Dropout(dropoutProb),
        #   nn.Linear(layer_array[9], layer_array[10]),
        #   nn.ReLU(),
        #   nn.Dropout(dropoutProb),
        #   nn.Linear(layer_array[10], layer_array[11]),
        #   nn.ReLU(),
        #   nn.Dropout(dropoutProb),
        #   nn.Linear(layer_array[11], layer_array[12]),
        #   nn.ReLU(),
        #   nn.Dropout(dropoutProb),
        #   nn.Linear(layer_array[12], layer_array[13]),
        #   nn.ReLU(),
        #   nn.Dropout(dropoutProb),
        #   nn.Linear(layer_array[13], layer_array[14]),
        #   nn.ReLU(),
        #   nn.Dropout(dropoutProb),
        #   nn.Linear(layer_array[14], layer_array[15]),
        #   nn.ReLU(),
        #   nn.Dropout(dropoutProb),
        #   nn.Linear(layer_array[15], layer_array[16]),
        #   nn.ReLU(),
        #   nn.Dropout(dropoutProb),
        #   nn.Linear(layer_array[16], layer_array[17]),
        #   nn.ReLU(),
        #   nn.Dropout(dropoutProb),
        #   nn.Linear(layer_array[17], layer_array[18]),
        #   nn.ReLU(),
        #   nn.Dropout(dropoutProb),
        #   nn.Linear(layer_array[18], layer_array[19]),
        #   nn.ReLU(),
        #   nn.Dropout(dropoutProb),
        #   nn.Linear(layer_array[19], layer_array[20]),
        #   nn.ReLU(),
        #   nn.Dropout(dropoutProb),

          nn.Linear(layer_array[-1], out_size)
        )
        
    def forward(self, xb):
        softmax = nn.LogSoftmax(dim=0)
        return softmax(self.network(xb))

    def training_step(self, batch):
        values, labels = batch 
        values = values.to(device)
        labels = labels.to(device)
        out = self(values)                  # Generate predictions
        loss = nn.functional.nll_loss(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        values, labels = batch 
        values = values.to(device)
        labels = labels.to(device)
        out = self(values)                    # Generate predictions
        loss = nn.functional.nll_loss(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

def predict(value, label, model):
    global correct_test_results
    global tp_p, tn_p, fp_p, fn_p
    global tp_c, tn_c, fp_c, fn_c
    # print("value process1\n\n\n", value)
    value_tensor = torch.from_numpy(value)
    # print("value process2\n\n\n", value_tensor)
    value = to_device(value_tensor, device)
    yb = model(value_tensor)
    _, preds  = torch.max(yb, dim=0)
    if (preds == label):
        correct_test_results += 1

    if(label == plasmid):
        if(preds == label):
            tp_p += 1
            tn_c += 1
        elif(preds != label):
            fn_p += 1
            fp_c += 1

    elif(label == chromosome):
        if(preds == label):
            tp_c += 1
            tn_p += 1
        elif(preds != label):
            fn_c += 1
            fp_c += 1

    return preds.item()

def plot_graphs(history):
    plot_accuracies(history)
    plot_losses(history)

def model_setup(trainingDataset):
    train_size = int(0.8 * len(trainingDataset))
    val_size = len(trainingDataset) - train_size
    print("training Data size and Validation Data sizes are", train_size, val_size)
    print('\nSplitting Training/Validation datasets....')
    train_ds, val_ds = random_split(trainingDataset, [train_size, val_size])

    # ind = torch.tensor(train_ds.indices)
    # tens = train_ds.dataset.get_label_values(ind)

    # weights = make_weights_for_balanced_classes(outputSize, tens)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, train_size)

    train_dl = DataLoader(train_ds, batchSize, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batchSize, num_workers=num_workers, pin_memory=True)

    model = Model(inputFeatures, layer_array, outputSize)
    model = to_device(model, device)
    model.double()

    return model, train_dl, val_dl

def model_fit(trainingDataset):

    # print("Training Dataset\n\n\n\n", len(trainingDataset))

    print("Start the training process of the model")
    model, train_dl, val_dl = model_setup(trainingDataset)
    print("Initialized the model")

    t = datetime.now()
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    print("Time for training the model", datetime.now()-t)

    plot_graphs(history)
    return model

def predict_model(testDataset, model):
    print("Start the testing process of the model")
    # print([element for element in testDataset])
    # testDataset = [(element[0].astype("float32"), element[1]) for element in testDataset]
    # print([element[1] for element in testDataset])
    test_results = [predict(element[0], element[1], model) for element in testDataset]
    return correct_test_results, test_results, tp_p, tn_p, fp_p, fn_p, tp_c, tn_c, fp_c, fn_c
    


