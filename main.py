import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models.resnet import ResNet18


import torch
from tqdm import tqdm
import torch.nn.functional as F

train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train_model(net, device, trainloader, optimizer, epoch):
    
    model.train()
    pbar = tqdm(trainloader)
    correct = 0
    processed = 0
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(pbar):
    # get samples
        data, target = data.to(device), target.to(device)

    # Init
        #optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
        y_pred = net(data)

    # Calculate loss

        loss = criterion(y_pred, target)
        #loss = F.normalize(loss)
        train_losses.append(loss)
        optimizer.zero_grad()
    # Backpropagation
        loss.backward()
        optimizer.step()

    # Update pbar-tqdm

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)
        
        return(train_losses,train_acc)
def test_model(net, device, testloader):
    
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += criterion(output, target).item()
            #test_loss = F.normalize(test_loss)
            pred =output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)
    test_losses.append(test_loss)

    print('\nTest set:  Accuracy: {}/{} ({:.2f}%)\n'.format(
         correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

    test_acc.append(100. * correct / len(testloader.dataset))
    return(test_losses,test_acc)


