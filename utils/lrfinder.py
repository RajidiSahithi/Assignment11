
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch_lr_finder import LRFinder

def lrfinder(model,device='cpu'):
  
  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

  lr_finder = LRFinder(model, optimizer, criterion, device="cpu")
  lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode="exp")
  lr_finder.plot() # to inspect the loss-learning rate graph
  lr_finder.reset() # to reset the model and optimizer to their initial state

  min_loss = min(lr_finder.history["loss"])
  max_lr = lr_finder.history["lr"][np.argmin(lr_finder.history["loss"], axis=0)]

  print("Min Loss = {}, Max LR = {}".format(min_loss, max_lr))


