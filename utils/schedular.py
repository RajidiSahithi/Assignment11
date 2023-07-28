import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
EPOCHS = 20
learning_rate=[]
scheduler = OneCycleLR(
        optimizer,
        max_lr=1.66E-03,
        steps_per_epoch=1,
        epochs=22,
        pct_start=5/EPOCHS,
        div_factor=100,
        three_phase=False,
        final_div_factor=100,
        anneal_strategy='linear'
    )


for epoch in range(EPOCHS):
    scheduler.step()
    print("EPOCH:", epoch)
    for param_group in optimizer.param_groups:
      print("lr= ",param_group['lr'])
    train_model(model, device, train_loader, optimizer, epoch)
    # scheduler.step()
    test_model(model, device, test_loader)


    learning_rate.append(param_group['lr']),