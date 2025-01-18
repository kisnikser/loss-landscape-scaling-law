import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

def train(model,
          optimizer,
          criterion,
          train_dataloader,
          num_epochs, 
          device):

        model.train()
        # Train the network
        losses = []
        for epoch in tqdm(range(num_epochs), desc='Train loop', leave=False):  # loop over the dataset multiple times
            for x, y in train_dataloader:
                x, y = x.to(device), y.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                losses.append(loss.detach().cpu().item())
        return model, losses
