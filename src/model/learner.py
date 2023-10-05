import lightning.pytorch as pl
import torch
from torch import nn
import sys

sys.path.insert(1, sys.path[0] + '/..')
from src.data.utils import crop

class Learner(pl.LightningModule):
    def __init__(self, net, loss):
        super().__init__()
        self.net = net
        self.loss = loss


    def forward(self, batch):
        x, y = batch
        x, y = x.float(), y.float()
        y_hat = self.net(x)
        # print('\nx.shape  ', x.shape, '\ny.shape  ', y.shape, '\ny_hat.shape', y_hat.shape)
        return y_hat, y
    
    def step(self, batch, mode = 'train'):

        # Forward pass
        y_hat, y = self.forward(batch)

        # Loss
        loss = self.loss(y_hat, y)

        # Metrics
        acc = ((y_hat > 0.5) == y).float().mean()
        # better metrics to come ...

        # Log
        self.log(f"{mode}_loss", loss, prog_bar=True)
        self.log(f"{mode}_acc",  acc,  prog_bar=True)

        return loss, y_hat

    def training_step(self, batch, batch_idx):
        loss, y_hat = self.step(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat = self.step(batch, 'val')

    def test_step(self, batch, batch_idx):
        loss, y_hat = self.step(batch, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def on_train_epoch_end(self):
        pass
