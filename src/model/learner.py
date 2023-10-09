import sys
import torch
import torch.nn as nn
import lightning.pytorch as pl

sys.path.insert(1, sys.path[0] + "/..")
from src.data.utils import crop
from src.model.modules import BoxLabelDecoder


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

    def step(self, batch, mode="train"):
        # Forward pass
        y_hat, y = self.forward(batch)

        # Loss
        loss = self.loss(y_hat, y)

        # Metrics
        acc = ((y_hat > 0.5) == y).float().mean()
        # better metrics to come ...

        # Log
        self.log(f"{mode}_loss", loss, prog_bar=True)
        self.log(f"{mode}_acc", acc, prog_bar=True)

        return loss, y_hat

    def training_step(self, batch, batch_idx):
        loss, y_hat = self.step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat = self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        loss, y_hat = self.step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def on_train_epoch_end(self):
        pass


class RetinanetLearner(pl.LightningModule):
    def __init__(self, net, loss):
        super().__init__()
        self.net = net
        self.loss = loss
        self.decoder = BoxLabelDecoder()

    def forward(self, batch):
        x, y_box, y_cls, info = batch
        x, y_box, y_cls = x.float(), y_box.float(), y_cls.long()
        y_box_hat, y_cls_hat = self.net(x)

        return y_box_hat, y_cls_hat, y_box, y_cls, info

    def step(self, batch, mode="train"):
        # Forward pass
        y_box_hat, y_cls_hat, y_box, y_cls, _ = self.forward(batch)

        # Loss
        box_loss, cls_loss = self.loss(y_box_hat, y_cls_hat, y_box, y_cls)
        box_loss, cls_loss = box_loss.mean(), cls_loss.mean()
        loss = box_loss + cls_loss

        # Log
        self.log(f"{mode}_box_loss", box_loss.item(), prog_bar=True)
        self.log(f"{mode}_cls_loss", cls_loss.item(), prog_bar=True)

        return loss, y_box_hat, y_cls_hat

    def training_step(self, batch, batch_idx):
        loss, y_box_hat, y_cls_hat = self.step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_box_hat, y_cls_hat = self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        loss, y_box_hat, y_cls_hat = self.step(batch, "test")

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            y_box_hat, y_cls_hat, _, _, info = self.forward(batch)
        y_box_hat, y_cls_hat = self.decoder.decode(y_box_hat, y_cls_hat)

        y_cls_hat_idx = y_cls_hat.argmax(dim=-1)
        y_cls_hat_idx = torch.zeros_like(y_cls_hat_idx)

        y_cls_hat_idx[:, :5] = 2

        y_box_hat = (torch.rand_like(y_box_hat) * 32).long()
        
        boxes_res = []
        for i, boxes in enumerate(y_box_hat):
            boxes_res.append(boxes[y_cls_hat_idx[i] == 2].long())


        return boxes_res, info

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    def on_train_epoch_end(self):
        pass
