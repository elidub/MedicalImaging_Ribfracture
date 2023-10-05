import sys
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from torchvision.transforms import ToTensor

sys.path.insert(1, sys.path[0] + '/..')
from src.data.dataset import CustomImageDataset, BoxesDataset

class DataModule(pl.LightningDataModule):
    def __init__(self, dir, dataset = 'images', batch_size = 2, num_workers = 0, splits = ['train', 'val', 'test']):
        super().__init__()
        self.dir = dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.splits = splits

    def setup(self, stage=None):

        if self.dataset == 'images':
            dataset = CustomImageDataset
            self.collate_fn = None
        elif self.dataset == 'boxes':
            dataset = BoxesDataset
            self.collate_fn = self.custom_collate
        else:
            raise ValueError('Unknown dataset type')
        
        self.datasets = { split : dataset(split = split, dir=self.dir) for split in self.splits }

    def custom_collate(self, batch):
        x, y = zip(*batch)  # Separate data and labels
        return torch.stack(x), torch.stack(y)

    def train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn = self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.datasets['val'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn = self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn = self.collate_fn)
    
    def predict_dataloader(self):
        return DataLoader(self.datasets['predict'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn = self.collate_fn)
    