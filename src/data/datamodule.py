from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from torchvision.transforms import ToTensor

from src.data.dataloader import CustomImageDataset

class DataModule(pl.LightningDataModule):
    def __init__(self, dir, batch_size = 1, num_workers = 0):
        super().__init__()
        self.dir = dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.datasets = { split : CustomImageDataset(split = split) for split in ['train', 'val', 'test'] }


    def train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.datasets['val'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    