from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from utils.log import *
from .osm_loader import OSMDataset

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig

class OSMDataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # data config for initializing the datamodule
        self.config: DictConfig  = DictConfig(kwargs)
        INFO("OSMDataModule initialized")

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = OSMDataset(config = self.config, mode='train')
        self.val_dataset = OSMDataset(config = self.config, mode='val')
        self.test_dataset = OSMDataset(config = self.config, mode='test')
        INFO("data setup complete!")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers, drop_last=True)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers, drop_last=True)