
import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningModule,Trainer, LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import Callback
from CityDM_lightning_model import PL_CityDM
from typing import List
from utils.log import *
import os
from datasets.osm_datamodule import OSMDataModule


def train(config: DictConfig):
    
    CityDM: LightningModule = hydra.utils.instantiate(config.CityDM)
    OSMData: LightningDataModule = hydra.utils.instantiate(config.Data)
    
    OSMData.setup()
    if config.CityDM.Main.debug :
        logger = None
    else:
        logger = hydra.utils.instantiate(config.private.config)

    if "callbacks" in config:
        INFO("Callbacks found in config")
        callbacks: List[Callback] = [hydra.utils.instantiate(config.callbacks[c]) for c in config.callbacks]
    else:
        callbacks = []
    
    
    trainer = hydra.utils.instantiate(config.Trainer, 
                                      logger=logger,
                                      callbacks=callbacks
                                      )
    
    trainer.fit(CityDM, datamodule=OSMData)
    
    exit()
    
    print('Training...')
    
def sample(config: DictConfig):
    print('Sampling...')

@hydra.main(config_path='../config', config_name='refactoring_train.yaml')
def main(config: DictConfig):
    if config.CityDM.Main.mode == 'train':
        train(config)
    elif config.CityDM.Main.mode == 'sample':
        sample(config)
    else:
        raise ValueError(f'Invalid mode: {config.mode}')

if __name__ == '__main__':
    main()