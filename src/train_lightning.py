
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
from rich.console import Console
console = Console()
import warnings
warnings.simplefilter('ignore')

def handle(config: DictConfig):
    try:
        
        CityDM: LightningModule = hydra.utils.instantiate(config.CityDM)
        OSMData: LightningDataModule = hydra.utils.instantiate(config.Data)
        
        OSMData.setup()
        if config.CityDM.Main.logger != "wandb":
            logger = None
        else:
            logger = hydra.utils.instantiate(config.private.wandb.config)

        if "callbacks" in config:
            INFO("Callbacks found in config")
            callbacks: List[Callback] = [hydra.utils.instantiate(config.callbacks[c]) for c in config.callbacks]
        else:
            callbacks = []
        
        
        trainer = hydra.utils.instantiate(config.Trainer, 
                                        logger=logger,
                                        callbacks=callbacks
                                        )
        
        if config.CityDM.Main.mode == "train":
            trainer.fit(CityDM, datamodule=OSMData)
        elif config.CityDM.Main.mode == "test":
            trainer.test(CityDM, datamodule=OSMData)
        elif config.CityDM.Main.mode == "sample":
            trainer.predict(CityDM, datamodule=OSMData)
        else:
            raise ValueError(f"mode {config.CityDM.Main.mode} not supported!")
    except Exception as e:
        ERROR(f"Error: {e}")
        console.print_exception()
        raise e
    
    
@hydra.main(config_path='../config', config_name='refactoring_train.yaml')
def main(config: DictConfig):
    handle(config)

if __name__ == '__main__':
    main()