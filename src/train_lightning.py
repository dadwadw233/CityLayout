
import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningModule,Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import Callback
from CityDM_lightning_model import PL_CityDM
from utils.log import *
import os


def train(config: DictConfig):
    
    CityDM: LightningModule = hydra.utils.instantiate(config.all)

    wandb_config = config.private.wandb
    trainer = Trainer(
        max_epochs=config.all.Main.max_epochs,
        devices=config.all.Main.gpus,
        logger=WandbLogger(project=wandb_config.project, 
                           group=wandb_config.group, 
                           entity=wandb_config.entity, 
                           job_type=wandb_config.job_type),
        precision=config.all.Main.precision,
        accumulate_grad_batches=config.all.Main.grad_accumulate,
        val_check_interval=config.all.Main.sample_frequency,
        
    )
    
    
    trainer.fit(CityDM)
    
    exit()
    
    print('Training...')
    
def sample(config: DictConfig):
    print('Sampling...')

@hydra.main(config_path='../config', config_name='refactoring_train.yaml')
def main(config: DictConfig):
    if config.all.Main.mode == 'train':
        train(config)
    elif config.all.Main.mode == 'sample':
        sample(config)
    else:
        raise ValueError(f'Invalid mode: {config.mode}')

if __name__ == '__main__':
    main()