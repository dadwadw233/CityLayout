
import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from CityDM_lightning_model import PL_CityDM
from utils.log import *
import os


def train(config: DictConfig):
    INFO(os.getcwd())
    CityDM: LightningModule = hydra.utils.instantiate(config.all)
    INFO(CityDM)
    print('Training...')
    
def sample(config: DictConfig):
    print('Sampling...')

@hydra.main(config_path='../config/new', config_name='refactoring_train.yaml')
def main(config: DictConfig):
    if config.all.Main.mode == 'train':
        train(config)
    elif config.all.Main.mode == 'sample':
        sample(config)
    else:
        raise ValueError(f'Invalid mode: {config.mode}')

if __name__ == '__main__':
    main()