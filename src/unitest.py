from CityDM import CityDM
import argparse
import os
import wandb
from accelerate import Accelerator
import torch
import torch.distributed as dist
from utils.log import *
from copy import deepcopy

# define global variables
accelerator = None
sweep_id = None
path = None

def train_main(sweep_config):
    citydm = CityDM(path, sweep_config=sweep_config)
    citydm.train()

def train_accelerate():
    cmd = "accelerate launch src/accelerate_handle.py --path {} --sweep_id {}".format(path, sweep_id)
    # launch the subprocess and wait for it to finish
    INFO("accelerate launch begin")
    os.system(cmd)
    INFO("accelerate launch finished")



parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="/home/admin/workspace/yuyuanhong/code/CityLayout/config/new/outpainting_train.yaml")
parser.add_argument("--path_sample", type=str, default="/home/admin/workspace/yuyuanhong/code/CityLayout/config/new/inpainting_sample.yaml")
parser.add_argument("--sample", action="store_true", default=False)
parser.add_argument("--train", action="store_true", default=False)
parser.add_argument("--cond", action="store_true", default=False)
parser.add_argument("--eval", action="store_true", default=False)
parser.add_argument("--best", action="store_true", default=False)
parser.add_argument("--multigpu", action="store_true", default=False)
parser.add_argument("--sweep", action="store_true", default=False)
parser.add_argument("--sweep_id", type=str, default=None, help="multi gpu for single task sweep training, \
                    by specifying the sweep id to record the training process in the same sweep")

parser.add_argument("--debug", action="store_true", default=False)

if parser.parse_args().sweep:
    sweep_config_train = {
        'method': 'bayes',
        'metric': {
            'name': 'val_cond.IS',
            'goal': 'maximize'
        },
        'parameters': {
            'lr': {
                'min': 0.00001,
                'max': 0.0001
            },
            'objective': {
                'values': ['pred_v', 'pred_noise']
            },
            'beta_schedule':
            {
                'values': ['sigmoid', 'cosine']
            },
            'timesteps': {
                'values': [2500]
            },
            'self_condition': {
                'values': [False]
            },
            'opt': {
                'values': ['adam']
            },
            'scheduler': {
                'values': ['cosine', 'step']
            },
        },
    }

else: 
    sweep_config_train = None
    sweep_config_sample = None

path = parser.parse_args().path
sample_path = parser.parse_args().path_sample

if not parser.parse_args().sweep:
    if parser.parse_args().sample:
        citydm = CityDM(sample_path)
        citydm.sample(cond=parser.parse_args().cond, eval=parser.parse_args().eval, best=parser.parse_args().best)    

    elif parser.parse_args().train:
        citydm = CityDM(path, debug = parser.parse_args().debug, ddp=parser.parse_args().multigpu)
        citydm.train()

else:
    if parser.parse_args().train:
        if parser.parse_args().multigpu:
            if parser.parse_args().sweep_id is None:
                sweep_id = wandb.sweep(sweep_config_train, project="CityLayout", entity="913217005")
            else:
                sweep_id = parser.parse_args().sweep_id

            wandb.agent(sweep_id, function=train_accelerate)

        else:
            if parser.parse_args().sweep_id is None:
                sweep_id = wandb.sweep(sweep_config_train, project="CityLayout", entity="913217005")
            else:
                sweep_id = parser.parse_args().sweep_id
            wandb.agent(sweep_id, function=lambda: train_main(sweep_config_train))

    elif parser.parse_args().sample:
        if parser.parse_args().sweep_id is None:
            sweep_id = wandb.sweep(sweep_config_sample, project="CityLayout", entity="913217005")
        else:
            sweep_id = parser.parse_args().sweep_id
        wandb.agent(sweep_id, function=lambda: train_main(sweep_config_sample))


        

    