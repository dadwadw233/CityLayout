from CityDM_re import CityDM
import argparse
import os
import wandb
from accelerate import Accelerator
import torch
import torch.distributed as dist
from utils.log import *
from copy import deepcopy
from utils.config import ConfigParser
# define global variables
accelerator = None
sweep_id = None
path = None

def train_main(sweep_config):
    citydm = CityDM(path, sweep_config=sweep_config)
    citydm.train()

def sample_main(sweep_config, parser):
    config_manager = ConfigParser(parser.parse_args().config_manager)
    sample_path = config_manager.get_config_by_name("sample")[sweep_config["config_prefix"]]
    citydm = CityDM(sample_path, sweep_config=sweep_config)
    citydm.sample(cond=parser.parse_args().cond, eval=parser.parse_args().eval, best=parser.parse_args().best)

def train_accelerate():
    cmd = "accelerate launch src/accelerate_handle.py --path {} --sweep_id {}".format(path, sweep_id)
    # launch the subprocess and wait for it to finish
    INFO("accelerate launch begin")
    os.system(cmd)
    INFO("accelerate launch finished")




parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="/home/admin/workspace/yuyuanhong/code/CityLayout/config/new/outpainting_train.yaml")
parser.add_argument("--path_sample", type=str, default="/home/admin/workspace/yuyuanhong/code/CityLayout/config/new/inpainting_sample_citygen.yaml")
parser.add_argument("--config_manager", type=str, default="./config/config_manage.yaml")
parser.add_argument("--config_prefix", type=str, default="op_sample")
parser.add_argument("--sample", action="store_true", default=False)
parser.add_argument("--sample_all", action="store_true", default=False)
parser.add_argument("--train", action="store_true", default=False)
parser.add_argument("--cond", action="store_true", default=False)
parser.add_argument("--eval", action="store_true", default=False)
parser.add_argument("--best", action="store_true", default=False)
parser.add_argument("--multigpu", action="store_true", default=False)
parser.add_argument("--sweep", action="store_true", default=False)
parser.add_argument("--sweep_id", type=str, default=None, help="multi gpu for single task sweep training, \
                    by specifying the sweep id to record the training process in the same sweep")

parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--wandb", action="store_true", default=False)


if parser.parse_args().sweep:
    sweep_config_train = {
        'method': 'bayes',
        'metric': {
            'name': 'val.IS',
            'goal': 'maximize'
        },
        'parameters': {
            'lr': {
                'min': 0.00001,
                'max': 0.0001
            },
            'objective': {
                'values': ['pred_v', 'pred_noise', 'pred_x0']
            },
            'beta_schedule':
            {
                'values': ['sigmoid', 'cosine', 'linear']
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

if parser.parse_args().config_manager is not None:
    config_manager = ConfigParser(parser.parse_args().config_manager)

    if parser.parse_args().config_prefix is not None:
        if parser.parse_args().sample:
            sample_path = config_manager.get_config_by_name("sample")[parser.parse_args().config_prefix]
        else:
            path = config_manager.get_config_by_name("train")[parser.parse_args().config_prefix]

if not parser.parse_args().sweep:
    if parser.parse_args().sample:
        if parser.parse_args().sample_all:
            INFO("sample all")
            for config_prefix in config_manager.get_config_by_name("sample"):
                INFO("sample {}".format(config_prefix))
                sample_path = config_manager.get_config_by_name("sample")[config_prefix]
                citydm = CityDM(sample_path)
                citydm.sample(cond=parser.parse_args().cond, eval=parser.parse_args().eval, best=parser.parse_args().best, use_wandb=parser.parse_args().wandb)
        else:
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
        raise NotImplementedError("sample with sweep is not implemented yet")


        

    