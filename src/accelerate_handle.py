from CityDM import CityDM
import argparse
import os
import wandb
from accelerate import Accelerator
import torch
import torch.distributed as dist
from utils.log import *
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="/home/admin/workspace/yuyuanhong/code/CityLayout/config/new/normal_train_multigpu.yaml")
parser.add_argument("--sweep_id", type=str, default=None)

# main function
def train_main():
    # wait for father process to send params
    
    citydm = CityDM(config_path=parser.parse_args().path, sweep_config=parser.parse_args().sweep_id, ddp=True)
    citydm.train()

if __name__ == "__main__":

    train_main()


        

    