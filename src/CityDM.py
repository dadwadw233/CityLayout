import math
import copy
from pathlib import Path
import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from typing import Any

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam, lr_scheduler, SGD


from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from utils.fid_evaluation import FIDEvaluation

from utils.utils import (
    exists,
    cycle,
    num_to_groups,
    has_int_squareroot,
    divisible_by,
    OSMVisulizer,
    cal_overlapping_rate,
)

from model.version import __version__

from data.osm_loader import OSMDataset

from torch.utils.tensorboard import SummaryWriter

import time

from utils.log import *

from utils.config import ConfigParser
from model.Unet import Unet
from model.DDPM import GaussianDiffusion
import os
import inspect
from utils.evaluation import Evaluation
from utils.asset import AssetGen
import numpy as np

class CityDM(object):

    def __init__(self, config_path) -> None:
        super().__init__()

        assert config_path is not None, "config path is None!"
        assert os.path.exists(config_path), f"config file {config_path} does not exist!"
        self.config_path = config_path
        self.config_parser = ConfigParser(config_path=self.config_path)
        INFO(f"config file {self.config_path} loaded!")

        # Init Accelerator
        acc_config = self.config_parser.get_config_by_name("Accelerator")
        self.accelerator = Accelerator(**acc_config)
        INFO(f"Accelerator initialized!")
        

        try:
            # Init Model
            model_config = self.config_parser.get_config_by_name("Model")
            self.backbone =  Unet(**model_config['backbone'])


            # Init Diffusion
            diffusion_config = model_config["diffusion"]
            diffusion_config['model'] = self.backbone
            self.diffusion = GaussianDiffusion(**diffusion_config)
        except Exception as e:
            ERROR(f"Init Model failed! {e}")
            raise e
        
        # Init Dataset
        main_config = self.config_parser.get_config_by_name("Main")

        data_config = self.config_parser.get_config_by_name("Data")

        # init some key params:
        
        self.batch_size = data_config["batch_size"]
        self.num_workers = data_config["num_workers"]
        
        self.epochs = main_config["max_epochs"]
        self.lr = float(main_config["lr"])
        
        # some ema params
        self.ema_decay = main_config["ema_decay"]
        self.ema_update_every = main_config["ema_update_every"]

        self.adam_betas = main_config["adam_betas"]
        self.opt_type = main_config["opt"]
        self.scheduler_type = main_config["scheduler"]

        self.sample_frequency = main_config["sample_frequency"]
        self.grad_accumulate = main_config["grad_accumulate"]
        self.max_grad_norm = main_config["max_grad_norm"]


        self.mode = main_config["mode"]
        self.seed = main_config["seed"]
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

            INFO(f"Seed set to {self.seed}")

        # Init EMA
        if self.accelerator.is_main_process:
            self.ema = EMA(self.diffusion, beta=self.ema_decay, update_every=self.ema_update_every)
            self.ema.to(self.device)
            INFO(f"EMA initialized!")
            
        # Init Dataloader
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        if self.mode == "train":
            self.train_dataset = OSMDataset(config=data_config, mode="train")
            self.val_dataset = OSMDataset(config=data_config, mode="val")
            
            self.train_dataloader = self.accelerator.prepare(
                DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
            )
            self.val_dataloader = self.accelerator.prepare(
                DataLoader(
                    self.val_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                )
            )
            self.train_dataloader = cycle(self.train_dataloader)
            self.val_dataloader = cycle(self.val_dataloader)
            INFO(f"Train dataset and dataloader initialized!")
        else:
            self.test_dataset = OSMDataset(config=data_config, mode="test")
            self.test_dataloader = self.accelerator.prepare(
                DataLoader(
                    self.test_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                )
            )
            self.test_dataloader = cycle(self.test_dataloader)
            INFO(f"Test dataset and dataloader initialized!")

        

        # Validation prepare
        val_config = self.config_parser.get_config_by_name("Validation")
        self.save_best_and_latest_only = val_config["save_best_and_latest_only"]
        self.num_samples = val_config["num_samples"]
        self.results_dir = val_config["results_dir"] 
        # add time stamp to results_dir
        self.results_dir = os.path.join(self.results_dir, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
        self.vis_results_dir = os.path.join(self.results_dir, "vis")
        self.ckpt_results_dir = os.path.join(self.results_dir, "ckpt")
        self.asset_results_dir = os.path.join(self.results_dir, "asset")
        self.sample_results_dir = os.path.join(self.results_dir, "sample")

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.vis_results_dir, exist_ok=True)
        os.makedirs(self.ckpt_results_dir, exist_ok=True)
        os.makedirs(self.asset_results_dir, exist_ok=True)
        os.makedirs(self.sample_results_dir, exist_ok=True)
        # log some info
        INFO(f"Results dir: {self.results_dir}")
        INFO(f"Vis results dir: {self.vis_results_dir}")
        INFO(f"Checkpoint results dir: {self.ckpt_results_dir}")
        INFO(f"Asset results dir: {self.asset_results_dir}")
        INFO(f"Sample results dir: {self.sample_results_dir}")

        # utils prepare
        self.vis_config = self.config_parser.get_config_by_name("Vis")
        self.vis = OSMVisulizer(self.vis_config["channel_to_rgb"], self.vis_config["threshold"], self.vis_results_dir)
        self.asset_gen = AssetGen(self.config_parser.get_config_by_name("Asset"), path=self.asset_results_dir)
        INFO(f"Utils initialized!")



        
        # evaluation prepare
        eval_config = self.config_parser.get_config_by_name("Evaluation")
        self.evaluation = Evaluation(batch_size=self.batch_size, device=self.device,
                                        dl=self.val_dataloader if self.mode == "train" else self.test_dataloader,
                                        sampler=self.ema.ema_model,
                                        accelerator=self.accelerator,
                                        num_fid_samples=eval_config["num_fid_samples"],
                                        inception_block_idx=eval_config["inception_block_idx"],
                                        data_type=eval_config["data_type"],
                                        mapping=eval_config["channel_to_rgb"],
                        )
        # 👆 代码evaluation初始化还可以再优化 优化成全config的形式
        INFO(f"Evaluation initialized!")
        # print some info
        self.evaluation.summary()

        self.best_evaluation_result = None

        # Init Optimizer
        if self.opt_type == "adam":
            self.optimizer = Adam(self.diffusion.parameters(), lr=self.lr, betas=self.adam_betas)
        elif self.opt_type == "sgd":
            self.optimizer = SGD(self.diffusion.parameters(), lr=self.lr)
        else:
            ERROR(f"Optimizer type {self.opt_type} not supported!")
            raise ValueError(f"Optimizer type {self.opt_type} not supported!")
        
        # Init Scheduler
        if self.scheduler_type == "cosine":
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        elif self.scheduler_type == "step":
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        else:
            ERROR(f"Scheduler type {self.scheduler_type} not supported!")
            raise ValueError(f"Scheduler type {self.scheduler_type} not supported!")
        
        # prepare model, optimizer, scheduler with accelerator
        self.diffusion, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.diffusion, self.optimizer, self.scheduler
        )
        
        

        # todo init wandb
            

        # last some preperation for train
        self.now_epoch = 0
        self.now_step = 0
        # Done
        INFO(f"CityDM initialized!")


    
    @property
    def device(self):
        return self.accelerator.device
    

    def save_ckpts(self, epoch, step, best=False):
        if self.accelerator.is_main_process:
            ckpt_path = os.path.join(self.ckpt_results_dir, f"ckpt_{epoch}_{step}.pth")
            
            ckpt = {
                "epoch": epoch,
                "step": step,
                "diffusion": self.diffusion.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "ema": self.ema.state_dict(),
                "best_evaluation_result": self.best_evaluation_result,
                "seed": self.seed,
            }

            if best:
                ckpt_path = os.path.join(self.ckpt_results_dir, f"best_ckpt.pth")
                torch.save(ckpt, ckpt_path)
                INFO(f"Best ckpt saved to {ckpt_path}")
            else:
                torch.save(ckpt, ckpt_path)
                INFO(f"Ckpt saved to {ckpt_path}")
    
    def load_ckpts(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        self.diffusion.load_state_dict(ckpt["diffusion"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.ema.load_state_dict(ckpt["ema"])
        self.best_evaluation_result = ckpt["best_evaluation_result"]
        self.seed = ckpt["seed"]
        INFO(f"Ckpt loaded from {ckpt_path}")

        self.now_epoch = ckpt["epoch"]
        self.now_step = ckpt["step"]

    def config_summarize(self):
        INFO(self.config_parser.get_summary())

    def model_summarize(self):
        
        # print model param
        INFO("Model Summary:")
        INFO("========================================")
        INFO(self.diffusion)
        INFO("========================================")
        INFO("Model Summary End")
    
    def train(self):
        
        self.config_summarize()



        
            
        

        





        
