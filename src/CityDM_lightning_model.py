import math
import copy
from pathlib import Path
import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from typing import Any
from datetime import timedelta
import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F

from torch.optim import Adam, lr_scheduler, SGD

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA
from omegaconf import OmegaConf

from utils.fid_evaluation import FIDEvaluation
from torch.cuda import amp
from utils.utils import (
    exists,
    cycle,
    num_to_groups,
    has_int_squareroot,
    divisible_by,
    cal_overlapping_rate,
)
from argparse import Namespace
from model.version import __version__

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
from utils.vis import OSMVisulizer
import numpy as np
import wandb
import json
import traceback
import torch.distributed as dist
from rich.console import Console
console = Console()
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from utils.model import init_backbone, init_diffuser
from rich.progress import Progress

class PL_CityDM(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # self.init_main()
    
    def init_basic_params(self):
        # init some key params:
        main_config = self.hparams.Main

        self.epochs = main_config["max_epochs"]
        self.lr = float(main_config["lr"])
        # some ema params
        self.ema_decay = main_config["ema_decay"]
        self.ema_update_every = main_config["ema_update_every"]

        self.adam_betas = main_config["adam_betas"]
        self.opt_type = main_config["opt"]
        self.scheduler_type = main_config["scheduler"]

        self.sample_frequency = main_config["sample_frequency"]
        self.max_grad_norm = main_config["max_grad_norm"]

        self.mode = main_config["mode"]
        self.seed = main_config["seed"]

        self.fine_tune = main_config["fine_tune"]

        if self.fine_tune:
            self.pretrain_model_type = main_config["pretrain_model_type"]
            self.pretrain_ckpt_type = main_config["pretrain_ckpt_type"]
            self.finetuning_type = main_config["finetuning_type"]
        else:
            self.pretrain_model_type = None
            self.pretrain_ckpt_type = None
            self.finetuning_type = None
            
        
            
    def get_ema(self, model):
        ema = EMA(model, beta=self.ema_decay, update_every=self.ema_update_every)
        return ema
    
    def opt_init(self):
        if self.opt_type == "adam":
            self.optimizer = Adam(self.generator.parameters(), lr=self.lr, betas=self.adam_betas)
        elif self.opt_type == "sgd":
            self.optimizer = SGD(self.generator.parameters(), lr=self.lr)
        else:
            ERROR(f"Optimizer type {self.opt_type} not supported!")
            raise ValueError(f"Optimizer type {self.opt_type} not supported!")
    
    def scheduler_init(self):
        if self.scheduler_type == "cosine":
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        elif self.scheduler_type == "step":
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.sample_frequency * 3, gamma=0.9)
        else:
            ERROR(f"Scheduler type {self.scheduler_type} not supported!")
            raise ValueError(f"Scheduler type {self.scheduler_type} not supported!")
    
    def create_folder(self, mode):
        
        self.asset_results_dir = os.path.join(self.results_dir, "asset")
        self.sample_results_dir = os.path.join(self.results_dir, "sample")
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.asset_results_dir, exist_ok=True)
        if mode == "train":
            self.val_results_dir = os.path.join(self.results_dir, "val")
            self.ckpt_results_dir = os.path.join(self.results_dir, "ckpt")
            os.makedirs(self.ckpt_results_dir, exist_ok=True)
            os.makedirs(self.val_results_dir, exist_ok=True)
        else:
            os.makedirs(self.sample_results_dir, exist_ok=True)
    
    def utils_init(self):
        # utils prepare
        self.vis_config = self.hparams.Vis
        if self.mode == "train":
            self.vis = OSMVisulizer(config=self.vis_config, path=self.val_results_dir)
        else:
            self.vis = OSMVisulizer(config=self.vis_config, path=self.sample_results_dir)
        self.asset_gen = AssetGen(self.hparams.Asset, path=self.asset_results_dir)
        INFO(f"Utils initialized!")

            
    def trainer_init(self):
        val_config = self.hparams.Validation
        self.save_best_and_latest_only = val_config["save_best_and_latest_only"]
        self.num_samples = val_config["num_samples"]
        self.results_dir = val_config["results_dir"]
        
        self.sample_type = val_config["sample_type"]

        
        self.results_dir = os.path.join(self.results_dir, self.model_type)
        self.results_dir = os.path.join(self.results_dir, time.strftime("%Y-%m-%d-%H-%M-%S",
                                                                        time.localtime())) + '-train-' + self.sample_type

        
        self.pretrain_ckpt_dir = os.path.join(val_config["fine_tune_dir"], "ckpt")
        self.pretrain_ckpt_type = self.hparams.Main["pretrain_ckpt_type"]

        # Init Optimizer
        self.opt_init()

        # Init Scheduler
        self.scheduler_init()
        
        
        self.completion = False
        self.refiner = None
    
    def sample_init(self):
        test_config = self.hparams.Test
        self.num_samples = test_config["num_samples"]
        self.results_dir = test_config["results_dir"]
        self.pretrain_ckpt_dir = test_config["ckpt_results_dir"]
        if self.pretrain_ckpt_dir is None or not os.path.exists(self.pretrain_ckpt_dir) or not os.path.isdir(
                self.pretrain_ckpt_dir):
            ERROR(f"pretrain_ckpt_dir {self.pretrain_ckpt_dir} does not exist!")
            raise ValueError(f"pretrain_ckpt_dir {self.pretrain_ckpt_dir} does not exist!")
        # add time stamp to results_dir
        # check if results_dir exists
        self.sample_type = test_config["sample_type"]


        self.results_dir = os.path.join(self.results_dir, self.model_type) + '-sample-' + self.sample_type
        self.results_dir = os.path.join(self.results_dir,
                                        time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())) + '-test'

        self.asset_results_dir = os.path.join(self.results_dir, "asset")
        self.sample_results_dir = os.path.join(self.results_dir, "sample")

        
            
        if test_config["use_completion"]:
            self.completion = True
            # LOAD COMPLETION MODEL
            backbone_config = self.hparams.Model["backbone"]
            backbone_config["channels"] = 6
            
            self.completion_backbone = init_backbone(backbone_config)
            self.refiner = init_diffuser(self.completion_backbone, self.hparams.Model["completion"])
            
            
            INFO(f"Completion model initialized!")
            self.refiner_ckpt = test_config["refiner_ckpt"]
            self.refiner_ckpt_type = test_config["refiner_ckpt_type"]
            
        else:
            self.completion = False
            self.refiner = None
        
        self.generator_ckpt_type = test_config["generator_ckpt_type"]
        
    def setup(self, stage):
        self.timesteps = self.hparams.Model.diffusion.timesteps
        self.data_type = self.hparams.Evaluation.data_type
        
        try:
            # Init Model
            model_config = self.hparams.Model
            backbone_config = model_config.backbone
            self.backbone = init_backbone(backbone_config)

            # Init Diffusion generator (aka. diffuser)
            diffusion_config = self.hparams.Model.diffusion
            self.generator = init_diffuser(self.backbone, diffusion_config)
            self.model_type = diffusion_config.model_type
            self.generator_ckpt_type = None
            
            
        except Exception as e:
            ERROR(f"Init Model failed! {e}")
            # log traceback
            console.print_exception(show_locals=True)
            raise e

        self.init_basic_params()
        
        if self.seed is not None:
            seed_everything(self.seed)

        self.batch_size = self.hparams.Main.batch_size
        # Validation prepare
        
        if self.mode == "train":
            self.trainer_init()
            
        elif self.mode == "test" or self.mode == "sample":
            self.sample_init()
            
            
        self.best_evaluation_result = None

        #  some preperation for train
        self.now_epoch = 0
        self.now_step = 0
        self.best_validation_result = None

        self.not_best = 0
        
        # step2: if finetuning or sampling, load ckpt
        if self.fine_tune or self.mode == "test" or self.mode == "sample":
            INFO(f"Fine tune mode or test mode, load ckpt from {self.pretrain_ckpt_dir}")
            if self.pretrain_ckpt_type == "best" or self.generator_ckpt_type == "best":
                ckpt_path = os.path.join(self.pretrain_ckpt_dir, "best_ckpt.pth")
            else:
                ckpt_path = os.path.join(self.pretrain_ckpt_dir, "latest_ckpt.pth")
            self.load_ckpts(self.generator, ckpt_path, mode=self.mode)
            
            INFO(f"ckpt loaded!")
            
            if (self.mode == "test" or self.mode == "sample") and self.refiner is not None:
                if self.refiner_ckpt_type == "best":
                    ckpt_path = os.path.join(self.refiner_ckpt, "best_ckpt.pth")
                else:
                    ckpt_path = os.path.join(self.refiner_ckpt, "latest_ckpt.pth")
                INFO(f"Refiner ckpt path: {ckpt_path}")
                self.load_ckpts(self.refiner, ckpt_path, mode=self.mode)
                INFO(f"Refiner ckpt loaded!")
        else:
            INFO(f"Train from scratch!")
            
        self.generator_ema = None
        self.evaluator = None
        self.refiner_ema = None
        self.folder_init = False
        self.outputs = []
        self.samples_for_test = {"real": [], "fake": [], "cond": []}
        # confirm sample type
        self.generator.set_sample_type(self.sample_type)
        if self.refiner is not None:
            self.refiner.set_sample_type("normal")

        # Init EMA
        self.generator_ema = self.get_ema(self.generator)
        INFO(f"EMA initialized!")
        
        if self.refiner is not None:
            self.refiner_ema = self.get_ema(self.refiner)
            INFO(f"Refiner EMA initialized!")
        INFO("CityDM (Lightning) init done!")
        
    def predict(self, *args, **kwargs):
        self.sample()
    
    def configure_optimizers(self):        
        return [self.optimizer], [self.scheduler]
    
    def training_step(self, batch, batch_idx):
        
        layout = batch["layout"].to(self.device)
        with amp.autocast(enabled=True):
            loss = self.generator(layout)
            
        self.log("train/loss", loss, logger=True, on_step=True, on_epoch=False, prog_bar=True, rank_zero_only=True)
        
        return {"loss": loss}
    
    def on_train_epoch_end(self):
        # renew ema model
        INFO(f"Epoch {self.current_epoch} finished!")
        # TODO:   check this work or not
        self.generator_ema.update()
    
    def convert_to_serializable(self, obj):
        if isinstance(obj, DictConfig):
            return {k: self.convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return [self.convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    def validation_step(self, batch, batch_idx):
        if self.trainer.is_global_zero and not self.folder_init:
            self.create_folder("train")
            # log some info
            INFO(f"Results dir: {self.results_dir}")
            # utils prepare
            self.utils_init()
            serializable_hparams = self.convert_to_serializable(self.hparams)
            with open(os.path.join(self.results_dir, "config.json"), "w") as f:
                f.write(json.dumps(serializable_hparams, indent=4))
            self.folder_init = True
            
        
        self.generator_ema.ema_model.eval()
        val_cond = False
        if self.model_type == "Completion" or self.model_type == "CityGen":
            val_cond = True
        if val_cond:
            layout = batch["layout"].to(self.device)
        else:
            layout = None
        with amp.autocast(enabled=True):
            sample = self.generator_ema.ema_model.sample(batch_size=self.batch_size, cond=layout)

        if self.trainer.num_devices > 1:
            sample = self.all_gather(sample).flatten(0, 1)
        
        self.outputs.append({"sample": sample})
        
        self.generator_ema.ema_model.train()
        
        return {"sample": sample}

    def on_validation_epoch_end(self):
        pass
        
    
    def test_step(self, batch, batch_idx):
        if not self.folder_init:
            if self.trainer.is_global_zero:
                self.create_folder("test")
                # log some info
                INFO(f"Results dir: {self.results_dir}")
                # utils prepare
                self.folder_init = True
                self.utils_init()
        
        
        
        self.generator_ema.ema_model.eval()
        if self.refiner is not None:
            self.refiner_ema.ema_model.eval()
            
        cond = False
        self_cond = self.model_type == "normal" and self.generator.get_sample_type() != "normal"
        if self.model_type == "Completion" or self.model_type == "CityGen" or self_cond:
            cond = True
        if cond:
            layout = batch["layout"].to(self.device)
        else:
            layout = None
        channel = batch['layout'].shape[1]
        with amp.autocast(enabled=True):
            sample = self.generator_ema.ema_model.sample(batch_size=self.batch_size, cond=layout)[:, :channel, ...]
            
            if self.refiner is not None:
                zero_mask = torch.zeros(sample.shape, device=self.device)
                sample = self.refiner_ema.ema_model.sample(batch_size=self.batch_size, cond=sample, mask=zero_mask)[:, :channel, ...]
            
        if self.trainer.num_devices > 1:
            sample = self.all_gather(sample).flatten(0, 1)
            real = self.all_gather(batch["layout"]).flatten(0, 1)
            if cond or self_cond:
                layout = self.all_gather(layout).flatten(0, 1)
        
        if self.global_rank == 0:
            self.samples_for_test["fake"].append(sample.detach().cpu())
            self.samples_for_test["real"].append(real.detach().cpu())
            if cond or self_cond:
                self.samples_for_test["cond"].append(layout.detach().cpu())        
        
        self.generator_ema.ema_model.train()
        if self.refiner is not None:
            self.refiner_ema.ema_model.train()
            
        
        
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if self.trainer.is_global_zero and not self.folder_init:
            self.create_folder("test")
            # log some info
            INFO(f"Results dir: {self.results_dir}")
            # utils prepare
            self.utils_init()
            
            self.folder_init = True
        pass
    
    def on_predict_epoch_end(self):
        self.sample(num_samples=self.num_samples)
        return


    def check_best_or_not(self, result):
        if self.best_evaluation_result is None and result is not None:
            self.best_evaluation_result = result
            return True
        elif result is None:
            return False
        else:
            if "SSIM" in result.keys():
                if self.best_evaluation_result["SSIM"] < result["SSIM"]:
                    self.best_evaluation_result = result
                    return True
                else:
                    return False
            elif self.best_evaluation_result["IS"] < result["IS"]:
                self.best_evaluation_result = result
                return True
            else:
                return False

    def load_model_params(self, tgt, model_state_dict, load_type):
        if load_type == "full" or load_type == None:
            tgt.load_state_dict(model_state_dict, strict=True)
        elif load_type == "partial":
            tgt.load_state_dict(self.partly_load(model_state_dict), strict=False)
        elif load_type == "LoRA":
            ERROR(f"LoRA not supported yet!")
            raise NotImplementedError(f"LoRA not supported yet!")
        else:
            ERROR(f"load_type {load_type} not supported!")
            raise ValueError(f"load_type {load_type} not supported!")

    def partly_load(self, param_dict):
        # pre-train model may not suitable for current model, so we load pre-train model's params
        # which is suitable for current model and zero init other new params

        padded_dict = self.generator.state_dict()
        for key in padded_dict.keys():
            if key not in param_dict.keys():
                padded_dict[key] = torch.zeros_like(padded_dict[key])
            else:
                # if shape of param_dict[key] is not equal to padded_dict[key], then pad it with zeros
                if padded_dict[key].shape != param_dict[key].shape:
                    padded_dict[key] = torch.zeros_like(padded_dict[key])
                    # fullfill the same shape part
                    padded_dict[key][:, 0:param_dict[key].shape[1], ...] = param_dict[key]
                else:
                    padded_dict[key] = param_dict[key]

        return padded_dict
    
    def registra_LoRAlayer(self, model, param_dict):
        # register LoRA layer in model
        # model: model to be registered
        # param_dict: param dict of pretrain model
        # return: model with LoRA layer registered
        for name, param in model.named_parameters():
            if name in param_dict.keys():
                param.requires_grad = False
        return model
    
    def LoRA_load(self, param_dict):
        
        # freeze param dict. (besides conv_in and feed forward layer in transformer)
        pass

    def save_ckpts(self, epoch, step, best=False, latest=False):
        if self.trainer.is_global_zero:
            if not best and not latest:
                ckpt_path = os.path.join(self.ckpt_results_dir, f"ckpt_{epoch}_{step}.pth")
            elif best:
                ckpt_path = os.path.join(self.ckpt_results_dir, f"best_ckpt.pth")
            else:
                ckpt_path = os.path.join(self.ckpt_results_dir, f"latest_ckpt.pth")

            ckpt = {
                "epoch": epoch,
                "step": step,
                "diffusion": self.generator.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "ema": self.generator_ema.state_dict(),
                "best_evaluation_result": self.best_evaluation_result,
                "seed": self.seed,
                'scaler': self.trainer.precision_plugin.state_dict() if self.trainer.precision_plugin is not None else None
            }
            

            if best:
                ckpt_path = os.path.join(self.ckpt_results_dir, f"best_ckpt.pth")
                torch.save(ckpt, ckpt_path)
                INFO(f"Best ckpt saved to {ckpt_path}")
            else:
                torch.save(ckpt, ckpt_path)
                INFO(f"Ckpt saved to {ckpt_path}")

    def load_ckpts(self, model, ckpt_path, mode="train"):
        ckpt = torch.load(ckpt_path)
        
        self.load_model_params(model, ckpt["diffusion"], self.finetuning_type)
        
            # self.optimizer.load_state_dict(ckpt["optimizer"])
        
        if 'scaler' in ckpt.keys() and self.trainer.precision_plugin is not None:
            INFO(f"load scaler state dict!")
            self.trainer.precision_plugin.load_state_dict(ckpt['scaler'])

        if mode == "train":
            self.opt_init()
            self.scheduler_init()
            INFO(f"reinit optimizer!")
            # self.optimizer.load_state_dict(ckpt["optimizer"])
            
        INFO(f"{ckpt['scaler']}")
        INFO(f"Ckpt loaded from {ckpt_path}")


    def get_op_mask(self, ratio=0.5, d='right', shape=(1, 3, 256, 256)):
        # ratio: the ratio of the mask
        # d: direction of the mask
        # shape: the shape of the layout patch

        # return binary mask 1 for op region, 0 for condtion region

        b, c, h, w = shape

        mask = torch.zeros((b, c, h, w), device=self.device)
        if d == 'left':
            mask[:, :, :, :int(w * ratio)] = 1
        elif d == 'right':
            mask[:, :, :, int(w * (1 - ratio)):] = 1
        elif d == 'up':
            mask[:, :, :int(h * ratio), :] = 1
        elif d == 'down':
            mask[:, :, int(h * (1 - ratio)):, :] = 1

        return mask

    def op_handle(self, org, padding=1, ratio=0.5, bar=None):
        # extend origin layout by sliding window 
        b, c, h, w = org.shape
        INFO(f"start extend layout with padding {padding} and ratio {ratio}...")
        extend_h = int(h + padding * h)
        extend_W = int(w + padding * w)

        extend_layout = torch.zeros((b, c, extend_h, extend_W), device=self.device)

        # org settle in the left top corner
        extend_layout[:, :, :h, :w] = org
        plane = extend_layout.clone()
        # slide direction: left to right; up to down
        init = False
        #with Progress() as progress:
            #task = progress.add_task("[red]Extending layout...", total=None)
        task = bar.add_task("[red]Extending layout...", total=None)
        for h_l_p in range(0, int(extend_h - h * ratio), int(h * ratio)):
            for w_l_p in range(0, int(extend_W - w * ratio), int(w * ratio)):
                # direction: left to right
                if init == False:
                    if w_l_p + int(w * ratio) + w > extend_W:
                        break
                    cond_region = extend_layout[:, :, h_l_p:h_l_p + h,
                                    w_l_p + int(w * ratio):w_l_p + int(w * ratio) + w]
                    # DEBUG(f"cond_region shape: {cond_region.shape}")
                    op_mask = self.get_op_mask(ratio=ratio, d='right', shape=(b, 1, h, w))
                    # DEBUG(f"op_mask shape: {op_mask.shape}")
                    # DEBUG(f"hlp: {h_l_p}, wlp: {w_l_p}")
                    # sample from model

                    extend_region = self.generator_ema.ema_model.sample(batch_size=b, cond=cond_region, mask=op_mask)[:, :c,
                                    ...]
                    
                    if self.refiner is not None:
                        zero_mask = torch.zeros((b, 1, 256, 256), device=self.device)
                        refine = self.refiner_ema.ema_model.sample(batch_size=b, cond= (extend_region > 0.7).float(), mask=zero_mask)[:, :3, ...]
                        extend_region = refine

                    extend_layout[:, :, h_l_p:h_l_p + h,
                    w_l_p + int(w * ratio):w_l_p + int(w * ratio) + w] = extend_region
                     

                else:
                    cond_region = extend_layout[:, :, h_l_p:h_l_p + h, w_l_p:w_l_p + w]

                    op_mask = self.get_op_mask(ratio=ratio, d='down', shape=(b, 1, h, w))
                    # DEBUG(f"hlp: {h_l_p}, wlp: {w_l_p}")
                    extend_region = self.generator_ema.ema_model.sample(batch_size=b, cond=cond_region, mask=op_mask)[:, :c,
                                    ...]
                    if self.refiner is not None:
                        zero_mask = torch.zeros((b, 1, 256, 256), device=self.device)
                        refine = self.refiner_ema.ema_model.sample(batch_size=b, cond= (extend_region > 0.7).float(), mask=zero_mask)[:, :3, ...]
                        extend_region = refine
                    extend_layout[:, :, h_l_p:h_l_p + h, w_l_p:w_l_p + w] = extend_region

                bar.update(task, advance=1)

            init = True

        if self.refiner is not None:
            zero_mask = torch.zeros((b, 1, extend_h, extend_W), device=self.device)
            # for cuda memory consern, refine batch samples on by one
            refine = extend_layout.clone()
            task = bar.add_task("[red]Refining layout...", total=b)
            for i in range(b):
                refine[i, :, :, :] = self.refiner_ema.ema_model.sample(batch_size=1, 
                                                                       cond=extend_layout[i:i+1, :3, ...].clone(), 
                                                                       mask=zero_mask[i:i+1, ...])[:, :3, ...]
                bar.update(task, advance=1)
            extend_layout = torch.cat((refine, extend_layout), dim=1)
            
        INFO(f"extend layout done!")    

        return torch.cat((extend_layout, plane), dim=1)

    def get_inpaint_mask(self, shape=(1, 3, 256, 256), bbox=None):
        # shape: the shape of the layout patch
        # bbox: the bbox of the inp region : shape (b, 4) (x1, y1, x2, y2)

        # return binary mask 1 for inp region, 0 for condtion region

        # if bbox is None, then random generate a bbox

        b, c, h, w = shape

        if bbox is None:
            bbox = torch.zeros((b, 4), device=self.device)
            for bid in range(b):
                # use same bbox for all batch
                x1 = torch.randint(0, w // 2, (1,)).item()
                y1 = torch.randint(0, h // 2, (1,)).item()
                x2 = torch.randint(w // 2, w, (1,)).item()
                y2 = torch.randint(h // 2, h, (1,)).item()

                bbox[bid, 0] = x1
                bbox[bid, 1] = y1
                bbox[bid, 2] = x2
                bbox[bid, 3] = y2

        else:
            # check the bbox region range 
            if (bbox[:, 0] < 0).any() or (bbox[:, 1] < 0).any() or (bbox[:, 2] > w).any() or (bbox[:, 3] > h).any():
                WARNING(f"bbox region out of range, reset bbox to None!")
                # clip the bbox region
                bbox[:, 0] = torch.clip(bbox[:, 0], 0, w)
                bbox[:, 1] = torch.clip(bbox[:, 1], 0, h)
                bbox[:, 2] = torch.clip(bbox[:, 2], 0, w)
                bbox[:, 3] = torch.clip(bbox[:, 3], 0, h)

        mask = torch.ones((b, 1, h, w), device=self.device)
        for i in range(b):
            mask[i, :, bbox[i, 1].int():bbox[i, 3].int(), bbox[i, 0].int():bbox[i, 2].int()] = 0  # condition region

        return mask

    def inp_handle(self, org, bbox=None):
        # handle inpainting situation
        # org: the original layout
        # bbox: the bbox of the inp region : shape (b, 4) (x1, y1, x2, y2)

        mask = self.get_inpaint_mask(shape=org.shape, bbox=bbox)
        if self.refiner is not None:
            sample = self.generator_ema.ema_model.sample(batch_size=org.shape[0], cond=org, mask=mask)
            zero_mask = torch.zeros((org.shape[0], 1, 256, 256), device=self.device)
            sample = (sample > 0.7).float()
            refine = self.refiner_ema.ema_model.sample(batch_size=org.shape[0], cond=sample[:, :3, ...].clone(), mask=zero_mask)[:, :3, ...]
            sample = torch.cat((refine, sample, org), dim=1)
            return sample
        else:
            sample = self.generator_ema.ema_model.sample(batch_size=org.shape[0], cond=org, mask=mask)
            return torch.cat((sample, org), dim=1)


    def sample(self, cond=False, num_samples=16):
        
        if self.global_rank == 0:
        
            INFO(f"Start sampling {num_samples} images...")
            INFO(F"Sample result save to {self.sample_results_dir}")

            INFO(f"sample mode: {'random' if not cond else 'conditional'}")

            with torch.inference_mode():

                batches = num_to_groups(num_samples, self.batch_size)

                if self.sample_type == "Outpainting":
                    all_images = None
                    padding = self.hparams.Test['op']["padding"]
                    ratio = self.hparams.Test['op']["ratio"]
                    with Progress() as progress:
                        task = progress.add_task("[green]outpainting sampling", total=len(batches))
                        for b in range(len(batches)):
                            cond_image = next(cycle(self.trainer.predict_dataloaders))["layout"].to(self.device)
                            # DEBUG(f"cond_image shape: {cond_image.shape}")
                            if all_images is None:
                                all_images = self.op_handle(cond_image, padding=padding, ratio=ratio, bar=progress)
                            else:
                                all_images = torch.cat(
                                    (all_images, self.op_handle(cond_image, padding=padding, ratio=ratio, bar=progress)), dim=0
                                )
                            progress.update(task, advance=1)


                elif self.sample_type == "Inpainting":
                    all_images = None
                    with Progress() as progress:
                        task = progress.add_task("[green]inpaiting sampling", total=len(batches))
                        for b in range(len(batches)):
                            cond_image = next(cycle(self.trainer.predict_dataloaders))["layout"].to(self.device)
                            # DEBUG(f"cond_image shape: {cond_image.shape}")
                            if all_images is None:
                                all_images = self.inp_handle(cond_image)
                            else:
                                all_images = torch.cat(
                                    (all_images, self.inp_handle(cond_image)), dim=0
                                )
                            progress.update(task, advance=1)

                else:
                    if cond is True:
                        # sample some real images from val_Dataloader as cond
                        all_images = None
                        with Progress() as progress:
                            task = progress.add_task("[green]sampling", total=len(batches))
                            for b in range(len(batches)):
                                cond_image = next(cycle(self.trainer.predict_dataloaders))["layout"].to(self.device)
                                sample = self.generator_ema.ema_model.sample(batch_size=batches[b], cond=cond_image)[:, :3, ...]
                                
                                if self.refiner is not None:
                                    zero_mask = torch.zeros((batches[b], 1, 256, 256), device=self.device)
                                    sample = (sample > 0.7).float()
                                    refiner_sample = self.refiner_ema.ema_model.sample(batch_size=batches[b], cond=sample[:, :3, ...].clone(), mask=zero_mask)[:, :3, ...]
                                    sample = torch.cat((refiner_sample, sample[:, :3, ...]), dim=1)
                                
                                if all_images is None:
                                    all_images = sample
                                
                                else:
                                    all_images = torch.cat(
                                        (all_images, sample), dim=0
                                    )
                                progress.update(task, advance=1)
                            
                    else:
                        if self.refiner is not None:
                            all_images = None
                            with Progress() as progress:
                                task = progress.add_task("[green]sampling", total=len(batches))
                                for b in range(len(batches)):
                                    sample = self.generator_ema.ema_model.sample(batch_size=batches[b], cond=None)
                                    zero_mask = torch.zeros((batches[b], 1, 256, 256), device=self.device)
                                    sample = (sample > 0.7).float()
                                    refiner_sample = self.refiner_ema.ema_model.sample(batch_size=batches[b], cond=sample[:, :3, ...].clone(), mask=zero_mask)[:, :3, ...]
                                    sample = torch.cat((refiner_sample, sample[:, :3, ...]), dim=1)
                                    if all_images is None:
                                        all_images = sample
                                    else:
                                        all_images = torch.cat(
                                            (all_images, sample), dim=0
                                        )
                                    progress.update(task, advance=1)
                        else:
                            all_images = list(
                                map(
                                    lambda n: self.generator_ema.ema_model.sample(batch_size=n, cond=None),
                                    batches,
                                )
                            )
                            all_images = torch.cat(
                                all_images, dim=0
                            )

                INFO(f"Sampling {num_samples} images done!")

                # save and evaluate
                with Progress() as progress:
                    task = progress.add_task("[cyan]Saving Validation imgs", total=len(all_images) // 4)
                    for idx in range(len(all_images) // 4):

                        if self.data_type == "rgb":
                            utils.save_image(
                                all_images[idx * 4:idx * 4 + 4],
                                os.path.join(
                                    self.sample_results_dir, f"sample-{idx}-c-rgb.png"
                                ),
                                nrow=2,
                            )
                            self.vis.visualize_rgb_layout(
                                all_images[idx * 4:idx * 4 + 4],
                                os.path.join(
                                    self.sample_results_dir, f"sample-{idx}-rgb.png"
                                )
                            )
                        elif self.data_type == "one-hot":
                            self.vis.visualize_onehot_layout(
                                all_images[idx * 4:idx * 4 + 4],
                                os.path.join(
                                    self.sample_results_dir, f"sample-{idx}-onehot.png"
                                )
                            )
                            self.vis.visualize_rgb_layout(
                                all_images[idx * 4:idx * 4 + 4],
                                os.path.join(
                                    self.sample_results_dir, f"sample-{idx}-rgb.png"
                                )
                            )
                        else:
                            raise ValueError(f"data_type {self.data_type} not supported!")
                        progress.advance(task,1)
                        # vectorize
                # bchw
                self.asset_gen.set_data(all_images[:, 0:3:, :])
                self.asset_gen.generate_geofiles()
            

                
                return all_images
        
            
            
