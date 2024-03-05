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
from utils.vis import OSMVisulizer
import numpy as np
import wandb
import json
import traceback
import torch.distributed as dist


class CityDM(object):

    def __init__(self, config_path, sweep_config=None, accelerator=None, ddp=False, debug=False) -> None:
        super().__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        assert config_path is not None, "config path is None!"
        assert os.path.exists(config_path), f"config file {config_path} does not exist!"
        self.config_path = config_path
        self.config_parser = ConfigParser(config_path=self.config_path)
        self.all_config = self.config_parser.get_config_all()
        self.serialized_config = None
        self.ddp = ddp
        self.debug = debug
        INFO(f"config file {self.config_path} loaded!")

        # Init Accelerator
        if accelerator is None:
            acc_config = self.config_parser.get_config_by_name("Accelerator")
            self.accelerator = Accelerator(**acc_config)
            INFO(f"Accelerator initialized!")
        else:
            self.accelerator = accelerator
            INFO(f"Accelerator passed in!")

        if sweep_config is not None and not debug:
            self.sweep_mode = True
            INFO(f"Running in sweep mode!")
            if ddp:
                config_size = 0
                if self.accelerator.is_main_process:
                    INFO(f"Running in multi-GPU mode!")
                    run = wandb.init()
                    self.config_parser.renew_by_sweep_config(wandb.config)
                    self.all_config = self.config_parser.get_config_all()
                    config_sweep_new = dict()
                    for key in wandb.config.keys():
                        config_sweep_new[key] = wandb.config[key]
                    self.serialized_config = json.dumps(config_sweep_new).encode("utf-8")

                    tensor_to_send = torch.tensor(list(self.serialized_config), dtype=torch.uint8,
                                                  device=self.accelerator.device)
                    config_size = len(tensor_to_send)
                self.accelerator.wait_for_everyone()

                config_size_tensor = torch.tensor([config_size], dtype=torch.long, device=self.accelerator.device)
                dist.broadcast(config_size_tensor, src=0)
                config_size = config_size_tensor.item()

                self.accelerator.wait_for_everyone()

                if not self.accelerator.is_main_process:
                    tensor_to_receive = torch.zeros([config_size], dtype=torch.uint8, device=self.accelerator.device)
                    DEBUG("OTHER PROCESS ready!")

                self.accelerator.wait_for_everyone()
                dist.broadcast(tensor_to_send if self.accelerator.is_main_process else tensor_to_receive, src=0)
                INFO("start to decode")
                if not self.accelerator.is_main_process:
                    received_bytes = tensor_to_receive.cpu().numpy().tobytes()
                    received_dict = json.loads(received_bytes.decode('utf-8'))  # 或 pickle.loads(received_bytes)
                    DEBUG("OTHER PROCESS received!")
                    self.config_parser.renew_by_sweep_config(received_dict)
                    self.all_config = self.config_parser.get_config_all()
                else:
                    DEBUG("MAIN PROCESS Done!")

                self.accelerator.wait_for_everyone()

            else:
                run = wandb.init()
                self.config_parser.renew_by_sweep_config(wandb.config)
                self.accelerator.wait_for_everyone()
        else:
            self.sweep_mode = False
            INFO(f"Running in normal mode without sweep!")

        self.timesteps = self.config_parser.get_config_by_name("Model")["diffusion"]["timesteps"]
        self.data_type = self.config_parser.get_config_by_name("Evaluation")["data_type"]

        try:
            # Init Model
            model_config = self.config_parser.get_config_by_name("Model")
            self.backbone = Unet(**model_config['backbone'])

            # Init Diffusion
            diffusion_config = model_config["diffusion"].copy()
            diffusion_config['model'] = self.backbone
            self.diffusion = GaussianDiffusion(**diffusion_config)
            self.generation_type = diffusion_config["model_type"]
        except Exception as e:
            ERROR(f"Init Model failed! {e}")
            raise e

        # init some key params:
        main_config = self.config_parser.get_config_by_name("Main")
        data_config = self.config_parser.get_config_by_name("Data")

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

        self.fine_tune = main_config["fine_tune"]

        if self.fine_tune:
            self.pretrain_model_type = main_config["pretrain_model_type"]
            self.pretrain_ckpt_type = main_config["pretrain_ckpt_type"]
            self.finetuning_type = main_config["finetuning_type"]
        else:
            self.pretrain_model_type = None
            self.pretrain_ckpt_type = None
            self.finetuning_type = None

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

        if self.mode == "train":

            val_config = self.config_parser.get_config_by_name("Validation")
            self.save_best_and_latest_only = val_config["save_best_and_latest_only"]
            self.num_samples = val_config["num_samples"]
            self.results_dir = val_config["results_dir"]
            self.sample_type = val_config["sample_type"]
            self.sample_mode = val_config["sample_mode"]

            # add time stamp to results_dir
            # TODO: check the finetuning type
            if self.fine_tune and self.pretrain_model_type is self.generation_type:
                self.results_dir = val_config["fine_tune_dir"]
            else:
                if self.accelerator.is_main_process:
                    self.results_dir = os.path.join(self.results_dir, self.generation_type)
                    self.results_dir = os.path.join(self.results_dir, time.strftime("%Y-%m-%d-%H-%M-%S",
                                                                                    time.localtime())) + '-train-' + self.sample_type

            self.val_results_dir = os.path.join(self.results_dir, "val")
            self.ckpt_results_dir = os.path.join(self.results_dir, "ckpt")
            self.asset_results_dir = os.path.join(self.results_dir, "asset")
            self.sample_results_dir = os.path.join(self.results_dir, "sample")

            self.pretrain_ckpt_dir = os.path.join(val_config["fine_tune_dir"], "ckpt")

            if self.accelerator.is_main_process:
                os.makedirs(self.results_dir, exist_ok=True)
                os.makedirs(self.val_results_dir, exist_ok=True)
                os.makedirs(self.ckpt_results_dir, exist_ok=True)
                os.makedirs(self.asset_results_dir, exist_ok=True)
                os.makedirs(self.sample_results_dir, exist_ok=True)
                # log some info
                INFO(f"Results dir: {self.results_dir}")
                INFO(f"Vis results dir: {self.val_results_dir}")
                INFO(f"Checkpoint results dir: {self.ckpt_results_dir}")
                INFO(f"Asset results dir: {self.asset_results_dir}")
                INFO(f"Sample results dir: {self.sample_results_dir}")

                # utils prepare
                self.vis_config = self.config_parser.get_config_by_name("Vis")
                self.vis = OSMVisulizer(config=self.vis_config, path=self.val_results_dir)
                self.asset_gen = AssetGen(self.config_parser.get_config_by_name("Asset"), path=self.asset_results_dir)
                INFO(f"Utils initialized!")

                self.ema.ema_model.set_sample_type(self.sample_type)
                self.ema.ema_model.set_sample_mode(self.sample_mode)

            # Init Optimizer
            if self.opt_type == "adam":
                self.optimizer = Adam(self.diffusion.parameters(), lr=self.lr, betas=self.adam_betas)
            elif self.opt_type == "sgd":
                self.optimizer = SGD(self.diffusion.parameters(), lr=self.lr)
            else:
                ERROR(f"Optimizer type {self.opt_type} not supported!")
                raise ValueError(f"Optimizer type {self.opt_type} not supported!")
            self.max_step = self.epochs * len(self.train_dataset) // (self.batch_size * self.grad_accumulate)

            # Init Scheduler
            if self.scheduler_type == "cosine":
                self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_step)
            elif self.scheduler_type == "step":
                self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.sample_frequency * 3, gamma=0.9)
            else:
                ERROR(f"Scheduler type {self.scheduler_type} not supported!")
                raise ValueError(f"Scheduler type {self.scheduler_type} not supported!")

            # prepare model, optimizer, scheduler with accelerator
            self.optimizer, self.scheduler = self.accelerator.prepare(
                self.optimizer, self.scheduler
            )

        elif self.mode == "test":
            test_config = self.config_parser.get_config_by_name("Test")
            self.num_samples = test_config["num_samples"]
            self.results_dir = test_config["results_dir"]
            self.ckpt_results_dir = test_config["ckpt_results_dir"]
            if self.ckpt_results_dir is None or not os.path.exists(self.ckpt_results_dir) or not os.path.isdir(
                    self.ckpt_results_dir):
                ERROR(f"ckpt_results_dir {self.ckpt_results_dir} does not exist!")
                raise ValueError(f"ckpt_results_dir {self.ckpt_results_dir} does not exist!")
            # add time stamp to results_dir
            # check if results_dir exists
            self.sample_type = test_config["sample_type"]
            self.sample_mode = test_config["sample_mode"]

            self.results_dir = os.path.join(self.results_dir, self.sample_mode) + '-sample-' + self.sample_type
            self.results_dir = os.path.join(self.results_dir,
                                            time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())) + '-test'

            self.asset_results_dir = os.path.join(self.results_dir, "asset")
            self.sample_results_dir = os.path.join(self.results_dir, "sample")

            os.makedirs(self.results_dir, exist_ok=True)
            os.makedirs(self.asset_results_dir, exist_ok=True)
            os.makedirs(self.sample_results_dir, exist_ok=True)
            # log some info
            INFO(f"Results dir: {self.results_dir}")
            INFO(f"Vis results dir: {self.sample_results_dir}")
            INFO(f"Asset results dir: {self.asset_results_dir}")
            INFO(f"Sample results dir: {self.sample_results_dir}")

            # utils prepare
            self.vis_config = self.config_parser.get_config_by_name("Vis")
            self.vis = OSMVisulizer(config=self.vis_config, path=self.sample_results_dir)
            self.asset_gen = AssetGen(self.config_parser.get_config_by_name("Asset"), path=self.asset_results_dir)
            INFO(f"Utils initialized!")

        # evaluation prepare
        eval_config = self.config_parser.get_config_by_name("Evaluation")
        if self.accelerator.is_main_process:
            self.evaluation = Evaluation(batch_size=self.batch_size // self.accelerator.num_processes,
                                         device=self.device,
                                         dl=self.val_dataloader if self.mode == "train" else self.test_dataloader,
                                         sampler=self.ema.ema_model,
                                         accelerator=self.accelerator,
                                         mapping=eval_config["channel_to_rgb"],
                                         config=eval_config,
                                         )
            # 👆 代码evaluation初始化还可以再优化 优化成全config的形式
            INFO(f"Evaluation initialized!")
            # print some info
            INFO(self.evaluation.summary())

        self.best_evaluation_result = None

        #  some preperation for train
        self.now_epoch = 0
        self.now_step = 0
        self.best_validation_result = None

        self.not_best = 0

        self.diffusion = self.accelerator.prepare(
            self.diffusion
        )

        # Done
        INFO(f"CityDM initialized!")

    def check_best_or_not(self, result):
        if self.best_evaluation_result is None and result is not None:
            self.best_evaluation_result = result
            return True
        elif result is None:
            return False
        else:
            if self.best_evaluation_result["IS"] < result["IS"]:
                self.best_evaluation_result = result
                return True
            else:
                return False

    @property
    def device(self):
        return self.accelerator.device

    def load_model_params(self, tgt, model_state_dict, load_type):
        if load_type == "full" or load_type == None:
            tgt.load_state_dict(model_state_dict)
        elif load_type == "partial":
            tgt.load_state_dict(self.partly_load(model_state_dict))
        elif load_type == "LoRA":
            ERROR(f"LoRA not supported yet!")
            raise NotImplementedError(f"LoRA not supported yet!")
        else:
            ERROR(f"load_type {load_type} not supported!")
            raise ValueError(f"load_type {load_type} not supported!")

    def partly_load(self, param_dict):
        # pre-train model may not suitable for current model, so we load pre-train model's params
        # which is suitable for current model and zero init other new params

        padded_dict = self.diffusion.state_dict()
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

    def save_ckpts(self, epoch, step, best=False, latest=False):
        if self.accelerator.is_main_process:
            if not best and not latest:
                ckpt_path = os.path.join(self.ckpt_results_dir, f"ckpt_{epoch}_{step}.pth")
            elif best:
                ckpt_path = os.path.join(self.ckpt_results_dir, f"best_ckpt.pth")
            else:
                ckpt_path = os.path.join(self.ckpt_results_dir, f"latest_ckpt.pth")

            ckpt = {
                "epoch": epoch,
                "step": step,
                "diffusion": self.accelerator.get_state_dict(self.diffusion),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "ema": self.ema.state_dict(),
                "best_evaluation_result": self.best_evaluation_result,
                "seed": self.seed,
                'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            }

            if best:
                ckpt_path = os.path.join(self.ckpt_results_dir, f"best_ckpt.pth")
                torch.save(ckpt, ckpt_path)
                INFO(f"Best ckpt saved to {ckpt_path}")
            else:
                torch.save(ckpt, ckpt_path)
                INFO(f"Ckpt saved to {ckpt_path}")

    def load_ckpts(self, ckpt_path, mode="train"):
        ckpt = torch.load(ckpt_path)
        # self.diffusion.load_state_dict(ckpt["diffusion"])
        if mode == "train":
            if self.pretrain_model_type == self.generation_type and self.fine_tune:
                WARNING(f"Fine tune mode and same training type, load pretrain model's params and continue training!")
                self.optimizer.load_state_dict(ckpt["optimizer"])
                self.scheduler.load_state_dict(ckpt["scheduler"])
                self.now_epoch = ckpt["epoch"]
                self.now_step = ckpt["step"]
                self.best_evaluation_result = ckpt["best_evaluation_result"]
                self.seed = ckpt["seed"]
        if self.fine_tune and mode == "train":
            INFO(f"Load ckpt from {ckpt_path} for fintunig {self.generation_type} model!")
            INFO(f"fintuning type: {self.finetuning_type}")

        self.load_model_params(self.diffusion, ckpt["diffusion"], self.finetuning_type)

        if self.accelerator.is_main_process:
            # reinit ema
            self.ema = EMA(self.diffusion, beta=self.ema_decay, update_every=self.ema_update_every)
            self.ema.to(self.device)
            INFO(f"EMA reinitialized!")

        self.ema.ema_model.set_sample_type(self.sample_type)
        self.ema.ema_model.set_sample_mode(self.sample_mode)

        INFO(f"Ckpt loaded from {ckpt_path}")

        if exists(self.accelerator.scaler) and exists(ckpt['scaler']):
            self.accelerator.scaler.load_state_dict(ckpt['scaler'])

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
        # dump training configs to result root dir
        if self.accelerator.is_main_process:
            with open(os.path.join(self.results_dir, "config.json"), "w") as f:
                f.write(json.dumps(self.all_config, indent=4))
            # replace result_dir's "/" with "-"

        if not self.debug:
            if self.ddp:
                if self.accelerator.is_main_process:
                    display_name = self.results_dir.replace("/", "-")
                    if self.sweep_mode:
                        wandb.init(config=self.config_parser.get_config_all(), name=display_name, reinit=True)
                        # wandb.watch(self.diffusion, log="all")
                    else:
                        wandb.init(project="CityLayout", entity="913217005", config=self.config_parser.get_config_all(),
                                   name=display_name)
                        # wandb.watch(self.diffusion, log="all")
            else:
                display_name = self.results_dir.replace("/", "-")
                if self.sweep_mode:
                    wandb.init(config=self.config_parser.get_config_all(), name=display_name, reinit=True)
                    # wandb.watch(self.diffusion, log="all")
                else:
                    wandb.init(project="CityLayout", entity="913217005", config=self.config_parser.get_config_all(),
                               name=display_name)
                    # wandb.watch(self.diffusion, log="all")

            if self.accelerator.is_main_process:
                experiment_title = "experiment_{}_lr{}_diffusion{}_maxepoch{}_resultfolder{}".format(
                    time.strftime("%Y%m%d_%H%M%S"),
                    self.lr,
                    self.timesteps,
                    self.epochs,
                    self.results_dir,
                )
                writer = SummaryWriter(log_dir=f"runs-DEBUG/{experiment_title}")
        else:
            writer = None

        if isinstance(self.diffusion, nn.DataParallel) or self.ddp:  # if model is wrapped by DataParallel, unwrap it
            DEBUG(f"Model is wrapped by DataParallel, unwrap it!")
            self.diffusion = self.diffusion.module
            self.ema = EMA(self.diffusion, beta=self.ema_decay, update_every=self.ema_update_every)
            self.ema.to(self.device)

        if self.fine_tune:
            INFO(f"Fine tune mode, load ckpt from {self.ckpt_results_dir}")
            if self.pretrain_ckpt_type == "best":
                ckpt_path = os.path.join(self.pretrain_ckpt_dir, "best_ckpt.pth")
            else:
                ckpt_path = os.path.join(self.pretrain_ckpt_dir, "latest_ckpt.pth")
            self.load_ckpts(ckpt_path, mode="train")

            INFO(f"ckpt loaded!")
            INFO(f"ckpt step & epoch: {self.now_step}, {self.now_epoch}")
        else:
            INFO(f"Train from scratch!")

        self.accelerator.wait_for_everyone()

        try:
            with tqdm(
                    initial=self.now_step,
                    total=self.max_step,
                    disable=not self.accelerator.is_main_process,
            ) as pbar:

                while self.now_step < self.max_step:
                    total_loss = 0.0

                    for _ in range(self.grad_accumulate):
                        # data = next(self.dl)["layout"].to(device)
                        data = next(self.train_dataloader)
                        layout = data["layout"].to(self.device)

                        with self.accelerator.autocast():
                            loss = self.diffusion(layout)
                            # DEBUG(f"loss: {loss}, GPU: {self.device}")
                            loss = loss / self.grad_accumulate
                            total_loss += loss.item()

                        self.accelerator.backward(loss)

                    if self.accelerator.is_main_process:

                        if not self.debug:
                            writer.add_scalar("loss", float(total_loss), self.now_step)
                            writer.add_scalar("lr", self.scheduler.get_last_lr()[0], self.now_step)
                            wandb.log({"loss": float(total_loss), "lr": self.scheduler.get_last_lr()[0]})
                        # if loss equals to nan, then stop training
                        if math.isnan(total_loss):
                            ERROR(f"Loss equals to nan, stop training!")
                            raise ValueError(f"Loss equals to nan, stop training!")

                    pbar.set_description(
                        f'loss: {total_loss:.5f}, lr: {self.optimizer.param_groups[0]["lr"]:.6f}, \
                        epoch: {self.now_step * self.batch_size * self.grad_accumulate / len(self.train_dataset):.2f}'
                    )

                    self.accelerator.wait_for_everyone()
                    self.accelerator.clip_grad_norm_(self.diffusion.parameters(), self.max_grad_norm)

                    self.scheduler.step()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.accelerator.wait_for_everyone()

                    self.now_step += 1
                    self.now_epoch = (self.now_step * self.batch_size * self.grad_accumulate) // len(self.train_dataset)
                    if self.accelerator.is_main_process:
                        self.ema.update()

                        if self.now_step != 0 and divisible_by(self.now_step, self.sample_frequency):
                            if self.generation_type == "uniDM":
                                self.validation(writer, cond=False)
                                INFO(f"Validation done!")
                                self.validation(writer, cond=True)
                                INFO(f"Validation with cond done!")
                            elif self.generation_type == "Outpainting":
                                self.validation(writer, cond=True)
                                INFO(f"Validation with cond done!")
                            elif self.generation_type == "normal":
                                self.validation(writer, cond=False)
                                INFO(f"Validation done!")

                    pbar.update(1)
            if self.accelerator.is_main_process:
                if not self.debug:
                    wandb.finish()
                    writer.close()
                self.accelerator.print("training complete")
        except Exception as e:
            ERROR(f"Training failed! {e}")
            # log traceback
            ERROR(f"Traceback: \n {traceback.format_exc()}")
            raise e

    def validation(self, writer, cond=False):
        if not self.accelerator.is_main_process:
            return

        self.ema.ema_model.eval()
        if cond:
            wandb_key = "val_cond"
        else:
            wandb_key = "val"

        with torch.inference_mode():
            milestone = self.now_step // self.sample_frequency
            now_val_path = os.path.join(self.val_results_dir, f"milestone_{milestone}_val_cond_{cond}")
            os.makedirs(now_val_path, exist_ok=True)
            if self.ddp:
                batches = num_to_groups(self.num_samples, self.batch_size // self.accelerator.num_processes)
            else:
                batches = num_to_groups(self.num_samples, self.batch_size)
            INFO(f"Start sampling {self.num_samples} images...")
            if cond == True:
                # sample some real images from val_Dataloader as cond
                all_images_list = None
                for b in range(len(batches)):
                    cond_image = next(self.val_dataloader)["layout"].to(self.device)
                    if all_images_list is None:
                        all_images_list = self.ema.ema_model.sample(batch_size=batches[b], cond=cond_image)
                    else:
                        all_images_list = torch.cat(
                            (all_images_list, self.ema.ema_model.sample(batch_size=batches[b], cond=cond_image)), dim=0
                        )
            else:
                all_images_list = list(
                    map(
                        lambda n: self.ema.ema_model.sample(batch_size=n, cond=None),
                        batches,
                    )
                )
            INFO(f"Sampling {self.num_samples} images done!")

        if cond == True:
            all_images = all_images_list
        else:
            all_images = torch.cat(
                all_images_list, dim=0
            )  # (num_samples, channel, image_size, image_size)

        for idx in tqdm(range(len(all_images) // 4), desc='Saving Validation imgs'):

            if self.data_type == "rgb":
                utils.save_image(
                    all_images[idx * 4:idx * 4 + 4],
                    os.path.join(
                        now_val_path, f"sample-{idx}-c-rgb.png"
                    ),
                    nrow=2,
                )
                self.vis.visualize_rgb_layout(
                    all_images[idx * 4:idx * 4 + 4],
                    os.path.join(
                        now_val_path, f"sample-{idx}-rgb.png"
                    )
                )
            elif self.data_type == "one-hot":
                self.vis.visulize_onehot_layout(
                    all_images[idx * 4:idx * 4 + 4],
                    os.path.join(
                        now_val_path, f"sample-{idx}-onehot.png"
                    )
                )
                self.vis.visualize_rgb_layout(
                    all_images[idx * 4:idx * 4 + 4],
                    os.path.join(
                        now_val_path, f"sample-{idx}-rgb.png"
                    )
                )
            else:
                raise ValueError(f"data_type {self.data_type} not supported!")
        # whether to calculate fid
        overlapping_rate = cal_overlapping_rate(all_images)
        self.accelerator.print(
            f"overlapping rate: {overlapping_rate:.5f}"
        )
        if not self.debug:
            writer.add_scalar(
                "overlapping_rate", overlapping_rate, self.now_step
            )
            wandb.log({wandb_key: {"overlapping_rate": overlapping_rate}}, commit=False)

        val_result = None
        try:
            if (self.evaluation.validation(cond, os.path.join(now_val_path, "data_analyse"))):
                val_result = self.evaluation.get_evaluation_dict()
                # self.accelerator.print(val_result)



        except Exception as e:
            self.accelerator.print("computation failed: \n")
            self.accelerator.print(e)

        if val_result is not None:
            if not self.debug:
                wandb.log({wandb_key: val_result}, commit=False)

        if self.save_best_and_latest_only:
            if self.check_best_or_not(val_result):
                self.save_ckpts(epoch=self.now_epoch, step=self.now_step, best=True)
            else:
                # self.not_best +=1
                # if self.not_best >= 5:
                #     wandb.alert(
                #         title="CityLayout",
                #         text="Model may be overfitting!, please check!",
                #         level=wandb.AlertLevel.WARNING,
                #         wait_duration=timedelta(minutes=5),  # 7 days
                #     )
                self.not_best = 0

            self.save_ckpts(epoch=self.now_epoch, step=self.now_step, latest=True)
        elif not self.save_best_and_latest_only:
            self.save_ckpts(epoch=self.now_epoch, step=self.now_step)

        self.ema.ema_model.train()

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

    def op_handle(self, org, padding=1, ratio=0.5):
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
        with tqdm(desc="extend layout", leave=False, colour="green") as pbar:
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

                        extend_region = self.ema.ema_model.sample(batch_size=b, cond=cond_region, mask=op_mask)[:, :c,
                                        ...]

                        extend_layout[:, :, h_l_p:h_l_p + h,
                        w_l_p + int(w * ratio):w_l_p + int(w * ratio) + w] = extend_region


                    else:
                        cond_region = extend_layout[:, :, h_l_p:h_l_p + h, w_l_p:w_l_p + w]

                        op_mask = self.get_op_mask(ratio=ratio, d='down', shape=(b, 1, h, w))
                        # DEBUG(f"hlp: {h_l_p}, wlp: {w_l_p}")
                        extend_region = self.ema.ema_model.sample(batch_size=b, cond=cond_region, mask=op_mask)[:, :c,
                                        ...]
                        extend_layout[:, :, h_l_p:h_l_p + h, w_l_p:w_l_p + w] = extend_region

                    pbar.update(1)

                init = True

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

            # use same bbox for all batch
            x1 = torch.randint(0, w // 2, (1,)).item()
            y1 = torch.randint(0, h // 2, (1,)).item()
            x2 = torch.randint(w // 2, w, (1,)).item()
            y2 = torch.randint(h // 2, h, (1,)).item()

            bbox[:, 0] = x1
            bbox[:, 1] = y1
            bbox[:, 2] = x2
            bbox[:, 3] = y2

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

        return self.ema.ema_model.sample(batch_size=org.shape[0], cond=org, mask=mask)

    def sample(self, cond=False, eval=True, best=False):
        INFO(f"Start sampling {self.num_samples} images...")
        INFO(F"Sample result save to {self.sample_results_dir}")

        INFO(f"sample mode: {'random' if not cond else 'conditional'}")

        # check and load ckpt
        INFO(f"ckpt path: {self.ckpt_results_dir}")
        if best:
            ckpt_path = os.path.join(self.ckpt_results_dir, "best_ckpt.pth")
        else:
            ckpt_path = os.path.join(self.ckpt_results_dir, "latest_ckpt.pth")

        INFO(f"ckpt path: {ckpt_path}")
        try:
            self.load_ckpts(ckpt_path, mode="test")
            INFO(f"ckpt loaded!")
            INFO(f"ckpt step & epoch: {self.now_step}, {self.now_epoch}")
        except Exception as e:
            ERROR(f"load ckpt failed! {e}")
            raise e

        # sample
        self.ema.ema_model.eval()

        self.ema.ema_model.set_sample_type(self.sample_type)
        self.ema.ema_model.set_sample_mode(self.sample_mode)

        with torch.inference_mode():

            batches = num_to_groups(self.num_samples, self.batch_size)

            if self.sample_mode == "Outpainting":
                all_images = None
                padding = self.config_parser.get_config_by_name("Test")['op']["padding"]
                ratio = self.config_parser.get_config_by_name("Test")['op']["ratio"]
                for b in tqdm(range(len(batches)), desc="outpainting sampling", leave=False, colour="green"):
                    cond_image = next(self.test_dataloader)["layout"].to(self.device)
                    # DEBUG(f"cond_image shape: {cond_image.shape}")
                    if all_images is None:
                        all_images = self.op_handle(cond_image, padding=padding, ratio=ratio)
                    else:
                        all_images = torch.cat(
                            (all_images, self.op_handle(cond_image, padding=padding, ratio=ratio)), dim=0
                        )

            elif self.sample_mode == "Inpainting":
                all_images = None
                for b in tqdm(range(len(batches)), desc="inpaiting sampling", leave=False, colour="green"):
                    cond_image = next(self.test_dataloader)["layout"].to(self.device)
                    # DEBUG(f"cond_image shape: {cond_image.shape}")
                    if all_images is None:
                        all_images = self.inp_handle(cond_image)
                    else:
                        all_images = torch.cat(
                            (all_images, self.inp_handle(cond_image)), dim=0
                        )

            else:
                if cond is True:
                    # sample some real images from val_Dataloader as cond
                    all_images = None
                    for b in tqdm(range(len(batches)), desc="sampling", leave=False, colour="green"):
                        cond_image = next(self.test_dataloader)["layout"].to(self.device)
                        if all_images is None:
                            all_images = self.ema.ema_model.sample(batch_size=batches[b], cond=cond_image)
                        else:
                            all_images = torch.cat(
                                (all_images, self.ema.ema_model.sample(batch_size=batches[b], cond=cond_image)), dim=0
                            )
                else:
                    all_images = list(
                        map(
                            lambda n: self.ema.ema_model.sample(batch_size=n, cond=None),
                            batches,
                        )
                    )

            if cond is True:
                all_images = all_images
            else:
                all_images = torch.cat(
                    all_images, dim=0
                )
            INFO(f"Sampling {self.num_samples} images done!")

            # save and evaluate
            for idx in tqdm(range(len(all_images) // 4), desc='Saving Sampled imgs'):

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
                    self.vis.visulize_onehot_layout(
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

                    # vectorize
            # bchw
            self.asset_gen.set_data(all_images[:, 0:3:, :])
            self.asset_gen.generate_geofiles()

            # evaluate
            result = None
            if eval:
                try:
                    self.evaluation.reset_sampler(self.ema.ema_model)
                    self.evaluation.validation(cond, os.path.join(self.sample_results_dir, "data_analyse"))
                    result = self.evaluation.get_evaluation_dict()
                    # write evaluation to file
                    path = os.path.join(self.sample_results_dir, "evaluation.json")
                    with open(path, "w") as f:
                        for key in result.keys():
                            f.write(f"{key}: {result[key]}\n")


                except Exception as e:
                    self.accelerator.print("evaluation failed: \n")
                    self.accelerator.print(e)

                return all_images, result
            else:
                return all_images, None
