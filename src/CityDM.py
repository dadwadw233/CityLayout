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

class CityDM(object):

    def __init__(self, config_path) -> None:
        super().__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        assert config_path is not None, "config path is None!"
        assert os.path.exists(config_path), f"config file {config_path} does not exist!"
        self.config_path = config_path
        self.config_parser = ConfigParser(config_path=self.config_path)
        INFO(f"config file {self.config_path} loaded!")

        # Init Accelerator
        acc_config = self.config_parser.get_config_by_name("Accelerator")
        self.accelerator = Accelerator(**acc_config)
        INFO(f"Accelerator initialized!")
        
        self.timesteps = self.config_parser.get_config_by_name("Model")["diffusion"]["timesteps"]
        self.data_type = self.config_parser.get_config_by_name("Evaluation")["data_type"]

        try:
            # Init Model
            model_config = self.config_parser.get_config_by_name("Model")
            self.backbone =  Unet(**model_config['backbone'])


            # Init Diffusion
            diffusion_config = model_config["diffusion"].copy()
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
        if self.mode == "train":
            val_config = self.config_parser.get_config_by_name("Validation")
            self.save_best_and_latest_only = val_config["save_best_and_latest_only"]
            self.num_samples = val_config["num_samples"]
            self.results_dir = val_config["results_dir"] 
            # add time stamp to results_dir
            self.results_dir = os.path.join(self.results_dir, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
            self.val_results_dir = os.path.join(self.results_dir, "val")
            self.ckpt_results_dir = os.path.join(self.results_dir, "ckpt")
            self.asset_results_dir = os.path.join(self.results_dir, "asset")
            self.sample_results_dir = os.path.join(self.results_dir, "sample")

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
            self.vis = OSMVisulizer(self.vis_config["channel_to_rgb"], self.vis_config["threshold"], self.val_results_dir)
            self.asset_gen = AssetGen(self.config_parser.get_config_by_name("Asset"), path=self.asset_results_dir)
            INFO(f"Utils initialized!")

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
                self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
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
            if self.ckpt_results_dir is None or not os.path.exists(self.ckpt_results_dir) or not os.path.isdir(self.ckpt_results_dir):
                ERROR(f"ckpt_results_dir {self.ckpt_results_dir} does not exist!")
                raise ValueError(f"ckpt_results_dir {self.ckpt_results_dir} does not exist!")
            # add time stamp to results_dir
            # check if results_dir exists
            
            self.results_dir = os.path.join(self.results_dir, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())) + '-test'


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
            self.vis = OSMVisulizer(self.vis_config["channel_to_rgb"], self.vis_config["threshold"], self.sample_results_dir)
            self.asset_gen = AssetGen(self.config_parser.get_config_by_name("Asset"), path=self.asset_results_dir)
            INFO(f"Utils initialized!")


        

        
        # evaluation prepare
        eval_config = self.config_parser.get_config_by_name("Evaluation")
        self.evaluation = Evaluation(batch_size=self.batch_size, device=self.device,
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
            if self.best_evaluation_result["FID"] < result["FID"]:
                self.best_evaluation_result = result
                return True
            else:
                return False

    
    @property
    def device(self):
        return self.accelerator.device
    

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
    
    def load_ckpts(self, ckpt_path, mode="train"):
        ckpt = torch.load(ckpt_path)
        self.diffusion.load_state_dict(ckpt["diffusion"])
        if mode == "train":
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
        if self.accelerator.is_main_process:
            # replace result_dir's "/" with "-"
            display_name = self.results_dir.replace("/", "-")
            wandb.init(project="CityLayout", entity="913217005", config = self.config_parser.get_config_all(), name=display_name)
            wandb.watch(self.diffusion, log="all")

        
        experiment_title = "experiment_{}_lr{}_diffusion{}_maxepoch{}_resultfolder{}".format(
            time.strftime("%Y%m%d_%H%M%S"),
            self.lr,
            self.timesteps,
            self.epochs,
            self.results_dir,
        )
        writer = SummaryWriter(log_dir=f"runs-DEBUG/{experiment_title}")
        
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
                        loss = loss / self.grad_accumulate
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                writer.add_scalar("loss", float(total_loss), self.now_step)
                writer.add_scalar("lr", self.scheduler.get_last_lr()[0], self.now_step)
                wandb.log({"loss": float(total_loss), "lr": self.scheduler.get_last_lr()[0]})

                pbar.set_description(
                    f'loss: {total_loss:.5f}, lr: {self.optimizer.param_groups[0]["lr"]:.6f}, \
                    epoch: {self.now_step* self.batch_size*self.grad_accumulate/len(self.train_dataset):.2f}'
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
                        self.validation(writer, cond=False)
                        INFO(f"Validation done!")
                        
                        self.validation(writer, cond=True)
                        INFO(f"Validation with cond done!")                        

                pbar.update(1)
        wandb.finish()
        self.accelerator.print("training complete")
        writer.close()

    def validation(self, writer, cond=False):
        self.ema.ema_model.eval()
        if cond:
            wandb_key = "val_cond"
        else:
            wandb_key = "val"

        
        with torch.inference_mode():
            milestone = self.now_step // self.sample_frequency
            now_val_path = os.path.join(self.val_results_dir, f"milestone_{milestone}_val_cond_{cond}")
            os.makedirs(now_val_path, exist_ok=True)
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

        image_for_show = all_images[:4] 
        if self.data_type == "rgb":
            utils.save_image(
                image_for_show,
                os.path.join(
                    now_val_path, f"sample-{milestone}-c-rgb.png"
                ),
                nrow=int(math.sqrt(self.num_samples)),
            )
            self.vis.visualize_rgb_layout(
                image_for_show,
                os.path.join(
                    now_val_path, f"sample-{milestone}-rgb.png"
                )
            )
        else:
            self.vis.visulize_onehot_layout(
                image_for_show,
                os.path.join(
                    now_val_path, f"sample-{milestone}-onehot.png"
                )
            )
            self.vis.visualize_rgb_layout(
                image_for_show,
                os.path.join(
                    now_val_path, f"sample-{milestone}-rgb.png"
                )
            )
        # whether to calculate fid
        overlapping_rate = cal_overlapping_rate(all_images)
        self.accelerator.print(
            f"overlapping rate: {overlapping_rate:.5f}"
        )
        writer.add_scalar(
            "overlapping_rate", overlapping_rate, self.now_step
        )
        wandb.log({wandb_key: {"overlapping_rate": overlapping_rate}}, commit=False)

        
        val_result = None
        try:
            if(self.evaluation.validation(cond, os.path.join(now_val_path, "data_analyse"))):
                val_result = self.evaluation.get_evaluation_dict()
                # self.accelerator.print(val_result)
            
                    
            
        except Exception as e:
            self.accelerator.print("computation failed: \n")
            self.accelerator.print(e)

        if val_result is not None:
            wandb.log({wandb_key: val_result}, commit=False)
                
        if self.save_best_and_latest_only and not cond:
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

    def sample(self, cond=False):
        INFO(f"Start sampling {self.num_samples} images...")
        INFO(F"Sample result save to {self.sample_results_dir}")

        INFO(f"sample mode: {'random' if not cond is None else 'conditional'}")

        # check and load ckpt
        INFO(f"ckpt path: {self.ckpt_results_dir}")
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
        with torch.inference_mode():
            batches = num_to_groups(self.num_samples, self.batch_size)
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

            if self.data_type == "rgb":
                utils.save_image(
                    all_images[:int(self.num_samples//4)],
                    os.path.join(
                        self.sample_results_dir, f"sample-c-rgb.png"
                    ),
                    nrow=int(math.sqrt(self.num_samples)),
                )
                self.vis.visualize_rgb_layout(
                    all_images[:int(self.num_samples//4)],
                    os.path.join(
                        self.sample_results_dir, f"sample-rgb.png"
                    )
                )
            else:
                self.vis.visulize_onehot_layout(
                    all_images[:int(self.num_samples//4)],
                    os.path.join(
                        self.sample_results_dir, f"sample-onehot.png"
                    )
                )
                self.vis.visualize_rgb_layout(
                    all_images[:int(self.num_samples//4)],
                    os.path.join(
                        self.sample_results_dir, f"sample-rgb.png"
                    )
                )

            # vectorize
            # bchw
            self.asset_gen.set_data(all_images[:, 0:3 :, :])
            self.asset_gen.generate_geojson()

            # evaluate
            try:
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

        
            
        

        





        
