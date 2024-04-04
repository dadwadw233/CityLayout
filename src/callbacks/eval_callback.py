from typing import Any
import pytorch_lightning as pl

from utils.evaluation import Evaluation
from omegaconf import DictConfig
from utils.log import *
from tqdm import tqdm
from utils.utils import *
import traceback
import os
import torch
from torchvision import transforms as T, utils
import hydra
import json
import wandb
from rich.progress import Progress

class EvalCallback(pl.Callback):
    def __init__(self, *args, **kwargs):
        self.config: DictConfig = DictConfig(kwargs)
        self.evaluator = Evaluation(**self.config.config)

    def save_generated_images(self, all_images, path, pl_module):
        with Progress() as progress:
            task = progress.add_task("[cyan]Saving Validation imgs", total=len(all_images) // 4)
            for idx in range(len(all_images) // 4):
                if pl_module.data_type == "rgb":
                    utils.save_image(
                        all_images[idx * 4:idx * 4 + 4],
                        os.path.join(
                            path, f"sample-{idx}-c-rgb.png"
                        ),
                        nrow=2,
                    )
                    pl_module.vis.visualize_rgb_layout(
                        all_images[idx * 4:idx * 4 + 4],
                        os.path.join(
                            path, f"sample-{idx}-rgb.png"
                        )
                    )
                elif pl_module.data_type == "one-hot":
                    pl_module.vis.visualize_onehot_layout(
                        all_images[idx * 4:idx * 4 + 4],
                        os.path.join(
                            path, f"sample-{idx}-onehot.png"
                        )
                    )
                    pl_module.vis.visualize_rgb_layout(
                        all_images[idx * 4:idx * 4 + 4],
                        os.path.join(
                            path, f"sample-{idx}-rgb.png"
                        )
                    )
                else:
                    raise ValueError(f"data_type {pl_module.data_type} not supported!")
                progress.update(task, advance=1)
        
            
        
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.is_global_zero:
            val_cond = False
            if pl_module.model_type == "completion" or pl_module.model_type == "CityGen":
                val_cond = True
            milestone = (pl_module.current_epoch)+1
            now_val_path = os.path.join(pl_module.val_results_dir, f"milestone_{milestone}_val_cond_{val_cond}")
            os.makedirs(now_val_path, exist_ok=True)
            all_images_list = list(map(lambda x: x["sample"], pl_module.outputs))
            all_images = torch.cat(all_images_list, dim=0)

            self.save_generated_images(all_images, now_val_path, pl_module)
        
                
            overlapping_rate = cal_overlapping_rate(all_images)
            pl_module.log("overlapping_rate", overlapping_rate, logger=True)
            
            val_result = None
            try:
                self.evaluator.reset_sampler(pl_module.generator_ema.ema_model)
                self.evaluator.reset_dl(cycle(trainer.val_dataloaders))
                
                if (self.evaluator.validation(val_cond, os.path.join(now_val_path, "data_analyse"))):
                    val_result = self.evaluator.get_evaluation_dict()
            except Exception as e:
                ERROR(f"Validation failed! {e}")
                # log traceback
                ERROR(f"Traceback: \n {traceback.format_exc()}")
                raise e
            
            if val_result is not None:
                try:
                    wandb.log(val_result)
                except Exception as e:
                    ERROR(f"Failed to log to wandb! {e}")
                    # log traceback
                    ERROR(f"Traceback: \n {traceback.format_exc()}")
                 
            if pl_module.save_best_and_latest_only:
                if pl_module.check_best_or_not(val_result):
                    pl_module.save_ckpts(epoch=pl_module.current_epoch , step=pl_module.global_step, best=True)
                pl_module.save_ckpts(epoch=pl_module.current_epoch , step=pl_module.global_step, latest=True)
                
            elif not pl_module.save_best_and_latest_only:
                pl_module.save_ckpts(epoch=pl_module.current_epoch , step=pl_module.global_step)
                
            pl_module.generator_ema.ema_model.train()
            pl_module.outputs.clear()
            
    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        
        if trainer.is_global_zero:
            path = os.path.join(pl_module.sample_results_dir, f"metrics/data_analyse")
            self.evaluator.image_evaluation_ddp(pl_module.samples_for_test, path)
            
            test_result = self.evaluator.get_evaluation_dict()
            
            result_json_path = os.path.join(pl_module.sample_results_dir, "metrics/result.json")
            os.makedirs(os.path.dirname(result_json_path), exist_ok=True)
            # dump the result to the path
            with open(result_json_path, "w") as f:
                json.dump(test_result, f, indent=4)
                
            
            