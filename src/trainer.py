import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam, lr_scheduler


from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from utils.fid_evaluation import FIDEvaluation

from utils.utils import exists, cycle, num_to_groups, has_int_squareroot, divisible_by, OSMVisulizer

from model.version import __version__

from data.osm_loader import OSMDataset

from torch.utils.tensorboard import SummaryWriter

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        *,
        convert_image_to = None,
        dataset_config = None,
        trainer_config = None
    ):
        super().__init__()
        assert trainer_config is not None, "trainer_config must be provided"
        self.vis = OSMVisulizer(mapping=trainer_config['vis']['channel_to_rgb'])
        # accelerator

        self.accelerator = Accelerator(
            split_batches = trainer_config['trainer']['split_batches'],
            mixed_precision = trainer_config['trainer']['mixed_precision_type'] if trainer_config['trainer']['amp'] else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # default convert_image_to depending on channels

        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters

        assert has_int_squareroot(trainer_config['trainer']['num_samples']), 'number of samples must have an integer square root'
        self.num_samples = trainer_config['trainer']['num_samples']
        self.save_and_sample_every = trainer_config['trainer']['sample_frequency']

        self.batch_size =  trainer_config['trainer']['batch_size']
        self.gradient_accumulate_every = trainer_config['trainer']['grad_accumulate']
        assert ( self.batch_size  *  self.gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.max_epochs = trainer_config['trainer']['max_epochs']
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = trainer_config['trainer']['max_grad_norm']

        # dataset and dataloader

        assert dataset_config is not None, "dataset_config must be provided"

        self.ds = OSMDataset(config=dataset_config, mode='train', transform=None)

        assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        dl = DataLoader(self.ds, batch_size = self.batch_size, shuffle = trainer_config['trainer']['shuffle'], pin_memory = trainer_config['trainer']['pin_memory'], num_workers = trainer_config['trainer']['num_workers'])

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        self.max_epochs = trainer_config['trainer']['max_epochs']
        self.max_step = self.max_epochs * len(self.ds)//(self.batch_size*self.gradient_accumulate_every)
        self.sample_step = self.save_and_sample_every*(len(self.ds)//(self.batch_size*self.gradient_accumulate_every))

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = float(trainer_config['trainer']['lr']), betas = trainer_config['trainer']['adam_betas'])

        self.scheduler = lr_scheduler.CosineAnnealingLR(self.opt, T_max = self.max_step, eta_min = 0)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = trainer_config['trainer']['ema_decay'], update_every = trainer_config['trainer']['ema_update_every'])
            self.ema.to(self.device)

        self.results_folder = Path(trainer_config['trainer']['results_dir'])
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt, self.scheduler = self.accelerator.prepare(self.model, self.opt, self.scheduler)

        # FID-score computation

        self.calculate_fid = trainer_config['trainer']['calculate_fid'] and self.accelerator.is_main_process
        

        if self.calculate_fid:
            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )
            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=self.results_folder,
                device=self.device,
                num_fid_samples=trainer_config['trainer']['num_fid_samples'],
                inception_block_idx=trainer_config['trainer']['inception_block_idx'],
            )
        self.save_best_and_latest_only = trainer_config['trainer']['save_best_and_latest_only']

        if self.save_best_and_latest_only:
            assert self.calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

        self.config = trainer_config
        self.data_type = dataset_config['data']['type']

        

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])


    def model_summarize(self):
        if self.accelerator.is_local_main_process:
            # self.accelerator.print(self.model)
            # self.accelerator.print(f'ema model: {self.ema.ema_model}')
            self.accelerator.print(f'number of parameters: {sum(p.numel() for p in self.model.parameters()):,}')
            self.accelerator.print(f'number of ema parameters: {sum(p.numel() for p in self.ema.ema_model.parameters()):,}')

            # print model architecture
            for name, param in self.model.named_parameters():
                self.accelerator.print(f'parameter: {name}, shape: {param.shape}')

    def trainer_config_summarize(self):
        if self.accelerator.is_local_main_process:
            self.accelerator.print(f'model arch: {self.config["model"]["arch"]}')
            self.accelerator.print(f'max epoch: {self.config["trainer"]["max_epochs"]}')
            self.accelerator.print(f'batch size: {self.config["trainer"]["batch_size"]}')
            self.accelerator.print(f'gradient accumulate: {self.config["trainer"]["grad_accumulate"]}')
            self.accelerator.print(f'init lr: {self.config["trainer"]["lr"]}')
            self.accelerator.print(f'ema decay: {self.config["trainer"]["ema_decay"]}')
            self.accelerator.print(f'ema update every: {self.config["trainer"]["ema_update_every"]}')
            
            self.accelerator.print(f'diffusion type: {self.config["diffusion"]["name"]}')
            self.accelerator.print(f'diffusion steps: {self.config["diffusion"]["timesteps"]}')
            self.accelerator.print(f'ddim sampling steps: {self.config["diffusion"]["ddim_timestep"]}')

            self.accelerator.print(f'sample per {self.sample_step} steps')

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        self.model_summarize()
        self.trainer_config_summarize()
        writer = SummaryWriter()

        with tqdm(initial = self.step, total = self.max_step, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.max_step:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)['layout'].to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                writer.add_scalar('loss', float(total_loss), self.step)
                writer.add_scalar('lr', self.scheduler.get_last_lr()[0], self.step)

                pbar.set_description(f'loss: {total_loss:.5f}, lr: {self.opt.param_groups[0]["lr"]:.6f}, epoch: {self.step* self.batch_size*self.gradient_accumulate_every/len(self.ds):.2f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scheduler.step()
                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.sample_step):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.sample_step
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim = 0) # (num_samples, channel, image_size, image_size)

                        
                        # utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                        if self.data_type == 'rgb':
                            utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}-rgb.png'), nrow = int(math.sqrt(self.num_samples)))
                            self.vis.visualize_rgb_layout(all_images, str(self.results_folder / f'sample-{milestone}-c-rgb.png'))
                        else:
                            self.vis.visulize_onehot_layout(all_images, str(self.results_folder / f'sample-{milestone}-onehot.png'))
                            self.vis.visualize_rgb_layout(all_images, str(self.results_folder / f'sample-{milestone}-rgb.png'))
                        # whether to calculate fid
                        
                        if self.calculate_fid:
                            try:
                                fid_score = self.fid_scorer.fid_score()
                                writer.add_scalar('fid_score', fid_score, self.step)
                                accelerator.print(f'fid_score: {fid_score}')
                            except:
                                accelerator.print('fid computation failed')
                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
        writer.close()


    def sample(self, num_samples=16, batch_size=16, milestone=None):

        self.load(milestone)
        self.model.eval()

        assert num_samples % batch_size == 0, f"num_samples ({num_samples}) must be divisible by batch_size ({batch_size})"


        batches = num_to_groups(num_samples, batch_size)
        all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

        all_images = torch.cat(all_images_list, dim = 0)

        if self.calculate_fid and self.accelerator.is_main_process:
            try:
                fid_score = self.fid_scorer.fid_score()
                self.accelerator.print(f'fid_score: {fid_score}')
            except:
                self.accelerator.print('fid computation failed')

        return all_images