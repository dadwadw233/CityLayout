import math
import copy
from pathlib import Path
import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from utils.fid_evaluation import FIDEvaluation
from itertools import combinations
from .version import __version__
from utils.utils import (
    cycle,
    exists,
    identity,
    convert_image_to_fn,
    default,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
    has_int_squareroot,
    divisible_by,
    num_to_groups,
)
from utils.log import *
# constantsF

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])

# helpers functions


# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        objective="pred_v",
        beta_schedule="sigmoid",
        schedule_fn_kwargs=dict(),
        ddim_sampling_eta=0.0,
        auto_normalize=True,
        offset_noise_strength=0.0,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight=False,  # https://arxiv.org/abs/2303.09556
        min_snr_gamma=5,
        model_type="uniDM", # or "Outpainting"
    ):
        super().__init__()
        
        assert not model.random_or_learned_sinusoidal_cond
        self.sample_mode = "normal"
        self.sample_type = "CityGen"
        self.model = model
        self.model_type = model_type
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])"

        if beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == "sigmoid":
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == "pred_noise":
            register_buffer("loss_weight", maybe_clipped_snr / snr)
        elif objective == "pred_x0":
            register_buffer("loss_weight", maybe_clipped_snr)
        elif objective == "pred_v":
            register_buffer("loss_weight", maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

        self.combinations = self.generate_all_channel_combinations(self.channels)
        # print("all channel combinations: ", self.combinations)
        
        
        
    def set_sample_type(self, sample_type):
        self.sample_type = sample_type
        INFO(f"set sample type to {self.sample_type}")
    
    def set_sample_mode(self, sample_mode):
        self.sample_mode = sample_mode
        INFO(f"set sample mode to {self.sample_mode}")

    def generate_all_channel_combinations(self, num_channels):
        # 生成所有可能的通道组合
        all_combinations = []
        for r in range(num_channels + 1):
            if r == 0:
                continue
            all_combinations.extend(combinations(range(num_channels), r))
        return all_combinations

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self, x, t, x_self_cond=None, op_mask=None, clip_x_start=False, rederive_pred_noise=False
    ):
        if op_mask is not None:
            model_input = torch.cat((x, op_mask), dim=1)
            model_output = self.model(model_input, t, x_self_cond)
        else:
            model_output = self.model(x, t, x_self_cond)

        maybe_clip = (
            partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        )

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start
    
    def get_random_noise_forward(self, image):
        # select random channel to add standard gaussian noise
        b, c, h, w = image.shape
        
        noise = torch.zeros_like(image, device=self.device)

        # 👇 maybe image in the batch have different conditional channel will lead to better performance
        # combinations_idx = torch.randint(0, len(self.combinations), (1,))

        combinations_idx = torch.randint(0, len(self.combinations), (b,))
        # combinations_idx shape : (b,)

        # then replace these channels with noise
        
        for i in range(b):
            noise[i, self.combinations[combinations_idx[i]], :, :] = torch.randn_like(image[i, self.combinations[combinations_idx[i]], :, :])

        return noise
    
    def get_random_outpainting_noise_forward(self, image):
        # crop some continue region from image and add noise
        b, c, h, w = image.shape

        noise = torch.zeros_like(image, device=self.device)

        # Define the size of the noise region
        # For simplicity, we will create a rectangular noise region
        # 50 % percent , mask(noise) region's height or width is equal to image's height or width
        if random.random() < 0.5:
            noise_height = h
            noise_width = random.randint(w // 8, w)
        else :
            noise_height = random.randint(h // 8, h)
            noise_width = w
        
        # Define the top left corner of the noise region
        if noise_height == h:
            if random.random() < 0.5:
                start_x = 0
            else:
                start_x = w - noise_width
            start_y = 0

        elif noise_width == w:
            if random.random() < 0.5:
                start_y = 0
            else:
                start_y = h - noise_height
            start_x = 0

        # Generate random noise
        noise_region = torch.randn(b, c, noise_height, noise_width, device=self.device)

        # Apply the noise to the corresponding region in the image
        noise[:, :, start_y:start_y + noise_height, start_x:start_x + noise_width] = noise_region

        # TODO: check whethter need to clip to [-1, 1]
        noise = torch.clamp(noise, -1.0, 1.0)

        # generate binary mask for noise region (b, 1, h, w)
        mask_shape = (b, 1, h, w)
        mask = torch.zeros(mask_shape, device=self.device)
        mask[:, :, start_y:start_y + noise_height, start_x:start_x + noise_width] = 1.0 # 1.0 means noise region(outpainting region)

        # normalize mask from [0, 1] to [-1, 1]
        mask = mask * 2.0 - 1.0

        return noise, mask

    @torch.inference_mode()
    def random_mask_image_backward(self, image):

        b, c, h, w = image.shape

        # combinations_idx = torch.randint(0, len(self.combinations), (1,))
        combinations_idx = torch.randint(0, len(self.combinations), (b,))
        # then replace these channels with noise
        # image[:, self.combinations[combinations_idx], :, :] = torch.randn_like(image[:, self.combinations[combinations_idx], :, :])
        for i in range(b):
            image[i, self.combinations[combinations_idx[i]], :, :] = torch.randn_like(image[i, self.combinations[combinations_idx[i]], :, :])

        return image
    
    @torch.inference_mode()
    def random_outpainting_noise_backward(self, image, mask=None):
        # crop some continue region from image and add noise
        b, c, h, w = image.shape

        # Define the size of the noise region
        # For simplicity, we will create a rectangular noise region
        # 50 % percent , mask(noise) region's height or width is equal to image's height or width
        if mask is None:
            
            if random.random() < 0.5:
                noise_height = h
                noise_width = random.randint(w // 8, w)
            else :
                noise_height = random.randint(h // 8, h)
                noise_width = w

            if noise_height == h:
                if random.random() < 0.5:
                    start_x = 0
                else:
                    start_x = w - noise_width
                start_y = 0

            elif noise_width == w:
                if random.random() < 0.5:
                    start_y = 0
                else:
                    start_y = h - noise_height
                start_x = 0

            # Generate random noise
            noise_region = torch.randn(b, c, noise_height, noise_width, device=self.device)

            # Apply the noise to the corresponding region in the image
            image[:, :, start_y:start_y + noise_height, start_x:start_x + noise_width] = noise_region
            image = torch.clamp(image, -1.0, 1.0)

            mask_shape = (b, 1, h, w)
            mask = torch.zeros(mask_shape, device=self.device)
            mask[:, :, start_y:start_y + noise_height, start_x:start_x + noise_width] = 1.0

            mask = mask * 2.0 - 1.0        
        
        else:
            # if mask is not None, then we use the given mask to generate noise
            # mask = torch.randn(b, 1, h, w, device=self.device)
            # mask = torch.clamp(mask, -1.0, 1.0)
            # mask = (mask + 1) / 2
            # mask = mask.bool()
            bool_mask = ((mask + 1) / 2).bool()
            noise = torch.randn(b, c, h, w, device=self.device)
            masked_noise = noise * bool_mask
            masked_img = image * ~bool_mask
            image = masked_img + masked_noise

        return image, mask





    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond=None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=True
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps=False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)
        return ret
    
    @torch.inference_mode()
    def get_sample_input(self, shape, org=None, mask=None):
        batch, device = shape[0], self.device
        if self.sample_mode == "uniDM":
            raise ValueError("uniDM model has been deprecated")
        elif self.sample_mode == "Outpainting":
            if org is None:
                # no condition supplied, generate random noise (normal mode)
                WARNING("no condition supplied, generate random noise")
                img = torch.randn(shape, device=device)
                org = img.clone()
            else:
                img = org.clone()
            
            if self.sample_type == "CityGen":
                # diffusion input : masked image wioth standard gaussian noise
                #DEBUG(self.sample_type)
                img, mask = self.random_outpainting_noise_backward(img, mask)
            else:
                # diffusion input : standard gaussian noise
                img = torch.randn(shape, device=device) 
        elif self.sample_mode == "normal":
            img = torch.randn(shape, device=device)
            org = img.clone()
        
        else:
            raise ValueError(f"unknown model type{self.model_type}")
            
        return img, org, mask
            

    @torch.inference_mode()
    def ddim_sample(self, shape, org=None, return_all_timesteps=False, mask=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = (
            shape[0],
            self.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )
        

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1], times[1:])
        )  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        op_mask = mask
        
        img, org, op_mask = self.get_sample_input(shape, org, mask)
            
        imgs = [img]

        x_start = None

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            # DEBUG(self.sample_type)
            # DEBUG(f"mask shape : {op_mask.shape}")
            if self.sample_type == "CityGen":
                pred_noise, x_start, *_ = self.model_predictions(
                    img, time_cond, self_cond, op_mask, clip_x_start=True, rederive_pred_noise=True
                )
            else:
                pred_noise, x_start, *_ = self.model_predictions(
                    img, time_cond, self_cond, clip_x_start=True, rederive_pred_noise=True
                )

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            imgs.append(img)
            
            # conditional sample
            if self.sample_mode == "Outpainting": 
                #DEBUG(img.shape)
                bool_mask = ((op_mask + 1) / 2).bool()
                cond_t = self.q_sample(org, torch.full((batch,), time_next, device=device, dtype=torch.long), noise)
                masked_cond_t = cond_t * ~bool_mask
                img = img * bool_mask + masked_cond_t
                #DEBUG(img.shape)
                
            
        

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        if not return_all_timesteps and org is not None:
            ret = torch.cat((ret, org), dim=1)
            if op_mask is not None:
                ret = torch.cat((ret, op_mask), dim=1)
        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, batch_size=16, cond=None, return_all_timesteps=False, mask=None):
        image_size, channels = self.image_size, self.channels
        sample_fn = (
            self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        )
        if cond is not None:
            cond = self.normalize(cond)
        if mask is not None:
            mask = self.normalize(mask)
        return sample_fn(
            (batch_size, channels, image_size, image_size), cond,
            return_all_timesteps=return_all_timesteps, mask=mask
        )

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device=device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(
            reversed(range(0, t)), desc="interpolation sample time step", total=t
        ):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled=False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise=None, offset_noise_strength=None):
        b, c, h, w = x_start.shape
        op_mask = None
        # noise = default(noise, lambda: torch.randn_like(x_start))
        if self.model_type == "uniDM":
            noise = default(noise, lambda: self.get_random_noise_forward(x_start))
        elif self.model_type == "Outpainting":
            noise, op_mask = self.get_random_outpainting_noise_forward(x_start)
        elif self.model_type == "normal":
            noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(
            offset_noise_strength, self.offset_noise_strength
        )

        if offset_noise_strength > 0.0:
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise += offset_noise_strength * rearrange(offset_noise, "b c -> b c 1 1")

        # noise sample
        

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # TODO: visualize for debug
        



        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond_copy = None
        if self.self_condition and random.random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
            
            x_self_cond_copy = x_self_cond.clone().requires_grad_()

        # predict and take gradient step

        if op_mask is not None:
            x = torch.concat((x, op_mask), dim=1)
        model_out = self.model(x, t, x_self_cond_copy)

        

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = F.mse_loss(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b", "mean")

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        (
            b,
            c,
            h,
            w,
            device,
            img_size,
        ) = (
            *img.shape,
            img.device,
            self.image_size,
        )
        assert (
            h == img_size and w == img_size
        ), f"height and width of image must be {img_size}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        
        return self.p_losses(img, t, *args, **kwargs)






