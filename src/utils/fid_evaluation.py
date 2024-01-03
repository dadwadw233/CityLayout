import math
import os

import numpy as np
import torch
from einops import rearrange, repeat
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from tqdm.auto import tqdm
from .utils import OSMVisulizer
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class FIDEvaluation:
    def __init__(
        self,
        batch_size,
        dl,
        sampler,
        channels=3,
        accelerator=None,
        stats_dir="./results",
        device="cuda",
        num_fid_samples=50000,
        inception_block_idx=2048,
        data_type="rgb",
        mapping=None,
        condition=False,
    ):
        self.batch_size = batch_size
        self.n_samples = num_fid_samples
        self.device = device
        self.channels = channels
        self.dl = dl
        self.sampler = sampler
        self.stats_dir = stats_dir
        self.print_fn = print if accelerator is None else accelerator.print
        assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
        self.inception_v3 = InceptionV3([block_idx]).to(device)
        self.dataset_stats_loaded = False
        self.vis = OSMVisulizer(mapping)
        self.data_type = data_type
        self.condition = condition
        self.fid_calculator = FrechetInceptionDistance(feature=inception_block_idx, normalize=True).to(device)
        self.kid_calculator = KernelInceptionDistance(feature=inception_block_idx, normalize=True).to(device)
        self.Is_caluculator = InceptionScore(normalize=True).to(device)

    def calculate_inception_features(self, samples):
        if self.channels == 1:
            samples = repeat(samples, "b 1 ... -> b c ...", c=3)

        self.inception_v3.eval()
        features = self.inception_v3(samples)[0]

        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
        return features

    def load_or_precalc_dataset_stats(self):
        path = os.path.join(self.stats_dir, "dataset_stats")
        try:
            ckpt = np.load(path + ".npz")
            self.m2, self.s2, self.real = ckpt["m2"], ckpt["s2"], ckpt["real_features"]
            self.print_fn("Dataset stats loaded from disk.")
            ckpt.close()
        except Exception as e:
            num_batches = int(math.ceil(self.n_samples / self.batch_size))
            stacked_real_features = []
            self.print_fn(
                f"Stacking Inception features for {self.n_samples} samples from the real dataset."
            )
            for _ in tqdm(range(num_batches)):
                cond = None
                try:
                    data = next(self.dl)
                    real_samples = data['layout']
                    if self.condition:
                        cond = data['condition'].to(self.device)
                    else:
                        cond = None
                    
                except StopIteration:
                    break
                real_samples = real_samples.to(self.device)
                if cond is not None:
                    real_samples = torch.cat([real_samples, cond], dim=1)
                if self.data_type == "one-hot":
                    real_samples = self.vis.onehot_to_rgb(real_samples)
                real_features = self.calculate_inception_features(real_samples)
                stacked_real_features.append(real_features)
            stacked_real_features = (
                torch.cat(stacked_real_features, dim=0).cpu().numpy()
            )
            m2 = np.mean(stacked_real_features, axis=0)
            s2 = np.cov(stacked_real_features, rowvar=False)
            np.savez_compressed(path, m2=m2, s2=s2, real_features=stacked_real_features)
            self.print_fn(f"Dataset stats cached to {path}.npz for future use.")
            self.m2, self.s2, self.real = m2, s2, stacked_real_features
        self.dataset_stats_loaded = True

    @torch.inference_mode()
    def fid_score(self):
        if not self.dataset_stats_loaded:
            self.load_or_precalc_dataset_stats()
        self.sampler.eval()
        batches = num_to_groups(self.n_samples, self.batch_size)
        stacked_fake_features = []
        self.print_fn(
            f"Stacking Inception features for {self.n_samples} generated samples."
        )
        for batch in tqdm(batches):
            if self.condition:
                cond = next(self.dl)['condition'].to(self.device)
            else:
                cond = None
            fake_samples = self.sampler.sample(batch_size=batch, cond=cond)
            if self.data_type == "one-hot":
                fake_samples = self.vis.onehot_to_rgb(fake_samples)
            fake_features = self.calculate_inception_features(fake_samples)
            stacked_fake_features.append(fake_features)
        stacked_fake_features = torch.cat(stacked_fake_features, dim=0).cpu().numpy()
        m1 = np.mean(stacked_fake_features, axis=0)
        s1 = np.cov(stacked_fake_features, rowvar=False)

        return calculate_frechet_distance(m1, s1, self.m2, self.s2)
    
    @torch.inference_mode()
    def evaluate(self):
        
        self.sampler.eval()
        batches = num_to_groups(self.n_samples, self.batch_size)

        self.print_fn(
            f"Stacking real sample..."
        )
        for batch in tqdm(batches):
            cond = None
            try:
                data = next(self.dl)
                real_samples = data['layout']
                if self.condition:
                    cond = data['condition'].to(self.device)
                else:
                    cond = None
                
            except StopIteration:
                break
            real_samples = real_samples.to(self.device)
            if cond is not None:
                real_samples = torch.cat([real_samples, cond], dim=1)
            if self.data_type == "one-hot":
                real_samples = self.vis.onehot_to_rgb(real_samples)
            
            real_samples = real_samples.to(self.device)
            self.fid_calculator.update(real_samples, True)
            self.kid_calculator.update(real_samples, True)

        self.print_fn(
            f"Stacking fake sample..."
        )
        for batch in tqdm(batches):
            if self.condition:
                cond = next(self.dl)['condition'].to(self.device)
            else:
                cond = None
            fake_samples = self.sampler.sample(batch_size=batch, cond=cond)
            if self.data_type == "one-hot":
                fake_samples = self.vis.onehot_to_rgb(fake_samples)
            
            fake_samples = fake_samples.to(self.device)
            self.fid_calculator.update(fake_samples, False)
            self.kid_calculator.update(fake_samples, False)
            self.Is_caluculator.update(fake_samples)
        
            

        fid = self.fid_calculator.compute().item()
        kid_mean, kid_std = self.kid_calculator.compute()
        kid = kid_mean.item()
        
        is_mean, is_std = self.Is_caluculator.compute()
        is_score = is_mean.item()
        
        return fid, kid, is_score
    