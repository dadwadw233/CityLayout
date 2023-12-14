import yaml
import torch
import numpy as np
import os
import math
import matplotlib.pyplot as plt


def cycle(dl):
    while True:
        for data in dl:
            yield data


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules
# visulization class
class OSMVisulizer:
    
    def __init__(self):
        self.name = "OSMVisulizer"
    def visulize_onehot_layout(self, data, path):
        b, c , h, w = data.shape
        print(path)
        
        fig, axes = plt.subplots(b, c, figsize=(20, 20))
        
        for i in range(b):
            for j in range(c):
                axes[i, j].imshow(data[i, j, :, :].cpu().numpy(), cmap='gray')
                axes[i, j].axis('off')
        plt.savefig(path)
        plt.close()
        

        
        