import yaml
import torch
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torchvision import transforms

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


def cast_tuple(t, length=1):
    if isinstance(t, tuple):
        return t
    return (t,) * length


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


def cal_overlapping_rate(tensor):
    """
    calculate the overlapping area of a tensor
    Args:
        tensor: shape (b, c, h, w)
    Returns:
        overlapping area: shape (b, c)
    """
    b, c, h, w = tensor.shape
    
    region = torch.zeros_like((tensor[:, 0, :, :]))
    for i in range(c):
        region += tensor[:, i, :, :]

    region = region > 1
    
    return (region.sum(dim=[1, 2]).float() / (h * w)).mean()
    



# small helper modules
# visulization class
class OSMVisulizer:
    def __init__(self, mapping):
        self.name = "OSMVisulizer"
        self.channel_to_rgb = mapping

    def minmax(self, data):
        data = data - data.min()
        data = data / data.max()
        return data
    
    def visulize_onehot_layout(self, data, path) -> None:
        b, c, h, w = data.shape
        print(path)

        fig, axes = plt.subplots(b, c, figsize=(20, 20))

        for i in range(b):
            for j in range(c):
                axes[i, j].imshow(data[i, j, :, :].cpu().numpy(), cmap="gray")
                axes[i, j].axis("off")
        plt.savefig(path)
        plt.close()

    def hex_or_name_to_rgb(self, color):

        # 使用matplotlib的颜色转换功能
        return mcolors.to_rgb(color)

    # need test
    def visualize_rgb_layout(self, data, path) -> np.ndarray:
        B, C, H, W = data.shape
        assert (
            self.channel_to_rgb.__len__() == C
        ), f"channel to rgb mapping length {self.channel_to_rgb.__len__()} does not match channel number {c}"
        data = data.cpu().numpy()

        fig, axs = plt.subplots(B, 1, figsize=(20, 20*B))  # 根据批次数量创建子图

        for b in range(B):
            combined_image = np.zeros((H, W, 3), dtype=np.float32)
            for c in range(C):
                color = np.array(self.hex_or_name_to_rgb(self.channel_to_rgb[c]))
                mask = data[b, c] > 0.7
                combined_image[mask, :] += color
            
            combined_image = np.clip(combined_image, 0, 1)  # 确保颜色值在0-1范围内
            axs[b].imshow(combined_image)
            axs[b].axis('off')  # 关闭坐标轴


        # 绘制图像
        plt.axis('off')
        plt.imshow(combined_image)
        plt.show()
        plt.savefig(path)
        plt.close()

        # return shape : (b, h, w, c)
        return  None
    
    def onehot_to_rgb(self, data):
        B, C, H, W = data.shape
        assert (
            self.channel_to_rgb.__len__() == C
        ), f"channel to rgb mapping length {self.channel_to_rgb.__len__()} does not match channel number {c}"


        combined_image = torch.zeros((B, H, W, 3), dtype=torch.float32, device=data.device)

        for b in range(B):
            for c in range(C):
                color = torch.tensor(self.hex_or_name_to_rgb(self.channel_to_rgb[c]), device=data.device)
                mask = data[b, c] > 0
                combined_image[b, mask, :] += color
        
        combined_image = self.minmax(combined_image.permute(0, 3, 1, 2))

        if torch.isnan(combined_image).any():
            combined_image[torch.isnan(combined_image)] = 0

        return combined_image
                

                
        
