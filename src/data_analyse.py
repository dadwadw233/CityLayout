from data.osm_loader import OSMDataset

import torch

from torch.utils.data import DataLoader

from torchvision import transforms

import numpy as np

import os

import argparse

from accelerate import Accelerator

import matplotlib.pyplot as plt
from utils.utils import cycle, load_config, DataAnalyser
from torchvision import transforms as T, utils
from tqdm import tqdm
from scipy import stats


def discretize_data(data, bin_width):
    # 确定数据的最小值和最大值
    min_val = min(data)
    max_val = max(data)
    
    # 根据步长创建区间边界
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    
    # 使用histogram函数离散化数据
    histogram, bin_edges = np.histogram(data, bins=bins)
    
    # 找出频率最高的区间
    max_freq_index = np.argmax(histogram)
    mode_bin = bin_edges[max_freq_index], bin_edges[max_freq_index + 1]
    
    return mode_bin, histogram[max_freq_index]

if __name__ == "__main__":
    device = "cpu"

    accelerator = Accelerator()

    ds_config = load_config(
        "/home/admin/workspace/yuyuanhong/code/CityLayout/config/data/data_analyse.yaml"
    )
    trainer_config = load_config(
        "/home/admin/workspace/yuyuanhong/code/CityLayout/config/train/osm_generator.yaml"
    )


    ds = OSMDataset(config=ds_config)

    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=1)

    dl = accelerator.prepare(dl)

    dl = cycle(dl)

    print("device:", accelerator.device)

    print("data len:", len(ds))

    # stastic all training data
    cnt = 0
    road = []
    building = []
    natural = []
    handle = DataAnalyser(config=ds_config)
    handle.init_folder()
    for _ in tqdm(range(len(ds))):
        data = next(dl)
        handle.add_data(data['layout'])
       
    
    # handle.contrast_analyse()
    handle.analyse()
        