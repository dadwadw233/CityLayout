from data.osm_loader import OSMDataset

import torch

from torch.utils.data import DataLoader

from torchvision import transforms

import numpy as np

import os

import argparse

from accelerate import Accelerator

import matplotlib.pyplot as plt
from utils.utils import cycle, load_config
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
        "/home/admin/workspace/yuyuanhong/code/CityLayout/config/data/osm_loader.yaml"
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
    for _ in tqdm(range(len(ds))):
        data = next(dl)
        road_rate = torch.count_nonzero(data["road"]) / (data["road"].shape[2] * data["road"].shape[3])

        building_rate = torch.count_nonzero(data["building"]) / (data["building"].shape[2] * data["building"].shape[3])

        natural_rate = torch.count_nonzero(data["natural"]) / (data["natural"].shape[2] * data["natural"].shape[3])

        road.append(road_rate.cpu())

        building.append(building_rate.cpu())

        natural.append(natural_rate.cpu())

        cnt += 1
        
        
    # print overlap rate satistic result
    print("road:", np.mean(road), np.std(road))
    print("building:", np.mean(building), np.std(building))
    print("natural:", np.mean(natural), np.std(natural))


    # plt overlap rate satistic result separately

    plt.figure()
    plt.hist(road, bins=100, color="red", label="road")
    plt.xlabel("overlap rate")
    plt.ylabel("count")
    plt.legend()
    plt.savefig("overlap_rate_road.png")
    plt.close()


    plt.figure()
    plt.hist(building, bins=100, color="blue", label="building")
    plt.xlabel("overlap rate")
    plt.ylabel("count")
    plt.legend()
    plt.savefig("overlap_rate_building.png")
    plt.close()

    plt.figure()
    plt.hist(natural, bins=100, color="green", label="natural")
    plt.xlabel("overlap rate")
    plt.ylabel("count")
    plt.legend()
    plt.savefig("overlap_rate_natural.png")
    plt.close()

    mode_bin_building, building_freq = discretize_data(building, 0.01)
    mode_bin_road, road_freq = discretize_data(road, 0.01)
    mode_bin_natural, natural_freq = discretize_data(natural, 0.01)

    print("building众数区间:", mode_bin_building, "频数:", building_freq)
    print("road众数区间:", mode_bin_road, "频数:", road_freq)
    print("natural众数区间:", mode_bin_natural, "频数:", natural_freq)