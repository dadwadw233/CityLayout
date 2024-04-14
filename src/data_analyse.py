from datasets.osm_loader import OSMDataset

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
import json
from utils.log import *
import random
import math


def discretize_data(data, bin_width):
    min_val = min(data)
    max_val = max(data)
    bins = np.arange(min_val, max_val + bin_width, bin_width)

    histogram, bin_edges = np.histogram(data, bins=bins)

    max_freq_index = np.argmax(histogram)
    mode_bin = bin_edges[max_freq_index], bin_edges[max_freq_index + 1]

    return mode_bin, histogram[max_freq_index]


## todo split bug need to fix
def data_spliter(data, data_names, _bin, train_ratio=0.9, test_ratio=0.05, val_ratio=0.05):
    # assert len(data.keys()) == 1
    data = data[list(data.keys())[0]]['overlap']
    total_size = len(data)
    DEBUG("total size: {}".format(total_size))
    bins_len = math.ceil(1 / _bin)
    bins = [[] for _ in range(bins_len)]

    # assert len(data) == len(data_names)

    DEBUG("data len: {}, data names len: {}".format(len(data), len(data_names)))

    for i in range(len(data)):
        bin_index = min(int(data[i] // _bin), bins_len - 1)
        bins[bin_index].append(data_names[i])

    DEBUG("bins len: {}".format(len(bins)))

    train, test, val = [], [], []
    for bin_data in bins:
        random.shuffle(bin_data)  
        bin_size = len(bin_data)
        bin_train_size = int(bin_size * train_ratio)
        bin_test_size = int(bin_size * test_ratio)
        bin_val_size = bin_size - bin_train_size - bin_test_size

        train.extend(bin_data[:bin_train_size])
        test.extend(bin_data[bin_train_size:bin_train_size + bin_test_size])
        val.extend(bin_data[bin_train_size + bin_test_size:])

    DEBUG("train size: {}, test size: {}, val size: {}".format(len(train), len(test), len(val)))
    with open('./data/train.json', 'w') as f:
        json.dump(train, f)
    with open('./data/test.json', 'w') as f:
        json.dump(test, f)
    with open('./data/val.json', 'w') as f:
        json.dump(val, f)

    return train, test, val


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

    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)

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
    data_names = []
    for _ in tqdm(range(len(ds))):
        data = next(dl)
        handle.add_data(data['layout'])
        data_names.append(data['name'][0])

    # handle.contrast_analyse()
    handle.analyse()
    # data_spliter(handle.get_data_dict(False), data_names, _bin=0.01, train_ratio=0.9, test_ratio=0.05, val_ratio=0.05)
