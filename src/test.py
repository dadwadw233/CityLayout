from data.osm_loader import OSMDataset

import torch

from torch.utils.data import DataLoader

from torchvision import transforms

import numpy as np

import os

import argparse

from accelerate import Accelerator

import matplotlib.pyplot as plt
from utils.utils import cycle, load_config, OSMVisulizer, cal_overlapping_rate, Vectorizer
from torchvision import transforms as T, utils


if __name__ == "__main__":
    accelerator = Accelerator()

    ds_config = load_config(
        "/home/admin/workspace/yuyuanhong/code/CityLayout/config/data/osm_loader.yaml"
    )
    trainer_config = load_config(
        "/home/admin/workspace/yuyuanhong/code/CityLayout/config/train/osm_generator.yaml"
    )


    ds = OSMDataset(config=ds_config)

    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=1)

    dl = accelerator.prepare(dl)

    dl = cycle(dl)

    print("device:", accelerator.device)

    print("data len:", len(ds))

    vis = OSMVisulizer(mapping=trainer_config["vis"]["channel_to_rgb"])
    vec = Vectorizer()
    for i in range(10):
        data = next(dl)
        print(data.keys())
        print(data["building"].shape)
        print(data["name"])
        print(data['condition'].shape)
        vis.visulize_onehot_layout(data['layout'], "/home/admin/workspace/yuyuanhong/code/CityLayout/test-{}.png".format(i))
        vis.visualize_rgb_layout(data['layout'], "/home/admin/workspace/yuyuanhong/code/CityLayout/test-rgb-{}.png".format(i))
        
        
        print(cal_overlapping_rate(data['layout']))
        # utils.save_image(data['layout'], './test.png', nrow = 4)
        print(data["natural"].shape)
        print(data["road"].shape)


        f = vec.vectorize(data["building"], 'building')

        vec.vectorize(data["road"], data_type='road', init_features=f, color='red')
        exit(0)
        print("-------------------")

    print("done")