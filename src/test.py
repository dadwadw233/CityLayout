from data.osm_loader import OSMDataset

import torch

from torch.utils.data import DataLoader

from torchvision import transforms

import numpy as np

import os

import argparse

from accelerate import Accelerator

import matplotlib.pyplot as plt
from utils.utils import cycle, load_config, OSMVisulizer


if __name__ == "__main__":
    accelerator = Accelerator()

    ds_config = load_config(
        "/home/admin/workspace/yuyuanhong/code/CityLayout/config/data/osm_loader.yaml"
    )

    ds = OSMDataset(config=ds_config)

    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=16)

    dl = accelerator.prepare(dl)

    dl = cycle(dl)

    print("device:", accelerator.device)

    print("data len:", len(ds))

    vis = OSMVisulizer()
    for i in range(10):
        data = next(dl)
        print(data.keys())
        print(data["building"].shape)
        print(data["name"])
        vis.visulize_onehot_layout(data, "/home/admin/workspace/yuyuanhong/code/CityLayout/test-{}.png".format(i))

        print(data["nature"].shape)
        print(data["road"].shape)
        exit(0)
        print("-------------------")

    print("done")
