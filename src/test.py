from data.osm_loader import OSMDataset

import torch

from torch.utils.data import DataLoader

from torchvision import transforms

import numpy as np

import os

import argparse

from accelerate import Accelerator

import matplotlib.pyplot as plt
from utils.utils import cycle, load_config, cal_overlapping_rate
from utils.vis import OSMVisulizer
from utils.asset import AssetGen


if __name__ == "__main__":
    accelerator = Accelerator()

    ds_config = load_config(
        "/home/admin/workspace/yuyuanhong/code/CityLayout/config/data/osm_loader.yaml"
    )
    trainer_config = load_config(
        "/home/admin/workspace/yuyuanhong/code/CityLayout/config/train/osm_generator_sample.yaml"
    )
    uni_config = load_config(
        "/home/admin/workspace/yuyuanhong/code/CityLayout/config/new/uniDM_train_new.yaml"
    )


    ds = OSMDataset(config=uni_config['Data'])

    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=1)

    dl = accelerator.prepare(dl)

    dl = cycle(dl)

    print("device:", accelerator.device)

    print("data len:", len(ds))

    vis = OSMVisulizer(config=uni_config["Vis"])
    asset = AssetGen(config=uni_config["Asset"])
    for i in range(3):
        data = next(dl)
        # print(data.keys())
        # print(data["building"].shape)
        # print(data["name"])
        # print(data['condition'].shape)
        print(data['layout'].shape)
        
        vis.visualize_onehot_layout(data['layout'], "/home/admin/workspace/yuyuanhong/code/CityLayout/test-{}.png".format(i))
        vis.visualize_rgb_layout(data['layout'], "/home/admin/workspace/yuyuanhong/code/CityLayout/test-rgb-{}.png".format(i))
        
        rgb_test = vis.onehot_to_rgb(data['layout'])
        rgb_test_for_show = rgb_test[0]
        print(rgb_test_for_show.max(), rgb_test_for_show.min())
        plt.imsave("/home/admin/workspace/yuyuanhong/code/CityLayout/test1-rgb-{}.png".format(i), (rgb_test_for_show.permute(1,2,0)*255).to(torch.uint8).cpu().numpy())
        
        asset.add_data(data['layout'])
        asset.generate_geofiles()

        exit(0)
        print("-------------------")

    print("done")