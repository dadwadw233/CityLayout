from data.osm_loader import OSMDataset

import torch

from torch.utils.data import DataLoader

from torchvision import transforms

import numpy as np

import os

import argparse

from accelerate import Accelerator

import matplotlib.pyplot as plt
from utils.utils import cycle


if __name__ == '__main__':
    accelerator = Accelerator()

    ds = OSMDataset(data_dir='/home/admin/workspace/yuyuanhong/code/CityLayout/data/train', mode='train', key_list=['building'])

    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

    dl = accelerator.prepare(dl)

    dl = cycle(dl)

    print('device:', accelerator.device)
    
    print('data len:', len(ds))
    for i in range(10):

        data = next(dl)
        print(data.keys())
        print(data['building'].shape)

        plt.imshow(data['building'][0][0].cpu().numpy())
        plt.show()
        plt.savefig('test.png')
        exit(0)
        print('-------------------')

    print('done')

