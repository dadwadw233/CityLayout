import yaml
import torch
import numpy as np
import os
import matplotlib.pyplot as plt


def cycle(dl):
    while True:
        for data in dl:
            yield data


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# visulization class
class OSMVisulizer:
    
    def __init__(self):
        self.color_dict = {
            'building': [0, 0, 0],
            'road': [255, 255, 255],
            'nature': [0, 255, 0],
            'water': [0, 0, 255],
            'landuse': [255, 0, 0],
            'unknown': [255, 255, 0]
        }
    def visulize_onehot_layout(self, data, path):
        b, c , h, w = data['layout'].shape

        fig, axes = plt.subplots(1, c, figsize=(20, 20))

        for i in range(c):
            axes[i].imshow(data['layout'][0, i, :, :].cpu().numpy(), cmap='gray')
            axes[i].axis('off')
        plt.savefig(path)
        plt.close()
        

        
        