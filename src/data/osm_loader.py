import h5py
import torch
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class OSMDataset(Dataset):
    def __init__(self,
                  data_dir, 
                  mode,
                  transform=None,
                  key_list=None
                 ):
        self.mode = mode
        self.data_dir = data_dir
        self.data_list = os.listdir(data_dir)
        self.transform = transform
        self.key_list = key_list
        self.default_transform = transforms.Compose([
            transforms.Resize((512, 512)),
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_name = self.data_list[idx]

        data_path = os.path.join(self.data_dir, data_name)

        data = h5py.File(data_path, 'r')
    
        data_dict = {}
        if self.data_list is None:
            data_dict = {
                            'building': torch.from_numpy(np.array(data['building'])).float().squeeze(2).permute(2, 0, 1),
                            'landuse': torch.from_numpy(np.array(data['landuse'])).float().squeeze(2).permute(2, 0, 1),
                            'nature': torch.from_numpy(np.array(data['nature'])).float().squeeze(2).permute(2, 0, 1),
                            'road': torch.from_numpy(np.array(data['road'])).float().squeeze(2).permute(2, 0, 1),
                        }

        else:
            for key in data.keys():
                if key in self.key_list:
                    data_dict[key] = (torch.from_numpy(np.array(data[key])).float()).squeeze(2).permute(2, 0, 1)
                else:
                    continue



        if self.transform:
            data_dict = self.transform(data_dict)
        else:
            for key in data_dict.keys():
                data_dict[key] = self.default_transform(data_dict[key])

        return data_dict