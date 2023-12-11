import h5py
import torch
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader


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

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_name = self.data_list[idx]

        data_path = os.path.join(self.data_dir, data_name)

        data = h5py.File(data_path, 'r')
    
        data_dict = {}
        if self.data_list is None:
            data_dict = {
                            'building': torch.from_numpy(np.array(data['building'])).float().squeeze(2),
                            'landuse': torch.from_numpy(np.array(data['landuse'])).float().squeeze(2),
                            'nature': torch.from_numpy(np.array(data['nature'])).float().squeeze(2),
                            'road': torch.from_numpy(np.array(data['road'])).float().squeeze(2)
                        }

        else:
            for key in data.keys():
                if key in self.key_list:
                    data_dict[key] = (torch.from_numpy(np.array(data[key])).float()).squeeze(2)
                else:
                    continue



        if self.transform:
            data_dict = self.transform(data_dict)

        return data_dict