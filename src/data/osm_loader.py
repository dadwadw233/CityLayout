import h5py
import torch
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class OSMDataset(Dataset):
    def __init__(self,
                  data_dir=None, 
                  mode=None,
                  transform=None,
                  key_list=None,
                  config=None
                 ):
        if data_dir is None:
            self.data_dir = config['path']['data_dir']
        else:
            self.data_dir = data_dir

        if mode is None:
            self.mode = config['params']['mode']
        else:
            self.mode = mode
        
        self.config = config
        self.custom = config['params']['custom']

        self.data_list = os.listdir(self.data_dir)
        self.transform = transform
        self.key_list = key_list
        self.default_transform = transforms.Compose([
            transforms.Resize((512, 512)),
        ])

    def custmize_layout(self, custom_list, key_map, data):
        assert data.shape[-1]==len(key_map)
        assert type(data)==torch.Tensor

        layout_list = []
        for item in custom_list:
            inner_layout = torch.zeros(data.shape[0], data.shape[1], 1)
            for label in item:
                inner_layout += data[:, :, key_map[label]:key_map[label]+1]
            layout_list.append(inner_layout)

        return torch.cat(layout_list, dim=-1).permute(2, 0, 1).float()


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_name = self.data_list[idx]

        data_path = os.path.join(self.data_dir, data_name)

        data = h5py.File(data_path, 'r')
    
        data_dict = {}
        if self.key_list is None:
            if self.custom:
                if self.config['data']['custom_dict']['building'].__len__()!=0:
                    data_dict['building'] = self.custmize_layout(self.config['data']['custom_dict']['building'], self.config['data']['key_map']['building'], torch.from_numpy(np.array(data['building'])).squeeze(2))
                if self.config['data']['custom_dict']['landuse'].__len__()!=0:
                    data_dict['landuse'] = self.custmize_layout(self.config['data']['custom_dict']['landuse'], self.config['data']['key_map']['landuse'], torch.from_numpy(np.array(data['landuse'])).squeeze(2))
                if self.config['data']['custom_dict']['nature'].__len__()!=0:
                    data_dict['nature'] = self.custmize_layout(self.config['data']['custom_dict']['nature'], self.config['data']['key_map']['nature'], torch.from_numpy(np.array(data['nature'])).squeeze(2))
                if self.config['data']['custom_dict']['road'].__len__()!=0:
                    data_dict['road'] = self.custmize_layout(self.config['data']['custom_dict']['road'], self.config['data']['key_map']['road'], torch.from_numpy(np.array(data['road'])).squeeze(2))

            else:
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

        data_dict['name'] = data_name

        return data_dict