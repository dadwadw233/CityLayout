import h5py
import torch
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.colors as mcolors
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
        self.resize = transforms.Compose([
            transforms.Resize((config['data']['resize'])),
        ])
        self.normalize_method = config['data']['normalizer']

        self.type = config['data']['type']
        self.channel_to_rgb = config['data']['channel_to_rgb']

        if self.normalize_method == 'minmax':
            self.normalize = self.minmax
        elif self.normalize_method == 'zscore':
            self.normalize = transforms.Compose([
                transforms.Normalize((0.0, ), (1.0, ))
            ])
        elif self.normalize_method == 'clamp':
            self.normalize = transforms.Compose([
                self.clamp,
                transforms.Normalize((0.0, ), (1.0, ))
            ])
        else :
            if config['data']['std'] == None or config['data']['mean'] == None:
                raise ValueError('std or mean is None')
            else:
                self.normalize = transforms.Compose([
                    transforms.Normalize(config['data']['mean'], config['data']['std'])
                ])

    def minmax(self, data):
        data = data - data.min()
        data = data / data.max()
        return data
    
    def clamp(self, data):
        data = data.clamp(0, 1)
        return data

    def custmize_layout(self, custom_list, key_map, data):
        assert data.shape[-1] == len(key_map)
        assert isinstance(data, torch.Tensor)

        layout_list = []
        for i, item in enumerate(custom_list):

            indices = [key_map[label] for label in item]
            
            layer = data[:, :, indices].sum(dim=-1, keepdim=True)
            layout_list.append(layer)


        return transforms.Normalize((0.0), (1.0))(torch.cat(layout_list, dim=-1).permute(2, 0, 1).float())

    def hex_or_name_to_rgb(self, color):

        # 使用matplotlib的颜色转换功能
        return mcolors.to_rgb(color)
    
    def generate_rgb_layout(self, data) -> torch.Tensor:
        C, H, W = data.shape
        assert (
            self.channel_to_rgb.__len__() == C
        ), f"channel to rgb mapping length {self.channel_to_rgb.__len__()} does not match channel number {c}"

        rgb_image = torch.zeros((H, W, 3), dtype=torch.float32,device=data.device)
        for c in range(C):
            color = torch.tensor(self.hex_or_name_to_rgb(self.channel_to_rgb[c]))
            mask = data[c] > 0
            rgb_image[mask, :] += color
            
        # combined_image = torch.clip(combined_image, 0, 1)  # 确保颜色值在0-1范围内
        

        # return shape : (, h, w, c)
        
        return  rgb_image.permute(2, 0, 1)

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
                data_dict[key] = self.normalize(self.resize(data_dict[key]))

        data_dict['name'] = data_name
        if self.type == 'rgb':
            data_dict['layout'] = self.generate_rgb_layout(torch.cat([data_dict[key] for key in data_dict.keys() if key != 'name'], dim=0))
        elif self.type == 'one-hot':
            data_dict['layout'] = torch.cat([data_dict[key] for key in data_dict.keys() if key != 'name'], dim=0)
        else:
            raise ValueError('type must be rgb or one-hot')

        h5py.File.close(data)

        return data_dict