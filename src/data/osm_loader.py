import h5py
import torch
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.colors as mcolors
import cv2
from torchvision.transforms.functional import InterpolationMode


class OSMDataset(Dataset):
    def __init__(
        self, data_dir=None, mode=None, transform=None, key_list=None, config=None
    ):
        

        if mode is None:
            self.mode = config["params"]["mode"]
        else:
            self.mode = mode

        if data_dir is None:
            if self.mode == "train":
                self.data_dir = config["path"]["train_dir"]
            else:
                self.data_dir = config["path"]["test_dir"]
        else:
            self.data_dir = data_dir

        self.config = config
        self.custom = config["params"]["custom"]

        self.data_list = os.listdir(self.data_dir)
        self.transform = transform
        self.key_list = key_list
        self.resize = transforms.Compose(
            [
                transforms.Resize((config["data"]["resize"]), interpolation=InterpolationMode.BICUBIC),
            ]
        )
        self.normalize_method = config["data"]["normalizer"]

        self.type = config["data"]["type"]
        self.channel_to_rgb = config["data"]["channel_to_rgb"]

        self.road_augment = transforms.Lambda(lambda x: self.widen_lines(x, kernel_size=3))

        if self.normalize_method == "minmax":
            self.normalize = self.minmax
        elif self.normalize_method == "zscore":
            self.normalize = transforms.Compose([transforms.Normalize((0.0,), (1.0,))])
        elif self.normalize_method == "clamp":
            self.normalize = transforms.Compose(
                [self.clamp, transforms.Normalize((0.0,), (1.0,))]
            )
        else:
            if config["data"]["std"] == None or config["data"]["mean"] == None:
                raise ValueError("std or mean is None")
            else:
                self.normalize = transforms.Compose(
                    [
                        transforms.Normalize(
                            config["data"]["mean"], config["data"]["std"]
                        )
                    ]
                )

    def minmax(self, data):
        data = data - data.min()
        data = data / data.max()
        return data

    def clamp(self, data):
        data = data.clamp(0, 1)
        return data

    def custmize_layout(self, custom_list, key_map, data):
        # print(data.shape)
        # print(len(key_map))
        data = data[:, :, : len(key_map)]
        
        assert data.shape[-1] == len(key_map)
        assert isinstance(data, torch.Tensor)

        layout_list = []
        for i, item in enumerate(custom_list):
            indices = [key_map[label] for label in item]

            layer = data[:, :, indices].sum(dim=-1, keepdim=True)
            layout_list.append(layer)

        ret = self.clamp(torch.cat(layout_list, dim=-1).permute(2, 0, 1).float())

       

        return ret

    def hex_or_name_to_rgb(self, color):
        return mcolors.to_rgb(color)

    def generate_rgb_layout(self, data) -> torch.Tensor:
        C, H, W = data.shape
        assert (
            self.channel_to_rgb.__len__() == C
        ), f"channel to rgb mapping length {self.channel_to_rgb.__len__()} does not match channel number {c}"

        rgb_image = torch.zeros((H, W, 3), dtype=torch.float32, device=data.device)
        for c in range(C):
            color = torch.tensor(self.hex_or_name_to_rgb(self.channel_to_rgb[c]))
            mask = data[c] > 0
            rgb_image[mask, :] += color

        # return shape : (, h, w, c)

        return self.clamp(rgb_image.permute(2, 0, 1))
    
    def widen_lines(self, image, kernel_size=3):
        
        image = image.numpy()

        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        dilated_image = cv2.dilate(image, kernel, iterations=1)



        dilated_image = torch.from_numpy(dilated_image)

        return dilated_image

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_name = self.data_list[idx]

        data_path = os.path.join(self.data_dir, data_name)

        data = h5py.File(data_path, "r")

        # recover data from packaged bool to float
        building = np.unpackbits(data["building"], axis=-1).astype(np.float32)
        landuse = np.unpackbits(data["landuse"], axis=-1).astype(np.float32)
        natural = np.unpackbits(data["natural"], axis=-1).astype(np.float32)
        road = np.unpackbits(data["road"], axis=-1).astype(np.float32)
        node = np.unpackbits(data["node"], axis=-1).astype(np.float32)
        
        data_dict = {}
        if self.key_list is None:
            if self.custom:
                if self.config["data"]["custom_dict"]["building"].__len__() != 0:
                    data_dict["building"] = self.custmize_layout(
                        self.config["data"]["custom_dict"]["building"],
                        self.config["data"]["key_map"]["building"],
                        torch.from_numpy(np.array(building)).squeeze(2),
                    )
                    if (
                        torch.count_nonzero(data_dict["building"])
                        / (
                            data_dict["building"].shape[1]
                            * data_dict["building"].shape[2]
                        )
                        <= self.config["data"]["filter"]["building"]
                    ):
                        return self.__getitem__(
                            np.random.randint(0, len(self.data_list))
                        )
                if self.config["data"]["custom_dict"]["landuse"].__len__() != 0:
                    data_dict["landuse"] = self.custmize_layout(
                        self.config["data"]["custom_dict"]["landuse"],
                        self.config["data"]["key_map"]["landuse"],
                        torch.from_numpy(np.array(landuse)).squeeze(2),
                    )
                    if (
                        torch.count_nonzero(data_dict["landuse"])
                        / (
                            data_dict["landuse"].shape[1]
                            * data_dict["landuse"].shape[2]
                        )
                        <= self.config["data"]["filter"]["landuse"]
                    ):
                        return self.__getitem__(
                            np.random.randint(0, len(self.data_list))
                        )
                if self.config["data"]["custom_dict"]["natural"].__len__() != 0:
                    data_dict["natural"] = self.custmize_layout(
                        self.config["data"]["custom_dict"]["natural"],
                        self.config["data"]["key_map"]["natural"],
                        torch.from_numpy(np.array(natural)).squeeze(2),
                    )
                    if (
                        torch.count_nonzero(data_dict["natural"])
                        / (
                            data_dict["natural"].shape[1]
                            * data_dict["natural"].shape[2]
                        )
                        <= self.config["data"]["filter"]["natural"]
                    ):
                        return self.__getitem__(
                            np.random.randint(0, len(self.data_list))
                        )
                if self.config["data"]["custom_dict"]["road"].__len__() != 0:
                    data_dict["road"] = self.custmize_layout(
                        self.config["data"]["custom_dict"]["road"],
                        self.config["data"]["key_map"]["road"],
                        torch.from_numpy(np.array(road)).squeeze(2),
                    )
                    data_dict["road"] = data_dict["road"] + self.custmize_layout(
                        self.config["data"]["custom_dict"]["road"],
                        self.config["data"]["key_map"]["road"],
                        torch.from_numpy(np.array(node)).squeeze(2),
                    )
                    if (
                        torch.count_nonzero(data_dict["road"])
                        / (data_dict["road"].shape[1] * data_dict["road"].shape[2])
                        <= self.config["data"]["filter"]["road"]
                    ):
                        return self.__getitem__(
                            np.random.randint(0, len(self.data_list))
                        )
                    data_dict["road"] = self.clamp(data_dict["road"])

            else:
                data_dict = {
                    "building": torch.from_numpy(np.array(building))
                    .float()
                    .squeeze(2)
                    .permute(2, 0, 1),
                    "landuse": torch.from_numpy(np.array(landuse))
                    .float()
                    .squeeze(2)
                    .permute(2, 0, 1),
                    "natural": torch.from_numpy(np.array(natural))
                    .float()
                    .squeeze(2)
                    .permute(2, 0, 1),
                    "road": torch.from_numpy(np.array(road))
                    .float()
                    .squeeze(2)
                    .permute(2, 0, 1),
                }

        else:
            for key in data.keys():
                if key in self.key_list:
                    data_dict[key] = (
                        (torch.from_numpy(np.array(data[key])).float())
                        .squeeze(2)
                        .permute(2, 0, 1)
                    )
                else:
                    continue

        if self.transform:
            data_dict = self.transform(data_dict)
        else:
            for key in data_dict.keys():
                
                data_dict[key] = self.resize(data_dict[key])
                if key == "road":
                    data_dict[key] = self.road_augment(data_dict[key])
                
                data_dict[key] = self.normalize(data_dict[key])


        data_dict["name"] = data_name
        if self.type == "rgb":
            data_dict["layout"] = self.generate_rgb_layout(
                torch.cat(
                    [data_dict[key] for key in data_dict.keys() if key != "name"], dim=0
                )
            )
        elif self.type == "one-hot":
            data_dict["layout"] = torch.cat(
                [data_dict[key] for key in data_dict.keys() if key != "name"], dim=0
            )
        else:
            raise ValueError("type must be rgb or one-hot")


    
        data_dict["layout"][torch.isnan(data_dict["layout"])] = 0
        data_dict["layout"][torch.isinf(data_dict["layout"])] = 0
        
        if self.config["params"]["condition"]:
            assert np.max(self.config["data"]["condition_dim"]) <= data_dict["layout"].shape[0], "condition dim must be smaller than layout dim"
            data_dict["condition"] = torch.concat([data_dict["layout"][i].unsqueeze(0) for i in self.config["data"]["condition_dim"]], dim=0)
            
        
            # delete condition data from layout (if dimmension is 1, keep it)
            data_dict["layout"] = torch.cat([data_dict["layout"][i].unsqueeze(0) for i in range(data_dict["layout"].shape[0]) if i not in self.config["data"]["condition_dim"]], dim=0)
        else:
            data_dict["condition"] = torch.zeros_like(data_dict["layout"])
            
        # print(data_dict["layout"].max(), data_dict["layout"].min())
        # print(data_dict["condition"].max(), data_dict["condition"].min())

        h5py.File.close(data)

        if (data_dict["layout"].max() == 0) and (data_dict["layout"].min() == 0.0):
            # print("data error")
            return self.__getitem__(np.random.randint(0, len(self.data_list)))

        return data_dict
