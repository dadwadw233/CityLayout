from typing import Any
import yaml
import torch
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torchvision import transforms
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon
import cv2
from pathlib import Path
import trimesh
from pyproj import Proj, transform, Transformer
import sys


def cycle(dl):
    while True:
        for data in dl:
            yield data


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cast_tuple(t, length=1):
    if isinstance(t, tuple):
        return t
    return (t,) * length


def divisible_by(numer, denom):
    return (numer % denom) == 0


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# normalization functions


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def cal_overlapping_rate(tensor):
    """
    calculate the overlapping area of a tensor
    Args:
        tensor: shape (b, c, h, w)
    Returns:
        overlapping area: shape (b, c)
    """
    b, c, h, w = tensor.shape

    region = torch.zeros_like((tensor[:, 0, :, :]))
    for i in range(c):
        mask = tensor[:, i, :, :] > 0.7
        one_hot_tensor = torch.zeros_like(tensor[:, i, :, :])
        one_hot_tensor[mask] = 1
        region += one_hot_tensor

    region = region > 1

    return (region.sum(dim=[1, 2]).float() / (h * w)).mean()


def hex_or_name_to_rgb(color):
    # 使用matplotlib的颜色转换功能
    return mcolors.to_rgb(color)


# small helper modules
# visulization class
class OSMVisulizer:
    def __init__(self, mapping, threshold=0.7, path="./"):
        self.name = "OSMVisulizer"
        self.channel_to_rgb = mapping
        self.threshold = threshold
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)

    def minmax(self, data):
        data = data - data.min()
        data = data / data.max()
        return data

    def visulize_onehot_layout(self, data, path=None) -> None:
        b, c, h, w = data.shape
        # print(data.shape)
        # exit(0)
        # print(path)
        assert (
            b > 1
        ), f"batch size {b} is too small, please set batch size larger than 1"

        fig, axes = plt.subplots(b, c, figsize=(20, 20))

        for i in range(b):
            for j in range(c):
                if c == 1:
                    axes[i].imshow(data[i, j, :, :].cpu().numpy(), cmap="gray")
                else:
                    axes[i, j].imshow(data[i, j, :, :].cpu().numpy(), cmap="gray")
                    axes[i, j].axis("off")
        if path is None:
            path = os.path.join(self.path, "onehot.png")
        plt.savefig(path)
        plt.close()

    def hex_or_name_to_rgb(self, color):
        # 使用matplotlib的颜色转换功能
        return mcolors.to_rgb(color)

    # need test
    # input must be a one hot layout
    def visualize_rgb_layout(self, data, path=None) -> np.ndarray:
        B, C, H, W = data.shape
        assert (
            self.channel_to_rgb.__len__() >= C
        ), f"channel to rgb mapping length {self.channel_to_rgb.__len__()} does not match channel number {C}"

        assert (
            B > 1
        ), f"batch size {B} is too small, please set batch size larger than 1"
        data = data.cpu().numpy()

        fig, axs = plt.subplots(B, 1, figsize=(20, 20 * B))  # 根据批次数量创建子图

        for b in range(B):
            combined_image = np.zeros((H, W, 3), dtype=np.float32)
            for c in range(C):
                color = np.array(self.hex_or_name_to_rgb(self.channel_to_rgb[c]))
                mask = data[b, c] > self.threshold
                combined_image[mask, :] += color

            combined_image = np.clip(combined_image, 0, 1)  # 确保颜色值在0-1范围内
            axs[b].imshow(combined_image)
            axs[b].axis("off")  # 关闭坐标轴

        # 绘制图像
        plt.axis("off")
        plt.imshow(combined_image)
        plt.show()
        if path is None:
            path = os.path.join(self.path, "rgb.png")
        plt.savefig(path)
        plt.close()

        # return shape : (b, h, w, c)
        return None

    def onehot_to_rgb(self, data):
        B, C, H, W = data.shape
        assert (
            self.channel_to_rgb.__len__() >= C
        ), f"channel to rgb mapping length {self.channel_to_rgb.__len__()} does not match channel number {C}"

        combined_image = torch.zeros(
            (B, H, W, 3), dtype=torch.float32, device=data.device
        )

        for b in range(B):
            for c in range(C):
                color = torch.tensor(
                    self.hex_or_name_to_rgb(self.channel_to_rgb[c]), device=data.device
                )
                mask = data[b, c] > self.threshold
                combined_image[b, mask, :] += color

        combined_image = self.minmax(
            combined_image.permute(0, 3, 1, 2)
        )  # normalize to 0-1

        if torch.isnan(combined_image).any():
            combined_image[torch.isnan(combined_image)] = 0

        return combined_image


class GeoJsonBuilder:
    def __init__(self, origin_lat_long, resolution, crs):
        self.origin_lat_long = origin_lat_long
        self.resolution = resolution  # meters per pixel
        self.features = []
        self.crs = crs

    def add_line(self, line_pixel_coordinates, properties=None):
        line_geo_coordinates = [
            self.pixel_to_geo(x, y) for x, y in line_pixel_coordinates
        ]
        self.features.append((LineString(line_geo_coordinates), properties or {}))

    def add_point(self, point_pixel_coordinates, properties=None):
        point_geo_coordinates = self.pixel_to_geo(*point_pixel_coordinates)
        self.features.append((Point(point_geo_coordinates), properties or {}))

    def add_polygon(self, polygon_pixel_coordinates, properties=None):
        polygon_geo_coordinates = [
            self.pixel_to_geo(x, y) for x, y in polygon_pixel_coordinates
        ]
        self.features.append((Polygon(polygon_geo_coordinates), properties or {}))

    def pixel_to_geo(self, x, y):
        origin_lat, origin_lon = self.origin_lat_long
        # 计算纬度变化（注意 y 轴是向下增加的，所以我们从原点纬度减去）
        delta_lat = y * self.resolution / 111320  # 约等于每度纬度的米数

        # 计算经度变化
        delta_lon = (
            x * self.resolution / (40075000 * np.cos(np.radians(origin_lat)) / 360)
        )

        # 新的地理坐标
        new_lat = origin_lat - delta_lat
        new_lon = origin_lon + delta_lon

        # geojson: (lon, lat)
        return new_lon, new_lat

    def get_geojson(self):
        gdf = gpd.GeoDataFrame(
            [{"geometry": geom, **props} for geom, props in self.features]
        )
        # gdf.to_file(output_file_path, driver='GeoJSON')

        return gdf.set_crs(epsg=self.crs)

    def get_features(self):
        return self.features

    def set_features(self, features):
        self.features = features

    def append_features(self, features):
        self.features += features

    def init_builder(self):
        self.features = []

    def empty(self):
        return self.features.__len__() == 0


class Vectorizer:
    def __init__(self, config=None):
        self.name = "Vectorizer"
        assert config is not None, "config must be provided"
        self.config = config
        self.geojson_builder_b = GeoJsonBuilder(
            origin_lat_long=self.config["origin"],
            resolution=self.config["resolution"],
            crs=self.config["crs"],
        )
        self.geojson_builder_c = GeoJsonBuilder(
            origin_lat_long=self.config["origin"],
            resolution=self.config["resolution"],
            crs=self.config["crs"],
        )
        self.mesh_builder = MeshBuilder(config=self.config)

        self.channel_to_rgb = self.config["channel_to_rgb"]
        self.channel_to_geo = self.config["channel_to_geo"]
        self.channel_to_key = self.config["channel_to_key"]

        self.threshold = self.config["threshold"]
        self.path = self.config["path"]

        self.dump_geojson = self.config["dump_geojson"]
        self.background = self.config["background"]

        self.crs = self.config["crs"]

    def __call__(self, data):
        return self.vectorize(data)

    def get_points_set(self, img, data_type):
        h, w = img.shape

        if data_type == "LineString":
            pts = []
            image = img[:, :].astype(np.uint8)
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 边缘检测
            edges = cv2.Canny(image, 1, 1, apertureSize=5)
            # dilated = cv2.dilate(edges, None, iterations=1)
            # eroded = cv2.erode(dilated, None, iterations=1)

            # cv2.imwrite('./canny.png', eroded)
            lines = cv2.HoughLinesP(
                image, 1, (np.pi / 180), 10, minLineLength=0.1, maxLineGap=10
            )
            show = np.zeros_like(image)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    start = (x1, y1)
                    end = (x2, y2)
                    # cv2.line(show, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    pts.append([start, end])

            return pts

        elif data_type == "Polygon":
            pts = []
            image = img[:, :].astype(np.uint8)
            # edges = cv2.Canny(image, 1, 1)
            # if b_id == 0:
            # cv2.imwrite('./canny.png', edges)

            # 查找轮廓
            contours, _ = cv2.findContours(
                image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours is not None:
                for contour in contours:
                    # 多边形拟合
                    perimeter = cv2.arcLength(contour, False)

                    # 设置逼近精度（例如，周长的1%）
                    epsilon = 0.01 * perimeter

                    # 轮廓逼近
                    approx = cv2.approxPolyDP(contour, epsilon, False)

                    approx = approx.reshape((-1, 2))
                    if len(approx) < 3:
                        continue
                    if approx[-1][0] != approx[0][0] or approx[-1][1] != approx[0][1]:
                        approx = np.concatenate((approx, approx[0:1, :]), axis=0)
                    pts.append(approx.tolist())

            return pts

    def vectorize(self, img):
        b, c, h, w = img.shape

        assert (
            c <= self.channel_to_rgb.__len__()
        ), f"channel number {c} is larger than channel to rgb mapping length {self.channel_to_rgb.__len__()}"

        # first step: one hot data augmentation
        # set data == 1 where data > threshold
        img[img > self.threshold] = 1
        img[img <= self.threshold] = 0

        img = img.cpu().numpy()
        # second step : discretize the image into point set

        if os.path.exists(self.path):
            # create vectorization result folder
            Path(os.path.join(self.path, "vec")).mkdir(parents=True, exist_ok=True)
            # create geojson folder

        save_path = os.path.join(self.path, "vec")

        for b_id in range(b):
            # create result folder
            Path(os.path.join(save_path, str(b_id))).mkdir(parents=True, exist_ok=True)
            temp_path = os.path.join(save_path, str(b_id))
            self.geojson_builder_b.init_builder()
            # plt image for every single batch, each channel need to be plotted with different color
            fig, ax = plt.subplots(figsize=(10, 10), facecolor=self.background)
            ax.axis("off")
            ax.set_aspect('equal')


            for c_id in range(c):
                color = hex_or_name_to_rgb(self.channel_to_rgb[c_id])

                pts = self.get_points_set(
                    img[b_id, c_id, :, :], self.channel_to_geo[c_id]
                )
                for pt in pts:
                    if self.channel_to_geo[c_id] == "LineString":
                        self.geojson_builder_c.add_line(
                            pt, properties={self.channel_to_key[c_id]: "fake"}
                        )
                    elif self.channel_to_geo[c_id] == "Polygon":
                        self.geojson_builder_c.add_polygon(
                            pt, properties={self.channel_to_key[c_id]: "fake"}
                        )

                if not self.geojson_builder_c.empty():
                    channel_gdf = self.geojson_builder_c.get_geojson()
                    channel_feature = self.geojson_builder_c.get_features()
                    self.geojson_builder_b.append_features(channel_feature)
                    if self.dump_geojson:
                        channel_gdf.to_file(
                            os.path.join(
                                temp_path,
                                f"{self.channel_to_key[c_id]}_geojson.geojson",
                            ),
                            driver="GeoJSON",
                        )
                    channel_gdf.plot(ax=ax, color=color)
                self.geojson_builder_c.init_builder()

            if self.dump_geojson and not self.geojson_builder_b.empty():
                self.geojson_builder_b.get_geojson().to_file(
                    os.path.join(temp_path, "fake.geojson"), driver="GeoJSON"
                )
                self.mesh_builder.geojson2mesh(
                    self.geojson_builder_b.get_geojson(),
                    os.path.join(temp_path, "fake.obj"),
                )

            # save image
            plt.savefig(
                os.path.join(temp_path, "fake.png"), bbox_inches="tight", pad_inches=0
            )
            plt.close("all")

        return None


class MeshBuilder:

    def __init__(self, config=None):
        self.name = "MeshBuilder"
        assert config is not None, "config must be provided"
        self.config = config
        self.origin_lat_long = self.config["origin"]
        self.resolution = self.config["resolution"]
        self.in_proj = Proj(init='epsg:4326')  # 输入投影，WGS84
        self.out_proj = Proj(init='epsg:3857')  # 输出投影，通常用于Web Mercator
        self.transformer = Transformer.from_proj(self.in_proj, self.out_proj, always_xy=True)

    def convert_coords(self, coords):
        """将地理坐标转换为笛卡尔坐标"""
        origin_lat, origin_lon = self.origin_lat_long
        reference_x, reference_y = self.transformer.transform(origin_lon, origin_lat)
        return [(self.transformer.transform(lon, lat) - np.array([reference_x, reference_y])) / self.resolution for lon, lat in coords]
        


    def create_3d_building(self, polygon, height):
        """为建筑物创建3D网格"""
        try: 
            if isinstance(polygon, Polygon):
                vertices = np.array(self.convert_coords(polygon.exterior.coords))
                vertices = np.c_[vertices, np.zeros(len(vertices))]

                top_vertices = vertices + np.array([0, 0, height])
                vertices = np.vstack([vertices, top_vertices])

                faces = [[i, i+1, len(vertices)//2 + i + 1, len(vertices)//2 + i] for i in range(len(vertices)//2 - 1)]
                return trimesh.Trimesh(vertices=vertices, faces=faces)
    
        except Exception as e:
            return None
    
    def create_3d_road(self, line, width, height):
        """为道路创建3D网格"""
        try:
            if isinstance(line, LineString):
                offset = width / 2.0
                polygon = line.buffer(offset, cap_style=2, join_style=2)
                
                vertices = np.array(self.convert_coords(polygon.exterior.coords))
                vertices = np.c_[vertices, np.zeros(len(vertices))]

                top_vertices = vertices + np.array([0, 0, height])
                vertices = np.vstack([vertices, top_vertices])

                faces = [[i, i+1, len(vertices)//2 + i + 1, len(vertices)//2 + i] for i in range(len(vertices)//2 - 1)]
                return trimesh.Trimesh(vertices=vertices, faces=faces)
        except Exception as e:
            return None

    def geojson2mesh(self, gdf, path):
        meshes = []

        for feature in gdf.geometry:
            # 确保几何类型是多边形
            if isinstance(feature, Polygon):
                # 将多边形的坐标转换为 [x, y, z] 格式
                height = np.random.randint(10, 30)
                mesh = self.create_3d_building(feature, height)
                if mesh is not None:
                    meshes.append(mesh)
            # elif isinstance(feature, LineString):
            #     # 将线的坐标转换为 [x, y, z] 格式
            #     width = np.random.randint(2, 5)
            #     height = np.random.randint(1, 2)
            #     mesh = self.create_3d_road(feature, width, height)
            #     if mesh is not None:
            #         meshes.append(mesh)
                

        
        combined_mesh = trimesh.util.concatenate(meshes)
        combined_mesh.export(path)
        

class DataAnalyser:
    GREAT_COLOR_SET = plt.cm.get_cmap("tab20").colors

    def __init__(self, config=None):
        self._parse_config(config)
        self.init_folder()
        self.data_dict = self._init_data_dict()
        self.real_size = 0
        self.fake_size = 0

    def _parse_config(self, config):
        assert config is not None, "config must be provided"
        self.path = config["analyse"]["path"]
        self.layout_keys = config["data"]["custom_dict"]
        self.analyse_types = config["analyse"]["types"]
        self.threshold = config["analyse"]["threshold"]
        self.limit = config["analyse"]["evaluate_data_limit"]
        self.mapping = []

    def init_folder(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
        else:
            for file in os.listdir(self.path):
                if file.endswith(".png") or file.endswith(".txt"):
                    os.remove(os.path.join(self.path, file))

    def _init_data_dict(self, fake=False):
        data_dict = {"real": {}, "fake": {}}
        
        for idx in self.layout_keys:
            for subgroup in self.layout_keys[idx]:
                subgroup_name = idx
                for key in subgroup:
                    subgroup_name += f"_{key}"
                    break

                self.mapping.append(subgroup_name)
                
                data_dict["real"][subgroup_name] = self._init_subgroup_data()
                data_dict["fake"][subgroup_name + "_fake"] = self._init_subgroup_data()
        
        for key in self.mapping.copy():
            self.mapping.append(key + "_fake")

        return data_dict

    def _init_subgroup_data(self):
        return {analyse_type: [] for analyse_type in self.analyse_types}

    @staticmethod
    def cal_overlap(data) -> np.float32:
        h, w = data.shape
        area = h * w
        overlap_rate = np.count_nonzero(data) / area
        return overlap_rate

    def _setup_plot(self, title=None, xlabel=None, ylabel=None, figsize=(10, 8)):
        """Prepare the plot with common settings."""
        plt.figure(figsize=figsize)
        plt.grid(True)
        if title:
            plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)

    def _save_plot(self, filename):
        """Save the plot to the specified path."""
        file_path = os.path.join(self.path, filename)
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    def _calculate_statistics(self, data):
        """Calculate statistics for the given data."""
        return {analyse_type: {"std": np.std(data[analyse_type]), "mean": np.mean(data[analyse_type])}
                for analyse_type in self.analyse_types}
    
    def _calculate_correlation(self, data1, data2):
        """Calculate correlation coefficient between two datasets."""
        try:
            corr_coef = np.corrcoef(data1, data2)[0, 1]
            return corr_coef if np.isfinite(corr_coef) else 0
        except Exception as e:
            print(f"Error calculating correlation: {e}")
            return 0

    def _flatten_data(self, data_dict, analyse_type):
        """Flatten the data for a specific analysis type."""
        return [data for values in data_dict.values() for data in values[analyse_type]]

    def _calculate_correlation_matrix(self, data_dict, analyse_type):
        """Calculate the correlation matrix for a specific analysis type."""
        sub_data = {key: values[analyse_type] for key, values in data_dict.items()}
        corr_matrix = np.zeros((len(sub_data), len(sub_data)))
        for i, (key1, values1) in enumerate(sub_data.items()):
            for j, (key2, values2) in enumerate(sub_data.items()):
                corr_matrix[i, j] = self._calculate_correlation(values1, values2)
        return corr_matrix

    def _cluster_data(self, data_dict):
        """Cluster the data based on correlation."""
        clusters = {}
        corr_matrix = {}
        for analyse_type in self.analyse_types:
            flattened_data = self._flatten_data(data_dict, analyse_type)
            corr_matrix[analyse_type] = self._calculate_correlation_matrix(data_dict, analyse_type)
            # ... Cluster the data based on the correlation matrix ...
        return clusters, corr_matrix

    def output_results_to_file(self, results, filename):
        """Write results to a text file."""
        with open(os.path.join(self.path, filename), "w") as file:
            for analyse_type in self.analyse_types:
                file.write(f"{analyse_type.upper()}\n")
                for key, values in results.items():
                    file.write(f"{key}: {values[analyse_type]}\n")
                file.write("\n")



    def plot_std_mean_all(self, data_dict, title="Mean and Standard Deviation for Each Category"):
        """Plot mean and standard deviation for all data categories."""
        for analyse_type in self.analyse_types:
            self._setup_plot(title=title, xlabel='Mean', ylabel='Standard Deviation')
            all_means = []
            all_stds = []
            for key, values in data_dict.items():
                mean = np.mean(values[analyse_type])
                std = np.std(values[analyse_type])
                if std == 0 and mean == 0:
                    continue
                all_means.append(mean)
                all_stds.append(std)
                plt.scatter(mean, std, label=key, color=self.GREAT_COLOR_SET[self.mapping.index(key)])

            plt.xlim(left=min(all_means)*0.8, right=max(all_means)*1.2)
            plt.ylim(bottom=min(all_stds)*0.8, top=max(all_stds)*1.2)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.tight_layout()
            self._save_plot(f"{analyse_type}_std_mean.png")
    
    def plot_hist_all(self, data_dict, title="Histogram for Each Category", bins=20, alpha=0.5):
        """Plot histograms for all data categories."""
        for analyse_type in self.analyse_types:
            self._setup_plot(title=title, xlabel='Value', ylabel='Frequency')
            for key, values in data_dict.items():
                all_values = [v for v in values[analyse_type] if v != 0]
                plt.hist(all_values, bins=bins, label=key, alpha=alpha, color=self.GREAT_COLOR_SET[self.mapping.index(key)])    

            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.tight_layout()
            self._save_plot(f"{analyse_type}_hist.png")


    def plot_curves_all(self, data_dict, title="Curves for Each Category"):
        """Plot curves for all data categories."""
        for analyse_type in self.analyse_types:
            self._setup_plot(title=title, xlabel='Epoch', ylabel='Value')
            for key, values in data_dict.items():
                plt.plot(values[analyse_type], label=key, color=self.GREAT_COLOR_SET[self.mapping.index(key)])

            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.tight_layout()
            self._save_plot(f"{analyse_type}_curves.png")
    

    def plot_correlation_matrix(self, corr_matrix, mapping, title="Correlation Matrix", cmap='coolwarm'):
        """Plot the correlation matrix."""
        for analyse_type in self.analyse_types:
            self._setup_plot(title=f"{title} - {analyse_type}", figsize=(10, 8))
            sub_corr_matrix = corr_matrix[analyse_type]
            plt.imshow(sub_corr_matrix, cmap=cmap, vmin=-1, vmax=1)
            plt.colorbar()
            plt.xticks(range(len(mapping)), mapping, rotation=90)
            plt.yticks(range(len(mapping)), mapping)
            plt.tight_layout()
            self._save_plot(f"{analyse_type}_corr_matrix.png")

    def plot_fake_and_real_std_mean(self, data_dict, fake_data_dict, title="Mean and Standard Deviation for Each Category"):
        '''plot the contrast between fake and real data'''
        for analyse_type in self.analyse_types:
            self._setup_plot(title=title, xlabel='Mean', ylabel='Standard Deviation')
            all_means = []
            all_stds = []
            for key, values in data_dict.items():
                mean = np.mean(values[analyse_type])
                std = np.std(values[analyse_type])
                if std == 0 and mean == 0:
                    continue
                all_means.append(mean)
                all_stds.append(std)
                plt.scatter(mean, std, label=key, color=self.GREAT_COLOR_SET[self.mapping.index(key)])

            for key, values in fake_data_dict.items():
                mean = np.mean(values[analyse_type])
                std = np.std(values[analyse_type])
                if std == 0 and mean == 0:
                    continue
                all_means.append(mean)
                all_stds.append(std)
                plt.scatter(mean, std, label=key, color=self.GREAT_COLOR_SET[self.mapping.index(key)//2], marker='x')

            plt.xlim(left=min(all_means)*0.8, right=max(all_means)*1.2)
            plt.ylim(bottom=min(all_stds)*0.8, top=max(all_stds)*1.2)
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.tight_layout()
            self._save_plot(f"{analyse_type}_std_mean_fake_and_real.png")
    
    def plot_fake_and_real_hist(self, data_dict, fake_data_dict, title="Histogram for Each Category"):
        """Plot histograms for all data categories."""
        

        for analyse_type in self.analyse_types:
            self._setup_plot(title=title, xlabel='Value', ylabel='Frequency')
            for key, values in data_dict.items():
                all_values = [v for v in values[analyse_type] if v != 0]
                plt.hist(all_values, bins=20, label=key, alpha=0.5, color=self.GREAT_COLOR_SET[self.mapping.index(key)])    

            for key, values in fake_data_dict.items():
                all_values = [v for v in values[analyse_type] if v != 0]
                plt.hist(all_values, bins=20, label=key, alpha=0.5, color=self.GREAT_COLOR_SET[self.mapping.index(key)//2], histtype='step', linestyle='dashed')    

            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.tight_layout()
            self._save_plot(f"{analyse_type}_hist_fake_and_real.png")

    def plot_fake_and_real_correlation_matrix(self, corr_matrix, mapping, title="Correlation Matrix"):
        """Plot the correlation matrix."""
        for analyse_type in self.analyse_types:
            sub_corr_matrix = corr_matrix[analyse_type]
            self._setup_plot(title=title, figsize=(10, 8))
            plt.imshow(sub_corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar()
            plt.xticks(range(len(mapping)), mapping, rotation=90)
            plt.yticks(range(len(mapping)), mapping)
            plt.tight_layout()
            self._save_plot(f"{analyse_type}_corr_matrix_fake_and_real.png")


    def analyse(self, fake: bool = False):
        """Main method to run the analysis."""
        assert self.real_size != 0 if not fake else self.fake_size != 0, "please add data first"
        # Perform calculations
        data_for_analysis = self.data_dict["fake"] if fake else self.data_dict["real"]
        statistics = {key: self._calculate_statistics(values) for key, values in data_for_analysis.items()}
        clusters, corr_matrix = self._cluster_data(data_for_analysis)
        
        # Plot results
        self.plot_std_mean_all(data_for_analysis)
        self.plot_hist_all(data_for_analysis)
        self.plot_curves_all(data_for_analysis)

        self.plot_correlation_matrix(corr_matrix, self.mapping[:self.mapping.__len__()//2])

        # Output results
        self.output_results_to_file(statistics, "statistics.txt")
        self.output_results_to_file(clusters, "clusters.txt")



    def contrast_analyse(self):
        """analyse the contrast between fake and real data"""
        assert self.real_size != 0 and self.fake_size != 0, "please add data first"


        evaluation_data_size = self.real_size if self.real_size < self.fake_size else self.fake_size
        print(f"evaluation data size: {evaluation_data_size}", f"\nbecause real data size: {self.real_size} and fake data size: {self.fake_size}")

        real_for_analysis = {}
        fake_for_analysis = {}
        for key, values in self.data_dict["real"].items():
            real_for_analysis[key] = {analyse_type: values[analyse_type][:evaluation_data_size] for analyse_type in self.analyse_types}
        for key, values in self.data_dict["fake"].items():
            fake_for_analysis[key] = {analyse_type: values[analyse_type][:evaluation_data_size] for analyse_type in self.analyse_types}


        statistics = {key: self._calculate_statistics(values) for key, values in real_for_analysis.items()}
        fake_statistics = {key: self._calculate_statistics(values) for key, values in fake_for_analysis.items()}

        uni_dict = {}
        for key, values in real_for_analysis.items():
            uni_dict[key] = values
        for key, values in fake_for_analysis.items():
            uni_dict[key] = values
        clusters, corr_matrix = self._cluster_data(uni_dict)
        

        self.plot_fake_and_real_hist(real_for_analysis, fake_for_analysis, title="Histogram for Each Category")  
        self.plot_fake_and_real_std_mean(real_for_analysis, fake_for_analysis, title="Mean and Standard Deviation for Each Category")  
        
        self.plot_fake_and_real_correlation_matrix(corr_matrix, self.mapping, title="Correlation Matrix")

        

        # Output results
        self.output_results_to_file(statistics, "real_statistics.txt")
        self.output_results_to_file(fake_statistics, "fake_statistics.txt")
        self.output_results_to_file(clusters, "uni_clusters.txt")

        
    # some helper functions 👇
    # some emoji for fun
    # 👨 👈 🤡
    # 👇 🚮
    # copilot 👉 🐂🍺


    def get_data_size(self) -> tuple:

        return self.real_size, self.fake_size
    

    def release_data (self, fake: bool = False):
        if fake:
            self.data_dict["fake"] = self._init_data_dict(fake=True)
        else:
            self.data_dict["real"] = self._init_data_dict(fake=False)
            

    
    def add_data(self, data: torch.tensor, fake: bool = False):
        if data.device != "cpu":
            data = data.cpu()
        
        b, c, h, w = data.shape

        if fake:
            self.fake_size += b
        else:
            self.real_size += b

        now_data_size = self.fake_size if fake else self.real_size
        if now_data_size > self.limit:
            return


        for idx in range(b):
            for c_id in range(c):
                if "overlap" in self.analyse_types:
                    if fake:
                        self.data_dict["fake"][self.mapping[c_id] + "_fake"]["overlap"].append(self.cal_overlap(data[idx, c_id, :, :]))
                    else:
                        self.data_dict["real"][self.mapping[c_id]]["overlap"].append(self.cal_overlap(data[idx, c_id, :, :]))


