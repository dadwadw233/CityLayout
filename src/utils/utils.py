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
    def __init__(self, mapping, thresold=0.7):
        self.name = "OSMVisulizer"
        self.channel_to_rgb = mapping
        self.thresold = thresold

    def minmax(self, data):
        data = data - data.min()
        data = data / data.max()
        return data

    def visulize_onehot_layout(self, data, path) -> None:
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
        plt.savefig(path)
        plt.close()

    def hex_or_name_to_rgb(self, color):
        # 使用matplotlib的颜色转换功能
        return mcolors.to_rgb(color)

    # need test
    # input must be a one hot layout
    def visualize_rgb_layout(self, data, path) -> np.ndarray:
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
                mask = data[b, c] > self.thresold 
                combined_image[mask, :] += color

            combined_image = np.clip(combined_image, 0, 1)  # 确保颜色值在0-1范围内
            axs[b].imshow(combined_image)
            axs[b].axis("off")  # 关闭坐标轴

        # 绘制图像
        plt.axis("off")
        plt.imshow(combined_image)
        plt.show()
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
                mask = data[b, c] > self.thresold 
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

        return new_lat, new_lon

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


class Vectorizer:
    def __init__(self, config=None):
        self.name = "Vectorizer"
        assert config is not None, "config must be provided"
        self.config = config
        self.geojson_builder_b = GeoJsonBuilder(
            origin_lat_long=self.config["vec"]["origin"],
            resolution=self.config["vec"]["resolution"],
            crs=self.config["vec"]["crs"],
        )
        self.geojson_builder_c = GeoJsonBuilder(
            origin_lat_long=self.config["vec"]["origin"],
            resolution=self.config["vec"]["resolution"],
            crs=self.config["vec"]["crs"],
        )

        self.channel_to_rgb = self.config["vec"]["channel_to_rgb"]
        self.channel_to_geo = self.config["vec"]["channel_to_geo"]
        self.channel_to_key = self.config["vec"]["channel_to_key"]

        self.thresold = self.config["vec"]["thresold"]
        self.path = self.config["vec"]["path"]

        self.dump_geojson = self.config["vec"]["dump_geojson"]
        self.background = self.config["vec"]["background"]

        self.crs = self.config["vec"]["crs"]

    def __call__(self, data):
        return self.vectorize(data)

    def get_points_set(self, img, data_type):
        h, w = img.shape

        if data_type == "LineString":
            pts = []
            image = img[b_id, :, :, :].transpose(1, 2, 0).astype(np.uint8)
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
            image = img[b_id, :, :, :].transpose(1, 2, 0).astype(np.uint8)
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
        # set data == 1 where data > thresold
        img[img > self.thresold] = 1
        img[img <= self.thresold] = 0

        img = img.cpu().numpy()
        # second step : discretize the image into point set

        if os.path.exists(self.path):
            # create vectorization result folder
            Path(os.join(self.path, "vec")).mkdir(parents=True, exist_ok=True)
            # create geojson folder

        save_path = os.path.join(self.path, "vec")

        for b_id in range(b):
            # create result folder
            Path(os.join(save_path, str(b_id))).mkdir(parents=True, exist_ok=True)
            temp_path = os.path.join(save_path, str(b_id))
            self.geojson_builder_b.init_builder()
            # plt image for every single batch, each channel need to be plotted with different color
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.axis("off")
            
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
                channel_gdf = self.geojson_builder_c.get_geojson()
                channel_feature = self.geojson_builder_c.get_features()
                self.geojson_builder_b.append_features(channel_feature)
                if self.dump_geojson:
                    channel_gdf.to_file(
                        os.path.join(temp_path, f"{self.channel_to_key[c_id]}_geojson.geojson"),
                        driver="GeoJSON",
                    )
                channel_gdf.plot(ax=ax, color=color)
                self.geojson_builder_c.init_builder()
            
            if self.dump_geojson:
                self.geojson_builder_b.get_geojson().to_file(
                    os.path.join(temp_path, "fake.geojson"), driver="GeoJSON"
                )
            
            # save image
            plt.savefig(os.path.join(temp_path, "fake.png"))
            plt.close("all")


        return None
