import numpy as np  
from shapely.geometry import Polygon, Point, LineString
import matplotlib.pyplot as plt
import os
import yaml
from pathlib import Path
import cv2
import geopandas as gpd
import trimesh
from pyproj import Proj, Transformer
from utils.utils import hex_or_name_to_rgb
from utils.log import *
from utils.config import ConfigParser
import torch
from tqdm import tqdm
from utils.log import *


class AssetGen:
    # unified assert generator which support :
    # image to geo file (geojson, shapefile, openstreetmap)
    # image to mesh file (obj, ply, stl)
    # image to image file (png, jpg, tiff)
    def __init__(self, config=None, path="./"):
        self.name = "AssetGen"
        assert config is not None, "config must be provided, which include params for vectorizer and mesh builder"
        self.config = config
        self.config_parser = ConfigParser()
        self.config_parser.set_config(config)
        self.path=path

        self.vectorizer = Vectorizer(config=self.config_parser.get_config_by_name("Vec"), path=self.path)
        self.mesh_builder = MeshBuilder(config=self.config_parser.get_config_by_name("Mesh"))

        self.asset = {}
        self.data = None

    def remove_asset(self, key):
        if key in self.asset:
            del self.asset[key]
        else:
            WARNING(f"key {key} does not exist in asset dict")

    def reset_all(self):
        self.asset = {}
        self.data = None
        INFO("reset all assets and data successfully, but config is still valid")

    def set_data(self, data):
        # check data is image or not
        assert data is not None, "data must be provided"
        if data.shape.__len__() != 4:
            ERROR(f"data shape {data.shape} is not valid, please provide data with shape [b, c, h, w]")
            return False

        self.data = data

        return True
    
    def add_data(self, data) -> bool:
        # check data is image or not
        assert data is not None, "data must be provided"
        if data.shape.__len__() != 4:
            ERROR(f"data shape {data.shape} is not valid, please provide data with shape [b, c, h, w]")
            return False
        if self.data is None:
            self.data = data
        else:
            self.data = torch.concat([self.data, data], dim=0)
        return True


    def generate_geofiles(self) -> bool:
        # check data type
        if self.data is None:
            ERROR("data is None, please set data first")
            return False

        self.asset["geojson"] = self.vectorizer(self.data) # return gdf files

        self.generate_shapefile(self.asset["geojson"]) 
    
        return True
    
    def generate_mesh(self) -> bool:
        # check data type
        if self.data is None:
            ERROR("data is None, please set data first")
            return False
        
        for item in tqdm(self.asset["geojson"], desc="Generate mesh"):
            if item['data'] is not None:
                self.mesh_builder.geojson2mesh(item['data'], os.path.join(item['path'], "fake.obj"))
            else:
                WARNING(f"geojson data is None, skip {item['path']}")
    
        return True
    

    # other asset generator will be added in the future
    def generate_osm(self):
        pass

    def generate_shapefile(self, gdf_list):
        if gdf_list is None:
            ERROR("geojson is None, please generate geojson first")
            return None
        else:
            for item in tqdm(gdf_list, desc="Generate shapefile"):
                if item['data'] is not None:
                    shp_root = os.path.join(item['path'], "shps")
                    # generate different shapefile for different geometry type
                    os.makedirs(shp_root, exist_ok=True)
                    # for each geometry type, we need to check the taret geometry type is exist or not
                    try:
                        # Polygon
                        item['data'][item['data'].geometry.type == 'Polygon'].to_file(os.path.join(shp_root,"fake_Pologon.shp"))
                        # LineString
                        item['data'][item['data'].geometry.type == 'LineString'].to_file(os.path.join(shp_root,"fake_LineString.shp"))
                        # Point
                        item['data'][item['data'].geometry.type == 'Point'].to_file(os.path.join(shp_root,"fake_Point.shp"))
                    except Exception as e:
                        WARNING(f"generate shapefile failed, skip {item['path']}")
                else:
                    WARNING(f"geojson data is None, skip {item['path']}")

    
        

    def get_asset(self):
        return self.asset
    
    def get_data(self):
        return self.data
    

    


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
    def __init__(self, config=None, path="./"):
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
        

        self.dump_geojson = self.config["dump_geojson"]
        self.background = self.config["background"]
        self.path = path
        self.crs = self.config["crs"]
        self.with_height = self.config["with_height"]

    def __call__(self, data):
        return self.vectorize(data)

    def merge_lines(self, lines, threshold_rho=5, threshold_theta=np.pi / 36):
        """
        合并相似的直线。
        :param lines: 检测到的线段数组，形式为 (x1, y1, x2, y2)。
        :param threshold_rho: 合并线段时的 rho 阈值。
        :param threshold_theta: 合并线段时的 theta 阈值。
        :return: 合并后的线段列表。
        """
        if lines is None:
            return []

        # 将直线端点转换为 (rho, theta)
        lines_polar = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            theta = np.arctan2(y2 - y1, x2 - x1)
            rho = x1 * np.cos(theta) + y1 * np.sin(theta)
            lines_polar.append((rho, theta))

        # 合并相似的线段
        merged_lines = []
        for rho, theta in lines_polar:
            # 查找一个已经存在的线段组，这个线段可以合并进去
            found = False
            for group in merged_lines:
                rho_m, theta_m, count = group
                if abs(rho - rho_m) < threshold_rho and abs(theta - theta_m) < threshold_theta:
                    # 更新组的平均值
                    rho_m = (rho_m * count + rho) / (count + 1)
                    theta_m = (theta_m * count + theta) / (count + 1)
                    group = (rho_m, theta_m, count + 1)
                    found = True
                    break
            if not found:
                # 如果没有找到相似的线段，创建一个新的组
                merged_lines.append((rho, theta, 1))

        # 转换回 (x1, y1, x2, y2) 形式的直线
        merged_lines_cartesian = []
        for rho, theta, _ in merged_lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            merged_lines_cartesian.append([x1, y1, x2, y2])

        return merged_lines_cartesian

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

            # 轮廓检测
            contours, _ = cv2.findContours(
                image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # 每个轮廓都是一个线段
            for contour in contours:
                if contour.shape[0] < 2:
                    continue
                else:
                    line = []
                for c_pts in contour:
                    line.append((c_pts[0][0], c_pts[0][1]))
                pts.append(line)
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
        
        elif data_type == "Point":
            pts = []
            image = img[:, :].astype(np.uint8)
            # convert to gray image
            
            # 关键点检测
            # adjust some parameters
            orb = cv2.ORB_create()
            # find the keypoints and descriptors with SIFT
            kp = orb.detect(image, None)
            # draw the keypoints
            DEBUG(f"key points: {kp}")
            pts = [kp[i].pt for i in range(len(kp))]

            


            return pts
        
        else :
            ERROR(f"Unknown data type {data_type}")
            
            return None    

    def vectorize(self, img):
        b, c, h, w = img.shape

        assert (
            c <= self.channel_to_rgb.__len__()
        ), f"channel number {c} is larger than channel to rgb mapping length {self.channel_to_rgb.__len__()}"

        # first step: one hot data augmentation
        # set data == 1 where data > threshold
        org = img.clone()
        org = org.cpu().numpy()
        img[img > self.threshold] = 1
        img[img <= self.threshold] = 0

        img = img.cpu().numpy()
        # second step : discretize the image into point set

        if os.path.exists(self.path):
            # create vectorization result folder
            Path(os.path.join(self.path, "vec")).mkdir(parents=True, exist_ok=True)
            # create geojson folder

        save_path = os.path.join(self.path, "vec")

        asset = []

        for b_id in tqdm(range(b), desc="Vectorizing images", colour="green"):
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
                if pts is None:
                    continue
                for pt in pts:
                    if self.channel_to_geo[c_id] == "LineString":
                        self.geojson_builder_c.add_line(
                            pt, properties={self.channel_to_key[c_id]: "fake", 'height': 0}
                        )
                    elif self.channel_to_geo[c_id] == "Polygon":
                        if self.channel_to_key[c_id] == "building":
                            if self.with_height is not None:

                                height = org[b_id, c_id][pt[0][1]][pt[0][0]] * 255.0
                                if height <= 0:
                                    height = 15
                                #DEBUG(f"building height: {height}")
                                self.geojson_builder_c.add_polygon(
                                    pt, properties={self.channel_to_key[c_id]: "fake", 'height': height}
                                )
                            else:
                                self.geojson_builder_c.add_polygon(
                                    pt, properties={self.channel_to_key[c_id]: "fake", 'height': np.random.randint(10, 70)}
                                )
                        else :
                            self.geojson_builder_c.add_polygon(
                                pt, properties={self.channel_to_key[c_id]: "fake", 'height': 0}
                            )
                    elif self.channel_to_geo[c_id] == "Point":
                        self.geojson_builder_c.add_point(
                            pt, properties={self.channel_to_key[c_id]: "fake", 'height': 0}
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
                asset.append({'data': self.geojson_builder_b.get_geojson(), 'path': temp_path})

            else:
                asset.append({'data': None, 'path': temp_path})
                


            # save image
            plt.savefig(
                os.path.join(temp_path, "fake.png"), bbox_inches="tight", pad_inches=0
            )
            plt.close("all")

        return asset


class MeshBuilder:

    def __init__(self, config=None):
        self.name = "MeshBuilder"
        assert config is not None, "config must be provided"
        self.config = config
        self.origin_lat_long = self.config["origin"]
        self.resolution = self.config["resolution"]
        self.in_proj = Proj('epsg:4326')  # 输入投影，WGS84
        self.out_proj = Proj('epsg:3857')  # 输出投影，通常用于Web Mercator
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