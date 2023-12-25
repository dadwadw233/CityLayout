import argparse
import osmnx as ox
import os
import tqdm
import geopandas as gpd
import matplotlib.pyplot as plt
import concurrent.futures
import logging
import pandas as pd
import rasterio
from shapely.geometry import Point, box
from PIL import Image
import numpy as np
from rasterio.features import rasterize
from io import BytesIO
import h5py
from collections import OrderedDict
import sys
import shutil
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.utils.utils import load_config
def escape_path(path):
    # 定义需要转义的字符及其转义后的形式
    escape_chars = {
        " ": "\\ ",  # 空格
        "(": "\\(",  # 左括号
        ")": "\\)",  # 右括号
        "&": "\\&",  # 和号
        "'": "\\'",  # 单引号
        '"': '\\"',  # 双引号
        "!": "\\!",  # 感叹号
        "@": "\\@",  # At
        "#": "\\#",  # 井号
        "$": "\\$",  # 美元符
        "%": "\\%",  # 百分号
        "^": "\\^",  # 脱字符
        "*": "\\*",  # 星号
        "=": "\\=",  # 等号
        "+": "\\+",  # 加号
        "|": "\\|",  # 竖线
        "{": "\\{",  # 左花括号
        "}": "\\}",  # 右花括号
        "[": "\\[",  # 左中括号
        "]": "\\]",  # 右中括号
        "\\": "\\\\",  # 反斜杠
        ":": "\\:",  # 冒号
        ";": "\\;",  # 分号
        "<": "\\<",  # 小于号
        ">": "\\>",  # 大于号
        "?": "\\?",  # 问号
        ",": "\\,",  # 逗号
        ".": "\\.",  # 英文句号
        "`": "\\`",  # 重音符
        "~": "\\~",  # 波浪号
    }

    # 对每个需要转义的字符进行替换
    for char, escaped_char in escape_chars.items():
        path = path.replace(char, '-')

    return path

def geo_data_validation(path, init=False):

    # path = escape_path(path)
    try: 
        if os.path.exists(os.path.join(path, "getdata_error.txt")):
            logging.error(f"Error occurred in {path}.")
            # remove data directory
            shutil.rmtree(path)
        if init:
            if os.path.exists(os.path.join(path)):
                for file in os.listdir(path):
                    if not file.endswith(".geojson"):
                        os.remove(os.path.join(path, file))
    except Exception as e:
        logging.error(f"Error occurred in {path}.")




def image_data_validation(path):
    try:
        if not os.path.exists(os.path.join(path, "plotting_img_finish.txt")):
            logging.error(f"Error occurred in {path}.")
            # remove data directory
            shutil.rmtree(path)
            # os.system(f"rm -rf {path}")
    except Exception as e:
        logging.error(f"Error occurred in {path}.")


def plot_dem(dem_file, output_filename, fig_size=(10, 10), xlim=None, ylim=None):
    with rasterio.open(dem_file) as src:
        # 读取第一个波段的数据
        dem = src.read(1)

    # 使用 Matplotlib 生成图片
    # print(dem)
    # fig, ax = plt.subplots(figsize=fig_size)
    # if xlim and ylim:
    #     ax.set_xlim(xlim)
    #     ax.set_ylim(ylim)

    plt.figure(figsize=fig_size)
    # if xlim and ylim:
    #     plt.xlim(xlim)
    #     plt.ylim(ylim)
    plt.imshow(dem, cmap="gray")  # 或者使用其他colormap，如 'terrain'
    plt.axis("off")
    plt.savefig(output_filename)
    plt.close()

def data_handle(gdf, data_type, config, out_path, conbained_ax, xlim, ylim):
    feature_img_dict = {}
    if data_type=='landuse':
        gdf["area"] = gdf.geometry.area
        gdf = gdf.sort_values(by="area", ascending=True)
    if data_type=='natural':
        nature_cols = ["natural"]
        for col in nature_cols:
            if col in gdf.columns:
                gdf[col] = gdf[col].fillna("")
            else:
                gdf[col] = ""

    color_dict = config["data_config"][data_type]["color_dict"]

    feature_list = config["data_config"][data_type]["feature_list"]

    fig_, ax_ = plt.subplots(figsize=config["plt_config"]["default"]["figsize"])
    ax_.set_xlim(xlim)
    ax_.set_ylim(ylim)
    ax_.axis("off")
    image_array = None

    if data_type == 'road' or data_type == 'node':
        gdf_key = 'highway'
    else:
        gdf_key = data_type
    for _type in gdf[gdf_key].unique():

        gdf_type = gdf[gdf[gdf_key] == _type]

        if _type in color_dict.keys():
            gdf_type.plot(
                ax=ax_,
                color=color_dict.get(_type, "#FFFFFF"),
                alpha=config["plt_config"][data_type]["alpha"],
            )
            gdf_type.plot(
                ax=conbained_ax,
                color=color_dict.get(_type, "#FFFFFF"),
                alpha=config["plt_config"][data_type]["alpha"],
            )

        if _type in feature_list:

            one_hot_fig, one_hot_ax = plt.subplots(figsize=config["plt_config"]["default"]["figsize"])
            one_hot_ax.set_xlim(xlim)
            one_hot_ax.set_ylim(ylim)
            one_hot_ax.axis("off")
            gdf_type.plot(ax=one_hot_ax, cmap="gray")
            # save image to buffer
            buf = BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            buf.seek(0)

            image = Image.open(buf)
            image_array = np.array(image)
            data = ~image_array[:, :, 0:1]
            data[data != 0] = 1
            feature_img_dict[_type] = data

            plt.close()
            buf.close()
    
    for feature in feature_list:
        if feature not in feature_img_dict.keys():
            feature_img_dict[feature] = np.zeros(
                (image_array.shape[0], image_array.shape[1], 1)
            )
        
    plt.savefig(
        os.path.join(out_path, f'{data_type}.jpg'),
        pad_inches=config["plt_config"][data_type]["pad_inches"],
        bbox_inches=config["plt_config"][data_type]["bbox_inches"],
        format=config["plt_config"][data_type]["format"],
    )
    plt.close(fig_)


    # fig, axs = plt.subplots(1, len(feature_img_dict), figsize=(20, 8))
    # for index,label in enumerate(feature_list):

    #     ax_inner = axs[index]
    #     ax_inner.axis('off')
    #     ax_inner.imshow(feature_img_dict[label], cmap='gray')
    #     ax_inner.set_title(f'Label {label}')

    # plt.savefig(os.path.join(os.path.dirname(output_filename), 'nature_multi_channel.jpg'))
    # plt.close(fig)

    feature_img_dict = {k: feature_img_dict[k] for k in feature_list}
    data_matrix = np.stack(list(feature_img_dict.values()), axis=-1)
    np.save(
        os.path.join(out_path, f'{data_type}.npy'),
        data_matrix,
    )

    #plt.close()

    return True



def plot_combined_map(
    roads_gdf,
    landuse_gdf,
    buildings_gdf,
    nature_gdf,
    output_filename,
    config,
    fig_size=(10, 10),
):
    fig, ax = plt.subplots(figsize=config["plt_config"]["default"]["figsize"])

    combined_gdf = gpd.GeoDataFrame(
        pd.concat(
            [roads_gdf, landuse_gdf, buildings_gdf, nature_gdf], ignore_index=True
        ),
        crs=roads_gdf.crs,
    )
    xlim = (
        combined_gdf.total_bounds[0],
        combined_gdf.total_bounds[0] + (config["plt_config"]["default"]["radius"] * 2),
    )  # Minx, Maxx
    ylim = (
        combined_gdf.total_bounds[1],
        combined_gdf.total_bounds[1] + (config["plt_config"]["default"]["radius"] * 2),
    )  # Miny, Maxy

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis("off")
    plt.axis("off")

    if not landuse_gdf.empty:
        data_handle(landuse_gdf, 'landuse', config, os.path.dirname(output_filename), ax, xlim, ylim)

    if not nature_gdf.empty:
        data_handle(nature_gdf, 'natural', config, os.path.dirname(output_filename), ax, xlim, ylim)


    if not roads_gdf.empty:
        data_handle(roads_gdf, 'road', config, os.path.dirname(output_filename), ax, xlim, ylim)    


        point_gdf = create_point_gdf_from_linestrings(roads_gdf)

        data_handle(point_gdf, 'node', config, os.path.dirname(output_filename), ax, xlim, ylim)


    if not buildings_gdf.empty:
        feature_img_dict = {}
        buildings_gdf.plot(
            ax=ax,
            color=config["plt_config"]["building"]["color"],
            edgecolor=config["plt_config"]["building"]["edgecolor"],
            alpha=config["plt_config"]["building"]["alpha"],
        )
        fig_, ax_ = plt.subplots(figsize=config["plt_config"]["default"]["figsize"])
        ax_.set_xlim(xlim)
        ax_.set_ylim(ylim)
        ax_.axis("off")

        buildings_gdf.plot(
            ax=ax_,
            color=config["plt_config"]["building"]["color"],
            edgecolor=config["plt_config"]["building"]["edgecolor"],
        )
        plt.savefig(
            os.path.join(os.path.dirname(output_filename), "building_location.jpg"),
            bbox_inches=config["plt_config"]["building"]["bbox_inches"],
            format=config["plt_config"]["building"]["format"],
            pad_inches=config["plt_config"]["building"]["pad_inches"],
        )

        plt.close(fig_)

        #  绘制高度数据
        # 判断'height'字段是否存在
        if "height" in buildings_gdf.columns:
            fig_, ax_ = plt.subplots(figsize=fig_size)
            ax_.set_xlim(xlim)
            ax_.set_ylim(ylim)
            ax_.axis("off")

            buildings_gdf.plot(
                ax=ax_, column="height", cmap=config["plt_config"]["building"]["cmap"]
            )
            plt.savefig(
                os.path.join(os.path.dirname(output_filename), "building_height.jpg"),
                bbox_inches=config["plt_config"]["building"]["bbox_inches"],
                format=config["plt_config"]["building"]["format"],
                pad_inches=config["plt_config"]["building"]["pad_inches"],
            )
            plt.close(fig_)

        buildings_gdf.plot(cmap="gray")

        # set axis off
        plt.axis("off")
        # set xlim and ylim
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.tight_layout()
        # save image to buffer
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)

        image = Image.open(buf)
        image_array = np.array(image)
        data = ~image_array[:, :, 0:1]
        data[data != 0] = 1
        feature_img_dict['building'] = data

        plt.close()
        buf.close()

        # fig, axs = plt.subplots(1, 1, figsize=(20, 8))
        # ax_inner = axs
        # ax_inner.axis('off')
        # ax_inner.imshow(feature_img_dict[road_type], cmap='gray')
        # ax_inner.set_title(f'Label building')

        # plt.savefig(os.path.join(os.path.dirname(output_filename), 'building_multi_channel.jpg'))
        # plt.close(fig)

        building_matrix = np.stack(list(feature_img_dict.values()), axis=-1)
        np.save(
            os.path.join(os.path.dirname(output_filename), "building.npy"),
            building_matrix,
        ) 
    plt.savefig(
        output_filename,
        bbox_inches=config["plt_config"]["combained"]["bbox_inches"],
        format=config["plt_config"]["combained"]["format"],
        pad_inches=config["plt_config"]["combained"]["pad_inches"],
    )
    
    plt.close("all")


    return xlim, ylim

def create_point_gdf_from_linestrings(line_gdf):

    points = []
    highway_values = []

    for _, row in line_gdf.iterrows():
        highway_type = row['highway']  
        line = row.geometry
        if line.geom_type == 'LineString':
            for xy in line.coords:
                points.append(Point(xy))
                highway_values.append(highway_type)
        elif line.geom_type == 'MultiLineString':
            for linestring in line.geoms:
                for xy in linestring.coords:
                    points.append(Point(xy))
                    highway_values.append(highway_type)


    point_gdf = gpd.GeoDataFrame({'highway': highway_values, 'geometry': points})


    return get_repeated_points(point_gdf)

def get_repeated_points(point_gdf):
    point_gdf['coords'] = point_gdf['geometry'].apply(lambda geom: str(geom.coords[:]))

    duplicated = point_gdf.duplicated(subset='coords', keep=False)
    repeated_points_gdf = point_gdf[duplicated]

    repeated_points_gdf = repeated_points_gdf.drop(columns=['coords'])

    return repeated_points_gdf

def plot_pop(pop_file, output_filename, fig_size=(10, 10), xlim=None, ylim=None):
    with rasterio.open(pop_file) as src:
        pop = src.read(1)

    plt.figure(figsize=fig_size)
    # if xlim and ylim:
    #     plt.xlim(xlim)
    #     plt.ylim(ylim)
    plt.imshow(pop, cmap="terrain")
    plt.axis("off")
    plt.savefig(output_filename)
    plt.close()


def process_city(city, input_root, output_root, config):
    try:
        # print(f"Processing {city}...")

        save_path = os.path.join(input_root, city)
        output_path = os.path.join(output_root, city)
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        # if(os.path.exists(os.path.join(save_path, 'plotting_img_finish.txt'))):
        #     print(f"Already processed {city}.")
        #     return

        radius = config["plt_config"]["default"]["radius"]

        try:
            roads_gdf = gpd.read_file(
                os.path.join(save_path, f"road_data_{str(radius)}.geojson")
            ).to_crs(epsg=config["geo"]["crs"])
            landuse_gdf = gpd.read_file(
                os.path.join(save_path, f"landuse_data_{str(radius)}.geojson")
            ).to_crs(epsg=config["geo"]["crs"])
            buildings_gdf = gpd.read_file(
                os.path.join(save_path, f"buildings_data_{str(radius)}.geojson")
            ).to_crs(epsg=config["geo"]["crs"])
            nature_gdf = gpd.read_file(
                os.path.join(save_path, f"nature_data_{str(radius)}.geojson")
            ).to_crs(epsg=config["geo"]["crs"])

        except Exception as e:
            with open(os.path.join(output_root, "plotting_img_error.txt"), "a") as file:
                file.write(f"Error occurred in {city}: {str(e)}\n")
            return

        xlim, ylim = plot_combined_map(
            roads_gdf,
            landuse_gdf,
            buildings_gdf,
            nature_gdf,
            os.path.join(output_root, city, "combined.jpg"),
            config=config,
        )

        with open(os.path.join(save_path, "plotting_img_finish.txt"), "a") as file:
            file.write(f"Success in {city}\n")

    except Exception as e:
        with open(os.path.join(save_path, "plotting_img_error.txt"), "a") as file:
            file.write(f"Error occurred in {city}: {str(e)}\n")

        return


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    lv = len(hex_color)
    return tuple(int(hex_color[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def dump_h5py(path, output_path):
    try:
        with h5py.File(output_path, "w") as f:
            for file in os.listdir(path):
                if file.endswith(".npy"):
                    data = np.load(os.path.join(path, file))
                    f.create_dataset(file[:-4], data=data)
    except Exception as e:
        print(f"Error occurred in {path}: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Input file",
        default="config/data/preprocess.yaml",
    )

    yaml_config = load_config(parser.parse_args().config)
    root_path = yaml_config["path"]["input"]
    output = yaml_config["path"]["output"]
    if not os.path.exists(root_path):
        raise Exception(f"Path {root_path} not exists.")
    cities = os.listdir(root_path)

    if yaml_config["params"]["osm_validation"]:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=yaml_config["params"]["max_processes"]
        ) as executor:
            futures = [
                executor.submit(geo_data_validation, os.path.join(root_path, city), yaml_config["params"]["init_osm"])
                for city in cities
            ]
            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures),
                total=len(cities),
                desc="Validating data",
            ):
                future.result()

    print(
        "Validation and initialize completed. Geo data total size:",
        len(os.listdir(root_path)),
    )

    # city_test_name = 'ZaanseSchans,Netherlands-7-3'

    # process_city(city_test_name, root_path, output, yaml_config)

    # exit(0)
    cities = os.listdir(root_path)
    if not yaml_config["params"]["debug"]:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=yaml_config["params"]["max_processes"]
        ) as executor:
            futures = [
                executor.submit(process_city, city, root_path, output, yaml_config)
                for city in cities
            ]
            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures),
                total=len(cities),
                desc="Processing cities",
            ):
                future.result()

    print("Processing completed.")

    if yaml_config["params"]["image_validation"]:
        for city in tqdm.tqdm(cities, desc="Validating images"):
            image_data_validation(os.path.join(output, city))

    print("Validation completed. Image data total size:", len(os.listdir(output)))

    cities = os.listdir(root_path)
    if yaml_config["params"]["dump_h5"]:
        
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=yaml_config["params"]["max_processes"]
        ) as executor:
            futures = [
                executor.submit(dump_h5py, os.path.join(output, city), os.path.join(output, city,  escape_path(city)+ ".h5"))
                for city in cities
            ]
            for future in tqdm.tqdm(
                concurrent.futures.as_completed(futures),
                total=len(cities),
                desc="Dumping h5py",
            ):
                future.result()
        
