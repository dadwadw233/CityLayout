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

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from src.utils.utils import load_config


def geo_data_validation(path, init=False):
    if os.path.exists(os.path.join(path, "getdata_error.txt")):
        logging.error(f"Error occurred in {path}.")
        os.system(f"rm -rf {path}")

    if os.path.exists(os.path.join(path, "plotting_img_finish.txt")):
        os.system(f"rm  {os.path.join(path, 'plotting_img_finish.txt')}")

    if os.path.exists(os.path.join(path, "plotting_img_error.txt")):
        os.system(f"rm  {os.path.join(path, 'plotting_img_error.txt')}")

    # delete file which is not geojson
    if init:
        for file in os.listdir(path):
            if not file.endswith(".geojson"):
                os.system(f"rm  {os.path.join(path, file)}")


def image_data_validation(path):
    if not os.path.exists(os.path.join(path, "plotting_img_finish.txt")):
        logging.error(f"Error occurred in {path}.")

        # remove data directory
        # os.system(f"rm -rf {path}")


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
    xlim = (combined_gdf.total_bounds[0], combined_gdf.total_bounds[2])  # Minx, Maxx
    ylim = (combined_gdf.total_bounds[1], combined_gdf.total_bounds[3])  # Miny, Maxy

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis("off")
    plt.axis("off")

    if not landuse_gdf.empty:
        feature_img_dict = {}
        landuse_gdf["area"] = landuse_gdf.geometry.area

        landuse_gdf = landuse_gdf.sort_values(by="area", ascending=True)

        # 绘制单独的landuse层
        color_dict = config["data_config"]["landuse"]["color_dict"]

        feature_list = config["data_config"]["landuse"]["feature_list"]

        fig_, ax_ = plt.subplots(figsize=config["plt_config"]["default"]["figsize"])
        ax_.set_xlim(xlim)
        ax_.set_ylim(ylim)

        ax_.axis("off")

        for landuse_type in landuse_gdf["landuse"].unique():
            # 只选取当前 landuse 类型的数据
            gdf_type = landuse_gdf[landuse_gdf["landuse"] == landuse_type]

            if landuse_type in color_dict.keys():
                gdf_type.plot(
                    ax=ax_,
                    color=color_dict.get(landuse_type, "#FFFFFF"),
                    alpha=config["plt_config"]["landuse"]["alpha"],
                )
                gdf_type.plot(
                    ax=ax,
                    color=color_dict.get(landuse_type, "#FFFFFF"),
                    alpha=config["plt_config"]["landuse"]["alpha"],
                )

            if landuse_type in feature_list:
                gdf_type.plot(cmap="gray")

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
                feature_img_dict[landuse_type] = data

                plt.close()
                buf.close()

        for feature in feature_list:
            if feature not in feature_img_dict.keys():
                feature_img_dict[feature] = np.zeros(
                    (image_array.shape[0], image_array.shape[1], 1)
                )

        plt.savefig(
            os.path.join(os.path.dirname(output_filename), "landuse.jpg"),
            pad_inches=config["plt_config"]["landuse"]["pad_inches"],
            bbox_inches=config["plt_config"]["landuse"]["bbox_inches"],
            format=config["plt_config"]["landuse"]["format"],
        )
        plt.close(fig_)

        # fig, axs = plt.subplots(1, len(feature_img_dict), figsize=(20, 8))
        # for index,label in enumerate(feature_list):

        #     ax_inner = axs[index]
        #     ax_inner.axis('off')
        #     ax_inner.imshow(feature_img_dict[label], cmap='gray')
        #     ax_inner.set_title(f'Label {label}')

        # plt.savefig(os.path.join(os.path.dirname(output_filename), 'landuse_multi_channel.jpg'))
        # plt.close(fig)

        feature_img_dict = {k: feature_img_dict[k] for k in feature_list}
        landuse_matrix = np.stack(list(feature_img_dict.values()), axis=-1)
        np.save(
            os.path.join(os.path.dirname(output_filename), "landuse.npy"),
            landuse_matrix,
        )

    if not nature_gdf.empty:
        feature_img_dict = {}
        nature_cols = ["natural"]
        for col in nature_cols:
            if col in nature_gdf.columns:
                nature_gdf[col] = nature_gdf[col].fillna("")
            else:
                nature_gdf[col] = ""

        color_dict = config["data_config"]["nature"]["color_dict"]

        feature_list = config["data_config"]["nature"]["feature_list"]
        fig_, ax_ = plt.subplots(figsize=config["plt_config"]["default"]["figsize"])
        ax_.set_xlim(xlim)
        ax_.set_ylim(ylim)
        ax_.axis("off")

        for nature_type in nature_gdf["natural"].unique():
            gdf_type = nature_gdf[nature_gdf["natural"] == nature_type]

            if nature_type in color_dict.keys():
                gdf_type.plot(ax=ax_, color=color_dict[nature_type], edgecolor="black")
                gdf_type.plot(
                    ax=ax,
                    color=color_dict[nature_type],
                    edgecolor="black",
                    alpha=config["plt_config"]["nature"]["alpha"],
                )

            if nature_type in feature_list:
                gdf_type.plot(cmap="gray")

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
                feature_img_dict[nature_type] = data

                plt.close()
                buf.close()

        for feature in feature_list:
            if feature not in feature_img_dict.keys():
                feature_img_dict[feature] = np.zeros(
                    (image_array.shape[0], image_array.shape[1], 1)
                )

        plt.savefig(
            os.path.join(os.path.dirname(output_filename), "nature.jpg"),
            bbox_inches=config["plt_config"]["nature"]["bbox_inches"],
            format=config["plt_config"]["nature"]["format"],
            pad_inches=config["plt_config"]["nature"]["pad_inches"],
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
        nature_matrix = np.stack(list(feature_img_dict.values()), axis=-1)
        np.save(
            os.path.join(os.path.dirname(output_filename), "nature.npy"), nature_matrix
        )

    if not roads_gdf.empty:
        feature_img_dict = {}

        color_dict = config["data_config"]["road"]["color_dict"]

        feature_list = config["data_config"]["road"]["feature_list"]

        fig_, ax_ = plt.subplots(figsize=config["plt_config"]["default"]["figsize"])
        ax_.set_xlim(xlim)
        ax_.set_ylim(ylim)
        ax_.axis("off")

        for road_type in roads_gdf["highway"].unique():
            gdf_type = roads_gdf[roads_gdf["highway"] == road_type]

            if road_type in color_dict.keys():
                gdf_type.plot(ax=ax_, color=color_dict[road_type])
                gdf_type.plot(
                    ax=ax,
                    color=color_dict[road_type],
                    alpha=config["plt_config"]["road"]["alpha"],
                )

            if road_type in feature_list:
                gdf_type.plot(cmap="gray")
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
                feature_img_dict[road_type] = data

                plt.close()
                buf.close()

        for feature in feature_list:
            if feature not in feature_img_dict.keys():
                feature_img_dict[feature] = np.zeros(
                    (image_array.shape[0], image_array.shape[1], 1)
                )

        plt.savefig(
            os.path.join(os.path.dirname(output_filename), "road.jpg"),
            bbox_inches=config["plt_config"]["road"]["bbox_inches"],
            format=config["plt_config"]["road"]["format"],
            pad_inches=config["plt_config"]["road"]["pad_inches"],
        )
        plt.close(fig_)

        # fig, axs = plt.subplots(1, len(feature_img_dict), figsize=(20, 8))
        # for index,label in enumerate(feature_list):

        #     ax_inner = axs[index]
        #     ax_inner.axis('off')
        #     ax_inner.imshow(feature_img_dict[label], cmap='gray')
        #     ax_inner.set_title(f'Label {label}')

        # plt.savefig(os.path.join(os.path.dirname(output_filename), 'road_multi_channel.jpg'))
        # plt.close(fig)

        feature_img_dict = {k: feature_img_dict[k] for k in feature_list}
        road_matrix = np.stack(list(feature_img_dict.values()), axis=-1)
        np.save(os.path.join(os.path.dirname(output_filename), "road.npy"), road_matrix)

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
        feature_img_dict[road_type] = data

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
    plt.close()

    return xlim, ylim


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

        try:
            roads_gdf = gpd.read_file(
                os.path.join(save_path, "road_data.geojson")
            ).to_crs(epsg=config["geo"]["crs"])
            landuse_gdf = gpd.read_file(
                os.path.join(save_path, "landuse_data.geojson")
            ).to_crs(epsg=config["geo"]["crs"])
            buildings_gdf = gpd.read_file(
                os.path.join(save_path, "buildings_data.geojson")
            ).to_crs(epsg=config["geo"]["crs"])
            nature_gdf = gpd.read_file(
                os.path.join(save_path, "nature_data.geojson")
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
    with h5py.File(output_path, "w") as f:
        for file in os.listdir(path):
            if file.endswith(".npy"):
                data = np.load(os.path.join(path, file))
                f.create_dataset(file[:-4], data=data)


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
        for city in tqdm.tqdm(cities, desc="Validating data"):
            geo_data_validation(
                os.path.join(root_path, city), init=yaml_config["params"]["init_osm"]
            )

    print(
        "Validation and initialize completed. Geo data total size:",
        len(os.listdir(root_path)),
    )

    # city_test_name = 'Zurich-7'

    # process_city(city_test_name, root_path, output, yaml_config)

    # exit(0)

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

    if yaml_config["params"]["dump_h5"]:
        for city in tqdm.tqdm(cities, desc="Dumping h5py"):
            dump_h5py(
                os.path.join(output, city), os.path.join(output, city, city + ".h5")
            )
