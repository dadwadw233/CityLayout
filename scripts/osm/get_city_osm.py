import geopandas as gpd
import argparse
import osmnx as ox
import tqdm
import os
import elevation
from shapely.geometry import Point, box
import requests
import rasterio
from rasterio.features import shapes
from rasterio.mask import mask
import pandas as pd
import aiohttp
import asyncio
from http.client import HTTPException
from requests.exceptions import RequestException
import time
import yaml
import concurrent.futures

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_area_polygon(lat, lon, radius):
    center = Point(lon, lat)

    area_polygon = box(
        center.x - radius, center.y - radius, center.x + radius, center.y + radius
    )

    return area_polygon


def get_dem(lat, lon, radius, output_path):
    bounds = (lon - radius, lat - radius, lon + radius, lat + radius)
    elevation.clip(bounds=bounds, output=output_path)


def clip_gdf_to_area(gdf, area_polygon):
    clipped_gdf = gdf.copy()
    clipped_gdf["geometry"] = gdf["geometry"].intersection(area_polygon)
    return clipped_gdf


def get_dem_data(
    city, south, north, west, east, output_path, API_Key=None, dem_type="SRTMGL1"
):
    base_url = "https://portal.opentopography.org/API/globaldem"
    if API_Key is None:
        raise Exception("API key is required to download data from OpenTopography.")
    params = {
        "demtype": dem_type,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
        "API_Key": API_Key,
    }
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        with open(os.path.join(output_path, f"dem_data.tif"), "wb") as file:
            file.write(response.content)
        # print("DEM data downloaded successfully.")
    elif response.status_code == 400 or response.status_code == 500:
        raise TimeoutError("Timeout error")
    else:
        raise Exception(f"Failed to retrieve data. Status code: {response.status_code}")


def clip_pop_file(filename, south, north, west, east, output_path):
    geom = box(south, north, west, east)
    geo_df = gpd.GeoDataFrame({"geometry": geom}, index=[0], crs="EPSG:4326")

    # 打开 GeoTIFF 文件
    with rasterio.open(filename) as src:
        # 将 GeoDataFrame 的坐标系转换为与 TIFF 相同的坐标系
        geo_df = geo_df.to_crs(crs=src.crs.data)

        # 使用 Rasterio 的 mask 方法裁剪 TIFF 文件
        out_image, out_transform = mask(src, shapes=geo_df.geometry, crop=True)
        out_meta = src.meta.copy()

    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        }
    )

    output_tif = os.path.join(output_path, "pop_data.tif")
    with rasterio.open(output_tif, "w", **out_meta) as dest:
        dest.write(out_image)


def convert_to_polygon(geom):
    if geom.geom_type == "Point":
        return geom.buffer(0.0001)
    elif geom.geom_type == "LineString":
        return geom.buffer(0.0001)
    return geom


def get_geo_data(region, geo_type, query_key, config, add_default=False):
    target_types = config["data"][geo_type]
    target_query = {query_key: target_types}
    if target_types == "all":
        target_query = {query_key: True}
    data = ox.features_from_bbox(
        north=region.bounds[3],
        south=region.bounds[1],
        east=region.bounds[2],
        west=region.bounds[0],
        tags=target_query,
    )

    data = clip_gdf_to_area(data, region)

    if add_default:
        data["geometry"] = data["geometry"].apply(convert_to_polygon)
        default_gdf = gpd.GeoDataFrame(
            {"geometry": [region], query_key: ["default"]}, crs="EPSG:4326"
        )
        default_gdf = default_gdf.to_crs(data.crs)
        filled_gdf = gpd.overlay(
            default_gdf, data, how="difference", keep_geom_type=False
        )
        data = gpd.GeoDataFrame(pd.concat([data, filled_gdf], ignore_index=True))

    return data

def data_handler(yaml_config, city, output_dir, radius, citys):
    retries = yaml_config["network"]["retries"]
    while retries > 0:
        try:
            if not os.path.exists(os.path.join(output_dir, city)):
                os.mkdir(os.path.join(output_dir, city))
            print(f"Processing {city}...")

            city_out_path = os.path.join(output_dir, city)
            lat, lon = citys[city]
            geo_redius = (radius / 1000) / 111.319444
            area_polygon = create_area_polygon(lat, lon, geo_redius)

            if not os.path.exists(
                os.path.join(city_out_path, f"road_data_{radius}.geojson")
            ):
                road_graph = get_geo_data(
                    area_polygon,
                    "road_types",
                    "highway",
                    yaml_config,
                    add_default=False,
                )
                problematic_cols = [
                    col
                    for col in road_graph.columns
                    if any(isinstance(item, list) for item in road_graph[col])
                ]
                road_graph = road_graph.drop(columns=problematic_cols)
                road_graph.to_file(
                    os.path.join(city_out_path, f"road_data_{radius}.geojson"),
                    driver="GeoJSON",
                )

            if not os.path.exists(
                os.path.join(city_out_path, f"landuse_data_{radius}.geojson")
            ):
                landuse_data = get_geo_data(
                    area_polygon,
                    "landuse_types",
                    "landuse",
                    yaml_config,
                    add_default=True,
                )
                problematic_cols = [
                    col
                    for col in landuse_data.columns
                    if any(isinstance(item, list) for item in landuse_data[col])
                ]
                landuse_data = landuse_data.drop(columns=problematic_cols)
                landuse_data.to_file(
                    os.path.join(city_out_path, f"landuse_data_{radius}.geojson"),
                    driver="GeoJSON",
                )

            if not os.path.exists(
                os.path.join(city_out_path, f"nature_data_{radius}.geojson")
            ):
                nature_data = get_geo_data(
                    area_polygon,
                    "nature_types",
                    "natural",
                    yaml_config,
                    add_default=True,
                )

                problematic_cols = [
                    col
                    for col in nature_data.columns
                    if any(isinstance(item, list) for item in nature_data[col])
                ]
                nature_data = nature_data.drop(columns=problematic_cols)
                nature_data.to_file(
                    os.path.join(city_out_path, f"nature_data_{radius}.geojson"),
                    driver="GeoJSON",
                )

            if not os.path.exists(
                os.path.join(city_out_path, f"buildings_data_{radius}.geojson")
            ):
                buildings_query = {"building": True}

                buildings_data = ox.features_from_bbox(
                    north=area_polygon.bounds[3],
                    south=area_polygon.bounds[1],
                    east=area_polygon.bounds[2],
                    west=area_polygon.bounds[0],
                    tags=buildings_query,
                )
                buildings_data = clip_gdf_to_area(buildings_data, area_polygon)

                problematic_cols = [
                    col
                    for col in buildings_data.columns
                    if any(isinstance(item, list) for item in buildings_data[col])
                ]
                buildings_data = buildings_data.drop(columns=problematic_cols)
                buildings_data.to_file(
                    os.path.join(city_out_path, f"buildings_data_{radius}.geojson"),
                    driver="GeoJSON",
                )

            # Get population data
            # pop_data = clip_gdf_to_area(pop_data, area_polygon)
            # pop_data.to_file(
            #     os.path.join(city_out_path, "pop_data.geojson"), driver="GeoJSON"
            # )

            break
        except TimeoutError as e:
            if retries == 0:
                print(f"Timeout error occured when processing {city}: {e}")
            else:
                print(
                    f"Timeout error occured when processing {city}: {e}, retrying..."
                )
                retries -= 1
                time.sleep(10)
                continue
        except HTTPException as e:
            if retries == 0:
                print(f"HTTPException error occured when processing {city}: {e}")
            else:
                print(
                    f"HTTPException error occured when processing {city}: {e}, retrying..."
                )
                retries -= 1
                time.sleep(10)
                continue
        except RequestException as e:
            if retries == 0:
                print(f"RequestException error occured when processing {city}: {e}")
            else:
                print(
                    f"RequestException error occured when processing {city}: {e}, retrying..."
                )
                retries -= 1
                time.sleep(10)
                continue
        except Exception as e:
            print(f"Error occured when processing {city}: {e}")
            with open(
                os.path.join(city_out_path, "getdata_error.txt"), "w"
            ) as file:
                file.write(str(e))
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="../../config/data/osm_cities.yaml",
        help="config file path",
    )

    yaml_config = load_config(parser.parse_args().config)

    input_file = yaml_config["path"]["input"]
    output_dir = yaml_config["path"]["output"]
    pop_file_path = yaml_config["path"]["pop_file_path"]
    # pop_data = gpd.read_file(pop_file_path)
    # print("Pop data loaded.")

    radius = yaml_config["data"]["radius"]

    # Load city locations
    citys = dict()
    ox.settings.all_oneway = True

    if not os.path.exists(input_file):
        raise Exception(f"Input file {input_file} does not exist.")

    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            city, lat, lon = line.split(" ")
            citys[city] = (float(lat), float(lon))

    print(f"{len(citys)} citie's locations loaded.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for city in citys.keys():
            futures.append(
                executor.submit(
                    data_handler,
                    yaml_config,
                    city,
                    output_dir,
                    radius,
                    citys,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error occured when processing {city}: {e}")

    print("Done.")
