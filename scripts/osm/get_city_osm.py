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

def create_area_polygon(lat, lon, radius):
    
    center = Point(lon, lat)
    
    area_polygon = box(center.x - radius, center.y - radius, center.x + radius, center.y + radius)
    
    return area_polygon

def get_dem(lat, lon, radius, output_path):
    
    bounds = (lon-radius, lat-radius, lon+radius, lat+radius)
    elevation.clip(bounds=bounds, output=output_path)

def clip_gdf_to_area(gdf, area_polygon):
    
    clipped_gdf = gdf.copy()
    clipped_gdf['geometry'] = gdf['geometry'].intersection(area_polygon)
    return clipped_gdf

def get_dem_data(city, south, north, west, east, output_path):
    base_url = "https://portal.opentopography.org/API/globaldem"

    params = {
        "demtype": "SRTMGL1",  
        "south": south,       
        "north": north,       
        "west": west,      
        "east": east,      
        "outputFormat": "GTiff", 
        "API_Key": "316dad05d6a595c83c9ee3864394ad85"
    }
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        with open(os.path.join(output_path, f"dem_data.tif"), "wb") as file:
            file.write(response.content)
        # print("DEM data downloaded successfully.")
    elif response.status_code == 400 or response.status_code == 500:
        raise TimeoutError("Timeout error")
    else :
        raise Exception(f"Failed to retrieve data. Status code: {response.status_code}")
    

def clip_pop_file(filename, south, north, west, east, output_path):
    geom = box(south, north, west, east)
    geo_df = gpd.GeoDataFrame({'geometry': geom}, index=[0], crs='EPSG:4326')


    # 打开 GeoTIFF 文件
    with rasterio.open(filename) as src:
        # 将 GeoDataFrame 的坐标系转换为与 TIFF 相同的坐标系
        geo_df = geo_df.to_crs(crs=src.crs.data)

        # 使用 Rasterio 的 mask 方法裁剪 TIFF 文件
        out_image, out_transform = mask(src, shapes=geo_df.geometry, crop=True)
        out_meta = src.meta.copy()

    out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})

    output_tif = os.path.join(output_path, 'pop_data.tif')
    with rasterio.open(output_tif, "w", **out_meta) as dest:
        dest.write(out_image)

def convert_to_polygon(geom):
    if geom.geom_type == 'Point':
        return geom.buffer(0.0001) 
    elif geom.geom_type == 'LineString':
        return geom.buffer(0.0001)
    return geom

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input file (for city locations)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output root directory')
    parser.add_argument('--radius', type=int, required=True, help='Radius of the city (meters))')
    pop_file_path = '../../data/raw/pop/GHS_POP_E2030_GLOBE_R2023A_54009_100_V1_0.tif'


    args = parser.parse_args()

    # Load city locations
    citys = dict()
    ox.settings.all_oneway=True

    with open(args.input, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            city, lat, lon = line.split(' ')
            citys[city] = (float(lat), float(lon))

    print(f"{len(citys)} citie's locations loaded.")
    

    for city in tqdm.tqdm(citys.keys(), desc='Processing cities'):
        retries = 10
        while retries > 0:
            try:
                if not os.path.exists(os.path.join(args.output_dir, city)):
                    os.mkdir(os.path.join(args.output_dir, city))
                else:
                    break
                city_out_path = os.path.join(args.output_dir, city)
                lat, lon = citys[city]
                geo_redius = (args.radius/1000) / 111.319444
                area_polygon = create_area_polygon(lat, lon, geo_redius)

                
                # road_graph = ox.features_from_polygon(area_polygon, tags={'highway': True})
                road_types = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'service']
                road_query = {'highway': road_types}
                road_graph = ox.features_from_bbox(north=area_polygon.bounds[3], south=area_polygon.bounds[1], east=area_polygon.bounds[2], west=area_polygon.bounds[0], tags=road_query)
                # road_graph = ox.features_from_point((lat, lon), dist=args.radius, tags=road_query)
                # road_graph = road_graph[(road_graph['highway'] == 'primary') | (road_graph['highway'] == 'secondary') | (road_graph['highway'] == 'tertiary') | (road_graph['highway'] == 'residential') | (road_graph['highway'] == 'service') ]  # | (road_graph['highway'] == 'motorway') | (road_graph['highway'] == 'trunk')
                road_graph = clip_gdf_to_area(road_graph, area_polygon)


                landuse_types = ['commercial', 'retail', 'industrial', 'depot', 'port', 'residual', 'farmland', 'meadow', 'orchard', 'vineyard', 'plant_nursery', 'forest', 'farmyard', 'grass', 'greenfield', 'military', 'railway', 'recreation_ground', 'fairground']
                landuse_query = {'landuse': landuse_types}
                landuse_data = ox.features_from_bbox(north=area_polygon.bounds[3], south=area_polygon.bounds[1], east=area_polygon.bounds[2], west=area_polygon.bounds[0], tags=landuse_query)
                # landuse_data = ox.features_from_point((lat, lon), dist=args.radius, tags=landuse_query)
                # landuse_data = ox.features_from_polygon(area_polygon, tags=landuse_query)
                landuse_data = clip_gdf_to_area(landuse_data, area_polygon)
                landuse_data['geometry'] = landuse_data['geometry'].apply(convert_to_polygon)

                default_gdf = gpd.GeoDataFrame({'geometry': [area_polygon], 'landuse': ['default']}, crs='EPSG:4326')
                default_gdf = default_gdf.to_crs(landuse_data.crs)
                filled_gdf  = gpd.overlay(default_gdf, landuse_data, how='difference', keep_geom_type=False)
                # print(landuse_data.geometry.type.unique())
                # print(default_gdf.geometry.type.unique())
                landuse_data = gpd.GeoDataFrame(pd.concat([landuse_data, filled_gdf], ignore_index=True))

                nature_types = ['tree', 'tree_row', 'wood', 'grassland', 'beach', 'water', 'wetland', 'bare_rock', 'hill', 'sand', 'valley']
                nature_query = {'natural': nature_types}
                nature_data = ox.features_from_bbox(north=area_polygon.bounds[3], south=area_polygon.bounds[1], east=area_polygon.bounds[2], west=area_polygon.bounds[0], tags=nature_query)
                # nature_data = ox.features_from_point((lat, lon), dist=args.radius, tags=nature_query)
                nature_data = clip_gdf_to_area(nature_data, area_polygon)
                nature_data['geometry'] = nature_data['geometry'].apply(convert_to_polygon)

                default_gdf = gpd.GeoDataFrame({'geometry': [area_polygon], 'natural': ['default']}, crs='EPSG:4326')
                default_gdf = default_gdf.to_crs(nature_data.crs)
                filled_gdf  = gpd.overlay(default_gdf, nature_data, how='difference', keep_geom_type=False)
                
                nature_data = gpd.GeoDataFrame(pd.concat([nature_data, filled_gdf], ignore_index=True))
                



                buildings_query = {'building': True}
                # buildings_data = ox.features_from_point((lat, lon), dist=args.radius, tags=buildings_query)
                # buildings_data = ox.features_from_polygon(area_polygon, tags=buildings_query)
                buildings_data = ox.features_from_bbox(north=area_polygon.bounds[3], south=area_polygon.bounds[1], east=area_polygon.bounds[2], west=area_polygon.bounds[0], tags=buildings_query)
                buildings_data = clip_gdf_to_area(buildings_data, area_polygon)





                problematic_cols = [col for col in road_graph.columns if any(isinstance(item, list) for item in road_graph[col])]
                road_graph = road_graph.drop(columns=problematic_cols)
                road_graph.to_file(os.path.join(city_out_path, "road_data.geojson"), driver='GeoJSON')
                


                problematic_cols = [col for col in landuse_data.columns if any(isinstance(item, list) for item in landuse_data[col])]
                landuse_data = landuse_data.drop(columns=problematic_cols)
                landuse_data.to_file(os.path.join(city_out_path, "landuse_data.geojson"), driver='GeoJSON')


                problematic_cols = [col for col in buildings_data.columns if any(isinstance(item, list) for item in buildings_data[col])]
                buildings_data = buildings_data.drop(columns=problematic_cols)
                buildings_data.to_file(os.path.join(city_out_path, "buildings_data.geojson"), driver='GeoJSON')


                problematic_cols = [col for col in nature_data.columns if any(isinstance(item, list) for item in nature_data[col])]
                nature_data = nature_data.drop(columns=problematic_cols)
                nature_data.to_file(os.path.join(city_out_path, "nature_data.geojson"), driver='GeoJSON')



                # print(f"OSM data for {city} saved.")
                break
            except TimeoutError as e:
                if retries == 0:
                    print(f"Timeout error occured when processing {city}: {e}")
                else:
                    print(f"Timeout error occured when processing {city}: {e}, retrying...")
                    retries -= 1
                    continue
            except Exception as e:
                print(f"Error occured when processing {city}: {e}")
                with open(os.path.join(city_out_path, 'getdata_error.txt'), 'w') as file:
                    file.write(str(e))
                break

    

    
    print('Done.')
        
