import geopandas as gpd
import argparse
import osmnx as ox
import tqdm
import os
import elevation
from shapely.geometry import Point, box

def create_area_polygon(lat, lon, radius):
    # 创建代表感兴趣区域的多边形
    center = Point(lon, lat)
    
    area_polygon = box(center.x - radius, center.y - radius, center.x + radius, center.y + radius)
    
    return area_polygon

def get_dem(lat, lon, radius, output_path):
    # 设置DEM数据的边界
    bounds = (lon-radius, lat-radius, lon+radius, lat+radius)
    elevation.clip(bounds=bounds, output=output_path)

def clip_gdf_to_area(gdf, area_polygon):
    
    clipped_gdf = gdf.copy()
    clipped_gdf['geometry'] = gdf['geometry'].intersection(area_polygon)
    return clipped_gdf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input file (for city locations)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output root directory')
    parser.add_argument('--radius', type=int, required=True, help='Radius of the city')

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
        
        try:

            if not os.path.exists(os.path.join(args.output_dir, city)):
                os.mkdir(os.path.join(args.output_dir, city))
            else:
                continue
            city_out_path = os.path.join(args.output_dir, city)
            lat, lon = citys[city]
            area_polygon = create_area_polygon(lat, lon, 0.00898)

            
            # 提取道路网络
            # road_graph = ox.graph_from_point((lat, lon), network_type='all')
            road_graph = ox.features_from_point((lat, lon), dist=args.radius, tags={'highway': True})
            # road_graph = ox.features_from_polygon(area_polygon, tags={'highway': True})
            road_types = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'service', 'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link']
            road_graph = road_graph[(road_graph['highway'] == 'primary') | (road_graph['highway'] == 'secondary') | (road_graph['highway'] == 'tertiary') | (road_graph['highway'] == 'residential') | (road_graph['highway'] == 'service') ]#  | (road_graph['highway'] == 'motorway') | (road_graph['highway'] == 'trunk')
            road_graph = clip_gdf_to_area(road_graph, area_polygon)


            landuse_query = {'landuse': True}
            landuse_data = ox.features_from_point((lat, lon), dist=args.radius, tags=landuse_query)
            # landuse_data = ox.features_from_polygon(area_polygon, tags=landuse_query)
            landuse_data = clip_gdf_to_area(landuse_data, area_polygon)


            nature_query = {'natural': True, 'water': True, 'waterway': True}
            nature_data = ox.features_from_point((lat, lon), dist=args.radius, tags=nature_query)
            # nature_data = ox.features_from_polygon(area_polygon, tags=nature_query)
            nature_data = clip_gdf_to_area(nature_data, area_polygon)


            buildings_query = {'building': True}
            buildings_data = ox.features_from_point((lat, lon), dist=args.radius, tags=buildings_query)
            # buildings_data = ox.features_from_polygon(area_polygon, tags=buildings_query)
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



            print(f"OSM data for {city} saved.")
        except Exception as e:
            print(f"Error occured when processing {city}: {e}")
            continue

    

    
    print('Done.')
        
