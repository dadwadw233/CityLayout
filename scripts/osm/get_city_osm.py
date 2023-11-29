import geopandas as gpd
import argparse
import osmnx as ox
import tqdm
import os


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

        lat, lon = citys[city]
        # 提取道路网络
        # road_graph = ox.graph_from_point((lat, lon), network_type='all')
        road_graph = ox.features_from_point((lat, lon), dist=args.radius, tags={'highway': True})
        # 提取土地利用数据
        landuse_query = {'landuse': True, 'natural': True, 'leisure': True, 'waterway': True, 'water': True}
        landuse_data = ox.features_from_point((lat, lon), dist=args.radius, tags=landuse_query)

        # 提取建筑物数据
        buildings_query = {'building': True}
        buildings_data = ox.features_from_point((lat, lon), dist=args.radius, tags=buildings_query)


        # 道路数据转换为GeoDataFrame并保存
        problematic_cols = [col for col in road_graph.columns if any(isinstance(item, list) for item in road_graph[col])]
        road_graph = road_graph.drop(columns=problematic_cols)
        road_graph.to_file("road_data.geojson", driver='GeoJSON')


        problematic_cols = [col for col in landuse_data.columns if any(isinstance(item, list) for item in landuse_data[col])]
        landuse_data = landuse_data.drop(columns=problematic_cols)
        landuse_data.to_file("landuse_data.geojson", driver='GeoJSON')

        problematic_cols = [col for col in buildings_data.columns if any(isinstance(item, list) for item in buildings_data[col])]
        buildings_data = buildings_data.drop(columns=problematic_cols)
        buildings_data.to_file("buildings_data.geojson", driver='GeoJSON')


        print(f"OSM data for {city} saved.")

    

    
    print('Done.')
        
