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

def geo_data_validation(path):

    if os.path.exists(os.path.join(path, 'getdata_error.txt')):
        logging.error(f"Error occurred in {path}.")
        os.system(f"rm -rf {path}")

    if os.path.exists(os.path.join(path, 'plotting_img_finish.txt')):
        os.system(f"rm  {os.path.join(path, 'plotting_img_finish.txt')}")

    if os.path.exists(os.path.join(path, 'plotting_img_error.txt')):
        os.system(f"rm  {os.path.join(path, 'plotting_img_error.txt')}")

    # delete file which is not geojson
    for file in os.listdir(path):
        if not file.endswith('.geojson'):
            os.system(f"rm  {os.path.join(path, file)}")


def image_data_validation(path):
    
    if not os.path.exists(os.path.join(path, 'plotting_img_finish.txt')):
        logging.error(f"Error occurred in {path}.")

        # remove data directory
        os.system(f"rm -rf {path}")


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
    plt.imshow(dem, cmap='gray')  # 或者使用其他colormap，如 'terrain'
    plt.axis('off')
    plt.savefig(output_filename)
    plt.close()        


def plot_combined_map(roads_gdf, landuse_gdf, buildings_gdf, nature_gdf, output_filename, fig_size=(10, 10)):
    fig, ax = plt.subplots(figsize=fig_size)
    combined_gdf = gpd.GeoDataFrame(pd.concat([roads_gdf, landuse_gdf, buildings_gdf, nature_gdf], ignore_index=True), crs=roads_gdf.crs)
    xlim = (combined_gdf.total_bounds[0], combined_gdf.total_bounds[2])  # Minx, Maxx
    ylim = (combined_gdf.total_bounds[1], combined_gdf.total_bounds[3])  # Miny, Maxy

    

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    plt.axis('off')

    if not landuse_gdf.empty:
        feature_img_dict = {}
        landuse_gdf['area'] = landuse_gdf.geometry.area

        
        landuse_gdf = landuse_gdf.sort_values(by='area', ascending=True)
        landuse_gdf.plot(ax=ax, column='landuse', cmap='Accent', alpha=0.5)

        # 绘制单独的landuse层
        color_dict = {'commercial': '#EFCAC0', 'retail': '#EFCAC0', 'education': '#EF826B', 'industrial':'#D6CCED',
                      'depot':'#D6CCED', 'port':'#D6CCED', 'residual':'#E5EBE8', 'farmland':'#DFF58D', 'meadow':'#DFF58D',
                      'orchard':'#DFF58D', 'vineyard':'#DFF58D', 'plant_nursery':'#DFF58D', 'forest':'#16F53E',
                      'farmyard': '#F0CB60', 'grass': '#B4F59D', 'greenfield':'#B4F59D', 'military':'#EF4631', 'railway':'#B884F0',
                      'recreation_ground':'#F07D00', 'fairground':'#F07D00', 'default':'#8F8F8F'}
        
        feature_list = ['commercial', 'retail', 'education', 'industrial', 
                        'residual', 'forest', 'grass', 'greenfield', 
                        'railway', 'recreation_ground', 'default']
        
        fig_, ax_ = plt.subplots(figsize=fig_size)
        ax_.set_xlim(xlim)
        ax_.set_ylim(ylim)

        ax_.axis('off')

        for landuse_type in landuse_gdf['landuse'].unique():
            # 只选取当前 landuse 类型的数据
            gdf_type = landuse_gdf[landuse_gdf['landuse'] == landuse_type]

            if landuse_type in color_dict.keys():
                gdf_type.plot(ax=ax_, color=color_dict.get(landuse_type, '#FFFFFF'), alpha=0.5)
                gdf_type.plot(ax=ax, color=color_dict.get(landuse_type, '#FFFFFF'), alpha=0.5)
            
            if landuse_type in feature_list:
                
                gdf_type.plot(cmap='gray')
                
                plt.axis('off')
                # set xlim and ylim
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.tight_layout()
                # save image to buffer
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
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
                feature_img_dict[feature] = np.zeros((image_array.shape[0], image_array.shape[1], 1))

        fig_.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.savefig(os.path.join(os.path.dirname(output_filename), 'landuse.jpg'), bbox_inches='tight', format='jpg')

        plt.close(fig_)

        # fig, axs = plt.subplots(1, len(feature_img_dict), figsize=(20, 8))  
        # for index,label in enumerate(feature_list):
            
        #     ax_inner = axs[index]
        #     ax_inner.axis('off')
        #     ax_inner.imshow(feature_img_dict[label], cmap='gray')
        #     ax_inner.set_title(f'Label {label}')
            
        # plt.savefig(os.path.join(os.path.dirname(output_filename), 'landuse_multi_channel.jpg'))
        # plt.close(fig)

        feature_img_dict = {k : feature_img_dict[k] for k in feature_list}
        landuse_matrix = np.stack(list(feature_img_dict.values()), axis=-1)
        np.save(os.path.join(os.path.dirname(output_filename), 'landuse.npy'), landuse_matrix)
        

    
    if not nature_gdf.empty:
        feature_img_dict = {}
        nature_cols = ['natural']
        for col in nature_cols:
            if col in nature_gdf.columns:
                nature_gdf[col] = nature_gdf[col].fillna('')
            else:
                nature_gdf[col] = ''
        # nature_gdf['nature_sum'] = nature_gdf[nature_cols].agg('_'.join, axis=1)

    
        # nature_gdf.plot(ax=ax, column='nature_sum', cmap='Set3', alpha=0.5)

        color_dict = {'grassland': '#A8EB83', 'tree': '#1CEF26', 'tree_row': '#1CEF26', 'wood': '#1CEF26'
                      , 'beach': '#D5E4ED', 'water': '#418DF0', 'wetland': '#51D5EB', 'bare_rock': '#E5F2D3'
                      , 'hill': '#CAF582', 'sand': '#EDE6B0', 'valley': '#F0BA60', 'default': '#BFBFBF'}
        
        feature_list = ['grassland', 'tree', 'beach', 'water','hill', 'sand', 'valley', 'default']

        fig_, ax_ = plt.subplots(figsize=fig_size)
        ax_.set_xlim(xlim)
        ax_.set_ylim(ylim)
        ax_.axis('off')
        
        for nature_type in nature_gdf['natural'].unique():
            
            gdf_type = nature_gdf[nature_gdf['natural'] == nature_type]

            if nature_type in color_dict.keys():
                gdf_type.plot(ax=ax_, color=color_dict[nature_type], edgecolor='black')
                gdf_type.plot(ax=ax, color=color_dict[nature_type], edgecolor='black', alpha=0.5)

            if nature_type in feature_list:
                    
                    gdf_type.plot(cmap='gray')

                    
                    plt.axis('off')
                    # set xlim and ylim
                    plt.xlim(xlim)
                    plt.ylim(ylim)
                    plt.tight_layout()
                    # save image to buffer
                    buf = BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
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
                    feature_img_dict[feature] = np.zeros((image_array.shape[0], image_array.shape[1], 1))


        fig_.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.savefig(os.path.join(os.path.dirname(output_filename), 'nature.jpg'), bbox_inches='tight', format='jpg')

        plt.close(fig_)

        # fig, axs = plt.subplots(1, len(feature_img_dict), figsize=(20, 8))
        # for index,label in enumerate(feature_list):
                
        #     ax_inner = axs[index]
        #     ax_inner.axis('off')
        #     ax_inner.imshow(feature_img_dict[label], cmap='gray')
        #     ax_inner.set_title(f'Label {label}')

        # plt.savefig(os.path.join(os.path.dirname(output_filename), 'nature_multi_channel.jpg'))
        # plt.close(fig)
        feature_img_dict = {k : feature_img_dict[k] for k in feature_list}
        nature_matrix = np.stack(list(feature_img_dict.values()), axis=-1)
        np.save(os.path.join(os.path.dirname(output_filename), 'nature.npy'), nature_matrix)

    
    if not roads_gdf.empty:
        feature_img_dict = {}

        color_dict = {'motorway': 'red', 'trunk': 'orange', 'primary': 'yellow', 'secondary': 'green',
                        'tertiary': 'pink', 'residential': 'blue', 'service': 'grey'
                        }

        feature_list = ['motorway', 'trunk', 'primary', 'secondary', 'residential', 'service']

        fig_, ax_ = plt.subplots(figsize=fig_size)
        ax_.set_xlim(xlim)
        ax_.set_ylim(ylim)
        ax_.axis('off')

        for road_type in roads_gdf['highway'].unique():
            gdf_type = roads_gdf[roads_gdf['highway'] == road_type]

            if road_type in color_dict.keys():
                gdf_type.plot(ax=ax_, color=color_dict[road_type])
                gdf_type.plot(ax=ax, color=color_dict[road_type], alpha=0.5)

            if road_type in feature_list:
                        
                gdf_type.plot(cmap='gray')
                
                # set axis off
                plt.axis('off')
                # set xlim and ylim
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.tight_layout()
                # save image to buffer
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
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
                        feature_img_dict[feature] = np.zeros((image_array.shape[0], image_array.shape[1], 1))

        # roads_gdf.plot(ax=ax_, column='highway', cmap='tab20', legend=True)
        # roads_gdf.plot(ax=ax, column='highway', cmap='tab20')
        fig_.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.savefig(os.path.join(os.path.dirname(output_filename), 'road.jpg'), bbox_inches='tight', format='jpg')
        plt.close(fig_)

        # fig, axs = plt.subplots(1, len(feature_img_dict), figsize=(20, 8))
        # for index,label in enumerate(feature_list):
                        
        #     ax_inner = axs[index]
        #     ax_inner.axis('off')
        #     ax_inner.imshow(feature_img_dict[label], cmap='gray')
        #     ax_inner.set_title(f'Label {label}')

        # plt.savefig(os.path.join(os.path.dirname(output_filename), 'road_multi_channel.jpg'))
        # plt.close(fig)

        feature_img_dict = {k : feature_img_dict[k] for k in feature_list}
        road_matrix = np.stack(list(feature_img_dict.values()), axis=-1)
        np.save(os.path.join(os.path.dirname(output_filename), 'road.npy'), road_matrix)

        
    
    if not buildings_gdf.empty:

        buildings_gdf.plot(ax=ax, color='grey', edgecolor='black', alpha=0.7)  

        fig_, ax_ = plt.subplots(figsize=fig_size)
        ax_.set_xlim(xlim)
        ax_.set_ylim(ylim)
        ax_.axis('off')

        buildings_gdf.plot(ax=ax_, color='grey', edgecolor='black')
        fig_.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.savefig(os.path.join(os.path.dirname(output_filename), 'building_location.jpg'), bbox_inches='tight', format='jpg')

        plt.close(fig_)

        #  绘制高度数据
        # 判断'height'字段是否存在
        if 'height' in buildings_gdf.columns:
            
            fig_, ax_ = plt.subplots(figsize=fig_size)
            ax_.set_xlim(xlim)
            ax_.set_ylim(ylim)
            ax_.axis('off')

            buildings_gdf.plot(ax=ax_, column='height', cmap='viridis')
            fig_.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.savefig(os.path.join(os.path.dirname(output_filename), 'building_height.jpg'), bbox_inches='tight', format='jpg')
            plt.close(fig_)

        buildings_gdf.plot(cmap='gray')

        # set axis off
        plt.axis('off')
        # set xlim and ylim
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.tight_layout()
        # save image to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
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

        feature_img_dict = {k : feature_img_dict[k] for k in feature_list}
        building_matrix = np.stack(list(feature_img_dict.values()), axis=-1)
        np.save(os.path.join(os.path.dirname(output_filename), 'building.npy'), building_matrix)



    
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.savefig(output_filename, bbox_inches='tight', format='jpg')
    plt.close()

    
    return xlim, ylim


def plot_pop(pop_file, output_filename, fig_size=(10, 10), xlim=None, ylim=None):
    with rasterio.open(pop_file) as src:
        pop = src.read(1)

    
    plt.figure(figsize=fig_size)
    # if xlim and ylim:
    #     plt.xlim(xlim)
    #     plt.ylim(ylim)
    plt.imshow(pop, cmap='terrain')
    plt.axis('off')
    plt.savefig(output_filename)
    plt.close()

def process_city(city, input_root, output_root):
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
            roads_gdf = gpd.read_file(os.path.join(save_path, "road_data.geojson")).to_crs(epsg=3857)
            landuse_gdf = gpd.read_file(os.path.join(save_path, "landuse_data.geojson")).to_crs(epsg=3857)
            buildings_gdf = gpd.read_file(os.path.join(save_path, "buildings_data.geojson")).to_crs(epsg=3857)
            nature_gdf = gpd.read_file(os.path.join(save_path, "nature_data.geojson")).to_crs(epsg=3857)
            
        except Exception as e:
            with open(os.path.join(output_root, 'plotting_img_error.txt'), 'a') as file:
                file.write(f"Error occurred in {city}: {str(e)}\n")
            return 

        xlim, ylim = plot_combined_map(roads_gdf, landuse_gdf, buildings_gdf, nature_gdf, os.path.join(args.output, city, 'combined.jpg'))


        with open(os.path.join(save_path, 'plotting_img_finish.txt'), 'a') as file:
            file.write(f"Success in {city}\n")

    except Exception as e:
        with open(os.path.join(save_path, 'plotting_img_error.txt'), 'a') as file:
            file.write(f"Error occurred in {city}: {str(e)}\n")
        
        return 
    

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    return tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def dump_h5py(path, output_path):
    with h5py.File(output_path, 'w') as f:
        for file in os.listdir(path):
            if file.endswith('.npy'):
                data = np.load(os.path.join(path, file))
                f.create_dataset(file[:-4], data=data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=False, help='Input file', default='/home/admin/workspace/yuyuanhong/code/CityLayout/data/raw/osm/cities')
    parser.add_argument('--output', type=str, required=False, help='Output file', default='/home/admin/workspace/yuyuanhong/code/CityLayout/data/raw/osm/cities')

    args = parser.parse_args()
    root_path = args.input

    cities = os.listdir(root_path)
    # todo 还需要统一一下labels
    for city in tqdm.tqdm(cities, desc='Validating data'):
        geo_data_validation(os.path.join(root_path, city))

    print('Validation and initialize completed. Geo data total size:', len(os.listdir(args.input)))

    # city_test_name = 'Zurich-7'

    # process_city(city_test_name, args.input, args.output)

    # exit(0)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_city, city, args.input, args.output) for city in cities]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(cities), desc='Processing cities'):
            future.result()

    print('Processing completed.')

    for city in tqdm.tqdm(cities, desc='Validating images'):
        image_data_validation(os.path.join(args.output, city))
        
    print('Validation completed. Image data total size:', len(os.listdir(args.output)))


    for city in tqdm.tqdm(cities, desc='Dumping h5py'):
        dump_h5py(os.path.join(args.output, city), os.path.join(args.output, city, city+'.h5'))


