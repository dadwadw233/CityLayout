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

def geo_data_validation(path):

    if os.path.exists(os.path.join(path, 'getdata_error.txt')):
        logging.error(f"Error occurred in {path}.")
        os.system(f"rm -rf {path}")


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




        

def plot_and_save_geojson(file_name, out_root, plot_type, xlim=None, ylim=None, fig_size=(10, 10)):

    gdf = gpd.read_file(file_name)
    gdf = gdf.to_crs(epsg=3857)
    
    # todo 更据full image的大小调整fig_size
    
    fig, ax = plt.subplots(figsize=fig_size)
    if xlim and ylim:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    ax.axis('off')

    if plot_type == 'landuse':
        gdf['area'] = gdf.geometry.area

        # 根据面积对地块进行排序（默认是升序，即小的在前）
        gdf_sorted = gdf.sort_values(by='area', ascending=True)


        # 绘制排序后的地块，使用透明度参数
        gdf_sorted.plot(ax=ax, column='landuse', cmap='Accent', alpha=0.5, legend=True)

        fig.savefig(os.path.join(out_root, 'landuse.jpg'), bbox_inches='tight', format='jpg')
        plt.close(fig)

    elif plot_type == 'building':

        # 位置图
        gdf.plot(ax=ax, color='grey', edgecolor='black')
        
        fig.savefig(os.path.join(out_root, 'building_location.jpg'), bbox_inches='tight', format='jpg')

        # 清除当前的ax和fig，为高度图重新创建
        plt.clf()
        plt.close(fig)
        fig, ax = plt.subplots(figsize=fig_size)
        if xlim and ylim:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        ax.axis('off')


        # 判断'height'字段是否存在
        if 'height' not in gdf.columns:
            raise KeyError("Column 'height' not found in the data.")

        
        # 高度图
        gdf.plot(ax=ax, column='height', cmap='viridis')
        fig.savefig(os.path.join(out_root, 'building_height.jpg'), bbox_inches='tight', format='jpg')
        plt.close(fig)

    elif plot_type == 'road':
        # 判断'highway'字段是否存在
        if 'highway' not in gdf.columns:
            raise KeyError("Column 'highway' not found in the data.")

        gdf.plot(ax=ax, column='highway', cmap='tab20', legend=True)
        fig.savefig(os.path.join(out_root, 'road.jpg'), bbox_inches='tight', format='jpg')
        plt.close(fig)

    elif plot_type == 'nature':
        # 创建组合列，仅当相关列存在时
        nature_cols = ['natural', 'water', 'waterway']
        for col in nature_cols:
            if col in gdf.columns:
                gdf[col] = gdf[col].fillna('')
            else:
                gdf[col] = ''
        gdf['nature_sum'] = gdf[nature_cols].agg('_'.join, axis=1)

    
        gdf.plot(ax=ax, column='nature_sum', cmap='Set3', legend=True)
        fig.savefig(os.path.join(out_root, 'nature.jpg'), bbox_inches='tight', format='jpg')
        plt.close(fig)

        
            
        


def plot_combined_map(roads_gdf, landuse_gdf, buildings_gdf, nature_gdf, output_filename, fig_size=(10, 10)):
    fig, ax = plt.subplots(figsize=fig_size)
    combined_gdf = gpd.GeoDataFrame(pd.concat([roads_gdf, landuse_gdf, buildings_gdf, nature_gdf], ignore_index=True), crs=roads_gdf.crs)
    xlim = (combined_gdf.total_bounds[0], combined_gdf.total_bounds[2])  # Minx, Maxx
    ylim = (combined_gdf.total_bounds[1], combined_gdf.total_bounds[3])  # Miny, Maxy
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    # 如果有landuse数据，先绘制landuse层
    if not landuse_gdf.empty:
        landuse_gdf['area'] = landuse_gdf.geometry.area

        
        landuse_gdf = landuse_gdf.sort_values(by='area', ascending=True)
        landuse_gdf.plot(ax=ax, column='landuse', cmap='Accent', alpha=0.5)

        # 绘制单独的landuse层
        color_dict = {'commercial': '#EFCAC0', 'retail': '#EFCAC0', 'education': '#EF826B', 'industrial':'#D6CCED',
                      'depot':'#D6CCED', 'port':'#D6CCED', 'residual':'#E5EBE8', 'farmland':'#DFF58D', 'meadow':'#DFF58D',
                      'orchard':'#DFF58D', 'vineyard':'#DFF58D', 'plant_nursery':'#DFF58D', 'forest':'#16F53E',
                      'farmyard': '#F0CB60', 'grass': '#B4F59D', 'greenfield':'#B4F59D', 'military':'#EF4631', 'railway':'#B884F0',
                      'recreation_ground':'#F07D00', 'fairground':'#F07D00', 'default':'#8F8F8F'}
        
        
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
        fig_.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.savefig(os.path.join(os.path.dirname(output_filename), 'landuse.jpg'), bbox_inches='tight', format='jpg')

        plt.close(fig_)


    
    if not nature_gdf.empty:
        
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

        fig_, ax_ = plt.subplots(figsize=fig_size)
        ax_.set_xlim(xlim)
        ax_.set_ylim(ylim)
        ax_.axis('off')
        
        for nature_type in nature_gdf['natural'].unique():
            
            gdf_type = nature_gdf[nature_gdf['natural'] == nature_type]

            if nature_type in color_dict.keys():
                gdf_type.plot(ax=ax_, color=color_dict[nature_type], edgecolor='black')
                gdf_type.plot(ax=ax, color=color_dict[nature_type], edgecolor='black', alpha=0.5)
        fig_.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.savefig(os.path.join(os.path.dirname(output_filename), 'nature.jpg'), bbox_inches='tight', format='jpg')

        plt.close(fig_)

    
    if not roads_gdf.empty:
        
        color_dict = {'motorway': 'red', 'trunk': 'orange', 'primary': 'yellow', 'secondary': 'green',
                        'tertiary': 'pink', 'residential': 'blue', 'service': 'grey'
                        }

        fig_, ax_ = plt.subplots(figsize=fig_size)
        ax_.set_xlim(xlim)
        ax_.set_ylim(ylim)
        ax_.axis('off')

        for road_type in roads_gdf['highway'].unique():
            gdf_type = roads_gdf[roads_gdf['highway'] == road_type]

            if road_type in color_dict.keys():
                gdf_type.plot(ax=ax_, color=color_dict[road_type])
                gdf_type.plot(ax=ax, color=color_dict[road_type], alpha=0.5)

        # roads_gdf.plot(ax=ax_, column='highway', cmap='tab20', legend=True)
        # roads_gdf.plot(ax=ax, column='highway', cmap='tab20')
        fig_.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.savefig(os.path.join(os.path.dirname(output_filename), 'road.jpg'), bbox_inches='tight', format='jpg')
        plt.close(fig_)
        
    
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


        # 处理每种类型的数据
        # plot_and_save_geojson(os.path.join(save_path, "road_data.geojson"), output_path, 'road', xlim, ylim)
        # plot_and_save_geojson(os.path.join(save_path, "landuse_data.geojson"), output_path, 'landuse', xlim, ylim)
        # plot_and_save_geojson(os.path.join(save_path, "buildings_data.geojson"), output_path, 'building', xlim, ylim)
        # plot_and_save_geojson(os.path.join(save_path, "nature_data.geojson"), output_path, 'nature', xlim, ylim)

        # plot_dem(os.path.join(save_path, 'dem_data.tif'), os.path.join(output_path, 'dem.jpg'), xlim=xlim, ylim=ylim)
        # plot_pop(os.path.join(save_path, 'pop_data.tif'), os.path.join(output_path, 'pop.jpg'), xlim=xlim, ylim=ylim)

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



def generate_trainning_data(path, color_to_label, type='nature'):

    original_img = np.array(Image.open(os.path.join(path, type + '.jpg')))

    # print(original_img.shape)
    # 初始化二值图像字典
    binary_images = {label: np.zeros(original_img.shape[:2], dtype=np.uint8) for label in color_to_label.values()}

    # 遍历原始图像的每个像素
    for i,j in np.ndindex(original_img.shape[:2]):
        pixel = original_img[i, j]
        
        pixel_tuple = tuple(pixel)
        
        if pixel_tuple in color_to_label.keys():
            label = color_to_label[pixel_tuple]
            # if (tuple(pixel)!=(191,191,191) and tuple(pixel)!=(255,255,255)):
            #     print(pixel)
            binary_images[label][i, j] = 1
    
    
    multi_channel_img = np.stack(list(binary_images.values()), axis=-1)

    num_channels = multi_channel_img.shape[2]
    
    if(num_channels == 1):
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(multi_channel_img[:, :, 0], cmap='gray')  # 使用灰度图展示
        ax.set_title(f'Channel {0+1}')
        ax.axis('off')
        
    else: 
        fig, axes = plt.subplots(1, num_channels, figsize=(20, 5))  # 调整大小

        for i in range(num_channels):
            ax = axes[i]
            ax.imshow(multi_channel_img[:, :, i], cmap='gray')  # 使用灰度图展示
            ax.set_title(f'Channel {i+1}')
            ax.axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(path, type + '_multi_channel.jpg'))
    print(np.count_nonzero(multi_channel_img))
    print(path)
    exit(0)
    np.save(os.path.join(path, type + '.npy'), multi_channel_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=False, help='Input file', default='/home/admin/workspace/yuyuanhong/code/CityLayout/data/raw/osm/cities')
    parser.add_argument('--output', type=str, required=False, help='Output file', default='/home/admin/workspace/yuyuanhong/code/CityLayout/data/raw/osm/cities')

    args = parser.parse_args()
    root_path = args.input

    cities = os.listdir(root_path)
    #todo 还需要统一一下labels
    # for city in tqdm.tqdm(cities, desc='Validating data'):
    #     geo_data_validation(os.path.join(root_path, city))

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(process_city, city, args.input, args.output) for city in cities]
    #     for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(cities), desc='Processing cities'):
    #         future.result()

    # print('Processing completed.')

    # for city in tqdm.tqdm(cities, desc='Validating images'):
    #     image_data_validation(os.path.join(args.output, city))
        
    # print('Validation completed. Image data total size:', len(os.listdir(args.output)))

    for city in tqdm.tqdm(cities, desc='Generating training data'):
        
        # nature_color_to_label = {
        #     "#A8EB83": "grassland",
        #     "#1CEF26": "tree",
        #     "#D5E4ED": "beach",
        #     "#418DF0": "water",
        #     "#51D5EB": "wetland",
        #     "#E5F2D3": "bare_rock",
        #     "#CAF582": "hill",
        #     "#EDE6B0": "sand",
        #     "#F0BA60": "valley",
        #     "#BFBFBF": "default"
        # }
        # nature_color_to_label = {hex_to_rgb(color): label for color, label in nature_color_to_label.items()}
        # # print(nature_color_to_label)
        # # exit(0)
        # generate_trainning_data(os.path.join(args.output, city), nature_color_to_label)

        building_color_to_label = {
            "#808080": "building",
        }
        building_color_to_label = {hex_to_rgb(color): label for color, label in building_color_to_label.items()}
        generate_trainning_data(os.path.join(args.output, city), building_color_to_label, type='building_location')




