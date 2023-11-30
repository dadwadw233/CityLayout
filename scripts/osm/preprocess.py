import argparse
import osmnx as ox
import os
import tqdm
import geopandas as gpd
import matplotlib.pyplot as plt

def plot_and_save_geojson(file_name, out_root, plot_type, xlim=None, ylim=None, fig_size=(10, 10)):
    try:
        gdf = gpd.read_file(file_name)

        
        
        fig, ax = plt.subplots(figsize=fig_size)
        ax.axis('off')

        if plot_type == 'landuse':
            gdf.plot(ax=ax, column='landuse', cmap='Set3', legend=True)
            fig.savefig(os.path.join(out_root, 'landuse.jpg'), bbox_inches='tight', format='jpg')

        elif plot_type == 'building':

            # 位置图
            gdf.plot(ax=ax, color='grey')
            if xlim and ylim:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            fig.savefig(os.path.join(out_root, 'building_location.jpg'), bbox_inches='tight', format='jpg')

            # 清除当前的ax和fig，为高度图重新创建
            plt.clf()
            fig, ax = plt.subplots(figsize=fig_size)
            ax.axis('off')


            # 判断'height'字段是否存在
            if 'height' not in gdf.columns:
                raise KeyError("Column 'height' not found in the data.")

            
            # 高度图
            gdf.plot(ax=ax, column='height', cmap='viridis')
            fig.savefig(os.path.join(out_root, 'building_height.jpg'), bbox_inches='tight', format='jpg')

        elif plot_type == 'road':
            # 判断'highway'字段是否存在
            if 'highway' not in gdf.columns:
                raise KeyError("Column 'highway' not found in the data.")

            gdf.plot(ax=ax, column='highway', cmap='tab20', legend=True)
            fig.savefig(os.path.join(out_root, 'road.jpg'), bbox_inches='tight', format='jpg')

        elif plot_type == 'nature':
            # 创建组合列，仅当相关列存在时
            nature_cols = ['natural', 'water', 'waterway']
            for col in nature_cols:
                if col in gdf.columns:
                    gdf[col] = gdf[col].fillna('')
                else:
                    gdf[col] = ''
            gdf['nature_sum'] = gdf[nature_cols].agg('_'.join, axis=1)

            fig, ax = plt.subplots(figsize=fig_size)
            ax.axis('off')
            gdf.plot(ax=ax, column='nature_sum', cmap='Set3', legend=True)
            fig.savefig(os.path.join(out_root, 'nature.jpg'), bbox_inches='tight', format='jpg')

    except Exception as e:
        # 输出错误信息到baddata.txt
        with open(os.path.join(out_root, 'baddata.txt'), 'w') as file:
            file.write(str(e))
        


    







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input file')
    parser.add_argument('--output', type=str, required=True, help='Output file')

    args = parser.parse_args()
    root_path = args.input

    citys = os.listdir(root_path)

    for city in tqdm.tqdm(citys):

        save_path = os.path.join(args.input, city)
        print(save_path)
        # 提取道路网络
        plot_and_save_geojson(os.path.join(save_path, "road_data.geojson"), os.path.join(args.output, city), 'road')
        # 提取土地利用数据
        plot_and_save_geojson(os.path.join(save_path, "landuse_data.geojson"), os.path.join(args.output, city), 'landuse')
        # 提取建筑物数据
        plot_and_save_geojson(os.path.join(save_path, "buildings_data.geojson"), os.path.join(args.output, city), 'building')

        # 提取自然地物数据
        plot_and_save_geojson(os.path.join(save_path, "nature_data.geojson"), os.path.join(args.output, city), 'nature')



