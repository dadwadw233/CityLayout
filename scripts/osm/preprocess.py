import argparse
import osmnx as ox


import geopandas as gpd
import matplotlib.pyplot as plt

def plot_and_save_geojson(file_name, output_image_name, plot_type, xlim=None, ylim=None):
    gdf = gpd.read_file(file_name)
    if plot_type == 'landuse':
        gdf['landuse_sum'] = gdf['landuse'].fillna('') + '_' + gdf['leisure'].fillna('') + '_' + gdf['natural'].fillna('') + '_' + gdf['water'].fillna('') + '_' + gdf['waterway'].fillna('')
    fig, ax = plt.subplots()

    if plot_type == 'landuse':
        
        gdf.plot(ax=ax, column='landuse_sum', cmap='Set3', legend=True)


    elif plot_type == 'building':
        # 位置图
        gdf.plot(ax=ax, color='grey')
        ax.axis('off')
        if xlim and ylim:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        fig.savefig(f'{output_image_name}_location.jpg', dpi=300, bbox_inches='tight', format='jpg')

        # 高度图
        gdf.plot(ax=ax, column='height', cmap='grey')  # 根据建筑高度着色

    elif plot_type == 'road':
        gdf.plot(ax=ax, column='highway', cmap='tab20')  # 根据道路类型着色

    ax.axis('off')
    if xlim and ylim:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.savefig(f'{output_image_name}.jpg', dpi=300, bbox_inches='tight', format='jpg')







if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input', type=str, required=True, help='Input file')
    # parser.add_argument('--output', type=str, required=True, help='Output file')
    plot_and_save_geojson('road_data.geojson', 'road_data', 'road')
    plot_and_save_geojson('landuse_data.geojson', 'landuse_data', 'landuse')
    plot_and_save_geojson('buildings_data.geojson', 'buildings_data', 'building')



