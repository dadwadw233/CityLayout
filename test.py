import osmnx as ox
import matplotlib.pyplot as plt

from elevation import get_elevation

lat, long = 39.92, 116.38  # 例子中的经纬度
elevation = get_elevation(lat, long)
print(elevation)