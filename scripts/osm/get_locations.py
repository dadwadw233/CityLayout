# coding: utf-8
from geopy.geocoders import Photon, Nominatim
from geopy.exc import GeocoderTimedOut
import time
import tqdm
import yaml


def get_city_coordinates(city_name):
    geolocator = Photon(user_agent="measurements")
    location = geolocator.geocode(city_name)
    if location:
        return (location.latitude, location.longitude)
    else:
        return (None, None)


def get_city_coordinates_withproxy(city_name):
    geolocator = Nominatim(user_agent="city_locator")
    try:
        location = geolocator.geocode(city_name)
        if location:
            return (location.latitude, location.longitude)
        else:
            return (None, None)
    except GeocoderTimedOut:
        return (None, None)


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


config = load_config(
    "/home/admin/workspace/yuyuanhong/code/CityLayout/config/data/city_landmark.yaml"
)
cities = config["name"]["cities"]
landmarks = config["name"]["landmarks"]

cities = list(set(cities))


landmarks = list(set(landmarks))

radius = 1000
geo_redius = (radius / 1000) / 111.319444
dx = [0, 0, -1, -1, 1, 1, -1, 1]
dy = [-1, 1, -1, 1, -1, 1, 0, 0]

with open("city_coordinates.txt", "w") as file:
    for city in tqdm.tqdm(cities, desc="Processing cities"):
        retries = 10
        while True:
            try:
                lat, lon = get_city_coordinates_withproxy(city)
                if lat and lon:
                    file.write(f'{city.replace(" ", "")} {lat} {lon}\n')
                    for i in range(8):
                        lat2 = lat + geo_redius * dx[i]
                        lon2 = lon + geo_redius * dy[i]
                        file.write(f'{city.replace(" ", "")}-{i} {lat2} {lon2}\n')
                else:
                    print(f"{city} not found")
                # time.sleep(0.5)
                break
            except:
                print(f"Error processing {city}, retrying...")
                time.sleep(10)
                retries -= 1
                if retries == 0:
                    break

print("Coordinates extraction completed.")


with open("landmark_coordinates.txt", "w") as file:
    for landmark in tqdm.tqdm(landmarks, desc="Processing landmarks"):
        retries = 10
        while True:
            try:
                lat, lon = get_city_coordinates_withproxy(landmark)
                if lat and lon:
                    file.write(
                        f'{landmark.replace(" ", "-").replace(",", "")} {lat} {lon}\n'
                    )
                    for i in range(8):
                        lat2 = lat + geo_redius * dx[i]
                        lon2 = lon + geo_redius * dy[i]
                        file.write(
                            f'{landmark.replace(" ", "-").replace(",", "")}-{i} {lat2} {lon2}\n'
                        )
                else:
                    print(f"{landmark} not found")
                # time.sleep(0.5)
                break
            except:
                print(f"Error processing {landmark}, retrying...")
                retries -= 1
                time.sleep(10)
                if retries == 0:
                    break

print("Coordinates extraction completed.")
