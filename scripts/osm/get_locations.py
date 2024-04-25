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
    "config/data/city_landmark.yaml"
)
cities = config["name"]["cities"]
landmarks = config["name"]["landmarks"]

cities = list(cities)


landmarks = list(set(landmarks))

radius = 128
geo_redius = (radius / 1000) / 111.319444
padding = 10


with open("city_coordinates.txt", "w") as file:
    for city in tqdm.tqdm(cities, desc="Processing cities"):
        retries = 10
        while True:
            try:
                lat, lon = get_city_coordinates_withproxy(city)
                if lat and lon:
                    # file.write(f'{city.replace(" ", "")} {lat} {lon}\n')
                    min_lat = lat - geo_redius * padding * 2
                    min_lon = lon - geo_redius * padding * 2
                    for i in range(padding * 2 + 1):
                        for j in range(padding * 2 + 1):
                            lat2 = min_lat + geo_redius * i * 2
                            lon2 = min_lon + geo_redius * j * 2
                            file.write(f'{city.replace(" ", "")}-{i}-{j} {lat2} {lon2}\n')
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

