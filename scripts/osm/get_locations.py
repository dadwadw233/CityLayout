# coding: utf-8
from geopy.geocoders import Photon, Nominatim
from geopy.exc import GeocoderTimedOut
import time
import tqdm

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

cities = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    "Austin", "Fort Worth", "Columbus", "San Francisco", "Indianapolis",
    "Seattle", "Denver", "Washington", "Boston", "Nashville",
    "Detroit", "Portland", "Las Vegas", "Memphis", "Baltimore",
    "Milwaukee", "Albuquerque", "Tucson", "Sacramento", "Atlanta",
    "Miami", "Raleigh", "Minneapolis", "Tampa", "Cincinnati",
    "Anchorage", "Pittsburgh", "Honolulu", "Santa Ana", "Virginia Beach",
    "Oakland", "Omaha", "Long Beach", "Mesa", "Tulsa",
    "Arlington", "Cleveland", "Aurora", "Anaheim", "Santa Clarita",
    "Riverside", "Lexington", "Stockton", "St. Louis", "Saint Paul",
    "Pittsburgh", "Columbus", "Overland Park", "Tempe", "McKinney",
    "Cape Coral", "Shreveport", "Knoxville", "Worcester", "Vancouver",
    "Fort Lauderdale", "Sioux Falls", "Ontario", "Providence",
    "Newport News", "Santa Rosa", "Oceanside", "Elk Grove", "Garden Grove",
    "Pembroke Pines", "Peoria", "Eugene", "Corona", "Cary",
    "Springfield", "Fort Collins", "Alexandria", "Hayward", "Lancaster",
    "Lakewood", "Clarksville", "Palmdale", "Salinas", "Springfield",
    "Pasadena", "Sunnyvale", "Macon", "Pomona", "Escondido",
    "Naperville", "Joliet", "Bellevue", "Rockford", "Savannah",
    "Paterson", "Torrance", "Bridgeport", "McAllen", "Mesquite",
    "Syracuse", "Midland", "Pasadena", "Murfreesboro", "Miramar",
    "Dayton", "Fullerton", "Olathe", "Thornton", "Roseville",
    "Denton", "Waco", "Surprise", "Carrollton", "Charleston",
    "Warren", "Hampton", "Gainesville", "Visalia", "Columbia",
    "Cedar Rapids", "Sterling Heights", "New Haven", "Barcelona",
    "Milan", "Munich", "Prague", "Vienna", "Budapest",
    "Amsterdam", "Warsaw", "Brussels", "Bucharest", "Saint Petersburg",
    "Hamburg", "Copenhagen", "Stockholm", "Kiev", "Athens",
    "Dublin", "Lisbon", "Manchester", "Helsinki", "Sofia",
    "Bratislava", "Cologne", "Naples", "Rotterdam", "Turin",
    "Marseille", "Zagreb", "Krakow", "Frankfurt", "Seville",
    "Oslo", "The Hague", "Düsseldorf", "Athens", "Palermo",
    "Rotterdam", "Genoa", "Helsinki", "Stuttgart", "Zurich",
    "Dortmund", "Málaga", "Leipzig", "Dresden", "Hanover",
    "Gothenburg", "Dublin", "Antwerp", "Bremen", "Sheffield",
    "Edinburgh", "Leeds", "Nuremberg", "Duisburg", "Alicante",
    "Bristol", "Glasgow", "Lyon", "Valencia", "Bologna",
    "Toronto", "Montreal", "Vancouver", "Calgary", "Ottawa",
    "Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide",
    "London", "Manchester", "Birmingham", "Glasgow", "Edinburgh",
    "Dublin", "Belfast", "Copenhagen", "Oslo", "Stockholm",
    "Berlin", "Munich", "Hamburg", "Frankfurt", "Düsseldorf",
    "Paris", "Marseille", "Lyon", "Nice", "Amsterdam",
    "Rotterdam", "The Hague", "Brussels", "Antwerp", "Ghent",
    "Vienna", "Salzburg", "Innsbruck", "Zurich", "Geneva",
    "Rome", "Milan", "Naples", "Florence", "Venice",
    "Madrid", "Barcelona", "Valencia", "Seville", "Bilbao",
    "Lisbon", "Porto", "Budapest", "Prague", "Krakow",
    "Warsaw", "Gdansk", "Helsinki", "Tampere", "Turku",
    "Athens", "Thessaloniki", "Istanbul", "Ankara", "Izmir",
    "Dubai", "Abu Dhabi", "Doha", "Kuwait City", "Riyadh",
    "Tokyo", "Osaka", "Kyoto", "Nagoya", "Yokohama",
    "Shanghai", "Beijing", "Guangzhou", "Shenzhen", "Hong Kong",
    "Seoul", "Busan", "Incheon", "Daegu", "Jeju",
    "Bangkok", "Chiang Mai", "Phuket", "Pattaya", "Hanoi",
    "Ho Chi Minh City", "Da Nang", "Hue", "Nha Trang", "Phnom Penh",
    "Siem Reap", "Vientiane", "Luang Prabang", "Yangon", "Bagan",
    "Kuala Lumpur", "Penang", "Johor Bahru", "Langkawi", "Manila",
    "Cebu City", "Makati", "Quezon City", "Jakarta", "Bali",
    "Singapore", "Sentosa", "Marina Bay Sands", "Kolkata", "Mumbai",
    "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune",
    "Sao Paulo", "Rio de Janeiro", "Buenos Aires", "Lima", "Bogotá",
    "Santiago", "Caracas", "Brasília", "Medellín", "Monterrey",
    "Guadalajara", "Belém", "Porto Alegre", "Recife", "Puebla",
    "Curitiba", "Fortaleza", "Manaus", "León", "Quito",
    "Salvador", "Maracaibo", "Barranquilla", "Santa Cruz de la Sierra", "Guayaquil",
    "Ciudad Juárez", "Tegucigalpa", "San José", "San Salvador", "San Pedro Sula"
]





cities = list(set(cities))
landmarks = [
    "Statue of Liberty, USA", "Empire State Building, USA", "Central Park, USA", "Times Square, USA", "Rockefeller Center, USA",
    "Golden Gate Bridge, USA", "Alcatraz Island, USA", "Fisherman's Wharf, USA", "Palace of Fine Arts, USA", "Golden Gate Park, USA",
    "White House, USA", "Lincoln Memorial, USA", "Washington Monument, USA", "Capitol Hill, USA", "Jefferson Memorial, USA",
    "Mount Rushmore National Memorial, USA", "Yellowstone National Park, USA", "Grand Canyon National Park, USA", "Yosemite National Park, USA", "Great Smoky Mountains National Park, USA",
    "Walt Disney World Resort, USA", "Universal Studios Hollywood, USA", "The Las Vegas Strip, USA", "Kennedy Space Center Visitor Complex, USA", "National Mall, USA",
    "Niagara Falls, USA/Canada", "The Freedom Trail, USA", "The Grand Ole Opry, USA", "The Alamo, USA", "Kennedy Space Center Visitor Complex, USA",
    "Mount Vernon, USA", "Independence Hall, USA", "Gettysburg National Military Park, USA", "National World War II Memorial, USA", "Hollywood Walk of Fame, USA",
    "Bryce Canyon National Park, USA", "Zion National Park, USA", "Acadia National Park, USA", "Everglades National Park, USA", "Haleakalā National Park, USA",
    "Glacier National Park, USA", "Denali National Park and Preserve, USA", "Arches National Park, USA", "Death Valley National Park, USA", "Biscayne National Park, USA",
    "Gateway Arch National Park, USA", "Mount St. Helens National Volcanic Monument, USA", "Yellowstone National Park, USA", "Grand Tetons National Park, USA", "Badlands National Park, USA",
    "Great Sand Dunes National Park and Preserve, USA", "Rocky Mountain National Park, USA", "Carlsbad Caverns National Park, USA", "Shenandoah National Park, USA", "Olympic National Park, USA",
    "Christ the Redeemer (Cristo Redentor), Brazil", "Machu Picchu, Peru", "Iguaçu Falls, Argentina/Brazil", "Galápagos Islands, Ecuador", "Amazon Rainforest, Multiple Countries",
    "Angel Falls (Salto Ángel), Venezuela", "Salar de Uyuni, Bolivia", "Easter Island (Rapa Nui), Chile", "Patagonia, Argentina/Chile", "Atacama Desert, Chile",
    "Torres del Paine National Park, Chile", "Mendoza Wine Region, Argentina", "Amazon River, Multiple Countries", "Aconcagua, Argentina", "Pantanal Wetlands, Brazil",
    "Angel Falls (Salto Ángel), Venezuela", "Margarita Island, Venezuela", "Cotopaxi, Ecuador", "Manú National Park, Peru", "Ecuador's Avenue of the Volcanoes, Ecuador", "Ilha Grande, Brazil",
    "São Paulo Museum of Art (MASP), Brazil", "Lencois Maranhenses National Park, Brazil", "Los Roques Archipelago, Venezuela", "Ushuaia, Argentina", "Huacachina, Peru",
    "Ecuador's Cajas National Park, Ecuador", "Tayrona National Natural Park, Colombia", "Colca Canyon, Peru", "Valle de la Luna (Moon Valley), Bolivia", "Chiloé Island, Chile",
    "Ecuador's Quilotoa Lake, Ecuador", "Cano Cristales, Colombia", "Maracanã Stadium, Brazil", "Amazon River, Multiple Countries", "Iberá Wetlands, Argentina", "Galápagos Islands, Ecuador",
    "Eiffel Tower, France", "Louvre Museum, France", "Arc de Triomphe, France", "Palace of Versailles, France", "Notre-Dame de Paris, France",
    "Leaning Tower of Pisa, Italy", "Colosseum, Italy", "Vatican City, Vatican", "Pantheon, Italy", "Trevi Fountain, Italy",
    "Acropolis of Athens, Greece", "Parthenon, Greece", "Ancient Agora of Athens, Greece", "Odeon of Herodes Atticus, Greece", "Mount Olympus, Greece",
    "Sagrada Família, Spain", "Park Güell, Spain", "Casa Batlló, Spain", "Camp Nou, Spain", "Tibidabo, Spain",
    "Amsterdam Canal Ring, Netherlands", "Rijksmuseum, Netherlands", "Van Gogh Museum, Netherlands", "Anne Frank House, Netherlands", "Zaanse Schans, Netherlands",
    "Prague Castle, Czech Republic", "Charles Bridge, Czech Republic", "Old Town Square, Czech Republic", "Astronomical Clock, Czech Republic", "Josefov (Jewish Quarter), Czech Republic",
    "Buda Castle, Hungary", "Chain Bridge, Hungary", "Hungarian Parliament Building, Hungary", "Fisherman's Bastion, Hungary", "Gellért Hill, Hungary",
    "Edinburgh Castle, United Kingdom", "Royal Mile, United Kingdom", "Palace of Holyroodhouse, United Kingdom", "National Museum of Scotland, United Kingdom", "Arthur's Seat, United Kingdom",
    "Trinity College Dublin, Ireland", "Dublin Castle, Ireland", "Guinness Storehouse, Ireland", "St. Patrick's Cathedral, Ireland", "Temple Bar, Ireland",
    "Giant's Causeway, United Kingdom", "Titanic Belfast, United Kingdom", "Ulster Museum, United Kingdom", "Botanic Gardens, United Kingdom", "Crumlin Road Gaol, United Kingdom",
    "Copenhagen Opera House, Denmark", "Tivoli Concert Hall, Denmark", "Church of Our Saviour, Denmark", "Christiania, Denmark", "Frederik's Church (The Marble Church), Denmark",
    "Neuschwanstein Castle, Germany", "Brandhorst Museum, Germany", "BMW Welt, Germany", "Olympiapark, Germany", "Hofbräuhaus München, Germany",
    "Hamburg Speicherstadt, Germany", "Miniatur Wunderland, Germany", "Elbphilharmonie, Germany", "Kunsthalle Hamburg, Germany", "St. Nicholas' Church, Germany",
    "Museum Island, Germany", "Berlin Cathedral, Germany", "Charlottenburg Palace, Germany", "Berlin State Opera, Germany", "Berlin Victory Column, Germany",
    "La Tour Eiffel, France", "Musée du Louvre, France", "Arc de Triomphe de l'Étoile, France", "Notre-Dame de Paris, France", "Château de Versailles, France",
    "Colosseo, Italy", "Vaticano, Vatican", "Pantheon, Italy", "Fontana di Trevi, Italy", "Piazza di Spagna, Italy",
    "Akropolis, Greece", "Parthenónas, Greece", "Agorá tou Athinai, Greece", "Ieros Naos tou Iródou tou Attikou, Greece", "Ólympos, Greece",
    "Great Wall of China, China", "Forbidden City, China", "Terracotta Army, China", "Summer Palace, China", "Temple of Heaven, China",
    "Taj Mahal, India", "Agra Fort, India", "Amber Fort, India", "Hawa Mahal, India", "Qutub Minar, India",
    "Petra, Jordan", "Wadi Rum, Jordan", "Jerash, Jordan", "Dead Sea, Jordan", "Mount Nebo, Jordan",
    "Pyramids of Giza, Egypt", "Great Sphinx of Giza, Egypt", "Karnak, Egypt", "Valley of the Kings, Egypt", "Luxor Temple, Egypt",
    "Angkor Wat, Cambodia", "Bayon Temple, Cambodia", "Ta Prohm, Cambodia", "Borobudur, Indonesia", "Prambanan, Indonesia",
    "Mount Fuji, Japan", "Kyoto Imperial Palace, Japan", "Kinkaku-ji (Golden Pavilion), Japan", "Kiyomizu-dera, Japan", "Fushimi Inari Taisha, Japan",
    "Terracotta Army, China", "Huangshan (Yellow Mountain), China", "Zhangjiajie National Forest Park, China", "Potala Palace, Tibet", "Jiuzhaigou Valley, China",
    "Mount Everest, Nepal", "Tigers Nest Monastery (Paro Taktsang), Bhutan", "Great Wall of China, China", "Forbidden City, China", "Terracotta Army, China",
    "Petra, Jordan", "Wadi Rum, Jordan", "Jerash, Jordan", "Dead Sea, Jordan", "Mount Nebo, Jordan",
    "Pyramids of Giza, Egypt", "Great Sphinx of Giza, Egypt", "Karnak, Egypt", "Valley of the Kings, Egypt", "Luxor Temple, Egypt",
    "Angkor Wat, Cambodia", "Bayon Temple, Cambodia", "Ta Prohm, Cambodia", "Borobudur, Indonesia", "Prambanan, Indonesia",
    "Mount Fuji, Japan", "Kyoto Imperial Palace, Japan", "Kinkaku-ji (Golden Pavilion), Japan", "Kiyomizu-dera, Japan", "Fushimi Inari Taisha, Japan",
    "Terracotta Army, China", "Huangshan (Yellow Mountain), China", "Zhangjiajie National Forest Park, China", "Potala Palace, Tibet", "Jiuzhaigou Valley, China",
    "Mount Everest, Nepal", "Tigers Nest Monastery (Paro Taktsang), Bhutan", "Great Wall of China, China", "Forbidden City, China", "Terracotta Army, China",
    "Petra, Jordan", "Wadi Rum, Jordan", "Jerash, Jordan", "Dead Sea, Jordan", "Mount Nebo, Jordan",
    "Pyramids of Giza, Egypt", "Great Sphinx of Giza, Egypt", "Karnak, Egypt", "Valley of the Kings, Egypt", "Luxor Temple, Egypt",
    "Al-Haram Mosque (Mecca), Saudi Arabia", "Madain Saleh, Saudi Arabia", "Wadi Rum, Jordan", "Jerash, Jordan", "Dead Sea, Jordan",
    "Sydney Opera House, Australia","Great Barrier Reef, Australia","Christchurch Botanic Gardens, New Zealand","Machu Picchu, Peru",
    "Santorini, Greece","Venice Canals, Italy","Neuschwanstein Castle, Germany","Sagrada Família, Spain","Kremlin, Russia",
    "Petronas Towers, Malaysia","Borobudur, Indonesia","Angkor Wat, Cambodia","Ha Long Bay, Vietnam","The Great Wall of China, China",
    "Mount Fuji, Japan","The Dead Sea, Jordan","Pyramids of Giza, Egypt","Victoria Falls, Zambia/Zimbabwe","Mount Everest, Nepal",
    "Taj Mahal, India","Chichen Itza, Mexico","The Colosseum, Italy","The Louvre, France","Stonehenge, United Kingdom",
    "Easter Island (Rapa Nui), Chile","Red Square, Russia","Petra, Jordan","The Grand Canyon, USA","Niagara Falls, USA/Canada",
    "The Great Sphinx of Giza, Egypt","Acropolis of Athens, Greece","The Parthenon, Greece","The Pantheon, Italy","The Trevi Fountain, Italy",
    "The Leaning Tower of Pisa, Italy","The Vatican City, Vatican","The Eiffel Tower, France","The Arc de Triomphe, France","The Palace of Versailles, France",
    "The Notre-Dame de Paris, France","The Louvre Museum, France","The Colosseo, Italy","The Vaticano, Vatican","The Piazza di Spagna, Italy",
    "The Akropolis, Greece","The Parthenónas, Greece","The Ieros Naos tou Iródou tou Attikou, Greece","The Ólympos, Greece",
    "The Pyramids of Giza, Egypt","The Karnak, Egypt","The Valley of the Kings, Egypt","The Luxor Temple, Egypt",
    "The Al-Haram Mosque (Mecca), Saudi Arabia","The Madain Saleh, Saudi Arabia",
]

landmarks = list(set(landmarks))

radius = 1000;
geo_redius = (radius/1000) / 111.319444
dx = [0,0,-1,-1,1,1,-1,1]
dy = [-1,1,-1,1,-1,1,0,0]

with open("city_coordinates.txt", "w") as file:

    
    for city in tqdm.tqdm(cities, desc='Processing cities'):
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
                    print(f'{city} not found')
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
    for landmark in tqdm.tqdm(landmarks, desc='Processing landmarks'):
        retries = 10
        while True:
            try:
                lat, lon = get_city_coordinates_withproxy(landmark)
                if lat and lon:
                    file.write(f'{landmark.replace(" ", "-").replace(",", "")} {lat} {lon}\n')
                    for i in range(8):
                        lat2 = lat + geo_redius * dx[i]
                        lon2 = lon + geo_redius * dy[i]
                        file.write(f'{landmark.replace(" ", "-").replace(",", "")}-{i} {lat2} {lon2}\n')
                else:
                    print(f'{landmark} not found')
                # time.sleep(0.5)  
                break
            except:
                print(f"Error processing {landmark}, retrying...")
                retries -= 1
                time.sleep(10)
                if retries == 0:
                    break

print("Coordinates extraction completed.")
