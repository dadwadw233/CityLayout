# coding: utf-8
from geopy.geocoders import Photon
import time
import tqdm

def get_city_coordinates(city_name):
    geolocator = Photon(user_agent="measurements")
    location = geolocator.geocode(city_name)
    if location:
        return (location.latitude, location.longitude)
    else:
        return (None, None)

cities = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    "Austin", "Jacksonville", "Fort Worth", "Columbus", "Charlotte",
    "San Francisco", "Indianapolis", "Seattle", "Denver", "Washington",
    "Boston", "El Paso", "Nashville", "Detroit", "Oklahoma City",
    "Portland", "Las Vegas", "Memphis", "Louisville", "Baltimore",
    "Milwaukee", "Albuquerque", "Tucson", "Fresno", "Mesa",
    "Sacramento", "Atlanta", "Kansas City", "Colorado Springs", "Miami",
    "Raleigh", "Omaha", "Long Beach", "Virginia Beach", "Oakland",
    "Minneapolis", "Tulsa", "Arlington", "Tampa", "New Orleans",
    "Wichita", "Cleveland", "Bakersfield", "Aurora", "Anaheim",
    "Honolulu", "Santa Ana", "Riverside", "Corpus Christi", "Lexington",
    "Stockton", "St. Louis", "Saint Paul", "Henderson", "Pittsburgh",
    "Cincinnati", "Anchorage", "Greensboro", "Plano", "Newark",
    "Lincoln", "Orlando", "Irvine", "Toledo", "Jersey City",
    "Chula Vista", "Durham", "Fort Wayne", "St. Petersburg", "Laredo",
    "Buffalo", "Madison", "Lubbock", "Chandler", "Scottsdale",
    "Reno", "Glendale", "Gilbert", "Winston–Salem", "North Las Vegas",
    "Norfolk", "Chesapeake", "Garland", "Irving", "Hialeah",
    "Fremont", "Boise", "Richmond", "Baton Rouge", "Spokane",
    "Des Moines", "Tacoma", "San Bernardino", "Modesto", "Fontana",
    "Santa Clarita", "Birmingham", "Oxnard", "Fayetteville", "Moreno Valley",
    "Rochester", "Glendale", "Huntington Beach", "Salt Lake City", "Grand Rapids",
    "Amarillo", "Yonkers", "Aurora", "Montgomery", "Akron",
    "Little Rock", "Huntsville", "Augusta", "Port St. Lucie", "Grand Prairie",
    "Columbus", "Tallahassee", "Overland Park", "Tempe", "McKinney",
    "Mobile", "Cape Coral", "Shreveport", "Frisco", "Knoxville",
    "Worcester", "Brownsville", "Vancouver", "Fort Lauderdale", "Sioux Falls",
    "Ontario", "Chattanooga", "Providence", "Newport News", "Rancho Cucamonga",
    "Santa Rosa", "Oceanside", "Salem", "Elk Grove", "Garden Grove",
    "Pembroke Pines", "Peoria", "Eugene", "Corona", "Cary",
    "Springfield", "Fort Collins", "Jackson", "Alexandria", "Hayward",
    "Lancaster", "Lakewood", "Clarksville", "Palmdale", "Salinas",
    "Springfield", "Hollywood", "Pasadena", "Sunnyvale", "Macon",
    "Kansas City", "Pomona", "Escondido", "Killeen", "Naperville",
    "Joliet", "Bellevue", "Rockford", "Savannah", "Paterson",
    "Torrance", "Bridgeport", "McAllen", "Mesquite", "Syracuse",
    "Midland", "Pasadena", "Murfreesboro", "Miramar", "Dayton",
    "Fullerton", "Olathe", "Orange", "Thornton", "Roseville",
    "Denton", "Waco", "Surprise", "Carrollton", "West Valley City",
    "Charleston", "Warren", "Hampton", "Gainesville", "Visalia",
    "Coral Springs", "Columbia", "Cedar Rapids", "Sterling Heights", "New Haven",
    "Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Chengdu",
    "Chongqing", "Tianjin", "Wuhan", "Hangzhou", "Nanjing",
    "Xi'an", "Qingdao", "Dalian", "Shenyang", "Suzhou",
    "Harbin", "Jinan", "Zhengzhou", "Changsha", "Kunming",
    "Shijiazhuang", "Changchun", "Ürümqi", "Taiyuan", "Hefei",
    "Nanning", "Guiyang", "Nanchang", "Lanzhou", "Fuzhou",
    "Xiamen", "Ningbo", "Haikou", "Hohhot", "Lhasa",
    "Yinchuan", "Wulumuqi", "Xining", "Yinchuan", "Wulumuqi",
    "Xining", "Taipei", "Hong Kong", "Macau", "Taichung",
    "Kaohsiung", "Tainan", "Hsinchu", "Taoyuan", "Keelung",
    "Chiayi", "Pingtung", "Yilan", "Hualien", "Taitung",
    "Penghu", "Kinmen", "Matsu", "Taiping", "Tonglu",
    "Jiande", "Fuyang", "Lin'an", "Hangzhou", "Shaoxing",
    "Ningbo", "Yuyao", "Cixi", "Yiwu", "Jinhua",
    "Quzhou", "Zhoushan", "Huzhou", "Jiaxing", "Tongxiang",
    "Wenzhou", "Lishui", "Shangrao", "Jingdezhen", "Pingxiang",
    "Jiujiang", "Xinyu", "Yichun", "Ji'an", "Fuzhou",
    "Nanchang", "Ganzhou", "Yingtan", "Shanghai", "Jiading",
    "Baoshan", "Jinshan", "Songjiang", "Qingpu", "Fengxian",
    "Chongming", "Nanhui", "Minhang", "Pudong", "Yangpu",
    "Hongkou", "Putuo", "Zhabei", "Jing'an", "Changning",
    "Xuhui", "Huangpu", "Luwan", "Barcelona", "Milan", "Munich", "Prague", "Vienna",
    "Budapest", "Amsterdam", "Warsaw", "Brussels", "Bucharest",
    "Saint Petersburg", "Hamburg", "Minsk", "Copenhagen", "Stockholm",
    "Kiev", "Athens", "Dublin", "Nizhny Novgorod", "Lisbon",
    "Manchester", "Helsinki", "Sofia", "Oporto", "Bratislava",
    "Cologne", "Naples", "Birmingham", "Rotterdam", "Turin",
    "Marseille", "Amsterdam", "Zagreb", "Krakow", "Frankfurt",
    "Seville", "Oslo", "The Hague", "Düsseldorf", "Athens",
    "Palermo", "Rotterdam", "Genoa", "Helsinki", "Stuttgart",
    "Zurich", "Dortmund", "Málaga", "Leipzig", "Dresden",
    "Hanover", "Gothenburg", "Dublin", "Antwerp", "Bremen",
    "Sheffield", "Edinburgh", "Leeds", "Nuremberg", "Duisburg",
    "Alicante", "Bristol", "Glasgow", "Lyon", "Valencia",
    "Bologna", "Cairo", "Alexandria", "Giza", "Shubra El-Kheima", "Port Said",
    "Suez", "Luxor", "El-Mahalla El-Kubra", "Tanta", "Asyut",
    "Ismailia", "Fayyum", "Zagazig", "Aswan", "Damietta",
    "São Paulo", "Rio de Janeiro", "Buenos Aires", "Lima", "Bogotá",
    "Santiago", "Caracas", "Brasília", "Medellín", "Monterrey",
    "Guadalajara", "Belém", "Porto Alegre", "Recife", "Puebla",
    "Curitiba", "Fortaleza", "Manaus", "León", "Quito",
    "Salvador", "Maracaibo", "Barranquilla", "Santa Cruz de la Sierra", "Guayaquil",
    "Ciudad Juárez", "Tegucigalpa", "San José", "San Salvador", "San Pedro Sula",
]





cities = list(set(cities))
landmarcks = [
    "Eiffel Tower", "Louvre Museum", "Arc de Triomphe", "Notre-Dame de Paris", "Palace of Versailles",
    "Big Ben", "Tower Bridge", "Buckingham Palace", "London Eye", "St Paul's Cathedral",
    "Empire State Building", "Statue of Liberty", "Central Park", "Times Square", "Rockefeller Center",
    "Tokyo Tower", "Tokyo Skytree", "Tokyo Imperial Palace", "Meiji Shrine", "Shibuya Crossing",
    "Hollywood Sign", "Hollywood Walk of Fame", "Griffith Observatory", "Santa Monica Pier", "Disneyland",
    "Willis Tower", "Navy Pier", "Millennium Park", "Art Institute of Chicago", "Cloud Gate",
    "Golden Gate Bridge", "Alcatraz Island", "Fisherman's Wharf", "Palace of Fine Arts", "Golden Gate Park",
    "CN Tower", "Rogers Centre", "Royal Ontario Museum", "Casa Loma", "Art Gallery of Ontario",
    "Sydney Opera House", "Sydney Harbour Bridge", "Bondi Beach", "Darling Harbour", "Taronga Zoo",
    "Federation Square", "Royal Exhibition Building", "Melbourne Cricket Ground", "National Gallery of Victoria", "Melbourne Zoo",
    "Stanley Park", "Canada Place", "Vancouver Aquarium", "Granville Island", "Science World",
    "Old Montreal", "Notre-Dame Basilica", "Montreal Botanical Garden", "Montreal Museum of Fine Arts", "Montreal Biosphere",
    "Calgary Tower", "Calgary Zoo", "Glenbow Museum", "Heritage Park Historical Village", "Calgary Stampede",
    "Parliament Hill", "Rideau Canal", "Canadian War Museum", "National Gallery of Canada", "Canadian Museum of History",
    "South Bank", "Lone Pine Koala Sanctuary", "Queensland Gallery of Modern Art", "Brisbane City Hall", "Wheel of Brisbane",
    "Kings Park", "Perth Zoo", "Art Gallery of Western Australia", "Perth Mint", "Perth Cultural Centre",
    "Sky Tower", "Auckland War Memorial Museum", "Auckland Art Gallery", "Auckland Domain", "Auckland Zoo",
    "Te Papa", "Wellington Cable Car", "Wellington Botanic Garden", "Wellington Zoo", "Museum of New Zealand Te Papa Tongarewa",
    "Christchurch Botanic Gardens", "Christchurch Gondola", "International Antarctic Centre", "Willowbank Wildlife Reserve", "Christchurch Art Gallery",
    "Forbidden City", "Temple of Heaven", "Tiananmen Square", "Summer Palace", "Beihai Park",
    "The Bund", "Yu Garden", "Shanghai Tower", "Oriental Pearl Tower", "Jin Mao Tower",
    "Canton Tower", "Chimelong Paradise", "Chimelong Safari Park", "Shamian Island", "Guangzhou Opera House",
    "Window of the World", "Happy Valley", "Overseas Chinese Town", "Shenzhen Museum", "Shenzhen Safari Park",
    "Victoria Harbour", "Hong Kong Disneyland", "Ocean Park", "Lantau Island", "Hong Kong Museum of History",
    "Taipei 101", "National Palace Museum", "Chiang Kai-shek Memorial Hall", "Taipei Zoo", "Taipei Fine Arts Museum",
    "Gyeongbokgung", "Myeong-dong", "N Seoul Tower", "Bukchon Hanok Village", "Changdeokgung",
    "Gardens by the Bay", "Singapore Zoo", "Universal Studios Singapore", "Singapore Botanic Gardens", "Singapore Flyer",
    "Grand Palace", "Wat Pho", "Wat Arun", "Chatuchak Weekend Market", "Temple of the Emerald Buddha",
    "Ben Thanh Market", "War Remnants Museum", "Saigon Notre-Dame Basilica", "Independence Palace", "Ho Chi Minh City Museum",
    "Hoan Kiem Lake", "Hanoi Opera House", "Ho Chi Minh Mausoleum", "Temple of Literature", "Hanoi Old Quarter",
    "National Monument", "National Museum of Indonesia", "National Gallery of Indonesia", "National Museum of World Cultures", "National Museum of Natural History",
    "Petronas Towers", "Batu Caves", "Kuala Lumpur Tower", "KLCC Park", "Islamic Arts Museum Malaysia",
    "Angkor Wat", "Bayon", "Ta Prohm", "Angkor Thom", "Banteay Srei",
    "Wat Phra Kaew", "Wat Arun", "Wat Pho", "Grand Palace", "Jim Thompson House",
    "Borobudur", "Prambanan", "Taman Sari", "Malioboro", "Mount Merapi",
    "Bagan", "Shwedagon Pagoda", "Inle Lake", "Mandalay Palace", "Ananda Temple",
    "Wat Phou", "Vat Phou", "Wat Xieng Thong", "Plain of Jars", "Pha That Luang",
    "Petra", "Wadi Rum", "Jerash", "Dead Sea", "Amman Citadel",
    "Pyramids of Giza", "Great Sphinx of Giza", "Karnak", "Valley of the Kings", "Luxor Temple",
    "Hagia Sophia", "Topkapı Palace", "Blue Mosque", "Grand Bazaar", "Basilica Cistern",
    "Taj Mahal", "Agra Fort", "Fatehpur Sikri", "Akbar's Tomb", "Mehtab Bagh",
    "Forbidden City", "Temple of Heaven", "Tiananmen Square", "Summer Palace", "Beihai Park",
    "lanzhou Bridge", "Yellow River Mother Sculpture", "Gansu Provincial Museum", "Baita Mountain Park", "Wuquan Mountain Park",
    "Jiayuguan Pass", "Jiayuguan City Walls", "Wei-Jin Art Gallery", "Xuanbi Great Wall", "Jiayuguan Museum",
    "Mogao Caves", "Mingsha Mountain and Crescent Spring", "Yumen Pass", "Yadan National Geological Park", "Dunhuang Museum",
    "Maijishan Grottoes", "Giant Buddha Temple", "Shuiliandong", "Shimen Cave", "Shimen Cave",
    "Labrang Monastery", "Sangke Grassland", "Gahai Lake", "Langmusi", "Xiahe",
    "Mount Wutai", "Yungang Grottoes", "Hanging Temple", "Mount Heng", "Mount Heng",
    "Mount Tai", "Confucius Temple", "Qufu Temple and Cemetery of Confucius", "Baotu Spring", "Daming Lake",
    "Mount Huang", "Hongcun", "Tunxi Ancient Street", "Xidi", "Tangyue Memorial Archway",
    "Mount Emei", "Leshan Giant Buddha", "Jiuzhaigou Valley", "Huanglong", "Mount Qingcheng",
    "Mount Lushan", "Longhu Mountain", "Sanqing Mountain", "Jingdezhen Ceramic Historical Museum", "Fuliang Ancient Town",
    "Mount Sanqingshan"
]
landmark = list(set(landmarcks))
with open("city_coordinates.txt", "w") as file:
    for city in tqdm.tqdm(cities, desc='Processing cities'):
        lat, lon = get_city_coordinates(city)
        if lat and lon:
            file.write(f"{city.replace(" ", "")} {lat} {lon}\n")
        else:
            print(f"{city} not found")
        time.sleep(1)  

print("Coordinates extraction completed.")


with open("landmark_coordinates.txt", "w") as file:
    for landmark in tqdm.tqdm(landmarcks, desc='Processing landmarks'):
        lat, lon = get_city_coordinates(landmark)
        if lat and lon:
            file.write(f"{landmark.replace(" ", "")} {lat} {lon}\n")
        else:
            print(f"{landmark} not found")
        time.sleep(1) 

print("Coordinates extraction completed.")
