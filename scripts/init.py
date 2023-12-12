import os

if not os.path.exists('../data'):
    os.mkdir('../data')
if not os.path.exists('../data/raw'):
    os.mkdir('../data/raw')
if not os.path.exists('../data/raw/osm'):
    os.mkdir('../data/raw/osm')
if not os.path.exists('../data/raw/osm/cities'):
    os.mkdir('../data/raw/osm/cities')
if not os.path.exists('../data/raw/osm/landmarks'):
    os.mkdir('../data/raw/osm/landmarks')