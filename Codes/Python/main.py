import json
import dasymetric_mapping
from dasymetric_mapping import Dasymetric
import landuse
import nightlight
import classifier

def main():

    with open('config.json') as config_file:
        config = json.load(config_file)

    das_map = Dasymetric(config)
    das_map.vectorize_raster('labelrsm2014.tif')
    #das_map.read_layers(config)

if __name__ == '__main__':
    main()