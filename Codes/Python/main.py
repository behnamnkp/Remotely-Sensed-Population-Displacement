import json
import dasymetric_mapping
from dasymetric_mapping import Dasymetric
import landuse
import nightlight
import classifier

def main():

    with open('H:/My Drive/population_displacement/config.json') as config_file:
        config = json.load(config_file)

    das_map = Dasymetric(config)

    # Classification patches were 50X50 meters with 0.5mX0.5m pixel sizes. We resample landuse to 50mX50m.
    # das_map.resample_landuse(50, 50, method='Majority')
    # das_map.vectorize_landuse()
    # das_map.vectorize_nightlight()
    das_map.dasymetic_mapping()

    #das_map.read_layers(config)

if __name__ == '__main__':
    main()