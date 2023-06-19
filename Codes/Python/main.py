import json
import dasymetric_mapping
import landuse
import nightlight
import classifier

def main():

    with open('config.json') as config_file:
        config = json.load(config_file)

    das_map = dasymetric_mapping.Dasymetric(config)
    das_map.read_layers(config)

if __name__ == '__main__':
    main()