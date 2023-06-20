from osgeo import gdal, ogr, osr
from scipy import ndimage
import geopandas as gp
import os
import utils
import pandas as pd

class Dasymetric:

    def __init__(self, config):
        """
        set up the falgs for the class.
        :param config:
        """

        self.MAIN = config['file_paths']['MAIN']
        self.INPUT = config['file_paths']['INPUT']
        self.OUTPUT = config['file_paths']['OUTPUT']
        self.DOCS = config['file_paths']['DOCS']
        self.FIGS = config['file_paths']['FIGS']
        self.CODES = config['file_paths']['CODES']
        self.EMPTY_SUFFIXES = config['flags']['EMPTY_SUFFIXES']
        self.START_YEAR = config['flags']['START_YEAR']
        self.END_YEAR = config['flags']['END_YEAR']
        self.LANDUSE = config['flags']['LANDUSE']
        self.NIGHTLIGHT = config['flags']['NIGHTLIGHT']
        self.INTERVAL = config['flags']['INTERVAL']
        self.STATISTIC = config['flags']['STATISTIC']
        self.NIGHTLIGHT_ANGLE_CORRECTION = config['flags']['NIGHTLIGHT_ANGLE_CORRECTION']

    def resample_landuse (self, dx, dy, method='Majority'):

        path = self.INPUT + "VHR/images/"
        f = os.listdir(path)
        f = [i for i in f if not any(i.endswith(suffix) for suffix in self.EMPTY_SUFFIXES)]

        for item in f:
            if 'rsm' not in item:
                try:
                    utils.resample_raster(path, self.INPUT + 'Temp/', item, dx, dy, method=method)
                except:
                    print("Having problem resampling item: " + item)

    def vectorize_landuse (self):

        path = self.INPUT + 'Temp/'
        f = os.listdir(path)
        f = [i for i in f if not any(i.endswith(suffix) for suffix in self.EMPTY_SUFFIXES)]

        for item in f:
            if 'rsm' in item:
                try:
                    utils.vectorize_raster(path, self.OUTPUT, item)
                except:
                    print("Having problem vectorizing item: " + item)

    def vectorize_nightlight(self):

        path = self.INPUT + 'VIIRS/VNP46A2/'
        f = os.listdir(path)
        f = [i for i in f if not any(i.endswith(suffix) for suffix in self.EMPTY_SUFFIXES)]

        for item in f:
            try:
                utils.vectorize_raster(path, self.OUTPUT, item)
            except:
                print("Having problem vectorizing item: " + item)

    def read_layers(self):
        """
        Read layers that are involved in dasymetric mapping. In the current pipeline we have two auxiliary data sets:
        land use and nightlight. We want to study dasymetric modeling of population using (landuse) and
        (landuse + nightlight). You can also have multiple layers for a single study area that represent different
        months, and year. The file name must include time information in any of the following formats.
        Annual(e.g., 2015), monthly(e.g., 2015-02), daily(e.g., 2015-02-01).
        """
        # Read and create geometry and index for auxiliary layers
        # Nightlight
        ntl = gp.read_file(self.INPUT + 'Temp/NTL.shp')
        ntl['ntl_id'] = ntl.index + 1
        ntl['ntl_area'] = ntl.area

        # Landuse
        landuse = gp.read_file(self.INPUT + 'Temp/landuse.shp')
        landuse['landuse_id'] = landuse.index + 1
        landuse['landuse_area'] = landuse.area

        # Census
        census = gp.read_file(self.INPUT + 'Temp/census.shp')
        census['census_id'] = census.index + 1
        census['census_area'] = census.area

        # Boundary
        boundary = gp.read_file(self.INPUT + 'Temp/CensusBoundary.shp')

        if self.LANDUSE == 1 and self.NIGHTLIGHT == 1 and self.INTERVAL=='ANNUAL' and \
                self.NIGHTLIGHT_ANGLE_CORRECTION==0 and self.STATISTIC=='MEDIAN':

            for year in [str(i) for i in range(self.START_YEAR, self.END_YEAR)]:
                image = gp.read_file(self.OUTPUT + 'ntlmed' + year + '.shp')
                image.rename({'Value':'Value' + year}, axis=1, inplace=True)
                ntl = pd.concat([ntl, image['Value' + year]], axis=1)
            ntl.rename({'Shape_Area':'ntl_area'}, axis=1, inplace=True)
            ntl.drop('Shape_Leng', axis=1, inplace=True)

            for year in [str(i) for i in range(self.START_YEAR, self.END_YEAR)]:
                if int(year) >= 2014:
                    image = gp.read_file(self.OUTPUT + 'rsmlabel' + year + '.shp')
                    image.rename({'Value': 'Value' + year}, axis=1, inplace=True)
                    landuse = pd.concat([landuse, image['Value' + year]], axis=1)
            landuse.rename({'Shape_Area':'landuse_area'}, axis=1, inplace=True)
            landuse.drop('Shape_Leng', axis=1, inplace=True)

            # Define the boundaries of layers
            ntl_clip = gp.clip(ntl, boundary)
            ntl_clip['ntl_clip_id'] = ntl_clip.index + 1
            ntl_clip['ntl_clip_area'] = ntl_clip.geometry.area
            landuse_clip = gp.clip(landuse, boundary)
            landuse_clip['landuse_clip_id'] = landuse_clip.index + 1
            landuse_clip['landuse_clip_area'] = landuse_clip.geometry.area

            # When working with dasymetric modeling, all layers must intersect to create a geometric layer that
            # represents the highest spatial solution. All objects within this new layer must have a value for all
            # layers involved in dasymetric mapping.
            intersect = gp.overlay(census, ntl_clip, how='intersection')
            intersect_all = gp.overlay(intersect, landuse_clip, how='intersection')

            # This intersection layer is the highest spatial resolution layer
            intersect_all['intersect_id'] = intersect_all.index + 1
            intersect_all['intersect_area'] = intersect_all.geometry.area

            # For boundary areas, we need to calculate nightlight values based on the area.
            for year in [str(i) for i in range(self.START_YEAR, self.END_YEAR)]:
                intersect_all['CNTL' + year] = (intersect_all['ntl_clip_area'] / intersect_all['ntl_area']) * \
                                               intersect_all['NTL' + year]

        return [intersect_all, ntl_clip, landuse_clip]

    def create_intersection_layer(self):
        """
        This function intersects all layers into an intersection layer. This intersection is geometric. That is
        no attributes are kept from the source layers. However, the indexes of the source layers are kept in the
        intersect layer. This is an important feature to later present new estimates the desired scale requested by
        the end user.
        :return:
        """

    def dasymetic_maping(self, target_scale):
        """
        Conducts dasymetric mapping into the desired scale. Desired scale could be at the level of source layer,
        axiliary layers, or the intersection layer.

        :param target_layer:
        :return:
        """
