from osgeo import gdal, ogr, osr
from scipy import ndimage
import geopandas as gp
import os
import utils
import pandas as pd
import statsmodels.api as sm
import patsy
from statsmodels.stats.outliers_influence import variance_inflation_factor
from libpysal.weights import Queen, Rook, KNN
from esda.moran import Moran
from esda.moran import Moran_Local
from splot.esda import lisa_cluster
from splot import _viz_utils

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
        self.TARGET_LAYER = config['flags']['TARGET_LAYER']

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

    def create_intersection_layer(self, main_lay, aux_lay1, aux_lay2, boundary):
        """
        This function intersects all layers into an intersection layer. This intersection is geometric. That is
        no attributes are kept from the source layers. However, the indexes of the source layers are kept in the
        intersect layer. This is an important feature to later present new estimates the desired scale requested by
        the end user.
        :return:
        """

        # Define the boundaries of layers
        lay1 = gp.clip(aux_lay1, boundary)
        lay1['ntl_clip_id'] = lay1.index + 1
        lay1['ntl_clip_area'] = lay1.geometry.area
        lay2 = gp.clip(aux_lay2, boundary)
        lay2['landuse_clip_id'] = lay2.index + 1
        lay2['landuse_clip_area'] = lay2.geometry.area

        # When working with dasymetric modeling, all layers must intersect to create a geometric layer that
        # represents the highest spatial solution. All objects within this new layer must have a value for all
        # layers involved in dasymetric mapping.
        intersect = gp.overlay(main_lay, lay1, how='intersection')
        intersect_all = gp.overlay(intersect, lay2, how='intersection')

        # This intersection layer is the highest spatial resolution layer
        intersect_all['intersect_id'] = intersect_all.index + 1
        intersect_all['intersect_area'] = intersect_all.geometry.area

        # If variables associated with each layer are continuous, we need to adjust their values based on area at the
        # border lines.
        for year in [str(i) for i in range(self.START_YEAR, self.END_YEAR)]:
            intersect_all['cntl' + year] = (intersect_all['ntl_clip_area'] / intersect_all['ntl_area']) * \
                                           intersect_all['ntl' + year]

        # If variables associated with each layer are categorical, we need to calculate area associate with each
        # category in the level of main layer.
        for year in [str(i) for i in range(self.START_YEAR, self.END_YEAR)]:
            if int(year) >= 2014:
                areas = intersect_all[['census_id', 'intersect_area']].groupby(['census_id']).sum().astype('float64')
                intersect_all = intersect_all.join(areas['intersect_area'],
                                                   on=['census_id'],
                                                   how='left',
                                                   lsuffix='_caller',
                                                   rsuffix='_other')
                intersect_all['census_res_area' + year] = intersect_all['intersect_area_other']
                intersect_all.drop('intersect_area_other', inplace=True, axis=1)
                intersect_all.rename({'intersect_area_caller':'intersect_area'}, axis=1, inplace=True)

        intersect_all = intersect_all.reset_index()[['id', 'census_id', 'census_area', 'ntl_id', 'ntl_area',
                                                     'ntl_clip_id', 'ntl_clip_area', 'landuse_id', 'landuse_area',
                                                     'landuse_clip_id', 'landuse_clip_area', 'intersect_id',
                                                     'intersect_area', 'MAX_popult', 'estPop2013', 'NTL2013', 'NTL2014',
                                                     'NTL2015', 'NTL2016', 'NTL2017', 'NTL2018', 'landuse2014',
                                                     'landuse2015', 'landuse2016', 'landuse2017', 'landuse2018',
                                                     'CNTL2013', 'CNTL2014', 'CNTL2015', 'CNTL2016', 'CNTL2017',
                                                     'CNTL2018', 'census_res_area2014', 'census_res_area2015',
                                                     'census_res_area2016', 'census_res_area2017', 'census_res_area2018'
                                                     ]]
        return [intersect_all, lay1, lay2]

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

        intersect_all, ntl_clip, landuse_clip = self.create_intersection_layer(census, ntl, landuse, boundary)

        return [intersect_all, ntl_clip, landuse_clip]

    def target_night_light(self, base, ntl, year):

        b = base.set_index('ntl_clip_id')

        b['countNTL'] = b['index'].groupby(b.index).transform('count')
        if int(year) < 2014:
            aux = b.groupby(['ntl_clip_id', 'landuse2014']).sum().loc[:, ['intersect_area']]
            aux2 = aux.unstack('landuse2014')
            aux2.columns = ['area_bg' + year, 'area_lr' + year, 'area_hr' + year, 'area_nr' + year]
            aux2.fillna(0, inplace=True)

        else:
            aux = b.groupby(['ntl_clip_id', 'landuse' + year]).sum().loc[:, ['intersect_area']]
            aux2 = aux.unstack('landuse' + year)
            aux2.columns = ['area_bg' + year, 'area_lr' + year, 'area_hr' + year, 'area_nr' + year]
            aux2.fillna(0, inplace=True)

        aux2['CNTL' + year] = b.groupby(b.index).max()['CNTL' + year]

        n = ntl.set_index('ntl_clip_id')
        aux2 = n.merge(aux2, left_on=n.index, right_on=aux2.index, how='left')
        aux2.drop(
            ['key_0', 'Shape_Leng', 'Shape_Area', 'ntl_area', 'NTL2014', 'NTL2015', 'ntl_id', 'NTL2016',
             'NTL2017', 'NTL2018', 'NTL2013', 'ntl_clip_area', 'level_0'], inplace=True, axis=1)
        n.drop(['level_0'], inplace=True, axis=1)

        aux2['X'] = aux2.geometry.centroid.x
        aux2['Y'] = aux2.geometry.centroid.y

        return aux2

    def upscale_nightlight (self, ntl_level, year):

        print(f"Multiple Linear Regression for upscaling nightlight {year}:")
        formula = f"CNTL{year} ~ area_hr{year} + area_nr{year} + area_bg{year} + area_lr{year}"
        y, X = patsy.dmatrices(formula, data=ntl_level, return_type='dataframe')
        model = sm.OLS(y, X).fit()
        print(model.summary())
        print("\nRetrieving manually the parameter estimates:")
        print(model.params)

        print("Checking collinearity (Variance Inflation Factor):")
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['Variable'] = X.columns
        print(vif)

        print("Investigating residuals and identifying clusters:")
        ntlresid = pd.concat((ntl_level, model.resid), axis=1)
        ntlresid.rename(columns={0: f'ntlresid{year}'}, inplace=True)
        W = Queen.from_dataframe(ntlresid)
        W.transform = 'r'
        moran_ntl = Moran(ntlresid[f'ntlresid{year}'], W)
        print(f'moran_ntl{year}: {moran_ntl.I}')
        moran_loc = Moran_Local(ntlresid[f'ntlresid{year}'], W)
        p = lisa_cluster(moran_loc, ntlresid, p=0.05, figsize=(9, 9))

        print("Naming clusters (1: HH, 2: LH, 3: LL, 4: HL):")
        cluster = _viz_utils.moran_hot_cold_spots(moran_loc, p=0.05)
        aux = pd.DataFrame(cluster)
        aux.rename(columns={0: f'clusters{year}'}, inplace=True)
        aux.loc[aux[f'clusters{year}'] == 0, [f'clusters{year}']] = 'NS'
        aux.loc[aux[f'clusters{year}'] == 1, [f'clusters{year}']] = 'HH'
        aux.loc[aux[f'clusters{year}'] == 2, [f'clusters{year}']] = 'LH'
        aux.loc[aux[f'clusters{year}'] == 3, [f'clusters{year}']] = 'LL'
        aux.loc[aux[f'clusters{year}'] == 4, [f'clusters{year}']] = 'HL'
        cluster = pd.concat((ntl_level, aux), axis=1)
        cluster = pd.get_dummies(cluster)

        print(f"Spatial Multiple Linear Regression for upscaling nightlight {year}:")
        formula = f"CNTL{year} ~ area_hr{year} + area_nr{year} + area_bg{year} + area_lr{year} + clusters{year}_HH + " \
                  f"clusters{year}_HL + clusters{year}_NS"
        y, X = patsy.dmatrices(formula, data=ntl_level, return_type='dataframe')
        spatial_model = sm.OLS(y, X).fit()
        print(spatial_model.summary())
        print("\nRetrieving manually the parameter estimates:")
        print(spatial_model.params)

        print("Checking collinearity (Variance Inflation Factor):")
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['Variable'] = X.columns
        print(vif)

        return cluster


    def dasymetic_mapping(self):
        """
        Conducts dasymetric mapping into the desired scale. Desired scale could be at the level of source layer,
        axiliary layers, or the intersection layer.
        :param target_layer:
        :return:
        """

        base, ntl, landuse = self.read_layers()

        for year in [str(i) for i in range(self.START_YEAR, self.END_YEAR)]:
            # We also need a target layer. Target layer is determined based on the decision-making problem or preference
            # of the end user.
            if self.TARGET_LAYER == "NIGHTLIGHT":
                ntl_level = self.target_night_light(base, ntl, year)
                cluster = self.upscale_nightlight(ntl_level, year)




