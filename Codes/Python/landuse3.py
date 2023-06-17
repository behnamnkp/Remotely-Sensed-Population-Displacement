import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import gdal, os
import matplotlib.pyplot as plt
from matplotlib import *
import pandas as pd
import scipy.stats
from pysal.lib import weights
import pysal as ps
import libpysal
from libpysal.weights import Queen, Rook, KNN
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
# import arcpy
# from arcpy import env
# from arcpy.sa import *
from simpledbf import Dbf5
from os import listdir
from os.path import isfile, join
# from simpledbf import Dbf5
import geopandas as gp
import libpysal
import mapclassify
# import pysal as ps
# import libpysal
# import esda
# from esda.moran import Moran
# from splot.esda import moran_scatterplot
# from splot.esda import plot_moran
# from esda.moran import Moran_Local
# from splot.esda import plot_local_autocorrelation
# from splot.esda import lisa_cluster
from mpl_toolkits.mplot3d import Axes3D
# For statistics. Requires statsmodels 5.0 or more
from statsmodels.formula.api import ols
import spreg
from statsmodels import regression
import statsmodels.api as sm
# Analysis of Variance (ANOVA) on linear models
from statsmodels.stats.anova import anova_lm
import seaborn as sns
# from pysal.contrib.viz import mapping as maps
# import mapclassify
from statsmodels.regression.linear_model import OLSResults
from matplotlib.pyplot import figure

image_path = 'G:/backupC27152020/Population_Displacement_Final/Resources/VHR/images/'
landuse_path = 'G:/backupC27152020/Population_Displacement_Final/Resources/VHR/landuse/'
viirs_path = 'G:/backupC27152020/Population_Displacement_Final/Resources/VIIRS/VNP46A2/'
geodb_path = 'G:/backupC27152020/Population_Displacement_Final/Resources/poulation_disp.gdb/Data/'
temp = 'G:/backupC27152020/Population_Displacement_Final/Resources/Temp/'
results = 'G:/backupC27152020/Population_Displacement_Final/Resources/Results/'
figures = 'G:/backupC27152020/Population_Displacement_Final/Results/Model/'
date = '01312021'

# Choose the year of the analysis:
years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018'] # 2014-15-16-17-18 is available now
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# arcpy.env.workspace = image_path
# # resample landuse
# for year in years:
#     if int(year) < 2014:
#         print('landue does not exist for ' + year)
#     else:
#         inputraster = 'label' + year +'.tif'
#         outpuraster = 'labelrsm' + year +'.tif'
#         arcpy.Resample_management(inputraster,outpuraster,"50 50", "Majority")
#
# convert to point
# for year in years:
#     for month in months:
#         arcpy.RasterToPoint_conversion(viirs_path + 'ntlMonthlyIncorrected_' + month  + year + '.tif', temp + 'ntlMonthlyIncorrected_' + month  + year, "VALUE")

# models = ['nontl', 'ntlmed', 'ntl_corrected_med_annualByMonth', 'ntl_corrected_med_monthly']
# models = ['ntlMonthlyIncorrected_', 'ntlMonthlyCorrected_']
models = ['ntlMonthlyIncorrected_']

for mdl in models:
    NTL = gp.read_file(temp + 'NTL.shp')
    NTL['ntl_id'] = NTL.index + 1
    NTL['ntl_area'] = NTL.area
    landuse = gp.read_file(temp + 'landuse.shp')
    landuse['landuse_id'] = landuse.index + 1
    landuse['landuse_area'] = landuse.area
    census = gp.read_file(temp + 'census.shp')
    census['census_id'] = census.index + 1
    census['census_area'] = census.area
    boundary = gp.read_file(temp + 'CensusBoundary.shp')

    if mdl == 'nontl':
        # assign the values from point features to areal
        for year in years:
            for month in months:
                image = gp.read_file(temp + 'ntlmed' + month + year + '.shp')
                NTL = gp.sjoin(NTL, image, how="inner", op='intersects')
                NTL.rename({'grid_code': 'NTL' + month + year}, inplace=True, axis=1)
                NTL.drop('pointid', inplace=True, axis=1)
                NTL.drop('index_right', inplace=True, axis=1)
    else:
        for year in years:
            for month in months:
                image = gp.read_file(temp + mdl + month + year + '.shp')
                NTL = gp.sjoin(NTL, image, how="inner", op='intersects')
                NTL.rename({'grid_code': 'NTL' + month + year}, inplace=True, axis=1)
                NTL.drop('pointid', inplace=True, axis=1)
                NTL.drop('index_right', inplace=True, axis=1)
    for year in years:
        if int(year) >= 2014:
            image = gp.read_file(temp + 'label' + year + '.shp')
            landuse = gp.sjoin(landuse, image, how="inner", op='intersects')
            landuse.rename({'grid_code': 'landuse' + year}, inplace=True, axis=1)
            landuse.drop('pointid', inplace=True, axis=1)
            landuse.drop('index_right', inplace=True, axis=1)

    NTL_clip = gp.clip(NTL, boundary)
    NTL_clip['ntl_clip_id'] = NTL_clip.index + 1
    NTL_clip['ntl_clip_area'] = NTL_clip.area
    landuse_clip = gp.clip(landuse, boundary)
    landuse_clip['landuse_clip_id'] = landuse_clip.index + 1
    landuse_clip['landuse_clip_area'] = landuse_clip.geometry.area

    intersect1 = gp.overlay(census, NTL_clip, how='intersection')
    intersect2 = gp.overlay(intersect1, landuse_clip, how='intersection')

    intersect2['intersect_id'] = intersect2.index + 1
    intersect2['intersect_area'] = intersect2.area

    for year in years:
        for month in months:
            intersect2['CNTL' + month + year] = (intersect2['ntl_clip_area'] /
                                                    intersect2['ntl_area'])*intersect2['NTL' + month + year]

    # Calculate residential area
    for year in years:
        if int(year) >= 2014:
            intersect2['intersect_area2'] = intersect2['intersect_area']
            # mask = ((intersect2['landuse' + year] == 1) | (intersect2['landuse' + year] == 4))
            # intersect2.loc[mask, ['intersect_area2']] = 0
            areas = intersect2.groupby(['census_id']).sum().astype('float64')
            intersect2 = intersect2.join(areas['intersect_area2'], on=['census_id'], how='left', lsuffix='_caller', rsuffix='_other')
            intersect2['census_res_area' + year] = intersect2['intersect_area2_other']
            intersect2.drop('intersect_area2_other', inplace=True, axis=1)

    intersect2 = intersect2.reset_index()

    for year in years:
        for month in months:
            # in the level of night light
            try:
                intersect2.set_index('ntl_clip_id', inplace=True)
            except:
                print('ntl_clip_id is already the index')

            intersect2['countNTL'] = intersect2['index'].groupby(intersect2.index).transform('count')
            if int(year) == 2012:
                ntl_scale_NTL = intersect2.groupby(['ntl_clip_id', 'landuse2014']).sum().loc[:,['intersect_area']]
                ntl_scale_NTL2 = ntl_scale_NTL.unstack('landuse2014')
                ntl_scale_NTL2.columns = ['area_bg' + month + year, 'area_lr' + month + year, 'area_hr' + month + year, 'area_nr' + month + year]
                ntl_scale_NTL2.fillna(0, inplace=True)

            elif int(year) == 2013:
                ntl_scale_NTL = intersect2.groupby(['ntl_clip_id', 'landuse2014']).sum().loc[:,['intersect_area']]
                ntl_scale_NTL2 = ntl_scale_NTL.unstack('landuse2014')
                ntl_scale_NTL2.columns = ['area_bg' + month + year, 'area_lr' + month + year, 'area_hr' + month + year, 'area_nr' + month + year]
                ntl_scale_NTL2.fillna(0, inplace=True)

            elif (int(year) == 2014) or (int(year) == 2015 and (month == 'Jan' or month == 'Feb')):
                ntl_scale_NTL = intersect2.groupby(['ntl_clip_id', 'landuse2014']).sum().loc[:, ['intersect_area']]
                ntl_scale_NTL2 = ntl_scale_NTL.unstack('landuse2014')
                ntl_scale_NTL2.columns = ['area_bg' + month + year, 'area_lr' + month + year, 'area_hr' + month + year, 'area_nr' + month + year]
                ntl_scale_NTL2.fillna(0, inplace=True)

            elif (int(year) == 2015 and (month != 'Jan' and month != 'Feb')) or (int(year) == 2016 and (month == 'Jan' or month == 'Feb' or month == 'Mar')):
                ntl_scale_NTL = intersect2.groupby(['ntl_clip_id', 'landuse2015']).sum().loc[:, ['intersect_area']]
                ntl_scale_NTL2 = ntl_scale_NTL.unstack('landuse2015')
                ntl_scale_NTL2.columns = ['area_bg' + month + year, 'area_lr' + month + year, 'area_hr' + month + year, 'area_nr' + month + year]
                ntl_scale_NTL2.fillna(0, inplace=True)

            elif (int(year) == 2016 and (month != 'Jan' and month != 'Feb' and month != 'Mar')):
                ntl_scale_NTL = intersect2.groupby(['ntl_clip_id', 'landuse2016']).sum().loc[:, ['intersect_area']]
                ntl_scale_NTL2 = ntl_scale_NTL.unstack('landuse2016')
                ntl_scale_NTL2.columns = ['area_bg' + month + year, 'area_lr' + month + year, 'area_hr' + month + year, 'area_nr' + month + year]
                ntl_scale_NTL2.fillna(0, inplace=True)

            elif int(year) == 2017:
                ntl_scale_NTL = intersect2.groupby(['ntl_clip_id', 'landuse2017']).sum().loc[:, ['intersect_area']]
                ntl_scale_NTL2 = ntl_scale_NTL.unstack('landuse2017')
                ntl_scale_NTL2.columns = ['area_bg' + month + year, 'area_lr' + month + year, 'area_hr' + month + year, 'area_nr' + month + year]
                ntl_scale_NTL2.fillna(0, inplace=True)

            elif int(year) == 2018:
                ntl_scale_NTL = intersect2.groupby(['ntl_clip_id', 'landuse2018']).sum().loc[:, ['intersect_area']]
                ntl_scale_NTL2 = ntl_scale_NTL.unstack('landuse2018')
                ntl_scale_NTL2.columns = ['area_bg' + month + year, 'area_lr' + month + year, 'area_hr' + month + year, 'area_nr' + month + year]
                ntl_scale_NTL2.fillna(0, inplace=True)

            ntl_scale_NTL2['CNTL' + month + year] = intersect2.groupby(intersect2.index).max()['CNTL' + month + year]

            ntl_scale_NTL2.reset_index(inplace=True)
            NTL_clip.reset_index(inplace=True)
            ntl_scale_NTL2 = NTL_clip.merge(ntl_scale_NTL2, left_on = NTL_clip.index, right_on = ntl_scale_NTL2.index, how='left')
            ntl_scale_NTL2.drop(['key_0', 'index', 'Shape_Leng', 'Shape_Area', 'ntl_area', 'ntl_id','ntl_clip_id_x', 'ntl_clip_area'],
                            inplace=True, axis=1)
            try:
                ntl_scale_NTL2.drop(['level_0'], inplace=True, axis=1)
                NTL_clip.drop(['level_0'], inplace=True, axis=1)
            except:
                print('level_0 is not in the columns')

            ntl_scale_NTL2['X'] = ntl_scale_NTL2.geometry.centroid.x
            ntl_scale_NTL2['Y'] = ntl_scale_NTL2.geometry.centroid.y

            if int(year) == 2013:
                olsmodelstring = 'CNTL' + month + year + ' ~ ' + 'area_bg' + month + year + ' + '  + \
                                'area_hr' + month + year + ' + ' + 'area_nr' + month + year
                model_NTL = ols(olsmodelstring,ntl_scale_NTL2).fit()
                # print(mdl + ': ' + olsmodelstring + "\n")
                # print(model_NTL.summary())
                # print("\nRetrieving manually the parameter estimates:")
                # print(model_NTL._results.params)
                model_NTL.save(results + 'ols_ntl.pickle', remove_data=False)

                mask = intersect2['landuse2014'] == 1
                intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL.params[1]
                mask = intersect2['landuse2014'] == 2
                intersect2.loc[mask, ['coef_ntl' + month + year]] = 0#model_NTL.params[2]
                mask = intersect2['landuse2014'] == 3
                intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL.params[2]
                mask = intersect2['landuse2014'] == 4
                intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL.params[3]
                intersect2['intercept_ntl' + month + year] = model_NTL.params[0]

            elif (int(year) == 2014) or (int(year) == 2015 and (month == 'Jan' or month == 'Feb')):

                olsmodelstring = 'CNTL' + month + year + ' ~ ' + 'area_bg' + month + year + ' + ' + \
                                'area_hr' + month + year + ' + ' + 'area_nr' + month + year
                model_NTL = ols(olsmodelstring, ntl_scale_NTL2).fit()
                # print(mdl + ': ' + olsmodelstring + "\n")
                # print(model_NTL.summary())
                # print("\nRetrieving manually the parameter estimates:")
                # print(model_NTL._results.params)
                model_NTL.save(results + 'ols_ntl.pickle', remove_data=False)

                mask = intersect2['landuse2014'] == 1
                intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL.params[1]
                mask = intersect2['landuse2014'] == 2
                intersect2.loc[mask, ['coef_ntl' + month + year]] = 0#model_NTL.params[2]
                mask = intersect2['landuse2014'] == 3
                intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL.params[2]
                mask = intersect2['landuse2014'] == 4
                intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL.params[3]
                intersect2['intercept_ntl' + month + year] = model_NTL.params[0]

            elif (int(year) == 2015 and (month != 'Jan' and month != 'Feb')) or \
                    (int(year) == 2016 and (month == 'Jan' or month == 'Feb' or month == 'Mar')):

                olsmodelstring = 'CNTL' + month + year + ' ~ ' + 'area_bg' + month + year + ' + ' + \
                                'area_hr' + month + year + ' + ' + 'area_nr' + month + year
                model_NTL = ols(olsmodelstring, ntl_scale_NTL2).fit()
                # print(mdl + ': ' + olsmodelstring + "\n")
                # print(model_NTL.summary())
                # print("\nRetrieving manually the parameter estimates:")
                # print(model_NTL._results.params)
                model_NTL.save(results + 'ols_ntl.pickle', remove_data=False)

                mask = intersect2['landuse2015'] == 1
                intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL.params[1]
                mask = intersect2['landuse2015'] == 2
                intersect2.loc[mask, ['coef_ntl' + month + year]] = 0#model_NTL.params[2]
                mask = intersect2['landuse2015'] == 3
                intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL.params[2]
                mask = intersect2['landuse2015'] == 4
                intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL.params[3]
                intersect2['intercept_ntl' + month + year] = model_NTL.params[0]

            elif (int(year) == 2016 and (month != 'Jan' and month != 'Feb' and month != 'Mar')):

                olsmodelstring = 'CNTL' + month + year + ' ~ ' + 'area_bg' + month + year + ' + ' + \
                                'area_hr' + month + year + ' + ' + 'area_nr' + month + year
                model_NTL = ols(olsmodelstring, ntl_scale_NTL2).fit()
                # print(mdl + ': ' + olsmodelstring + "\n")
                # print(model_NTL.summary())
                # print("\nRetrieving manually the parameter estimates:")
                # print(model_NTL._results.params)
                model_NTL.save(results + 'ols_ntl.pickle', remove_data=False)

                mask = intersect2['landuse2016'] == 1
                intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL.params[1]
                mask = intersect2['landuse2016'] == 2
                intersect2.loc[mask, ['coef_ntl' + month + year]] = 0#model_NTL.params[2]
                mask = intersect2['landuse2016'] == 3
                intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL.params[2]
                mask = intersect2['landuse2016'] == 4
                intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL.params[3]
                intersect2['intercept_ntl' + month + year] = model_NTL.params[0]

            elif int(year) == 2017:

                olsmodelstring = 'CNTL' + month + year + ' ~ ' + 'area_bg' + month + year + ' + ' + \
                                'area_hr' + month + year + ' + ' + 'area_nr' + month + year
                model_NTL = ols(olsmodelstring, ntl_scale_NTL2).fit()
                # print(mdl + ': ' + olsmodelstring + "\n")
                # print(model_NTL.summary())
                # print("\nRetrieving manually the parameter estimates:")
                # print(model_NTL._results.params)
                model_NTL.save(results + 'ols_ntl.pickle', remove_data=False)

                mask = intersect2['landuse2017'] == 1
                intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL.params[1]
                mask = intersect2['landuse2017'] == 2
                intersect2.loc[mask, ['coef_ntl' + month + year]] = 0#model_NTL.params[2]
                mask = intersect2['landuse2017'] == 3
                intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL.params[2]
                mask = intersect2['landuse2017'] == 4
                intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL.params[3]
                intersect2['intercept_ntl' + month + year] = model_NTL.params[0]

            elif int(year) == 2018:

                olsmodelstring = 'CNTL' + month + year + ' ~ ' + 'area_bg' + month + year + ' + ' + \
                                'area_hr' + month + year + ' + ' + 'area_nr' + month + year
                model_NTL = ols(olsmodelstring, ntl_scale_NTL2).fit()
                # print(mdl + ': ' + olsmodelstring + "\n")
                # print(model_NTL.summary())
                # print("\nRetrieving manually the parameter estimates:")
                # print(model_NTL._results.params)
                model_NTL.save(results + 'ols_ntl.pickle', remove_data=False)

                mask = intersect2['landuse2018'] == 1
                intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL.params[1]
                mask = intersect2['landuse2018'] == 2
                intersect2.loc[mask, ['coef_ntl' + month + year]] = 0#model_NTL.params[2]
                mask = intersect2['landuse2018'] == 3
                intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL.params[2]
                mask = intersect2['landuse2018'] == 4
                intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL.params[3]
                intersect2['intercept_ntl' + month + year] = model_NTL.params[0]
            if int(year) != 2012:
                intersect2['disNTL' + month + year] = intersect2['landuse_clip_area']*intersect2['coef_ntl' + month + year] + \
                                       (intersect2['intercept_ntl' + month + year]/intersect2['countNTL'])
                intersect2.reset_index(inplace=True)
                # We have to set the negative values to 0

                mask = intersect2['disNTL' + month + year] < 0
                mask2 = intersect2['disNTL' + month + year] >= 0
                print('Percentage error caused by removing negative values in nightlight: ',
                      abs(intersect2[mask].sum()['disNTL' + month + year] / intersect2[mask2].sum()['disNTL' + month + year])*100)

                intersect2.loc[mask, ['disNTL' + month + year]] = 0

                intersect2['disNTL_verify' + month + year] = intersect2['disNTL' + month + year].groupby(intersect2.ntl_clip_id).transform('sum')

                intersect2['disNTL_prime' + month + year] = \
                    intersect2['disNTL' + month + year] * (intersect2['CNTL' + month + year] / intersect2['disNTL_verify' + month + year])

    # in the level of census
    intersect2.set_index('census_id', inplace=True)
    intersect2['countPop'] = intersect2['index'].groupby(intersect2.index).transform('count')
    ntl_scale_Pop = intersect2.groupby(['census_id', 'landuse2014']).sum().loc[:,['intersect_area']]
    ntl_scale_Pop2 = ntl_scale_Pop.unstack('landuse2014')
    ntl_scale_Pop2.columns = ['area_bg', 'area_lr', 'area_hr', 'area_nr']
    ntl_scale_Pop2.fillna(0, inplace=True)

    ntl_scale_NTL2['target_pop2013'] = intersect2.groupby(intersect2.index).max()['estPop2013']

    ntl_scale_Pop2.reset_index(inplace=True)
    census.reset_index(inplace=True)
    ntl_scale_Pop2 = census.merge(ntl_scale_Pop2, left_on = census.index, right_on = ntl_scale_Pop2.index, how='left')
    ntl_scale_Pop2.drop(['key_0', 'index', 'Shape_Leng', 'Shape_Area', 'MAX_popult', 'census_id_x', 'census_area'],
                    inplace=True, axis=1)
    ntl_scale_Pop2['X'] = ntl_scale_Pop2.geometry.centroid.x
    ntl_scale_Pop2['Y'] = ntl_scale_Pop2.geometry.centroid.y

    print('Multiple Linear Regression for disaggregating population:')
    model_Pop = ols("estPop2013 ~ area_hr", ntl_scale_Pop2).fit()
    print(model_Pop.summary())
    print("\nRetrieving manually the parameter estimates:")
    print(model_Pop._results.params)

    model_Pop.save(results + 'ols_census.pickle', remove_data=False)

    from statsmodels.regression.linear_model import OLSResults
    model_Pop = OLSResults.load(results + 'ols_census.pickle')

    mask = intersect2['landuse2014'] == 1
    intersect2.loc[mask, ['coef_Pop2014']] = 0#model_Pop.params[1]
    mask = intersect2['landuse2014'] == 2
    intersect2.loc[mask, ['coef_Pop2014']] = 0#model_Pop.params[2]
    mask = intersect2['landuse2014'] == 3
    intersect2.loc[mask, ['coef_Pop2014']] = model_Pop.params[1]
    mask = intersect2['landuse2014'] == 4
    intersect2.loc[mask, ['coef_Pop2014']] = 0#model_Pop.params[4]
    intersect2['intercept_Pop2014'] = model_Pop.params[0]

    intersect2['disPop2013'] = intersect2['landuse_clip_area']*intersect2['coef_Pop2014'] + \
                            (intersect2['intercept_Pop2014']/intersect2['countPop'])

    intersect2['disPop2013_verify'] = intersect2['disPop2013'].groupby(intersect2.index).transform('sum')

    intersect2['disPop2013_prime'] = intersect2['disPop2013'] * (intersect2['estPop2013'] / intersect2['disPop2013_verify'])

    # in the level of night light
    for year in years:
        for month in months:
            if int(year) == 2012:
                try:
                    intersect2.set_index('ntl_clip_id', inplace=True)
                except:
                    print('ntl_clip_id is already the index')
                ntl_scale_NTL = intersect2.groupby(['ntl_clip_id', 'landuse2014']).sum().loc[:,['intersect_area']]
                ntl_scale_NTL2 = ntl_scale_NTL.unstack('landuse2014')
                ntl_scale_NTL2.columns = ['area_bg2012', 'area_lr2012', 'area_hr2012', 'area_nr2012']
                ntl_scale_NTL2.fillna(0, inplace=True)

                # print('NTL2013_lr sum of the values: ', ntl_scale_NTL2['NTL2013_lr'].sum())
                ntl_scale_NTL2['CNTL' + month + '2012'] = intersect2.groupby(intersect2.index).max()['CNTL' + month + '2012']

                ntl_scale_NTL2.to_csv(results + 'observations_median_' + mdl + month + year + date + '.csv')

            elif int(year) == 2013:
                try:
                    intersect2.set_index('ntl_clip_id', inplace=True)
                except:
                    print('ntl_clip_id is already the index')
                ntl_scale_NTL = intersect2.groupby(['ntl_clip_id', 'landuse2014']).sum().loc[:,['intersect_area','disNTL_prime' + month + '2013', 'disPop2013_prime']]
                ntl_scale_NTL2 = ntl_scale_NTL.unstack('landuse2014')
                ntl_scale_NTL2.columns = ['area_bg', 'area_lr', 'area_hr', 'area_nr','NTL2013_bg', 'NTL2013_lr', 'NTL2013_hr', 'NTL2013_nr',
                                        'Pop2013_bg', 'Pop2013_lr', 'Pop2013_hr', 'Pop2013_nr']
                ntl_scale_NTL2.fillna(0, inplace=True)
                ntl_scale_NTL2['NTL2013'] = ntl_scale_NTL2['NTL2013_bg'] + ntl_scale_NTL2['NTL2013_lr'] + \
                                            ntl_scale_NTL2['NTL2013_hr'] + ntl_scale_NTL2['NTL2013_nr']
                ntl_scale_NTL2['Pop2013'] = ntl_scale_NTL2['Pop2013_bg'] + ntl_scale_NTL2['Pop2013_lr'] + \
                                            ntl_scale_NTL2['Pop2013_hr'] + ntl_scale_NTL2['Pop2013_nr']

                # print('NTL2013_lr sum of the values: ', ntl_scale_NTL2['NTL2013_lr'].sum())
                ntl_scale_NTL2['CNTL' + month + '2013'] = intersect2.groupby(intersect2.index).max()['CNTL' + month + '2013']

                ntl_scale_NTL2.reset_index(inplace=True)
                NTL_clip.reset_index(inplace=True)
                ntl_scale_NTL2 = NTL_clip.merge(ntl_scale_NTL2, left_on = NTL_clip.index, right_on = ntl_scale_NTL2.index, how='left')
                ntl_scale_NTL2.drop(['key_0', 'index', 'Shape_Leng', 'Shape_Area', 'ntl_id','ntl_area', 'ntl_clip_id_x', 'ntl_clip_area'],inplace=True, axis=1)
                try:
                    ntl_scale_NTL2.drop(['level_0'], inplace=True, axis=1)
                    NTL_clip.drop(['level_0'], inplace=True, axis=1)
                except:
                    print('level_0 is not in the columns')
                ntl_scale_NTL2['X'] = ntl_scale_NTL2.geometry.centroid.x
                ntl_scale_NTL2['Y'] = ntl_scale_NTL2.geometry.centroid.y

                if mdl == 'nontl':
                    print(mdl + ': ' + month)
                    model_NTL_Pop = ols("Pop2013 ~  area_hr",ntl_scale_NTL2).fit()
                    print(model_NTL_Pop.summary())
                    print("\nRetrieving manually the parameter estimates:")
                    print(model_NTL_Pop._results.params)

                    model_NTL_Pop.save(results + 'ols_ntl_pop.pickle', remove_data=False)
                    from statsmodels.regression.linear_model import OLSResults
                    model_NTL_Pop = OLSResults.load(results + 'ols_ntl_pop.pickle')

                    ntl_scale_NTL2.to_csv(results + 'observations_' + 'lndus_' + date + '.csv')
                elif mdl == 'ntlMonthlyIncorrected_':
                    olsmodelstring = 'Pop2013' + ' ~ ' + 'area_hr' + ' + ' + 'CNTL' + month + year
                    print(mdl + '_'+ month + ': ' + olsmodelstring)
                    model_NTL_Pop = ols(olsmodelstring,ntl_scale_NTL2).fit()
                    print(model_NTL_Pop.summary())
                    print("\nRetrieving manually the parameter estimates:")
                    print(model_NTL_Pop._results.params)

                    model_NTL_Pop.save(results + 'ols_ntl_pop.pickle', remove_data=False)
                    from statsmodels.regression.linear_model import OLSResults
                    model_NTL_Pop = OLSResults.load(results + 'ols_ntl_pop.pickle')

                    ntl_scale_NTL2.to_csv(results + 'observations_median_' + mdl + month + year + date + '.csv')
                elif mdl == 'ntl_corrected_med_annualByMonth':
                    print(mdl + ': ' + month)
                    model_NTL_Pop = ols("Pop2013 ~  area_hr + NTL2013_hr + NTL2013_nr",ntl_scale_NTL2).fit()
                    print(model_NTL_Pop.summary())
                    print("\nRetrieving manually the parameter estimates:")
                    print(model_NTL_Pop._results.params)

                    model_NTL_Pop.save(results + 'ols_ntl_pop.pickle', remove_data=False)
                    from statsmodels.regression.linear_model import OLSResults
                    model_NTL_Pop = OLSResults.load(results + 'ols_ntl_pop.pickle')

                    ntl_scale_NTL2.to_csv(results + 'observations_median_' + 'ntl_annual_corrected_' +  year + date + '.csv')
                else:
                    olsmodelstring = 'Pop2013' + ' ~ ' + 'area_hr' + ' + ' + 'CNTL' + month + year
                    print(mdl + '_'+ month + ': ' + olsmodelstring)
                    model_NTL_Pop = ols(olsmodelstring,ntl_scale_NTL2).fit()
                    print(model_NTL_Pop.summary())
                    print("\nRetrieving manually the parameter estimates:")
                    print(model_NTL_Pop._results.params)

                    model_NTL_Pop.save(results + 'ols_ntl_pop.pickle', remove_data=False)
                    from statsmodels.regression.linear_model import OLSResults

                    ntl_scale_NTL2.to_csv(results + 'observations_median_' + mdl + month +  year + date + '.csv')

    # Prepare all years and months:
    try:
        NTL_clip.drop(['level_0'], inplace=True, axis=1)

    except:
        print('level_0 is not in the columns')

    NTL_clip.set_index('ntl_clip_id', inplace=True)
    NTL_clip_aux = NTL_clip
    # for all years
    for year in years:
        for month in months:
            if (int(year) == 2014) or (int(year) == 2015 and (month == 'Jan' or month == 'Feb')):
                ntl_prediction = intersect2.groupby(['ntl_clip_id', 'landuse2014']).sum().loc[:,['intersect_area','disNTL_prime' + month + year]]
                ntl_prediction2 = ntl_prediction.unstack('landuse2014')
                ntl_prediction2.columns = ['area_bg' + month + year, 'area_lr' + month + year, 'area_hr' + month + year, 'area_nr' + month + year,
                                                   'NTL_bg' + month + year, 'NTL_lr' + month + year, 'NTL_hr' + month + year, 'NTL_nr' + month + year]
                ntl_prediction2['CNTL' + month + year] = intersect2.groupby(intersect2.index).max()['CNTL' + month + year]
                NTL_clip_aux['ntl_clip_id_copy'] = NTL_clip_aux.index
                NTL_clip_aux = NTL_clip_aux.merge(ntl_prediction2, left_on=NTL_clip_aux.index, right_on=ntl_prediction2.index, how='left')
                NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
                NTL_clip_aux.drop('key_0', inplace=True, axis=1)

            elif (int(year) == 2015 and (month != 'Jan' and month != 'Feb')) or (int(year) == 2016 and (month == 'Jan' or month == 'Feb' or month == 'Mar')):
                ntl_prediction = intersect2.groupby(['ntl_clip_id', 'landuse2015']).sum().loc[:,['intersect_area','disNTL_prime' + month + year]]
                ntl_prediction2 = ntl_prediction.unstack('landuse2015')
                ntl_prediction2.columns = ['area_bg' + month + year, 'area_lr' + month + year, 'area_hr' + month + year, 'area_nr' + month + year,
                                                   'NTL_bg' + month + year, 'NTL_lr' + month + year, 'NTL_hr' + month + year, 'NTL_nr' + month + year]
                ntl_prediction2['CNTL' + month + year] = intersect2.groupby(intersect2.index).max()['CNTL' + month + year]
                NTL_clip_aux['ntl_clip_id_copy'] = NTL_clip_aux.index
                NTL_clip_aux = NTL_clip_aux.merge(ntl_prediction2, left_on=NTL_clip_aux.index, right_on=ntl_prediction2.index, how='left')
                NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
                NTL_clip_aux.drop('key_0', inplace=True, axis=1)

            elif (int(year) == 2016 and (month != 'Jan' and month != 'Feb' and month != 'Mar')):
                ntl_prediction = intersect2.groupby(['ntl_clip_id', 'landuse2016']).sum().loc[:,['intersect_area','disNTL_prime' + month + year]]
                ntl_prediction2 = ntl_prediction.unstack('landuse2016')
                ntl_prediction2.columns = ['area_bg' + month + year, 'area_lr' + month + year, 'area_hr' + month + year, 'area_nr' + month + year,
                                                   'NTL_bg' + month + year, 'NTL_lr' + month + year, 'NTL_hr' + month + year, 'NTL_nr' + month + year]
                ntl_prediction2['CNTL' + month + year] = intersect2.groupby(intersect2.index).max()['CNTL' + month + year]
                NTL_clip_aux['ntl_clip_id_copy'] = NTL_clip_aux.index
                NTL_clip_aux = NTL_clip_aux.merge(ntl_prediction2, left_on=NTL_clip_aux.index, right_on=ntl_prediction2.index, how='left')
                NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
                NTL_clip_aux.drop('key_0', inplace=True, axis=1)

            elif int(year) == 2017:
                ntl_prediction = intersect2.groupby(['ntl_clip_id', 'landuse2017']).sum().loc[:,['intersect_area','disNTL_prime' + month + year]]
                ntl_prediction2 = ntl_prediction.unstack('landuse2017')
                ntl_prediction2.columns = ['area_bg' + month + year, 'area_lr' + month + year, 'area_hr' + month + year, 'area_nr' + month + year,
                                                   'NTL_bg' + month + year, 'NTL_lr' + month + year, 'NTL_hr' + month + year, 'NTL_nr' + month + year]
                ntl_prediction2['CNTL' + month + year] = intersect2.groupby(intersect2.index).max()['CNTL' + month + year]
                NTL_clip_aux['ntl_clip_id_copy'] = NTL_clip_aux.index
                NTL_clip_aux = NTL_clip_aux.merge(ntl_prediction2, left_on=NTL_clip_aux.index, right_on=ntl_prediction2.index, how='left')
                NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
                NTL_clip_aux.drop('key_0', inplace=True, axis=1)

            elif int(year) == 2018:
                ntl_prediction = intersect2.groupby(['ntl_clip_id', 'landuse2018']).sum().loc[:,
                                 ['intersect_area', 'disNTL_prime' + month + year]]
                ntl_prediction2 = ntl_prediction.unstack('landuse2018')
                ntl_prediction2.columns = ['area_bg' + month + year, 'area_lr' + month + year, 'area_hr' + month + year,'area_nr' + month + year,
                                           'NTL_bg' + month + year, 'NTL_lr' + month + year, 'NTL_hr' + month + year, 'NTL_nr' + month + year]
                ntl_prediction2['CNTL' + month + year] = intersect2.groupby(intersect2.index).max()[
                    'CNTL' + month + year]
                NTL_clip_aux['ntl_clip_id_copy'] = NTL_clip_aux.index
                NTL_clip_aux = NTL_clip_aux.merge(ntl_prediction2, left_on=NTL_clip_aux.index,
                                                  right_on=ntl_prediction2.index, how='left')
                NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
                NTL_clip_aux.drop('key_0', inplace=True, axis=1)

    NTL_clip_aux.fillna(0, inplace=True)
    NTL_clip_aux['X'] = NTL_clip_aux.geometry.centroid.x
    NTL_clip_aux['Y'] = NTL_clip_aux.geometry.centroid.y

    if mdl == 'nontl':
        NTL_clip_aux.to_csv(results + 'NTL_Level_All_years_' + 'lndus_' + date + '.csv')
    elif mdl == 'ntlMonthlyIncorrected_':
        NTL_clip_aux.to_csv(results + 'NTL_Level_All_years_median_' + mdl + date + '.csv')
    elif mdl == 'ntl_corrected_med_annualByMonth':
        NTL_clip_aux.to_csv(results + 'NTL_Level_All_years_median_' + 'ntl_annual_corrected_' + date + '.csv')
    else:
        NTL_clip_aux.to_csv(results + 'NTL_Level_All_years_median_' + mdl + date + '.csv')

# Let's look at different predictions of the popualtion in 2013
for mdl in models:
    try:
        NTL_clip.set_index('ntl_clip_id', inplace=True)
    except:
        print('ntl_clip_id is already index')

    fig, axs = plt.subplots(4, 3, figsize=(20, 20))
    mm = 0
    nn = 0
    for month in months:
        print(month + ' ...')
        if mm < 3:
            gwr_model = gp.GeoDataFrame(pd.read_csv(results + 'GWR_All_years_' + mdl + month + '01312021.csv'))
            mae = mean_absolute_error(gwr_model['Pop2013'], gwr_model['pred'])
            mask1 = gwr_model['pred'] >= 0
            sumerr = np.abs(gwr_model['Pop2013'].sum() - gwr_model.loc[mask1, ['pred']].sum()[0])
            gwr_model.set_index('ntl_clip_id', inplace=True)
            df = NTL_clip.loc[:, ['Shape_Leng', 'geometry']].merge(gwr_model, left_on=NTL_clip.index, right_on=gwr_model.index, how='left')
            pp = df.plot(column='pred', cmap='Spectral_r', linewidth=0.1, ax=axs[nn, mm], edgecolor='white',legend=True)
            axs[nn,mm].get_xaxis().set_visible(False)
            axs[nn,mm].get_yaxis().set_visible(False)
            axs[nn,mm].title.set_text(mdl + ': 2013 Population Estimation (' + month + ')' + '\n MAE=' + str(mae) + '\n Overall Population Error=' + str(sumerr))
            mm = mm + 1
        else:
            mm=0
            nn=nn+1
            gwr_model = gp.GeoDataFrame(pd.read_csv(results + 'GWR_All_years_' + mdl + month + '01312021.csv'))
            mae = mean_absolute_error(gwr_model['Pop2013'], gwr_model['pred'])
            mask1 = gwr_model['pred'] >= 0
            sumerr = np.abs(gwr_model['Pop2013'].sum() - gwr_model.loc[mask1, ['pred']].sum()[0])
            gwr_model.set_index('ntl_clip_id', inplace=True)
            df = NTL_clip.loc[:, ['Shape_Leng', 'geometry']].merge(gwr_model, left_on=NTL_clip.index, right_on=gwr_model.index, how='left')
            pp = df.plot(column='pred', cmap='Spectral_r', linewidth=0.1, ax=axs[nn, mm], edgecolor='white',legend=True)
            axs[nn,mm].get_xaxis().set_visible(False)
            axs[nn,mm].get_yaxis().set_visible(False)
            axs[nn,mm].title.set_text(mdl + ': 2013 Population Estimation (' + month + ')' + '\n MAE=' + str(mae) + '\n Overall Population Error=' + str(sumerr))
            mm = mm + 1
    pp.get_figure().savefig(figures + mdl + '_population_Prediction_map_2013.png')

    # Check for seasonality
    df = pd.DataFrame(columns = ['MAE'], index=months)
    for month in months:
        print(month + ' ...')
        gwr_model = gp.GeoDataFrame(pd.read_csv(results + 'GWR_All_years_' + mdl + month + '01312021.csv'))
        mae = mean_absolute_error(gwr_model['Pop2013'], gwr_model['pred'])
        mask1=gwr_model['pred']>=0
        sumerr = np.abs(gwr_model['Pop2013'].sum() - gwr_model.loc[mask1, ['pred']].sum()[0])
        df.loc[month, 'MAE'] = mae
        df.loc[month, 'sumerr'] = sumerr

    # df.reset_index(inplace=True)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    sns.pointplot(data=df, x=df.index, y='MAE', color='green', ax = axs[0])
    axs[0].title.set_text(mdl + ': Mean Absolute Error')
    pp = sns.pointplot(data=df, x=df.index, y='sumerr', color='green', ax = axs[1])
    axs[1].title.set_text('Difference in Overall Population Estimation')
    axs[1].set(ylabel=mdl + ': Difference in Overall Population')

    pp.get_figure().savefig(figures + mdl + '_Seasonality_in_Estimated_Population_by_Different_Month_From_2013.png')


    # Landuse model + annual ntl
    # Overall popualtion predictions by different reference months (one population model)
    aux1 = []
    for year in years:
        if int(year) >= 2014:
            for month in months:
                aux1.append('estpop' + month + year)
    nn = 0
    predpoulations = pd.DataFrame(index=aux1)

    for mm in months:
        ntl_scale_NTL2 = pd.read_csv(results + 'observations_median_' + mdl + month + date + '.csv')
        NTL_clip_aux = pd.read_csv(results + 'NTL_Level_All_years_median_' + mdl + date + '.csv')
        gwr_model = pd.read_csv(results + 'GWR_All_years_' + mdl + mm + '01312021.csv')
        NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
        gwr_model.set_index('ntl_clip_id', inplace=True)
        NTL_clip_aux['ntl_clip_id'] = NTL_clip_aux.index
        predict_all_years = NTL_clip_aux.merge(gwr_model, left_on=NTL_clip_aux.index, right_on=gwr_model.index, how='left')
        predict_all_years.rename({})

        for year in years:
            for month in months:
                if int(year) >= 2014:
                    predict_all_years['estpop' + month + year] = predict_all_years['X.Intercept.'] + \
                                                     (predict_all_years['area_hr'] * predict_all_years['area_hr' + month + year]) + \
                                                     (predict_all_years['area_nr'] * predict_all_years['area_nr' + month + year]) + \
                                                     (predict_all_years['CNTL' + mm + '2013'] * predict_all_years['CNTL' + month + year])
        predpoulations[mm] = predict_all_years.iloc[:, -60:].apply(lambda x: x[x >= 0].sum(), axis=0)

    ntl_scale_NTL2 = pd.read_csv(results + 'observations_median_' + mdl + month + date + '.csv')
    NTL_clip_aux = pd.read_csv(results + 'NTL_Level_All_years_median_' + mdl + date + '.csv')
    gwr_model = pd.read_csv(results + 'GWR_All_years_lndus_' + mm + '01312021.csv')
    NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
    gwr_model.set_index('ntl_clip_id', inplace=True)
    NTL_clip_aux['ntl_clip_id'] = NTL_clip_aux.index
    predict_all_years = NTL_clip_aux.merge(gwr_model, left_on=NTL_clip_aux.index, right_on=gwr_model.index, how='left')
    predict_all_years.rename({})
    for year in years:
        for month in months:
            if int(year) >= 2014:
                predict_all_years['estpop' + month + year] = predict_all_years['X.Intercept.'] + \
                                                (predict_all_years['area_hr'] * predict_all_years['area_hr' + month + year]) + \
                                                (predict_all_years['area_nr'] * predict_all_years['area_nr' + month + year])

    predpoulations['Landuse' + mm] = predict_all_years.iloc[:, -60:].apply(lambda x: x[x >= 0].sum(), axis=0)

    predpoulations.reset_index(inplace=True)
    predpoulations.rename({'index': 'monthyear'}, axis=1, inplace=1)

    lables = list(predpoulations.monthyear)
    fig, ax = plt.subplots(figsize=(20,15))
    pp=predpoulations.plot(cmap="Spectral", ax = ax)
    plt.xlabel('Month/Year', fontsize=18)
    plt.ylabel('Estimated Population', fontsize=16)
    plt.xticks(range(0,len(list(predpoulations.monthyear))), lables, rotation='vertical')

    pp.get_figure().savefig(figures + mdl + '_Overall_popualtion_predictions_by_A_Single_reference_months_From_2013.png')


    # Map of each Predicted month by considering 12 months of reference (12 population model on 12 months):
    aux1 = []
    for year in years:
        if int(year) >= 2014:
            for month in months:
                aux1.append('estpop' + month + year)

    predpoulations = pd.DataFrame(columns = ['Values'], index=aux1)

    for year in years:
        if int(year) >= 2014:
            for month in months:
                ntl_scale_NTL2 = pd.read_csv(
                        results + 'observations_median_' + mdl + month + date + '.csv')
                NTL_clip_aux = pd.read_csv(
                        results + 'NTL_Level_All_years_median_' + mdl + date + '.csv')
                gwr_model = pd.read_csv(results + 'GWR_All_years_' + mdl + month + '01312021.csv')
                NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
                gwr_model.set_index('ntl_clip_id', inplace=True)
                NTL_clip_aux['ntl_clip_id'] = NTL_clip_aux.index
                predict_all_years = NTL_clip_aux.merge(gwr_model, left_on=NTL_clip_aux.index, right_on=gwr_model.index,
                                                           how='left')
                predict_all_years.rename({})
                predict_all_years['estpop' + month + year] = predict_all_years['X.Intercept.'] + \
                                                     (predict_all_years['area_hr'] * predict_all_years['area_hr' + month + year]) + \
                                                     (predict_all_years['area_nr'] * predict_all_years['area_nr' + month + year]) + \
                                                     (predict_all_years['CNTL' + month + '2013'] * predict_all_years['CNTL' + month + year])
                predpoulations.loc['estpop' + month + year, 'Values'] = predict_all_years.loc[:, ['estpop' + month + year]].apply(lambda x: x[x >= 0].sum(), axis=0)[0]

    ntl_scale_NTL2 = pd.read_csv(results + 'observations_median_' + mdl + month + date + '.csv')
    NTL_clip_aux = pd.read_csv(results + 'NTL_Level_All_years_median_' + mdl + date + '.csv')
    gwr_model = pd.read_csv(results + 'GWR_All_years_lndus_' + month + '01312021.csv')
    NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
    gwr_model.set_index('ntl_clip_id', inplace=True)
    NTL_clip_aux['ntl_clip_id'] = NTL_clip_aux.index
    predict_all_years = NTL_clip_aux.merge(gwr_model, left_on=NTL_clip_aux.index, right_on=gwr_model.index, how='left')
    predict_all_years.rename({})
    for year in years:
        for month in months:
            if int(year) >= 2014:
                predict_all_years['estpop' + month + year] = predict_all_years['X.Intercept.'] + \
                                                (predict_all_years['area_hr'] * predict_all_years['area_hr' + month + year]) + \
                                                (predict_all_years['area_nr'] * predict_all_years['area_nr' + month + year])

    predpoulations['Landuse' + month] = predict_all_years.iloc[:, -60:].apply(lambda x: x[x >= 0].sum(), axis=0)

    predpoulations.reset_index(inplace=True)
    predpoulations.rename({'index': 'monthyear'}, axis=1, inplace=1)

    lables = list(predpoulations.monthyear)
    fig, ax = plt.subplots(figsize=(20,15))
    pp=predpoulations.plot(cmap="Spectral", ax = ax)
    plt.xlabel('Month/Year', fontsize=18)
    plt.ylabel('Estimated Population', fontsize=16)
    plt.xticks(range(0,len(list(predpoulations.monthyear))), lables, rotation='vertical')

    pp.get_figure().savefig(figures + mdl + '_Overall_popualtion_predictions_by_different_reference_months_From_2013.png')

    # plot nightights sum
    aux2 = []
    for year in years:
        if int(year) >= 2014:
            for month in months:
                aux2.append('CNTL' + month + year)

    nightlights = pd.DataFrame(index=aux2)
    NTL_clip_aux = pd.read_csv(results + 'NTL_Level_All_years_median_' + mdl + date + '.csv')
    nightlights = NTL_clip_aux.loc[:, aux2].apply(lambda x: x[x >= 0].sum(), axis=0)

    lables = list(nightlights.index)
    fig, ax = plt.subplots(figsize=(20,15))
    pp = nightlights.plot(cmap="Spectral", ax = ax)
    plt.xlabel('Month/Year', fontsize=18)
    plt.ylabel('Sum of Nightlight For The City', fontsize=16)
    plt.xticks(range(0,len(list(nightlights.index))), lables, rotation='vertical')

    pp.get_figure().savefig(figures + mdl + '_Night_light_sum_other_years.png')

    # plot 2013 sum of light
    aux3 = []
    for year in years:
        if int(year) <= 2013:
            for month in months:
                aux3.append('CNTL' + month + year)

    nightlights20122013 = pd.DataFrame(columns=['Value'],index=aux3)
    for year in years:
        if int(year) == 2012:
            for mm in months:
                print(mm)
                ntl_scale_NTL2 = pd.read_csv(results + 'observations_median_' + mdl + mm + year + date + '.csv')
                nightlights20122013.loc['CNTL' + mm + '2012', ['Value']] = ntl_scale_NTL2.loc[:, 'CNTL' + mm + '2012'].sum()
        elif int(year) == 2013:
            for mm in months:
                print(mm)
                ntl_scale_NTL2 = pd.read_csv(results + 'observations_median_' + mdl + mm + year  + date + '.csv')
                nightlights20122013.loc['CNTL' + mm + '2013', ['Value']] = ntl_scale_NTL2.loc[:, 'CNTL' + mm + '2013'].sum()

    lables = list(nightlights20122013.index)
    fig, ax = plt.subplots(figsize=(20,15))
    pp = nightlights20122013.plot(cmap="Spectral", ax = ax)
    plt.xlabel('Month/Year', fontsize=18)
    plt.ylabel('Estimated Population', fontsize=16)
    plt.xticks(range(0,len(list(nightlights20122013.index))), lables, rotation='vertical')

    pp.get_figure().savefig(figures + mdl + '_Night_light_sum_2013.png')

    nightlights20122013.loc['CNTLJan2013_', ['Value']] = nightlights20122013.loc['CNTLJan2013', ['Value']] - \
                                                         nightlights20122013.loc['CNTLJan2012', ['Value']]
    nightlights20122013.loc['CNTLFeb2013_', ['Value']] = nightlights20122013.loc['CNTLFeb2013', ['Value']] - \
                                                         nightlights20122013.loc['CNTLFeb2012', ['Value']]
    nightlights20122013.loc['CNTLMar2013_', ['Value']] = nightlights20122013.loc['CNTLMar2013', ['Value']] - \
                                                         nightlights20122013.loc['CNTLMar2012', ['Value']]
    nightlights20122013.loc['CNTLApr2013_', ['Value']] = nightlights20122013.loc['CNTLApr2013', ['Value']] - \
                                                         nightlights20122013.loc['CNTLApr2012', ['Value']]
    nightlights20122013.loc['CNTLMay2013_', ['Value']] = nightlights20122013.loc['CNTLMay2013', ['Value']] - \
                                                         nightlights20122013.loc['CNTLMay2012', ['Value']]
    nightlights20122013.loc['CNTLJun2013_', ['Value']] = nightlights20122013.loc['CNTLJun2013', ['Value']] - \
                                                         nightlights20122013.loc['CNTLJun2012', ['Value']]
    nightlights20122013.loc['CNTLJul2013_', ['Value']] = nightlights20122013.loc['CNTLJul2013', ['Value']] - \
                                                         nightlights20122013.loc['CNTLJul2012', ['Value']]
    nightlights20122013.loc['CNTLAug2013_', ['Value']] = nightlights20122013.loc['CNTLAug2013', ['Value']] - \
                                                         nightlights20122013.loc['CNTLAug2012', ['Value']]
    nightlights20122013.loc['CNTLSep2013_', ['Value']] = nightlights20122013.loc['CNTLSep2013', ['Value']] - \
                                                         nightlights20122013.loc['CNTLSep2012', ['Value']]
    nightlights20122013.loc['CNTLOct2013_', ['Value']] = nightlights20122013.loc['CNTLOct2013', ['Value']] - \
                                                         nightlights20122013.loc['CNTLOct2012', ['Value']]
    nightlights20122013.loc['CNTLNov2013_', ['Value']] = nightlights20122013.loc['CNTLNov2013', ['Value']] - \
                                                         nightlights20122013.loc['CNTLNov2012', ['Value']]
    nightlights20122013.loc['CNTLDec2013_', ['Value']] = nightlights20122013.loc['CNTLDec2013', ['Value']] - \
                                                         nightlights20122013.loc['CNTLDec2012', ['Value']]

    lables = list(nightlights20122013.index[24:])
    fig, ax = plt.subplots(figsize=(20,15))
    pp = nightlights20122013.iloc[24:, :].plot(cmap="Spectral", ax = ax)
    plt.xlabel('Month/Year', fontsize=18)
    plt.ylabel('Estimated Population', fontsize=16)
    plt.xticks(range(0,len(list(nightlights20122013.index[24:]))), lables, rotation='vertical')

    # plot nightlight by pixel
    NTL_clip_aux = pd.read_csv(results + 'NTL_Level_All_years_median_' + mdl + date + '.csv')

    sns.set(rc={'figure.figsize': (20, 15)}, style="whitegrid")
    fig, ax = plt.subplots(1, 1)
    for n in range(0, NTL_clip_aux.loc[:, aux2].shape[0]):
        pp = sns.lineplot(x=list(NTL_clip_aux.loc[:, aux2].columns), y = NTL_clip_aux.loc[:, aux2].iloc[n, :].astype('float'), ax=ax)
        ax.set(ylabel='Nightlight Value')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    pp.get_figure().savefig(figures + mdl + '_Night_light_other_years_by_pixel.png')


    # Map of each Predicted month
    NTL_clip_aux = pd.read_csv(results + 'NTL_Level_All_years_median_' + mdl + date + '.csv')
    NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
    predpoulations = pd.DataFrame(index=NTL_clip_aux.index)

    for year in years:
        if int(year) >= 2014:
            fig, axs = plt.subplots(4, 3, figsize=(20, 20))
            mm = 0
            nn = 0
            for month in months:
                print(month + ' ...')
                if mm < 3:
                    gwr_model = gp.GeoDataFrame(pd.read_csv(results + 'GWR_All_years_' + mdl + month + '01312021.csv'))
                    ntl_scale_NTL2 = pd.read_csv(results + 'observations_median_' + mdl + month + date + '.csv')
                    NTL_clip_aux = pd.read_csv(results + 'NTL_Level_All_years_median_' + mdl + date + '.csv')
                    NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
                    gwr_model.set_index('ntl_clip_id', inplace=True)
                    NTL_clip_aux['ntl_clip_id'] = NTL_clip_aux.index
                    predict_all_years = NTL_clip_aux.merge(gwr_model, left_on=NTL_clip_aux.index, right_on=gwr_model.index,
                                                               how='left')
                    predict_all_years.drop('key_0', inplace=True, axis=1)
                    predict_all_years.drop('geometry', inplace=True, axis=1)
                    predict_all_years.rename({})
                    predict_all_years['estpop' + month + year] = predict_all_years['X.Intercept.'] + \
                                                                     (predict_all_years['area_hr'] * predict_all_years['area_hr' + month + year]) + \
                                                                     (predict_all_years['area_nr'] * predict_all_years['area_nr' + month + year]) + \
                                                                     (predict_all_years['CNTL' + month + '2013'] * predict_all_years['CNTL' + month + year])
                    predict_all_years.set_index('ntl_clip_id', inplace=True)

                    predpoulations['estpop' + month + year] = predict_all_years.loc[:, ['estpop' + month + year]]

                    df = NTL_clip.loc[:, ['Shape_Leng', 'geometry']].merge(predict_all_years, left_on=NTL_clip.index, right_on=predict_all_years.index, how='left')
                    df.loc[df['estpop' + month + year] < 0, ['estpop' + month + year]] = 0
                    pp = df.plot(column='estpop' + month + year, cmap='Spectral_r', linewidth=0.1, ax=axs[nn, mm], edgecolor='white',legend=True, vmin=0, vmax = 8000)
                    axs[nn,mm].get_xaxis().set_visible(False)
                    axs[nn,mm].get_yaxis().set_visible(False)
                    axs[nn,mm].title.set_text(mdl + ': ' + year + ' Population Estimation (' + month + ')')
                    mm = mm + 1
                else:
                    mm=0
                    nn=nn+1
                    gwr_model = gp.GeoDataFrame(pd.read_csv(results + 'GWR_All_years_' + mdl + month + '01312021.csv'))
                    ntl_scale_NTL2 = pd.read_csv(results + 'observations_median_' + mdl + month + date + '.csv')
                    NTL_clip_aux = pd.read_csv(results + 'NTL_Level_All_years_median_' + mdl + date + '.csv')
                    NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
                    gwr_model.set_index('ntl_clip_id', inplace=True)
                    NTL_clip_aux['ntl_clip_id'] = NTL_clip_aux.index
                    predict_all_years = NTL_clip_aux.merge(gwr_model, left_on=NTL_clip_aux.index, right_on=gwr_model.index,
                                                               how='left')
                    predict_all_years.drop('key_0', inplace=True, axis=1)
                    predict_all_years.drop('geometry', inplace=True, axis=1)
                    predict_all_years.rename({})
                    predict_all_years['estpop' + month + year] = predict_all_years['X.Intercept.'] + \
                                                                     (predict_all_years['area_hr'] * predict_all_years['area_hr' + month + year]) + \
                                                                     (predict_all_years['area_nr'] * predict_all_years['area_nr' + month + year]) + \
                                                                     (predict_all_years['CNTL' + month + '2013'] * predict_all_years['CNTL' + month + year])
                    predict_all_years.set_index('ntl_clip_id', inplace=True)

                    predpoulations['estpop' + month + year] = predict_all_years.loc[:, ['estpop' + month + year]]

                    df = NTL_clip.loc[:, ['Shape_Leng', 'geometry']].merge(predict_all_years, left_on=NTL_clip.index, right_on=predict_all_years.index, how='left')
                    df.loc[df['estpop' + month + year] < 0, ['estpop' + month + year]] = 0
                    pp = df.plot(column='estpop' + month + year, cmap='Spectral_r', linewidth=0.1, ax=axs[nn, mm], edgecolor='white',legend=True, vmin=0, vmax = 8000)
                    axs[nn,mm].get_xaxis().set_visible(False)
                    axs[nn,mm].get_yaxis().set_visible(False)
                    axs[nn,mm].title.set_text(mdl + ': ' + year + ' Population Estimation (' + month + ')')
                    mm = mm + 1

        pp.get_figure().savefig(figures + mdl + '_population_Prediction_maps_' + year + '.png')

    # Map of difference for each predicted month:
    predpoulationsdiff = predpoulations.diff(axis=1)
    predpoulations[predpoulations < 0] = 0

    predpoulationsdiff.columns = predpoulations.columns + 'Chng'
    predpoulationsdiff.drop('estpopJan2014Chng', inplace=True, axis=1)
    # NTL_clip.set_index('ntl_clip_id', inplace=True)

    vmin = np.min(np.array(predpoulationsdiff))
    vmax = np.max(np.array(predpoulationsdiff))

    for year in years:
        if int(year) >= 2014:
            fig, axs = plt.subplots(4, 3, figsize=(20, 20))
            mm = 0
            nn = 0
            for month in months:
                print(month + ' ...')
                if mm < 3:
                    NTL_clip_aux = pd.read_csv(results + 'NTL_Level_All_years_median_' + mdl + date + '.csv')
                    NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
                    NTL_clip_aux['ntl_clip_id'] = NTL_clip_aux.index
                    predict_all_years_diff = NTL_clip_aux.merge(predpoulationsdiff, left_on=NTL_clip_aux.index, right_on=predpoulationsdiff.index,
                                                               how='left')
                    predict_all_years_diff.drop('key_0', inplace=True, axis=1)
                    predict_all_years_diff.drop('geometry', inplace=True, axis=1)
                    predict_all_years.rename({})
                    predict_all_years_diff.set_index('ntl_clip_id', inplace=True)

                    df = NTL_clip.loc[:, ['Shape_Leng', 'geometry']].merge(predict_all_years_diff, left_on=NTL_clip.index,
                                                                           right_on=predict_all_years_diff.index, how='left')
                    if int(year)==2014 and month == 'Jan':
                        mm = mm + 1
                    else:
                        pp = df.plot(column='estpop' + month + year + 'Chng', cmap='Spectral_r', linewidth=0.1, ax=axs[nn, mm], edgecolor='white',legend=True)
                        axs[nn,mm].get_xaxis().set_visible(False)
                        axs[nn,mm].get_yaxis().set_visible(False)
                        axs[nn,mm].title.set_text(mdl + ': ' + year + ' Population Change Estimation (' + month + ')')
                        mm = mm + 1
                else:
                    mm=0
                    nn=nn+1
                    NTL_clip_aux = pd.read_csv(results + 'NTL_Level_All_years_median_' + mdl + date + '.csv')
                    NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
                    NTL_clip_aux['ntl_clip_id'] = NTL_clip_aux.index
                    predict_all_years_diff = NTL_clip_aux.merge(predpoulationsdiff, left_on=NTL_clip_aux.index, right_on=predpoulationsdiff.index,
                                                               how='left')
                    predict_all_years_diff.drop('key_0', inplace=True, axis=1)
                    predict_all_years_diff.drop('geometry', inplace=True, axis=1)
                    predict_all_years.rename({})
                    predict_all_years_diff.set_index('ntl_clip_id', inplace=True)
                    df = NTL_clip.loc[:, ['Shape_Leng', 'geometry']].merge(predict_all_years_diff, left_on=NTL_clip.index,right_on=predict_all_years_diff.index, how='left')
                    pp = df.plot(column='estpop' + month + year + 'Chng', cmap='Spectral_r', linewidth=0.1, ax=axs[nn, mm],
                                 edgecolor='white', legend=True)
                    axs[nn, mm].get_xaxis().set_visible(False)
                    axs[nn, mm].get_yaxis().set_visible(False)
                    axs[nn, mm].title.set_text(mdl + ': ' + year + ' Population Change Estimation (' + month + ')')
                    mm = mm + 1

        pp.get_figure().savefig(figures + mdl + '_population_Difference_Prediction_maps_' + year + '.png')

#
# vmin = np.min(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014', 'estpop2015', 'estpop2016', 'estpop2017', 'estpop2018']]))
# vmax = np.max(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014', 'estpop2015', 'estpop2016', 'estpop2017', 'estpop2018']]))
# # vmin=-6000
# # vmax=7000
#
# fig, axs = plt.subplots(2, 3, figsize=(20, 10))
# NTL_clip_aux3_noNeg.plot(column='pred', cmap='Spectral_r', linewidth=0.1, ax=axs[0,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[0,0].get_xaxis().set_visible(False)
# axs[0,0].get_yaxis().set_visible(False)
# axs[0,0].title.set_text('2013 Population Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2014', cmap='Spectral_r', linewidth=0.1, ax=axs[0,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[0,1].get_xaxis().set_visible(False)
# axs[0,1].get_yaxis().set_visible(False)
# axs[0,1].title.set_text('2014 Population Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2015', cmap='Spectral_r', linewidth=0.1, ax=axs[0,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[0,2].get_xaxis().set_visible(False)
# axs[0,2].get_yaxis().set_visible(False)
# axs[0,2].title.set_text('2015 Population Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2016', cmap='Spectral_r', linewidth=0.1, ax=axs[1,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[1,0].get_xaxis().set_visible(False)
# axs[1,0].get_yaxis().set_visible(False)
# axs[1,0].title.set_text('2016 Population Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2017', cmap='Spectral_r', linewidth=0.1, ax=axs[1,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[1,1].get_xaxis().set_visible(False)
# axs[1,1].get_yaxis().set_visible(False)
# axs[1,1].title.set_text('2017 Population Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2018', cmap='Spectral_r', linewidth=0.1, ax=axs[1,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[1,2].get_xaxis().set_visible(False)
# axs[1,2].get_yaxis().set_visible(False)
# axs[1,2].title.set_text('2018 Population Estimation')
# plt.suptitle(mdl + ": Landuse and NTL (One Year + Incorrected)", size=16)
#
# # Landuse model + corrected annual ntl
# ntl_scale_NTL2 = pd.read_csv(results + 'observations_median_' + 'ntl_annual_corrected_' + date + '.csv')
# ntl_scale_NTL2.drop('CNTL2013', inplace=True, axis=1)
# NTL_clip_aux = pd.read_csv(results + 'NTL_Level_All_years_median_ntl_annual_corrected_01312021.csv')
# gwr_model = pd.read_csv(results + 'GWR_median_ntl_annual_corrected_01312021.csv')
# NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
# gwr_model.set_index('ntl_clip_id', inplace=True)
# NTL_clip_aux['ntl_clip_id'] = NTL_clip_aux.index
# predict_all_years = NTL_clip_aux.merge(gwr_model, left_on=NTL_clip_aux.index, right_on=gwr_model.index, how='left')
# predict_all_years.rename({})
# for year in years:
#     if int(year) >= 2014:
#         predict_all_years['estpop' + year] = predict_all_years['X.Intercept.'] + \
#                                              (predict_all_years['area_hr'] * predict_all_years['area_hr' + year]) + \
#                                              (predict_all_years['CNTL2013'] * predict_all_years['CNTL' + year])
# predict_all_years.drop(['key_0'],inplace=True, axis=1)
# mask = predict_all_years['pred'] >=0
# df.iloc[1, 2] = predict_all_years.loc[mask, ['pred']].sum()[0]
# mask = predict_all_years['estpop2014'] >=0
# df.iloc[2, 2] = predict_all_years.loc[mask, ['estpop2014']].sum()[0]
# mask = predict_all_years['estpop2015'] >=0
# df.iloc[3, 2] = predict_all_years.loc[mask, ['estpop2015']].sum()[0]
# mask = predict_all_years['estpop2016'] >=0
# df.iloc[4, 2] = predict_all_years.loc[mask, ['estpop2016']].sum()[0]
# mask = predict_all_years['estpop2017'] >=0
# df.iloc[5, 2] = predict_all_years.loc[mask, ['estpop2017']].sum()[0]
# mask = predict_all_years['estpop2018'] >=0
# df.iloc[6, 2] = predict_all_years.loc[mask, ['estpop2018']].sum()[0]
# df.iloc[-3, 2] = mean_squared_error(ntl_scale_NTL2.Pop2013, predict_all_years.pred, squared=True)
# df.iloc[-2:, 2] = mean_absolute_error(ntl_scale_NTL2.Pop2013, predict_all_years.pred)
# predict_all_years.to_csv(results + 'predict_all_years_ntl_annual_corrected_01312021.csv')
#
# predict_all_years.drop('geometry', inplace=True, axis=1)
# NTL_clip_aux2 = NTL_clip
# predict_all_years.set_index('ntl_clip_id', inplace=True)
# NTL_clip_aux3 = NTL_clip_aux2.merge(predict_all_years, left_on=NTL_clip_aux2.index, right_on=predict_all_years.index)
# NTL_clip_aux3.drop('key_0', inplace=True, axis=1)
#
# NTL_clip_aux3_noNeg = NTL_clip_aux3
# mask = NTL_clip_aux3_noNeg['pred'] < 0
# NTL_clip_aux3_noNeg.loc[mask, ['pred']] = 0
#
# for year in years:
#     if int(year) >= 2014:
#         mask = NTL_clip_aux3_noNeg['estpop' + year] < 0
#         NTL_clip_aux3_noNeg.loc[mask, ['estpop' + year]] = 0
#
# NTL_clip_aux3_noNeg['estpop2014change'] = NTL_clip_aux3_noNeg['estpop2014'] - NTL_clip_aux3_noNeg['pred']
# NTL_clip_aux3_noNeg['estpop2015change'] = NTL_clip_aux3_noNeg['estpop2015'] - NTL_clip_aux3_noNeg['estpop2014']
# NTL_clip_aux3_noNeg['estpop2016change'] = NTL_clip_aux3_noNeg['estpop2016'] - NTL_clip_aux3_noNeg['estpop2015']
# NTL_clip_aux3_noNeg['estpop2017change'] = NTL_clip_aux3_noNeg['estpop2017'] - NTL_clip_aux3_noNeg['estpop2016']
# NTL_clip_aux3_noNeg['estpop2018change'] = NTL_clip_aux3_noNeg['estpop2018'] - NTL_clip_aux3_noNeg['estpop2017']
#
# vmin = np.min(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014', 'estpop2015', 'estpop2016', 'estpop2017', 'estpop2018']]))
# vmax = np.max(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014', 'estpop2015', 'estpop2016', 'estpop2017', 'estpop2018']]))
# # vmin=-6000
# # vmax=7000
#
# fig, axs = plt.subplots(2, 3, figsize=(20, 10))
# NTL_clip_aux3_noNeg.plot(column='pred', cmap='Spectral_r', linewidth=0.1, ax=axs[0,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[0,0].get_xaxis().set_visible(False)
# axs[0,0].get_yaxis().set_visible(False)
# axs[0,0].title.set_text('2013 Population Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2014', cmap='Spectral_r', linewidth=0.1, ax=axs[0,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[0,1].get_xaxis().set_visible(False)
# axs[0,1].get_yaxis().set_visible(False)
# axs[0,1].title.set_text('2014 Population Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2015', cmap='Spectral_r', linewidth=0.1, ax=axs[0,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[0,2].get_xaxis().set_visible(False)
# axs[0,2].get_yaxis().set_visible(False)
# axs[0,2].title.set_text('2015 Population Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2016', cmap='Spectral_r', linewidth=0.1, ax=axs[1,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[1,0].get_xaxis().set_visible(False)
# axs[1,0].get_yaxis().set_visible(False)
# axs[1,0].title.set_text('2016 Population Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2017', cmap='Spectral_r', linewidth=0.1, ax=axs[1,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[1,1].get_xaxis().set_visible(False)
# axs[1,1].get_yaxis().set_visible(False)
# axs[1,1].title.set_text('2017 Population Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2018', cmap='Spectral_r', linewidth=0.1, ax=axs[1,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[1,2].get_xaxis().set_visible(False)
# axs[1,2].get_yaxis().set_visible(False)
# axs[1,2].title.set_text('2018 Population Estimation')
# plt.suptitle(mdl + ": Landuse and NTL (One Year + Corrected)", size=16)
#
# # Landuse model + corrected monthly ntl
# ntl_scale_NTL2 = pd.read_csv(results + 'observations_median_' + 'ntl_monthly_corrected_' + date + '.csv')
# NTL_clip_aux = pd.read_csv(results + 'NTL_Level_All_years_median_ntl_monthly_corrected_01312021.csv')
# gwr_model = pd.read_csv(results + 'GWR_median_ntl_monthly_corrected_01312021.csv')
# NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
# gwr_model.set_index('ntl_clip_id', inplace=True)
# NTL_clip_aux['ntl_clip_id'] = NTL_clip_aux.index
# predict_all_years = NTL_clip_aux.merge(gwr_model, left_on=NTL_clip_aux.index, right_on=gwr_model.index, how='left')
# predict_all_years.rename({})
# for year in years:
#     if int(year) >= 2014:
#         predict_all_years['estpop' + year] = predict_all_years['X.Intercept.'] + \
#                                              (predict_all_years['area_hr'] * predict_all_years['area_hr' + year]) + \
#                                              (predict_all_years['CNTL2013'] * predict_all_years['CNTL' + year])
# predict_all_years.drop(['key_0'],inplace=True, axis=1)
# mask = predict_all_years['pred'] >=0
# df.iloc[1, 3] = predict_all_years.loc[mask, ['pred']].sum()[0]
# mask = predict_all_years['estpop2014'] >=0
# df.iloc[2, 3] = predict_all_years.loc[mask, ['estpop2014']].sum()[0]
# mask = predict_all_years['estpop2015'] >=0
# df.iloc[3, 3] = predict_all_years.loc[mask, ['estpop2015']].sum()[0]
# mask = predict_all_years['estpop2016'] >=0
# df.iloc[4, 3] = predict_all_years.loc[mask, ['estpop2016']].sum()[0]
# mask = predict_all_years['estpop2017'] >=0
# df.iloc[5, 3] = predict_all_years.loc[mask, ['estpop2017']].sum()[0]
# mask = predict_all_years['estpop2018'] >=0
# df.iloc[6, 3] = predict_all_years.loc[mask, ['estpop2018']].sum()[0]
# df.iloc[-3, 3] = mean_squared_error(ntl_scale_NTL2.Pop2013, predict_all_years.pred, squared=True)
# df.iloc[-2:, 3] = mean_absolute_error(ntl_scale_NTL2.Pop2013, predict_all_years.pred)
# predict_all_years.to_csv(results + 'predict_all_years_ntl_monthly_corrected_01312021.csv')
#
# predict_all_years.drop('geometry', inplace=True, axis=1)
# NTL_clip_aux2 = NTL_clip
# predict_all_years.set_index('ntl_clip_id', inplace=True)
# NTL_clip_aux3 = NTL_clip_aux2.merge(predict_all_years, left_on=NTL_clip_aux2.index, right_on=predict_all_years.index)
# NTL_clip_aux3.drop('key_0', inplace=True, axis=1)
#
# NTL_clip_aux3_noNeg = NTL_clip_aux3
# mask = NTL_clip_aux3_noNeg['pred'] < 0
# NTL_clip_aux3_noNeg.loc[mask, ['pred']] = 0
#
# for year in years:
#     if int(year) >= 2014:
#         mask = NTL_clip_aux3_noNeg['estpop' + year] < 0
#         NTL_clip_aux3_noNeg.loc[mask, ['estpop' + year]] = 0
#
# NTL_clip_aux3_noNeg['estpop2014change'] = NTL_clip_aux3_noNeg['estpop2014'] - NTL_clip_aux3_noNeg['pred']
# NTL_clip_aux3_noNeg['estpop2015change'] = NTL_clip_aux3_noNeg['estpop2015'] - NTL_clip_aux3_noNeg['estpop2014']
# NTL_clip_aux3_noNeg['estpop2016change'] = NTL_clip_aux3_noNeg['estpop2016'] - NTL_clip_aux3_noNeg['estpop2015']
# NTL_clip_aux3_noNeg['estpop2017change'] = NTL_clip_aux3_noNeg['estpop2017'] - NTL_clip_aux3_noNeg['estpop2016']
# NTL_clip_aux3_noNeg['estpop2018change'] = NTL_clip_aux3_noNeg['estpop2018'] - NTL_clip_aux3_noNeg['estpop2017']
#
# vmin = np.min(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014', 'estpop2015', 'estpop2016', 'estpop2017', 'estpop2018']]))
# vmax = np.max(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014', 'estpop2015', 'estpop2016', 'estpop2017', 'estpop2018']]))
# # vmin=-6000
# # vmax=7000
#
# fig, axs = plt.subplots(2, 3, figsize=(20, 10))
# NTL_clip_aux3_noNeg.plot(column='pred', cmap='Spectral_r', linewidth=0.1, ax=axs[0,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[0,0].get_xaxis().set_visible(False)
# axs[0,0].get_yaxis().set_visible(False)
# axs[0,0].title.set_text('2013 Population Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2014', cmap='Spectral_r', linewidth=0.1, ax=axs[0,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[0,1].get_xaxis().set_visible(False)
# axs[0,1].get_yaxis().set_visible(False)
# axs[0,1].title.set_text('2014 Population Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2015', cmap='Spectral_r', linewidth=0.1, ax=axs[0,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[0,2].get_xaxis().set_visible(False)
# axs[0,2].get_yaxis().set_visible(False)
# axs[0,2].title.set_text('2015 Population Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2016', cmap='Spectral_r', linewidth=0.1, ax=axs[1,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[1,0].get_xaxis().set_visible(False)
# axs[1,0].get_yaxis().set_visible(False)
# axs[1,0].title.set_text('2016 Population Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2017', cmap='Spectral_r', linewidth=0.1, ax=axs[1,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[1,1].get_xaxis().set_visible(False)
# axs[1,1].get_yaxis().set_visible(False)
# axs[1,1].title.set_text('2017 Population Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2018', cmap='Spectral_r', linewidth=0.1, ax=axs[1,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[1,2].get_xaxis().set_visible(False)
# axs[1,2].get_yaxis().set_visible(False)
# axs[1,2].title.set_text('2018 Population Estimation')
# plt.suptitle("Landuse and NTL (One Month + Corrected)", size=16)
#
# df.iloc[0, :] = ntl_scale_NTL2.Pop2013.sum()
# df.iloc[-1, :] = [0.9831893, 0.9838901, 0.9767996, 0.9688029]
# df.rename({'pred':'estpop2013'}, inplace=True, axis=0)
# df.reset_index(inplace=True)
#
# sns.set(rc={'figure.figsize': (20, 11)}, style="whitegrid")
# f, axes = plt.subplots(2, 2)
# f.subplots_adjust(hspace=.5)
# sns.barplot(x=df.columns[1:], y=list(df.iloc[1, 1:]-df.iloc[0, 1:]), ax=axes[0, 0], color='red')
# axes[0, 0].set(xlabel='model')
# axes[0, 0].set(ylabel='Difference in Overall Population')
# axes[0, 0].set(title='Predicted Population of 2013 - Census Population of 2013')
# axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation = 20)
#
# sns.barplot(x=df.columns[1:], y=list(df.iloc[-2, 1:]), ax=axes[0, 1], color='green')
# axes[0, 1].set(xlabel='Model')
# axes[0, 1].set(ylabel='Mean Absolute Error')
# axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation = 20)
#
# sns.barplot(x=df.columns[1:], y=list(df.iloc[-1, 1:]), ax=axes[1, 0], color='blue')
# axes[1, 0].set(xlabel='Model')
# axes[1, 0].set(ylabel='GWR_R2')
# axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation = 20)
#
# sns.lineplot(x=range(2013, 2019), y=list(df.iloc[1:-3, 1]), ax=axes[1, 1], color='green')
# axes[1, 1].set(xlabel='Year')
# axes[1, 1].set(ylabel='Population')
#
# sns.lineplot(x=range(2013, 2019), y=list(df.iloc[1:-3, 2]), ax=axes[1, 1], color='orange')
# axes[1, 1].set(xlabel='Year')
# axes[1, 1].set(ylabel='Population')
#
# sns.lineplot(x=range(2013, 2019), y=list(df.iloc[1:-3, 3]), ax=axes[1, 1], color='blue')
# axes[1, 1].set(xlabel='Year')
# axes[1, 1].set(ylabel='Population')
#
# sns.lineplot(x=range(2013, 2019), y=list(df.iloc[1:-3, 4]), ax=axes[1, 1], color='red')
# axes[1, 1].set(xlabel='Year')
# axes[1, 1].set(ylabel='Population')
# axes[1, 1].legend(df.columns[1:], loc='lower left')
# axes[1, 1].set(title='Predictions over 5 years')
#
#
#
#
# # Predict values for all years (when not disaggregating nithglight)
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.utils import *
# df = pd.DataFrame(columns=['lndus', 'lndus_ntl_annual', 'lndus_ntl_annual_corrected', 'lndus_ntl_monthly_corrected'],
#                   index=['censuspop2013', 'pred', 'estpop2014', 'estpop2015', 'estpop2016', 'estpop2017', 'estpop2018', 'RMSE', 'MAE', 'GWR_R2'])
# try:
#     NTL_clip.set_index('ntl_clip_id', inplace=True)
# except:
#     print('ntl_clip_id is already index')
# # Landuse model
# ntl_scale_NTL2 = pd.read_csv(results + 'observations_' + 'lndus_' + date + '.csv')
# NTL_clip_aux = pd.read_csv(results + 'NTL_Level_All_years_lndus_01312021.csv')
# gwr_model = pd.read_csv(results + 'GWR_lndus_01312021.csv')
# NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
# gwr_model.set_index('ntl_clip_id', inplace=True)
# NTL_clip_aux['ntl_clip_id'] = NTL_clip_aux.index
# predict_all_years = NTL_clip_aux.merge(gwr_model, left_on=NTL_clip_aux.index, right_on=gwr_model.index, how='left')
# predict_all_years.rename({})
#
# for year in years:
#     if int(year) >= 2014:
#         predict_all_years['estpop' + year] = predict_all_years['X.Intercept.'] + \
#                                              (predict_all_years['area_hr'] * predict_all_years['area_hr' + year])
# predict_all_years.drop(['key_0'],inplace=True, axis=1)
# mask = predict_all_years['pred'] >=0
# df.iloc[1, 0] = predict_all_years.loc[mask, ['pred']].sum()[0]
# mask = predict_all_years['estpop2014'] >=0
# df.iloc[2, 0] = predict_all_years.loc[mask, ['estpop2014']].sum()[0]
# mask = predict_all_years['estpop2015'] >=0
# df.iloc[3, 0] = predict_all_years.loc[mask, ['estpop2015']].sum()[0]
# mask = predict_all_years['estpop2016'] >=0
# df.iloc[4, 0] = predict_all_years.loc[mask, ['estpop2016']].sum()[0]
# mask = predict_all_years['estpop2017'] >=0
# df.iloc[5, 0] = predict_all_years.loc[mask, ['estpop2017']].sum()[0]
# mask = predict_all_years['estpop2018'] >=0
# df.iloc[6, 0] = predict_all_years.loc[mask, ['estpop2018']].sum()[0]
#
# df.iloc[-3, 0] = mean_squared_error(ntl_scale_NTL2.Pop2013, predict_all_years.pred, squared=True)
# df.iloc[-2:, 0] = mean_absolute_error(ntl_scale_NTL2.Pop2013, predict_all_years.pred)
#
# predict_all_years.to_csv(results + 'predict_all_years_lndus_01312021.csv')
# predict_all_years.drop('geometry', inplace=True, axis=1)
# NTL_clip_aux2 = NTL_clip
# predict_all_years.set_index('ntl_clip_id', inplace=True)
# NTL_clip_aux3 = NTL_clip_aux2.merge(predict_all_years, left_on=NTL_clip_aux2.index, right_on=predict_all_years.index)
# NTL_clip_aux3.drop('key_0', inplace=True, axis=1)
#
# NTL_clip_aux3_noNeg = NTL_clip_aux3
# mask = NTL_clip_aux3_noNeg['pred'] < 0
# NTL_clip_aux3_noNeg.loc[mask, ['pred']] = 0
#
# for year in years:
#     if int(year) >= 2014:
#         mask = NTL_clip_aux3_noNeg['estpop' + year] < 0
#         NTL_clip_aux3_noNeg.loc[mask, ['estpop' + year]] = 0
#
# NTL_clip_aux3_noNeg['estpop2014change'] = NTL_clip_aux3_noNeg['estpop2014'] - NTL_clip_aux3_noNeg['pred']
# NTL_clip_aux3_noNeg['estpop2015change'] = NTL_clip_aux3_noNeg['estpop2015'] - NTL_clip_aux3_noNeg['estpop2014']
# NTL_clip_aux3_noNeg['estpop2016change'] = NTL_clip_aux3_noNeg['estpop2016'] - NTL_clip_aux3_noNeg['estpop2015']
# NTL_clip_aux3_noNeg['estpop2017change'] = NTL_clip_aux3_noNeg['estpop2017'] - NTL_clip_aux3_noNeg['estpop2016']
# NTL_clip_aux3_noNeg['estpop2018change'] = NTL_clip_aux3_noNeg['estpop2018'] - NTL_clip_aux3_noNeg['estpop2017']
#
# vmin = np.min(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014change', 'estpop2015change', 'estpop2016change', 'estpop2017change', 'estpop2018change']]))
# vmax = np.max(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014change', 'estpop2015change', 'estpop2016change', 'estpop2017change', 'estpop2018change']]))
# # vmin=-6000
# # vmax=7000
#
# fig, axs = plt.subplots(2, 3, figsize=(20, 10))
# NTL_clip_aux3_noNeg.plot(column='pred', cmap='Spectral_r', linewidth=0.1, ax=axs[0,0], edgecolor='white', legend=True)
# axs[0,0].get_xaxis().set_visible(False)
# axs[0,0].get_yaxis().set_visible(False)
# axs[0,0].title.set_text('2013 Population Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2014change', cmap='Spectral_r', linewidth=0.1, ax=axs[0,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[0,1].get_xaxis().set_visible(False)
# axs[0,1].get_yaxis().set_visible(False)
# axs[0,1].title.set_text('2014 Population Change Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2015change', cmap='Spectral_r', linewidth=0.1, ax=axs[0,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[0,2].get_xaxis().set_visible(False)
# axs[0,2].get_yaxis().set_visible(False)
# axs[0,2].title.set_text('2015 Population Change Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2016change', cmap='Spectral_r', linewidth=0.1, ax=axs[1,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[1,0].get_xaxis().set_visible(False)
# axs[1,0].get_yaxis().set_visible(False)
# axs[1,0].title.set_text('2016 Population Change Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2017change', cmap='Spectral_r', linewidth=0.1, ax=axs[1,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[1,1].get_xaxis().set_visible(False)
# axs[1,1].get_yaxis().set_visible(False)
# axs[1,1].title.set_text('2017 Population Change Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2018change', cmap='Spectral_r', linewidth=0.1, ax=axs[1,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[1,2].get_xaxis().set_visible(False)
# axs[1,2].get_yaxis().set_visible(False)
# axs[1,2].title.set_text('2018 Population Change Estimation')
# plt.suptitle("Landuse", size=16)
#
# # Landuse model + annual ntl
# ntl_scale_NTL2 = pd.read_csv(results + 'observations_median_' + 'ntl_annual_incorrected_' + date + '.csv')
# NTL_clip_aux = pd.read_csv(results + 'NTL_Level_All_years_median_ntl_annual_incorrected_01312021.csv')
# gwr_model = pd.read_csv(results + 'GWR_median_ntl_annual_incorrected_01312021.csv')
# NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
# gwr_model.set_index('ntl_clip_id', inplace=True)
# NTL_clip_aux['ntl_clip_id'] = NTL_clip_aux.index
# predict_all_years = NTL_clip_aux.merge(gwr_model, left_on=NTL_clip_aux.index, right_on=gwr_model.index, how='left')
# predict_all_years.rename({})
# for year in years:
#     if int(year) >= 2014:
#         predict_all_years['estpop' + year] = predict_all_years['X.Intercept.'] + \
#                                              (predict_all_years['area_hr'] * predict_all_years['area_hr' + year]) + \
#                                              (predict_all_years['NTL2013_bg'] * predict_all_years['NTL_bg' + year]) + \
#                                              (predict_all_years['NTL2013_hr'] * predict_all_years['NTL_hr' + year]) + \
#                                              (predict_all_years['NTL2013_nr'] * predict_all_years['NTL_nr' + year])
#
# predict_all_years.drop(['key_0'],inplace=True, axis=1)
# mask = predict_all_years['pred'] >=0
# df.iloc[1, 1] = predict_all_years.loc[mask, ['pred']].sum()[0]
# mask = predict_all_years['estpop2014'] >=0
# df.iloc[2, 1] = predict_all_years.loc[mask, ['estpop2014']].sum()[0]
# mask = predict_all_years['estpop2015'] >=0
# df.iloc[3, 1] = predict_all_years.loc[mask, ['estpop2015']].sum()[0]
# mask = predict_all_years['estpop2016'] >=0
# df.iloc[4, 1] = predict_all_years.loc[mask, ['estpop2016']].sum()[0]
# mask = predict_all_years['estpop2017'] >=0
# df.iloc[5, 1] = predict_all_years.loc[mask, ['estpop2017']].sum()[0]
# mask = predict_all_years['estpop2018'] >=0
# df.iloc[6, 1] = predict_all_years.loc[mask, ['estpop2018']].sum()[0]
# df.iloc[-3, 1] = mean_squared_error(ntl_scale_NTL2.Pop2013, predict_all_years.pred, squared=True)
# df.iloc[-2:, 1] = mean_absolute_error(ntl_scale_NTL2.Pop2013, predict_all_years.pred)
# predict_all_years.to_csv(results + 'predict_all_years_ntl_annual_incorrected_01312021.csv')
#
# predict_all_years.drop('geometry', inplace=True, axis=1)
# NTL_clip_aux2 = NTL_clip
# predict_all_years.set_index('ntl_clip_id', inplace=True)
# NTL_clip_aux3 = NTL_clip_aux2.merge(predict_all_years, left_on=NTL_clip_aux2.index, right_on=predict_all_years.index)
# NTL_clip_aux3.drop('key_0', inplace=True, axis=1)
#
# NTL_clip_aux3_noNeg = NTL_clip_aux3
# mask = NTL_clip_aux3_noNeg['pred'] < 0
# NTL_clip_aux3_noNeg.loc[mask, ['pred']] = 0
#
# for year in years:
#     if int(year) >= 2014:
#         mask = NTL_clip_aux3_noNeg['estpop' + year] < 0
#         NTL_clip_aux3_noNeg.loc[mask, ['estpop' + year]] = 0
#
# NTL_clip_aux3_noNeg['estpop2014change'] = NTL_clip_aux3_noNeg['estpop2014'] - NTL_clip_aux3_noNeg['pred']
# NTL_clip_aux3_noNeg['estpop2015change'] = NTL_clip_aux3_noNeg['estpop2015'] - NTL_clip_aux3_noNeg['estpop2014']
# NTL_clip_aux3_noNeg['estpop2016change'] = NTL_clip_aux3_noNeg['estpop2016'] - NTL_clip_aux3_noNeg['estpop2015']
# NTL_clip_aux3_noNeg['estpop2017change'] = NTL_clip_aux3_noNeg['estpop2017'] - NTL_clip_aux3_noNeg['estpop2016']
# NTL_clip_aux3_noNeg['estpop2018change'] = NTL_clip_aux3_noNeg['estpop2018'] - NTL_clip_aux3_noNeg['estpop2017']
#
# vmin = np.min(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014change', 'estpop2015change', 'estpop2016change', 'estpop2017change', 'estpop2018change']]))
# vmax = np.max(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014change', 'estpop2015change', 'estpop2016change', 'estpop2017change', 'estpop2018change']]))
# # vmin=-6000
# # vmax=7000
#
# fig, axs = plt.subplots(2, 3, figsize=(20, 10))
# NTL_clip_aux3_noNeg.plot(column='pred', cmap='Spectral_r', linewidth=0.1, ax=axs[0,0], edgecolor='white', legend=True)
# axs[0,0].get_xaxis().set_visible(False)
# axs[0,0].get_yaxis().set_visible(False)
# axs[0,0].title.set_text('2013 Population Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2014change', cmap='Spectral_r', linewidth=0.1, ax=axs[0,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[0,1].get_xaxis().set_visible(False)
# axs[0,1].get_yaxis().set_visible(False)
# axs[0,1].title.set_text('2014 Population Change Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2015change', cmap='Spectral_r', linewidth=0.1, ax=axs[0,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[0,2].get_xaxis().set_visible(False)
# axs[0,2].get_yaxis().set_visible(False)
# axs[0,2].title.set_text('2015 Population Change Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2016change', cmap='Spectral_r', linewidth=0.1, ax=axs[1,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[1,0].get_xaxis().set_visible(False)
# axs[1,0].get_yaxis().set_visible(False)
# axs[1,0].title.set_text('2016 Population Change Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2017change', cmap='Spectral_r', linewidth=0.1, ax=axs[1,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[1,1].get_xaxis().set_visible(False)
# axs[1,1].get_yaxis().set_visible(False)
# axs[1,1].title.set_text('2017 Population Change Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2018change', cmap='Spectral_r', linewidth=0.1, ax=axs[1,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[1,2].get_xaxis().set_visible(False)
# axs[1,2].get_yaxis().set_visible(False)
# axs[1,2].title.set_text('2018 Population Change Estimation')
# plt.suptitle("Landuse and NTL (One Year + Incorrected)", size=16)
#
# # Landuse model + corrected annual ntl
# ntl_scale_NTL2 = pd.read_csv(results + 'observations_median_' + 'ntl_annual_corrected_' + date + '.csv')
# ntl_scale_NTL2.drop('CNTL2013', inplace=True, axis=1)
# NTL_clip_aux = pd.read_csv(results + 'NTL_Level_All_years_median_ntl_annual_corrected_01312021.csv')
# gwr_model = pd.read_csv(results + 'GWR_median_ntl_annual_corrected_01312021.csv')
# NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
# gwr_model.set_index('ntl_clip_id', inplace=True)
# NTL_clip_aux['ntl_clip_id'] = NTL_clip_aux.index
# predict_all_years = NTL_clip_aux.merge(gwr_model, left_on=NTL_clip_aux.index, right_on=gwr_model.index, how='left')
# predict_all_years.rename({})
# for year in years:
#     if int(year) >= 2014:
#         predict_all_years['estpop' + year] = predict_all_years['X.Intercept.'] + \
#                                              (predict_all_years['area_hr'] * predict_all_years['area_hr' + year]) + \
#                                              (predict_all_years['NTL2013_bg'] * predict_all_years['NTL_bg' + year]) + \
#                                              (predict_all_years['NTL2013_hr'] * predict_all_years['NTL_hr' + year]) + \
#                                              (predict_all_years['NTL2013_nr'] * predict_all_years['NTL_nr' + year])
#
# predict_all_years.drop(['key_0'],inplace=True, axis=1)
# mask = predict_all_years['pred'] >=0
# df.iloc[1, 2] = predict_all_years.loc[mask, ['pred']].sum()[0]
# mask = predict_all_years['estpop2014'] >=0
# df.iloc[2, 2] = predict_all_years.loc[mask, ['estpop2014']].sum()[0]
# mask = predict_all_years['estpop2015'] >=0
# df.iloc[3, 2] = predict_all_years.loc[mask, ['estpop2015']].sum()[0]
# mask = predict_all_years['estpop2016'] >=0
# df.iloc[4, 2] = predict_all_years.loc[mask, ['estpop2016']].sum()[0]
# mask = predict_all_years['estpop2017'] >=0
# df.iloc[5, 2] = predict_all_years.loc[mask, ['estpop2017']].sum()[0]
# mask = predict_all_years['estpop2018'] >=0
# df.iloc[6, 2] = predict_all_years.loc[mask, ['estpop2018']].sum()[0]
# df.iloc[-3, 2] = mean_squared_error(ntl_scale_NTL2.Pop2013, predict_all_years.pred, squared=True)
# df.iloc[-2:, 2] = mean_absolute_error(ntl_scale_NTL2.Pop2013, predict_all_years.pred)
# predict_all_years.to_csv(results + 'predict_all_years_ntl_annual_corrected_01312021.csv')
#
# predict_all_years.drop('geometry', inplace=True, axis=1)
# NTL_clip_aux2 = NTL_clip
# predict_all_years.set_index('ntl_clip_id', inplace=True)
# NTL_clip_aux3 = NTL_clip_aux2.merge(predict_all_years, left_on=NTL_clip_aux2.index, right_on=predict_all_years.index)
# NTL_clip_aux3.drop('key_0', inplace=True, axis=1)
#
# NTL_clip_aux3_noNeg = NTL_clip_aux3
# mask = NTL_clip_aux3_noNeg['pred'] < 0
# NTL_clip_aux3_noNeg.loc[mask, ['pred']] = 0
#
# for year in years:
#     if int(year) >= 2014:
#         mask = NTL_clip_aux3_noNeg['estpop' + year] < 0
#         NTL_clip_aux3_noNeg.loc[mask, ['estpop' + year]] = 0
#
# NTL_clip_aux3_noNeg['estpop2014change'] = NTL_clip_aux3_noNeg['estpop2014'] - NTL_clip_aux3_noNeg['pred']
# NTL_clip_aux3_noNeg['estpop2015change'] = NTL_clip_aux3_noNeg['estpop2015'] - NTL_clip_aux3_noNeg['estpop2014']
# NTL_clip_aux3_noNeg['estpop2016change'] = NTL_clip_aux3_noNeg['estpop2016'] - NTL_clip_aux3_noNeg['estpop2015']
# NTL_clip_aux3_noNeg['estpop2017change'] = NTL_clip_aux3_noNeg['estpop2017'] - NTL_clip_aux3_noNeg['estpop2016']
# NTL_clip_aux3_noNeg['estpop2018change'] = NTL_clip_aux3_noNeg['estpop2018'] - NTL_clip_aux3_noNeg['estpop2017']
#
# vmin = np.min(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014change', 'estpop2015change', 'estpop2016change', 'estpop2017change', 'estpop2018change']]))
# vmax = np.max(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014change', 'estpop2015change', 'estpop2016change', 'estpop2017change', 'estpop2018change']]))
# # vmin=-6000
# # vmax=7000
#
# fig, axs = plt.subplots(2, 3, figsize=(20, 10))
# NTL_clip_aux3_noNeg.plot(column='pred', cmap='Spectral_r', linewidth=0.1, ax=axs[0,0], edgecolor='white', legend=True)
# axs[0,0].get_xaxis().set_visible(False)
# axs[0,0].get_yaxis().set_visible(False)
# axs[0,0].title.set_text('2013 Population Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2014change', cmap='Spectral_r', linewidth=0.1, ax=axs[0,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[0,1].get_xaxis().set_visible(False)
# axs[0,1].get_yaxis().set_visible(False)
# axs[0,1].title.set_text('2014 Population Change Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2015change', cmap='Spectral_r', linewidth=0.1, ax=axs[0,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[0,2].get_xaxis().set_visible(False)
# axs[0,2].get_yaxis().set_visible(False)
# axs[0,2].title.set_text('2015 Population Change Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2016change', cmap='Spectral_r', linewidth=0.1, ax=axs[1,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[1,0].get_xaxis().set_visible(False)
# axs[1,0].get_yaxis().set_visible(False)
# axs[1,0].title.set_text('2016 Population Change Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2017change', cmap='Spectral_r', linewidth=0.1, ax=axs[1,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[1,1].get_xaxis().set_visible(False)
# axs[1,1].get_yaxis().set_visible(False)
# axs[1,1].title.set_text('2017 Population Change Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2018change', cmap='Spectral_r', linewidth=0.1, ax=axs[1,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[1,2].get_xaxis().set_visible(False)
# axs[1,2].get_yaxis().set_visible(False)
# axs[1,2].title.set_text('2018 Population Change Estimation')
# plt.suptitle("Landuse and NTL (One Year + Corrected)", size=16)
#
# # Landuse model + corrected monthly ntl
# ntl_scale_NTL2 = pd.read_csv(results + 'observations_median_' + 'ntl_monthly_corrected_' + date + '.csv')
# NTL_clip_aux = pd.read_csv(results + 'NTL_Level_All_years_median_ntl_monthly_corrected_01312021.csv')
# gwr_model = pd.read_csv(results + 'GWR_median_ntl_monthly_corrected_01312021.csv')
# NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
# gwr_model.set_index('ntl_clip_id', inplace=True)
# NTL_clip_aux['ntl_clip_id'] = NTL_clip_aux.index
# predict_all_years = NTL_clip_aux.merge(gwr_model, left_on=NTL_clip_aux.index, right_on=gwr_model.index, how='left')
# predict_all_years.rename({})
# for year in years:
#     if int(year) >= 2014:
#         predict_all_years['estpop' + year] = predict_all_years['X.Intercept.'] + \
#                                              (predict_all_years['area_hr'] * predict_all_years['area_hr' + year]) + \
#                                              (predict_all_years['NTL2013_bg'] * predict_all_years['NTL_bg' + year]) + \
#                                              (predict_all_years['NTL2013_nr'] * predict_all_years['NTL_nr' + year])
#
# predict_all_years.drop(['key_0'],inplace=True, axis=1)
# mask = predict_all_years['pred'] >=0
# df.iloc[1, 3] = predict_all_years.loc[mask, ['pred']].sum()[0]
# mask = predict_all_years['estpop2014'] >=0
# df.iloc[2, 3] = predict_all_years.loc[mask, ['estpop2014']].sum()[0]
# mask = predict_all_years['estpop2015'] >=0
# df.iloc[3, 3] = predict_all_years.loc[mask, ['estpop2015']].sum()[0]
# mask = predict_all_years['estpop2016'] >=0
# df.iloc[4, 3] = predict_all_years.loc[mask, ['estpop2016']].sum()[0]
# mask = predict_all_years['estpop2017'] >=0
# df.iloc[5, 3] = predict_all_years.loc[mask, ['estpop2017']].sum()[0]
# mask = predict_all_years['estpop2018'] >=0
# df.iloc[6, 3] = predict_all_years.loc[mask, ['estpop2018']].sum()[0]
# df.iloc[-3, 3] = mean_squared_error(ntl_scale_NTL2.Pop2013, predict_all_years.pred, squared=True)
# df.iloc[-2:, 3] = mean_absolute_error(ntl_scale_NTL2.Pop2013, predict_all_years.pred)
# predict_all_years.to_csv(results + 'predict_all_years_ntl_monthly_corrected_01312021.csv')
#
# predict_all_years.drop('geometry', inplace=True, axis=1)
# NTL_clip_aux2 = NTL_clip
# predict_all_years.set_index('ntl_clip_id', inplace=True)
# NTL_clip_aux3 = NTL_clip_aux2.merge(predict_all_years, left_on=NTL_clip_aux2.index, right_on=predict_all_years.index)
# NTL_clip_aux3.drop('key_0', inplace=True, axis=1)
#
# NTL_clip_aux3_noNeg = NTL_clip_aux3
# mask = NTL_clip_aux3_noNeg['pred'] < 0
# NTL_clip_aux3_noNeg.loc[mask, ['pred']] = 0
#
# for year in years:
#     if int(year) >= 2014:
#         mask = NTL_clip_aux3_noNeg['estpop' + year] < 0
#         NTL_clip_aux3_noNeg.loc[mask, ['estpop' + year]] = 0
#
# NTL_clip_aux3_noNeg['estpop2014change'] = NTL_clip_aux3_noNeg['estpop2014'] - NTL_clip_aux3_noNeg['pred']
# NTL_clip_aux3_noNeg['estpop2015change'] = NTL_clip_aux3_noNeg['estpop2015'] - NTL_clip_aux3_noNeg['estpop2014']
# NTL_clip_aux3_noNeg['estpop2016change'] = NTL_clip_aux3_noNeg['estpop2016'] - NTL_clip_aux3_noNeg['estpop2015']
# NTL_clip_aux3_noNeg['estpop2017change'] = NTL_clip_aux3_noNeg['estpop2017'] - NTL_clip_aux3_noNeg['estpop2016']
# NTL_clip_aux3_noNeg['estpop2018change'] = NTL_clip_aux3_noNeg['estpop2018'] - NTL_clip_aux3_noNeg['estpop2017']
#
# vmin = np.min(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014change', 'estpop2015change', 'estpop2016change', 'estpop2017change', 'estpop2018change']]))
# vmax = np.max(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014change', 'estpop2015change', 'estpop2016change', 'estpop2017change', 'estpop2018change']]))
# # vmin=-6000
# # vmax=7000
#
# fig, axs = plt.subplots(2, 3, figsize=(20, 10))
# NTL_clip_aux3_noNeg.plot(column='pred', cmap='Spectral_r', linewidth=0.1, ax=axs[0,0], edgecolor='white', legend=True)
# axs[0,0].get_xaxis().set_visible(False)
# axs[0,0].get_yaxis().set_visible(False)
# axs[0,0].title.set_text('2013 Population Change Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2014change', cmap='Spectral_r', linewidth=0.1, ax=axs[0,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[0,1].get_xaxis().set_visible(False)
# axs[0,1].get_yaxis().set_visible(False)
# axs[0,1].title.set_text('2014 Population Change Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2015change', cmap='Spectral_r', linewidth=0.1, ax=axs[0,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[0,2].get_xaxis().set_visible(False)
# axs[0,2].get_yaxis().set_visible(False)
# axs[0,2].title.set_text('2015 Population Change Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2016change', cmap='Spectral_r', linewidth=0.1, ax=axs[1,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[1,0].get_xaxis().set_visible(False)
# axs[1,0].get_yaxis().set_visible(False)
# axs[1,0].title.set_text('2016 Population Change Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2017change', cmap='Spectral_r', linewidth=0.1, ax=axs[1,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[1,1].get_xaxis().set_visible(False)
# axs[1,1].get_yaxis().set_visible(False)
# axs[1,1].title.set_text('2017 Population Change Estimation')
# NTL_clip_aux3_noNeg.plot(column='estpop2018change', cmap='Spectral_r', linewidth=0.1, ax=axs[1,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[1,2].get_xaxis().set_visible(False)
# axs[1,2].get_yaxis().set_visible(False)
# axs[1,2].title.set_text('2018 Population Change Estimation')
# plt.suptitle("Landuse and NTL (One Month + Corrected)", size=16)
#
# df.iloc[0, :] = ntl_scale_NTL2.Pop2013.sum()
# df.iloc[-1, :] = [0.9831893, 0.9722858, 0.9733809, 0.9690868]
# df.rename({'pred':'estpop2013'}, inplace=True, axis=0)
# df.reset_index(inplace=True)
#
# sns.set(rc={'figure.figsize': (20, 11)}, style="whitegrid")
# f, axes = plt.subplots(2, 2)
# f.subplots_adjust(hspace=.5)
# sns.barplot(x=df.columns[1:], y=list(df.iloc[1, 1:]-df.iloc[0, 1:]), ax=axes[0, 0], color='red')
# axes[0, 0].set(xlabel='model')
# axes[0, 0].set(ylabel='Difference in Overall Population')
# axes[0, 0].set(title='Predicted Population of 2013 - Census Population of 2013')
# axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation = 20)
#
# sns.barplot(x=df.columns[1:], y=list(df.iloc[-2, 1:]), ax=axes[0, 1], color='green')
# axes[0, 1].set(xlabel='Model')
# axes[0, 1].set(ylabel='Mean Absolute Error')
# axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation = 20)
#
# sns.barplot(x=df.columns[1:], y=list(df.iloc[-1, 1:]), ax=axes[1, 0], color='blue')
# axes[1, 0].set(xlabel='Model')
# axes[1, 0].set(ylabel='GWR_R2')
# axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation = 20)
#
# sns.lineplot(x=range(2013, 2019), y=list(df.iloc[1:-3, 1]), ax=axes[1, 1], color='green')
# axes[1, 1].set(xlabel='Year')
# axes[1, 1].set(ylabel='Population')
#
# sns.lineplot(x=range(2013, 2019), y=list(df.iloc[1:-3, 2]), ax=axes[1, 1], color='orange')
# axes[1, 1].set(xlabel='Year')
# axes[1, 1].set(ylabel='Population')
#
# sns.lineplot(x=range(2013, 2019), y=list(df.iloc[1:-3, 3]), ax=axes[1, 1], color='blue')
# axes[1, 1].set(xlabel='Year')
# axes[1, 1].set(ylabel='Population')
#
# sns.lineplot(x=range(2013, 2019), y=list(df.iloc[1:-3, 4]), ax=axes[1, 1], color='red')
# axes[1, 1].set(xlabel='Year')
# axes[1, 1].set(ylabel='Population')
# axes[1, 1].legend(df.columns[1:], loc='lower left')
# axes[1, 1].set(title='Predictions over 5 years')