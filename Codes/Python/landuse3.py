import numpy as np
import matplotlib.pyplot as plt
from matplotlib import *
import pandas as pd
import scipy.stats
# from pysal.lib import weights
# import pysal as ps
import libpysal
from libpysal.weights import Queen, Rook, KNN
from esda.moran import Moran
from esda.moran import Moran_Local
from splot.esda import moran_scatterplot
import os
# import arcpy
# from arcpy import env
# from arcpy.sa import *
# from simpledbf import Dbf5
from os import listdir
from os.path import isfile, join
# from simpledbf import Dbf5
import geopandas as gp
import mapclassify
# import pysal as ps
# import libpysal
# import esda
# from esda.moran import Moran
# from splot.esda import moran_scatterplot
# from splot.esda import plot_moran
from splot.esda import lisa_cluster
# from esda.moran import Moran_Local
# from splot.esda import plot_local_autocorrelation
# from splot.esda import lisa_cluster
from mpl_toolkits.mplot3d import Axes3D
# For statistics. Requires statsmodels 5.0 or more
from statsmodels.formula.api import ols
from splot import _viz_utils
from patsy import dmatrices
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import *
from statsmodels.stats.outliers_influence import variance_inflation_factor
import spreg
from statsmodels import regression
import statsmodels.api as sm
# Analysis of Variance (ANOVA) on linear models
from statsmodels.stats.anova import anova_lm
import seaborn as sns
# from pysal.contrib.viz import mapping as maps
# import mapclassify
# image_path = 'G:/backupC27152020/Population_Displacement_Final/Resources/VHR/images/'
# landuse_path = 'G:/backupC27152020/Population_Displacement_Final/Resources/VHR/landuse/'
# viirs_path = 'G:/backupC27152020/Population_Displacement_Final/Resources/VIIRS/VNP46A2/'
# geodb_path = 'G:/backupC27152020/Population_Displacement_Final/Resources/poulation_disp.gdb/Data/'
# temp = 'G:/backupC27152020/Population_Displacement_Final/Resources/Temp/'
# results = 'G:/backupC27152020/Population_Displacement_Final/Resources/Results/'
# date = '03292020'

image_path = 'C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Sources/VHR/images/'
landuse_path = 'C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Sources/VHR/landuse/'
viirs_path = 'C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Sources/VIIRS/VNP46A2/'
geodb_path = 'C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Sources/poulation_disp.gdb/Data/'
temp = 'C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Sources/Temp/'
results = 'C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Sources/Results/'
figures = 'C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Sources/Results/Model/'
date = '03292020'

# Choose the year of the analysis:
years = ['2014', '2016', '2017'] # 2014-15-16-17-18 is available now
months = ['Jan','Feb','Mar','Jun','Jul','Oct','Nov']

month_year = ['Jun2014', 'Jul2014', 'Oct2016', 'Nov2016', 'Jan2017','Feb2017', 'Mar2017', 'Jun2017', 'Jul2017']

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
mdl = 'ntlMonthlyIncorrected_'

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

for my in month_year:
    year = my[-4:]
    month = my[:3]
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

for my in month_year:
    year = my[-4:]
    month = my[:3]
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

# Jun - 2014 if int(year) == 2014 and month == 'Jun':  # All significant
# Jul - 2014 elif int(year) == 2014 and month == 'Jul':  # lr is not significant
# Oct - 2016 elif int(year) == 2016 and month == 'Oct':  # All significant
# Nov - 2016 elif int(year) == 2016 and month == 'Nov':  # lr is significant
# Jan - 2017 elif int(year) == 2017 and month == 'Jan':  # HL and LH are not significant
# Feb - 2017 elif int(year) == 2017 and month == 'Feb':  # LH is insignificant
# Mar - 2017 elif int(year) == 2017 and month == 'Mar':  # HL and LH are not significant
# Jun - 2017 elif int(year) == 2017 and month == 'Jun':  # LH is insignificant
# Jul - 2017 elif int(year) == 2017 and month == 'Jul':  # LH is insignificant

for my in month_year:
    year = my[-4:]
    month = my[:3]
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

    else:
        ntl_scale_NTL = intersect2.groupby(['ntl_clip_id', 'landuse2018']).sum().loc[:, ['intersect_area']]
        ntl_scale_NTL2 = ntl_scale_NTL.unstack('landuse2018')
        ntl_scale_NTL2.columns = ['area_bg' + month + year, 'area_lr' + month + year, 'area_hr' + month + year, 'area_nr' + month + year]
        ntl_scale_NTL2.fillna(0, inplace=True)

    ntl_scale_NTL2['CNTL' + month + year] = intersect2.groupby(intersect2.index).max()['CNTL' + month + year]

    # ntl_scale_NTL2.reset_index(inplace=True)
    # NTL_clip.reset_index(inplace=True)
    NTL_clip.set_index('ntl_clip_id', inplace=True)
    ntl_scale_NTL2 = NTL_clip.merge(ntl_scale_NTL2, left_on = NTL_clip.index, right_on = ntl_scale_NTL2.index, how='left')
    # ntl_scale_NTL2.drop(['key_0', 'index', 'Shape_Leng', 'Shape_Area', 'ntl_area', 'ntl_id','ntl_clip_id_x', 'ntl_clip_area'],
    #                 inplace=True, axis=1)
    ntl_scale_NTL2.drop(['key_0', 'Shape_Leng', 'Shape_Area', 'ntl_area', 'ntl_id','ntl_clip_area'],
                    inplace=True, axis=1)

    try:
        ntl_scale_NTL2.drop(['level_0'], inplace=True, axis=1)
        NTL_clip.drop(['level_0'], inplace=True, axis=1)
    except:
        print('level_0 is not in the columns')

    ntl_scale_NTL2['X'] = ntl_scale_NTL2.geometry.centroid.x
    ntl_scale_NTL2['Y'] = ntl_scale_NTL2.geometry.centroid.y

    if int(year) == 2014 and month == 'Jun':  # All significant
        olsmodelstring = 'CNTL' + month + year + ' ~ ' + 'area_hr' + month + year + ' + '  + \
                        'area_nr' + month + year + ' + ' + 'area_bg' + month + year + ' + ' + 'area_lr' + month + year
        model_NTL = ols(olsmodelstring,ntl_scale_NTL2).fit()
        # print(mdl + ': ' + olsmodelstring + "\n")
        # print(model_NTL.summary())
        # print("\nRetrieving manually the parameter estimates:")
        print(model_NTL._results.params)
        model_NTL.save(results + 'ols_ntl.pickle', remove_data=False)

        y, X = dmatrices(olsmodelstring,data=ntl_scale_NTL2, return_type='dataframe')
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['variable'] = X.columns
        print(vif)

        ntlresid = pd.concat((ntl_scale_NTL2, model_NTL.resid), axis=1)
        ntlresid.rename({0: 'ntlresid' + month + year}, axis=1, inplace=True)
        W = Queen.from_dataframe(ntlresid)
        W.transform = 'r'
        moran_ntl = Moran(ntlresid['ntlresid' + month + year], W)
        print('moran_ntl' + month + year + ': ' + str(moran_ntl.I))
        moran_loc = Moran_Local(ntlresid['ntlresid' + month + year], W)
        p = lisa_cluster(moran_loc, ntlresid, p=0.05, figsize=(9, 9))
        plt.title('Cluster Map of Nightlight Residuals(' + month +  '-' + year + ')', size=20)
        plt.show()
        plt.savefig(
            'C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Results/Cluster Map of Nightlight Residuals_' + month + '-' + year + '.png',
            dpi=500, bbox_inches='tight')

        # 1 HH, 2 LH, 3 LL, 4 HL
        cluster = _viz_utils.moran_hot_cold_spots(moran_loc, p=0.05)
        aux = pd.DataFrame(cluster)
        aux.rename({0: 'clusters' + month + year}, inplace=True, axis=1)
        aux.loc[aux['clusters' + month + year] == 0, ['clusters' + month + year]] = 'NS'
        aux.loc[aux['clusters' + month + year] == 1, ['clusters' + month + year]] = 'HH'
        aux.loc[aux['clusters' + month + year] == 2, ['clusters' + month + year]] = 'LH'
        aux.loc[aux['clusters' + month + year] == 3, ['clusters' + month + year]] = 'LL'
        aux.loc[aux['clusters' + month + year] == 4, ['clusters' + month + year]] = 'HL'
        cluster = pd.concat((ntl_scale_NTL2, aux), axis=1)
        cluster = pd.get_dummies(cluster)

        olsmodelstring_spatial = 'CNTL' + month + year + ' ~ ' + 'area_hr' + month + year + ' + '  + \
                                 'area_nr' + month + year + ' + ' + 'area_bg' + month + year + ' + ' + \
                                 'clusters' + month + year + '_HH'  + ' + '  + 'clusters' + month + year + '_HL'  + ' + ' + \
                                 'clusters' + month + year + '_NS' + ' + '  + 'clusters' + month + year + '_LH' + ' + '  + \
                                 'area_lr' + month + year

        print('Spatial Multiple Linear Regression for disaggregating nightlight'  + month + '-' + year + ':')
        model_NTL_spatial = ols(olsmodelstring_spatial,cluster).fit()
        print(model_NTL_spatial.summary())
        print("\nRetrieving manually the parameter estimates:")
        print(model_NTL_spatial._results.params)

        y, X = dmatrices(olsmodelstring_spatial,data=cluster, return_type='dataframe')
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['variable'] = X.columns
        print(vif)

        model_NTL_spatial.save(results + 'ols_ntl_spatial.pickle', remove_data=False)

        NTL_clip.reset_index(inplace=True)
        cluster = cluster.merge(NTL_clip[['ntl_clip_id']], left_on=cluster.index, right_on=NTL_clip.index.array,
                                how='left')
        cluster.drop('key_0', axis=1, inplace=True)
        cluster.set_index('ntl_clip_id', inplace=True)
        intersect2 = intersect2.merge(
            cluster.loc[:, ['clusters' + month + year + '_HH', 'clusters' + month + year + '_HL', 'clusters' + month + year + '_LL', 'clusters' + month + year + '_NS', 'clusters' + month + year + '_LH']],
            left_on=intersect2.index, right_on=cluster.index, how='left')
        intersect2.rename({'key_0': 'ntl_clip_id'}, inplace=True, axis=1)
        try:
            intersect2.drop('level_0', inplace=True, axis=1)
        except:
            print('done!')
        cluster.reset_index(inplace=True)

        model_NTL_pred = cluster['area_bg' + month + year] * model_NTL.params[3] + cluster['area_hr' +  month + year] * \
                         model_NTL.params[1] + \
                         cluster['area_nr' +  month + year] * model_NTL.params[2] + cluster['area_lr' +  month + year] * \
                         model_NTL.params[4]
        model_NTL_pred = pd.DataFrame(model_NTL_pred, columns=['ntlpred' +  month + year])
        Predictions = pd.concat((ntl_scale_NTL2, model_NTL_pred), axis=1)

        model_NTL_spatial_pred = cluster['area_bg' + month + year] * model_NTL_spatial.params[3] + cluster[
            'area_hr' + month + year] * model_NTL_spatial.params[1] + \
                                 cluster['area_nr' + month + year] * model_NTL_spatial.params[2] + cluster[
                                     'clusters' + month + year + '_HH'] * model_NTL_spatial.params[4] + \
                                 cluster['clusters' + month + year + '_HL'] * model_NTL_spatial.params[5] + \
                                 cluster['clusters' + month + year + '_NS'] * model_NTL_spatial.params[6] + \
                                 cluster['clusters' + month + year + '_LH'] * model_NTL_spatial.params[7] + \
                                 cluster['area_lr' + month + year] * model_NTL_spatial.params[8]
        model_NTL_spatial_pred = pd.DataFrame(model_NTL_spatial_pred, columns=['ntlpred' + month + year])
        Predictions_spatial = pd.concat((ntl_scale_NTL2, model_NTL_spatial_pred), axis=1)

        sns.set(rc={'figure.figsize': (5, 10)}, style="whitegrid")
        f, axes = plt.subplots(2, 1)
        f.subplots_adjust(hspace=.5)
        sns.scatterplot(x=Predictions['ntlpred' + month + year], y=Predictions['CNTL' + month + year], ax=axes[0],
                        color='black')
        axes[0].set(xlabel='Nightlight emmission ' + month + year + ' (OLS model)')
        axes[0].set(ylabel='Nightlight emmission ' + month + year)
        sns.scatterplot(x=Predictions_spatial['ntlpred' + month + year], y=Predictions['CNTL' + month + year], ax=axes[1],
                        color='black')
        axes[1].set(xlabel='Nightlight emmission ' + month + year + ' (Spatial model)')
        axes[1].set(ylabel='Nightlight emmission ' + month + year)

    elif int(year) == 2014 and month == 'Jul': # lr is not significant
        olsmodelstring = 'CNTL' + month + year + ' ~ ' + 'area_hr' + month + year + ' + '  + \
                        'area_nr' + month + year + ' + ' + 'area_bg' + month + year + ' + ' + 'area_lr' + month + year
        model_NTL = ols(olsmodelstring,ntl_scale_NTL2).fit()
        # print(mdl + ': ' + olsmodelstring + "\n")
        # print(model_NTL.summary())
        # print("\nRetrieving manually the parameter estimates:")
        print(model_NTL._results.params)
        model_NTL.save(results + 'ols_ntl.pickle', remove_data=False)

        y, X = dmatrices(olsmodelstring,data=ntl_scale_NTL2, return_type='dataframe')
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['variable'] = X.columns
        print(vif)

        ntlresid = pd.concat((ntl_scale_NTL2, model_NTL.resid), axis=1)
        ntlresid.rename({0: 'ntlresid' + month + year}, axis=1, inplace=True)
        W = Queen.from_dataframe(ntlresid)
        W.transform = 'r'
        moran_ntl = Moran(ntlresid['ntlresid' + month + year], W)
        print('moran_ntl' + month + year + ': ' + str(moran_ntl.I))
        moran_loc = Moran_Local(ntlresid['ntlresid' + month + year], W)
        p = lisa_cluster(moran_loc, ntlresid, p=0.05, figsize=(9, 9))
        plt.title('Cluster Map of Nightlight Residuals(' + month +  '-' + year + ')', size=20)
        plt.show()
        plt.savefig(
            'C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Results/Cluster Map of Nightlight Residuals_' + month + '-' + year + '.png',
            dpi=500, bbox_inches='tight')

        # 1 HH, 2 LH, 3 LL, 4 HL
        cluster = _viz_utils.moran_hot_cold_spots(moran_loc, p=0.05)
        aux = pd.DataFrame(cluster)
        aux.rename({0: 'clusters' + month + year}, inplace=True, axis=1)
        aux.loc[aux['clusters' + month + year] == 0, ['clusters' + month + year]] = 'NS'
        aux.loc[aux['clusters' + month + year] == 1, ['clusters' + month + year]] = 'HH'
        aux.loc[aux['clusters' + month + year] == 2, ['clusters' + month + year]] = 'LH'
        aux.loc[aux['clusters' + month + year] == 3, ['clusters' + month + year]] = 'LL'
        aux.loc[aux['clusters' + month + year] == 4, ['clusters' + month + year]] = 'HL'
        cluster = pd.concat((ntl_scale_NTL2, aux), axis=1)
        cluster = pd.get_dummies(cluster)

        olsmodelstring_spatial = 'CNTL' + month + year + ' ~ ' + 'area_hr' + month + year + ' + '  + \
                                 'area_nr' + month + year + ' + ' + 'area_bg' + month + year + ' + ' + \
                                 'clusters' + month + year + '_HH'  + ' + '  + 'clusters' + month + year + '_HL'  + ' + ' + \
                                 'clusters' + month + year + '_NS' + ' + '  + 'clusters' + month + year + '_LH'

        print('Spatial Multiple Linear Regression for disaggregating nightlight'  + month + '-' + year + ':')
        model_NTL_spatial = ols(olsmodelstring_spatial,cluster).fit()
        print(model_NTL_spatial.summary())
        print("\nRetrieving manually the parameter estimates:")
        print(model_NTL_spatial._results.params)

        y, X = dmatrices(olsmodelstring_spatial,data=cluster, return_type='dataframe')
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['variable'] = X.columns
        print(vif)

        model_NTL_spatial.save(results + 'ols_ntl_spatial.pickle', remove_data=False)

        NTL_clip.reset_index(inplace=True)
        cluster = cluster.merge(NTL_clip[['ntl_clip_id']], left_on=cluster.index, right_on=NTL_clip.index.array,
                                how='left')
        cluster.drop('key_0', axis=1, inplace=True)
        cluster.set_index('ntl_clip_id', inplace=True)
        intersect2 = intersect2.merge(
            cluster.loc[:, ['clusters' + month + year + '_HH', 'clusters' + month + year + '_HL', 'clusters' + month + year + '_LL', 'clusters' + month + year + '_NS', 'clusters' + month + year + '_LH']],
            left_on=intersect2.index, right_on=cluster.index, how='left')
        intersect2.rename({'key_0': 'ntl_clip_id'}, inplace=True, axis=1)
        try:
            intersect2.drop('level_0', inplace=True, axis=1)
        except:
            print('done!')
        cluster.reset_index(inplace=True)

        model_NTL_pred = cluster['area_bg' + month + year] * model_NTL.params[3] + cluster['area_hr' +  month + year] * \
                         model_NTL.params[1] + \
                         cluster['area_nr' +  month + year] * model_NTL.params[2] + cluster['area_lr' +  month + year] * \
                         model_NTL.params[4]
        model_NTL_pred = pd.DataFrame(model_NTL_pred, columns=['ntlpred' +  month + year])
        Predictions = pd.concat((ntl_scale_NTL2, model_NTL_pred), axis=1)

        model_NTL_spatial_pred = cluster['area_bg' + month + year] * model_NTL_spatial.params[3] + \
        cluster['area_hr' + month + year] * model_NTL_spatial.params[1] + \
        cluster['area_nr' + month + year] * model_NTL_spatial.params[2] + \
        cluster['clusters' + month + year + '_HH'] * model_NTL_spatial.params[4] + \
        cluster['clusters' + month + year + '_HL'] * model_NTL_spatial.params[5] + \
        cluster['clusters' + month + year + '_NS'] * model_NTL_spatial.params[6] + \
        cluster['clusters' + month + year + '_LH'] * model_NTL_spatial.params[7]

        model_NTL_spatial_pred = pd.DataFrame(model_NTL_spatial_pred, columns=['ntlpred' + month + year])
        Predictions_spatial = pd.concat((ntl_scale_NTL2, model_NTL_spatial_pred), axis=1)

        sns.set(rc={'figure.figsize': (5, 10)}, style="whitegrid")
        f, axes = plt.subplots(2, 1)
        f.subplots_adjust(hspace=.5)
        sns.scatterplot(x=Predictions['ntlpred' + month + year], y=Predictions['CNTL' + month + year], ax=axes[0],
                        color='black')
        axes[0].set(xlabel='Nightlight emmission ' + month + year + ' (OLS model)')
        axes[0].set(ylabel='Nightlight emmission ' + month + year)
        sns.scatterplot(x=Predictions_spatial['ntlpred' + month + year], y=Predictions['CNTL' + month + year], ax=axes[1],
                        color='black')
        axes[1].set(xlabel='Nightlight emmission ' + month + year + ' (Spatial model)')
        axes[1].set(ylabel='Nightlight emmission ' + month + year)

    elif int(year) == 2016 and month == 'Oct':  # All significant
        olsmodelstring = 'CNTL' + month + year + ' ~ ' + 'area_hr' + month + year + ' + '  + \
                        'area_nr' + month + year + ' + ' + 'area_bg' + month + year + ' + ' + 'area_lr' + month + year
        model_NTL = ols(olsmodelstring,ntl_scale_NTL2).fit()
        # print(mdl + ': ' + olsmodelstring + "\n")
        # print(model_NTL.summary())
        # print("\nRetrieving manually the parameter estimates:")
        print(model_NTL._results.params)
        model_NTL.save(results + 'ols_ntl.pickle', remove_data=False)

        y, X = dmatrices(olsmodelstring,data=ntl_scale_NTL2, return_type='dataframe')
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['variable'] = X.columns
        print(vif)

        ntlresid = pd.concat((ntl_scale_NTL2, model_NTL.resid), axis=1)
        ntlresid.rename({0: 'ntlresid' + month + year}, axis=1, inplace=True)
        W = Queen.from_dataframe(ntlresid)
        W.transform = 'r'
        moran_ntl = Moran(ntlresid['ntlresid' + month + year], W)
        print('moran_ntl' + month + year + ': ' + str(moran_ntl.I))
        moran_loc = Moran_Local(ntlresid['ntlresid' + month + year], W)
        p = lisa_cluster(moran_loc, ntlresid, p=0.05, figsize=(9, 9))
        plt.title('Cluster Map of Nightlight Residuals(' + month +  '-' + year + ')', size=20)
        plt.show()
        plt.savefig(
            'C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Results/Cluster Map of Nightlight Residuals_' + month + '-' + year + '.png',
            dpi=500, bbox_inches='tight')

        # 1 HH, 2 LH, 3 LL, 4 HL
        cluster = _viz_utils.moran_hot_cold_spots(moran_loc, p=0.05)
        aux = pd.DataFrame(cluster)
        aux.rename({0: 'clusters' + month + year}, inplace=True, axis=1)
        aux.loc[aux['clusters' + month + year] == 0, ['clusters' + month + year]] = 'NS'
        aux.loc[aux['clusters' + month + year] == 1, ['clusters' + month + year]] = 'HH'
        aux.loc[aux['clusters' + month + year] == 2, ['clusters' + month + year]] = 'LH'
        aux.loc[aux['clusters' + month + year] == 3, ['clusters' + month + year]] = 'LL'
        aux.loc[aux['clusters' + month + year] == 4, ['clusters' + month + year]] = 'HL'
        cluster = pd.concat((ntl_scale_NTL2, aux), axis=1)
        cluster = pd.get_dummies(cluster)

        olsmodelstring_spatial = 'CNTL' + month + year + ' ~ ' + 'area_hr' + month + year + ' + '  + \
                                 'area_nr' + month + year + ' + ' + 'area_bg' + month + year + ' + ' + \
                                 'clusters' + month + year + '_HH'  + ' + '  + 'clusters' + month + year + '_HL'  + ' + ' + \
                                 'clusters' + month + year + '_NS' + ' + '  + 'clusters' + month + year + '_LH' + ' + '  + \
                                 'area_lr' + month + year

        print('Spatial Multiple Linear Regression for disaggregating nightlight'  + month + '-' + year + ':')
        model_NTL_spatial = ols(olsmodelstring_spatial,cluster).fit()
        print(model_NTL_spatial.summary())
        print("\nRetrieving manually the parameter estimates:")
        print(model_NTL_spatial._results.params)

        y, X = dmatrices(olsmodelstring_spatial,data=cluster, return_type='dataframe')
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['variable'] = X.columns
        print(vif)

        model_NTL_spatial.save(results + 'ols_ntl_spatial.pickle', remove_data=False)

        NTL_clip.reset_index(inplace=True)
        cluster = cluster.merge(NTL_clip[['ntl_clip_id']], left_on=cluster.index, right_on=NTL_clip.index.array,
                                how='left')
        cluster.drop('key_0', axis=1, inplace=True)
        cluster.set_index('ntl_clip_id', inplace=True)
        intersect2 = intersect2.merge(
            cluster.loc[:, ['clusters' + month + year + '_HH', 'clusters' + month + year + '_HL', 'clusters' + month + year + '_LL', 'clusters' + month + year + '_NS', 'clusters' + month + year + '_LH']],
            left_on=intersect2.index, right_on=cluster.index, how='left')
        intersect2.rename({'key_0': 'ntl_clip_id'}, inplace=True, axis=1)
        try:
            intersect2.drop('level_0', inplace=True, axis=1)
        except:
            print('done!')
        cluster.reset_index(inplace=True)

        model_NTL_pred = cluster['area_bg' + month + year] * model_NTL.params[3] + cluster['area_hr' +  month + year] * \
                         model_NTL.params[1] + \
                         cluster['area_nr' +  month + year] * model_NTL.params[2] + cluster['area_lr' +  month + year] * \
                         model_NTL.params[4]
        model_NTL_pred = pd.DataFrame(model_NTL_pred, columns=['ntlpred' +  month + year])
        Predictions = pd.concat((ntl_scale_NTL2, model_NTL_pred), axis=1)

        model_NTL_spatial_pred = cluster['area_bg' + month + year] * model_NTL_spatial.params[3] + cluster[
            'area_hr' + month + year] * model_NTL_spatial.params[1] + \
                                 cluster['area_nr' + month + year] * model_NTL_spatial.params[2] + cluster[
                                     'clusters' + month + year + '_HH'] * model_NTL_spatial.params[4] + \
                                 cluster['clusters' + month + year + '_HL'] * model_NTL_spatial.params[5] + \
                                 cluster['clusters' + month + year + '_NS'] * model_NTL_spatial.params[6] + \
                                 cluster['clusters' + month + year + '_LH'] * model_NTL_spatial.params[7] + \
                                 cluster['area_lr' + month + year] * model_NTL_spatial.params[8]
        model_NTL_spatial_pred = pd.DataFrame(model_NTL_spatial_pred, columns=['ntlpred' + month + year])
        Predictions_spatial = pd.concat((ntl_scale_NTL2, model_NTL_spatial_pred), axis=1)

        sns.set(rc={'figure.figsize': (5, 10)}, style="whitegrid")
        f, axes = plt.subplots(2, 1)
        f.subplots_adjust(hspace=.5)
        sns.scatterplot(x=Predictions['ntlpred' + month + year], y=Predictions['CNTL' + month + year], ax=axes[0],
                        color='black')
        axes[0].set(xlabel='Nightlight emmission ' + month + year + ' (OLS model)')
        axes[0].set(ylabel='Nightlight emmission ' + month + year)
        sns.scatterplot(x=Predictions_spatial['ntlpred' + month + year], y=Predictions['CNTL' + month + year], ax=axes[1],
                        color='black')
        axes[1].set(xlabel='Nightlight emmission ' + month + year + ' (Spatial model)')
        axes[1].set(ylabel='Nightlight emmission ' + month + year)

    elif int(year) == 2016 and month == 'Nov':  # lr is insignificant
        olsmodelstring = 'CNTL' + month + year + ' ~ ' + 'area_hr' + month + year + ' + '  + \
                        'area_nr' + month + year + ' + ' + 'area_bg' + month + year + ' + ' + 'area_lr' + month + year
        model_NTL = ols(olsmodelstring,ntl_scale_NTL2).fit()
        # print(mdl + ': ' + olsmodelstring + "\n")
        # print(model_NTL.summary())
        # print("\nRetrieving manually the parameter estimates:")
        print(model_NTL._results.params)
        model_NTL.save(results + 'ols_ntl.pickle', remove_data=False)

        y, X = dmatrices(olsmodelstring,data=ntl_scale_NTL2, return_type='dataframe')
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['variable'] = X.columns
        print(vif)

        ntlresid = pd.concat((ntl_scale_NTL2, model_NTL.resid), axis=1)
        ntlresid.rename({0: 'ntlresid' + month + year}, axis=1, inplace=True)
        W = Queen.from_dataframe(ntlresid)
        W.transform = 'r'
        moran_ntl = Moran(ntlresid['ntlresid' + month + year], W)
        print('moran_ntl' + month + year + ': ' + str(moran_ntl.I))
        moran_loc = Moran_Local(ntlresid['ntlresid' + month + year], W)
        p = lisa_cluster(moran_loc, ntlresid, p=0.05, figsize=(9, 9))
        plt.title('Cluster Map of Nightlight Residuals(' + month +  '-' + year + ')', size=20)
        plt.show()
        plt.savefig(
            'C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Results/Cluster Map of Nightlight Residuals_' + month + '-' + year + '.png',
            dpi=500, bbox_inches='tight')

        # 1 HH, 2 LH, 3 LL, 4 HL
        cluster = _viz_utils.moran_hot_cold_spots(moran_loc, p=0.05)
        aux = pd.DataFrame(cluster)
        aux.rename({0: 'clusters' + month + year}, inplace=True, axis=1)
        aux.loc[aux['clusters' + month + year] == 0, ['clusters' + month + year]] = 'NS'
        aux.loc[aux['clusters' + month + year] == 1, ['clusters' + month + year]] = 'HH'
        aux.loc[aux['clusters' + month + year] == 2, ['clusters' + month + year]] = 'LH'
        aux.loc[aux['clusters' + month + year] == 3, ['clusters' + month + year]] = 'LL'
        aux.loc[aux['clusters' + month + year] == 4, ['clusters' + month + year]] = 'HL'
        cluster = pd.concat((ntl_scale_NTL2, aux), axis=1)
        cluster = pd.get_dummies(cluster)

        olsmodelstring_spatial = 'CNTL' + month + year + ' ~ ' + 'area_hr' + month + year + ' + '  + \
                                 'area_nr' + month + year + ' + ' + 'area_bg' + month + year + ' + ' + \
                                 'clusters' + month + year + '_HH'  + ' + '  + 'clusters' + month + year + '_HL'  + ' + ' + \
                                 'clusters' + month + year + '_NS' + ' + '  + 'clusters' + month + year + '_LH'

        print('Spatial Multiple Linear Regression for disaggregating nightlight'  + month + '-' + year + ':')
        model_NTL_spatial = ols(olsmodelstring_spatial,cluster).fit()
        print(model_NTL_spatial.summary())
        print("\nRetrieving manually the parameter estimates:")
        print(model_NTL_spatial._results.params)

        y, X = dmatrices(olsmodelstring_spatial,data=cluster, return_type='dataframe')
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['variable'] = X.columns
        print(vif)

        model_NTL_spatial.save(results + 'ols_ntl_spatial.pickle', remove_data=False)

        NTL_clip.reset_index(inplace=True)
        cluster = cluster.merge(NTL_clip[['ntl_clip_id']], left_on=cluster.index, right_on=NTL_clip.index.array,
                                how='left')
        cluster.drop('key_0', axis=1, inplace=True)
        cluster.set_index('ntl_clip_id', inplace=True)
        intersect2 = intersect2.merge(
            cluster.loc[:, ['clusters' + month + year + '_HH', 'clusters' + month + year + '_HL', 'clusters' + month + year + '_LL', 'clusters' + month + year + '_NS', 'clusters' + month + year + '_LH']],
            left_on=intersect2.index, right_on=cluster.index, how='left')
        intersect2.rename({'key_0': 'ntl_clip_id'}, inplace=True, axis=1)
        try:
            intersect2.drop('level_0', inplace=True, axis=1)
        except:
            print('done!')
        cluster.reset_index(inplace=True)

        model_NTL_pred = cluster['area_bg' + month + year] * model_NTL.params[3] + cluster['area_hr' +  month + year] * \
                         model_NTL.params[1] + \
                         cluster['area_nr' +  month + year] * model_NTL.params[2] + cluster['area_lr' +  month + year] * \
                         model_NTL.params[4]
        model_NTL_pred = pd.DataFrame(model_NTL_pred, columns=['ntlpred' +  month + year])
        Predictions = pd.concat((ntl_scale_NTL2, model_NTL_pred), axis=1)

        model_NTL_spatial_pred = cluster['area_bg' + month + year] * model_NTL_spatial.params[3] + \
        cluster['area_hr' + month + year] * model_NTL_spatial.params[1] + \
        cluster['area_nr' + month + year] * model_NTL_spatial.params[2] + \
        cluster['clusters' + month + year + '_HH'] * model_NTL_spatial.params[4] + \
        cluster['clusters' + month + year + '_HL'] * model_NTL_spatial.params[5] + \
        cluster['clusters' + month + year + '_NS'] * model_NTL_spatial.params[6] + \
        cluster['clusters' + month + year + '_LH'] * model_NTL_spatial.params[7]

        model_NTL_spatial_pred = pd.DataFrame(model_NTL_spatial_pred, columns=['ntlpred' + month + year])
        Predictions_spatial = pd.concat((ntl_scale_NTL2, model_NTL_spatial_pred), axis=1)

        sns.set(rc={'figure.figsize': (5, 10)}, style="whitegrid")
        f, axes = plt.subplots(2, 1)
        f.subplots_adjust(hspace=.5)
        sns.scatterplot(x=Predictions['ntlpred' + month + year], y=Predictions['CNTL' + month + year], ax=axes[0],
                        color='black')
        axes[0].set(xlabel='Nightlight emmission ' + month + year + ' (OLS model)')
        axes[0].set(ylabel='Nightlight emmission ' + month + year)
        sns.scatterplot(x=Predictions_spatial['ntlpred' + month + year], y=Predictions['CNTL' + month + year], ax=axes[1],
                        color='black')
        axes[1].set(xlabel='Nightlight emmission ' + month + year + ' (Spatial model)')
        axes[1].set(ylabel='Nightlight emmission ' + month + year)


    elif int(year) == 2017 and month == 'Jan':  # HL and LH are not significant
        olsmodelstring = 'CNTL' + month + year + ' ~ ' + 'area_hr' + month + year + ' + '  + \
                        'area_nr' + month + year + ' + ' + 'area_bg' + month + year + ' + ' + 'area_lr' + month + year
        model_NTL = ols(olsmodelstring,ntl_scale_NTL2).fit()
        # print(mdl + ': ' + olsmodelstring + "\n")
        # print(model_NTL.summary())
        # print("\nRetrieving manually the parameter estimates:")
        print(model_NTL._results.params)
        model_NTL.save(results + 'ols_ntl.pickle', remove_data=False)

        y, X = dmatrices(olsmodelstring,data=ntl_scale_NTL2, return_type='dataframe')
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['variable'] = X.columns
        print(vif)

        ntlresid = pd.concat((ntl_scale_NTL2, model_NTL.resid), axis=1)
        ntlresid.rename({0: 'ntlresid' + month + year}, axis=1, inplace=True)
        W = Queen.from_dataframe(ntlresid)
        W.transform = 'r'
        moran_ntl = Moran(ntlresid['ntlresid' + month + year], W)
        print('moran_ntl' + month + year + ': ' + str(moran_ntl.I))
        moran_loc = Moran_Local(ntlresid['ntlresid' + month + year], W)
        p = lisa_cluster(moran_loc, ntlresid, p=0.05, figsize=(9, 9))
        plt.title('Cluster Map of Nightlight Residuals(' + month +  '-' + year + ')', size=20)
        plt.show()
        plt.savefig(
            'C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Results/Cluster Map of Nightlight Residuals_' + month + '-' + year + '.png',
            dpi=500, bbox_inches='tight')

        # 1 HH, 2 LH, 3 LL, 4 HL
        cluster = _viz_utils.moran_hot_cold_spots(moran_loc, p=0.05)
        aux = pd.DataFrame(cluster)
        aux.rename({0: 'clusters' + month + year}, inplace=True, axis=1)
        aux.loc[aux['clusters' + month + year] == 0, ['clusters' + month + year]] = 'NS'
        aux.loc[aux['clusters' + month + year] == 1, ['clusters' + month + year]] = 'HH'
        aux.loc[aux['clusters' + month + year] == 2, ['clusters' + month + year]] = 'LH'
        aux.loc[aux['clusters' + month + year] == 3, ['clusters' + month + year]] = 'LL'
        aux.loc[aux['clusters' + month + year] == 4, ['clusters' + month + year]] = 'HL'
        cluster = pd.concat((ntl_scale_NTL2, aux), axis=1)
        cluster = pd.get_dummies(cluster)

        olsmodelstring_spatial = 'CNTL' + month + year + ' ~ ' + 'area_hr' + month + year + ' + '  + \
                                 'area_nr' + month + year + ' + ' + 'area_bg' + month + year + ' + ' + \
                                 'clusters' + month + year + '_HH'  + ' + ' + \
                                 'clusters' + month + year + '_NS' + ' + ' + \
                                 'area_lr' + month + year

        print('Spatial Multiple Linear Regression for disaggregating nightlight'  + month + '-' + year + ':')
        model_NTL_spatial = ols(olsmodelstring_spatial,cluster).fit()
        print(model_NTL_spatial.summary())
        print("\nRetrieving manually the parameter estimates:")
        print(model_NTL_spatial._results.params)

        y, X = dmatrices(olsmodelstring_spatial,data=cluster, return_type='dataframe')
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['variable'] = X.columns
        print(vif)

        model_NTL_spatial.save(results + 'ols_ntl_spatial.pickle', remove_data=False)

        NTL_clip.reset_index(inplace=True)
        cluster = cluster.merge(NTL_clip[['ntl_clip_id']], left_on=cluster.index, right_on=NTL_clip.index.array,
                                how='left')
        cluster.drop('key_0', axis=1, inplace=True)
        cluster.set_index('ntl_clip_id', inplace=True)
        intersect2 = intersect2.merge(
            cluster.loc[:, ['clusters' + month + year + '_HH', 'clusters' + month + year + '_HL', 'clusters' + month + year + '_LL', 'clusters' + month + year + '_NS', 'clusters' + month + year + '_LH']],
            left_on=intersect2.index, right_on=cluster.index, how='left')
        intersect2.rename({'key_0': 'ntl_clip_id'}, inplace=True, axis=1)
        try:
            intersect2.drop('level_0', inplace=True, axis=1)
        except:
            print('done!')
        cluster.reset_index(inplace=True)

        model_NTL_pred = cluster['area_bg' + month + year] * model_NTL.params[3] + cluster['area_hr' +  month + year] * \
                         model_NTL.params[1] + \
                         cluster['area_nr' +  month + year] * model_NTL.params[2] + cluster['area_lr' +  month + year] * \
                         model_NTL.params[4]
        model_NTL_pred = pd.DataFrame(model_NTL_pred, columns=['ntlpred' +  month + year])
        Predictions = pd.concat((ntl_scale_NTL2, model_NTL_pred), axis=1)

        model_NTL_spatial_pred = cluster['area_bg' + month + year] * model_NTL_spatial.params[3] + cluster[
            'area_hr' + month + year] * model_NTL_spatial.params[1] + \
                                 cluster['area_nr' + month + year] * model_NTL_spatial.params[2] + cluster[
                                     'clusters' + month + year + '_HH'] * model_NTL_spatial.params[4] + \
                                 cluster['clusters' + month + year + '_NS'] * model_NTL_spatial.params[5] + \
                                 cluster['area_lr' + month + year] * model_NTL_spatial.params[6]
        model_NTL_spatial_pred = pd.DataFrame(model_NTL_spatial_pred, columns=['ntlpred' + month + year])
        Predictions_spatial = pd.concat((ntl_scale_NTL2, model_NTL_spatial_pred), axis=1)

        sns.set(rc={'figure.figsize': (5, 10)}, style="whitegrid")
        f, axes = plt.subplots(2, 1)
        f.subplots_adjust(hspace=.5)
        sns.scatterplot(x=Predictions['ntlpred' + month + year], y=Predictions['CNTL' + month + year], ax=axes[0],
                        color='black')
        axes[0].set(xlabel='Nightlight emmission ' + month + year + ' (OLS model)')
        axes[0].set(ylabel='Nightlight emmission ' + month + year)
        sns.scatterplot(x=Predictions_spatial['ntlpred' + month + year], y=Predictions['CNTL' + month + year], ax=axes[1],
                        color='black')
        axes[1].set(xlabel='Nightlight emmission ' + month + year + ' (Spatial model)')
        axes[1].set(ylabel='Nightlight emmission ' + month + year)

    elif int(year) == 2017 and month == 'Feb':  # LH is insignificant
        olsmodelstring = 'CNTL' + month + year + ' ~ ' + 'area_hr' + month + year + ' + '  + \
                        'area_nr' + month + year + ' + ' + 'area_bg' + month + year + ' + ' + 'area_lr' + month + year
        model_NTL = ols(olsmodelstring,ntl_scale_NTL2).fit()
        # print(mdl + ': ' + olsmodelstring + "\n")
        # print(model_NTL.summary())
        # print("\nRetrieving manually the parameter estimates:")
        print(model_NTL._results.params)
        model_NTL.save(results + 'ols_ntl.pickle', remove_data=False)

        y, X = dmatrices(olsmodelstring,data=ntl_scale_NTL2, return_type='dataframe')
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['variable'] = X.columns
        print(vif)

        ntlresid = pd.concat((ntl_scale_NTL2, model_NTL.resid), axis=1)
        ntlresid.rename({0: 'ntlresid' + month + year}, axis=1, inplace=True)
        W = Queen.from_dataframe(ntlresid)
        W.transform = 'r'
        moran_ntl = Moran(ntlresid['ntlresid' + month + year], W)
        print('moran_ntl' + month + year + ': ' + str(moran_ntl.I))
        moran_loc = Moran_Local(ntlresid['ntlresid' + month + year], W)
        p = lisa_cluster(moran_loc, ntlresid, p=0.05, figsize=(9, 9))
        plt.title('Cluster Map of Nightlight Residuals(' + month +  '-' + year + ')', size=20)
        plt.show()
        plt.savefig(
            'C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Results/Cluster Map of Nightlight Residuals_' + month + '-' + year + '.png',
            dpi=500, bbox_inches='tight')

        # 1 HH, 2 LH, 3 LL, 4 HL
        cluster = _viz_utils.moran_hot_cold_spots(moran_loc, p=0.05)
        aux = pd.DataFrame(cluster)
        aux.rename({0: 'clusters' + month + year}, inplace=True, axis=1)
        aux.loc[aux['clusters' + month + year] == 0, ['clusters' + month + year]] = 'NS'
        aux.loc[aux['clusters' + month + year] == 1, ['clusters' + month + year]] = 'HH'
        aux.loc[aux['clusters' + month + year] == 2, ['clusters' + month + year]] = 'LH'
        aux.loc[aux['clusters' + month + year] == 3, ['clusters' + month + year]] = 'LL'
        aux.loc[aux['clusters' + month + year] == 4, ['clusters' + month + year]] = 'HL'
        cluster = pd.concat((ntl_scale_NTL2, aux), axis=1)
        cluster = pd.get_dummies(cluster)

        olsmodelstring_spatial = 'CNTL' + month + year + ' ~ ' + 'area_hr' + month + year + ' + '  + \
                                 'area_nr' + month + year + ' + ' + 'area_bg' + month + year + ' + ' + \
                                 'clusters' + month + year + '_HH'  + ' + '  + 'clusters' + month + year + '_HL'  + ' + ' + \
                                 'clusters' + month + year + '_NS' + ' + '  + \
                                 'area_lr' + month + year

        print('Spatial Multiple Linear Regression for disaggregating nightlight'  + month + '-' + year + ':')
        model_NTL_spatial = ols(olsmodelstring_spatial,cluster).fit()
        print(model_NTL_spatial.summary())
        print("\nRetrieving manually the parameter estimates:")
        print(model_NTL_spatial._results.params)

        y, X = dmatrices(olsmodelstring_spatial,data=cluster, return_type='dataframe')
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['variable'] = X.columns
        print(vif)

        model_NTL_spatial.save(results + 'ols_ntl_spatial.pickle', remove_data=False)

        NTL_clip.reset_index(inplace=True)
        cluster = cluster.merge(NTL_clip[['ntl_clip_id']], left_on=cluster.index, right_on=NTL_clip.index.array,
                                how='left')
        cluster.drop('key_0', axis=1, inplace=True)
        cluster.set_index('ntl_clip_id', inplace=True)
        intersect2 = intersect2.merge(
            cluster.loc[:, ['clusters' + month + year + '_HH', 'clusters' + month + year + '_HL', 'clusters' + month + year + '_LL', 'clusters' + month + year + '_NS', 'clusters' + month + year + '_LH']],
            left_on=intersect2.index, right_on=cluster.index, how='left')
        intersect2.rename({'key_0': 'ntl_clip_id'}, inplace=True, axis=1)
        try:
            intersect2.drop('level_0', inplace=True, axis=1)
        except:
            print('done!')
        cluster.reset_index(inplace=True)

        model_NTL_pred = cluster['area_bg' + month + year] * model_NTL.params[3] + cluster['area_hr' +  month + year] * \
                         model_NTL.params[1] + \
                         cluster['area_nr' +  month + year] * model_NTL.params[2] + cluster['area_lr' +  month + year] * \
                         model_NTL.params[4]
        model_NTL_pred = pd.DataFrame(model_NTL_pred, columns=['ntlpred' +  month + year])
        Predictions = pd.concat((ntl_scale_NTL2, model_NTL_pred), axis=1)

        model_NTL_spatial_pred = cluster['area_bg' + month + year] * model_NTL_spatial.params[3] + cluster[
            'area_hr' + month + year] * model_NTL_spatial.params[1] + \
                                 cluster['area_nr' + month + year] * model_NTL_spatial.params[2] + cluster[
                                     'clusters' + month + year + '_HH'] * model_NTL_spatial.params[4] + \
                                 cluster['clusters' + month + year + '_HL'] * model_NTL_spatial.params[5] + \
                                 cluster['clusters' + month + year + '_NS'] * model_NTL_spatial.params[6] + \
                                 cluster['area_lr' + month + year] * model_NTL_spatial.params[7]
        model_NTL_spatial_pred = pd.DataFrame(model_NTL_spatial_pred, columns=['ntlpred' + month + year])
        Predictions_spatial = pd.concat((ntl_scale_NTL2, model_NTL_spatial_pred), axis=1)

        sns.set(rc={'figure.figsize': (5, 10)}, style="whitegrid")
        f, axes = plt.subplots(2, 1)
        f.subplots_adjust(hspace=.5)
        sns.scatterplot(x=Predictions['ntlpred' + month + year], y=Predictions['CNTL' + month + year], ax=axes[0],
                        color='black')
        axes[0].set(xlabel='Nightlight emmission ' + month + year + ' (OLS model)')
        axes[0].set(ylabel='Nightlight emmission ' + month + year)
        sns.scatterplot(x=Predictions_spatial['ntlpred' + month + year], y=Predictions['CNTL' + month + year], ax=axes[1],
                        color='black')
        axes[1].set(xlabel='Nightlight emmission ' + month + year + ' (Spatial model)')
        axes[1].set(ylabel='Nightlight emmission ' + month + year)

    elif int(year) == 2017 and month == 'Mar':  # HL and LH are not significant
        olsmodelstring = 'CNTL' + month + year + ' ~ ' + 'area_hr' + month + year + ' + '  + \
                        'area_nr' + month + year + ' + ' + 'area_bg' + month + year + ' + ' + 'area_lr' + month + year
        model_NTL = ols(olsmodelstring,ntl_scale_NTL2).fit()
        # print(mdl + ': ' + olsmodelstring + "\n")
        # print(model_NTL.summary())
        # print("\nRetrieving manually the parameter estimates:")
        print(model_NTL._results.params)
        model_NTL.save(results + 'ols_ntl.pickle', remove_data=False)

        y, X = dmatrices(olsmodelstring,data=ntl_scale_NTL2, return_type='dataframe')
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['variable'] = X.columns
        print(vif)

        ntlresid = pd.concat((ntl_scale_NTL2, model_NTL.resid), axis=1)
        ntlresid.rename({0: 'ntlresid' + month + year}, axis=1, inplace=True)
        W = Queen.from_dataframe(ntlresid)
        W.transform = 'r'
        moran_ntl = Moran(ntlresid['ntlresid' + month + year], W)
        print('moran_ntl' + month + year + ': ' + str(moran_ntl.I))
        moran_loc = Moran_Local(ntlresid['ntlresid' + month + year], W)
        p = lisa_cluster(moran_loc, ntlresid, p=0.05, figsize=(9, 9))
        plt.title('Cluster Map of Nightlight Residuals(' + month +  '-' + year + ')', size=20)
        plt.show()
        plt.savefig(
            'C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Results/Cluster Map of Nightlight Residuals_' + month + '-' + year + '.png',
            dpi=500, bbox_inches='tight')

        # 1 HH, 2 LH, 3 LL, 4 HL
        cluster = _viz_utils.moran_hot_cold_spots(moran_loc, p=0.05)
        aux = pd.DataFrame(cluster)
        aux.rename({0: 'clusters' + month + year}, inplace=True, axis=1)
        aux.loc[aux['clusters' + month + year] == 0, ['clusters' + month + year]] = 'NS'
        aux.loc[aux['clusters' + month + year] == 1, ['clusters' + month + year]] = 'HH'
        aux.loc[aux['clusters' + month + year] == 2, ['clusters' + month + year]] = 'LH'
        aux.loc[aux['clusters' + month + year] == 3, ['clusters' + month + year]] = 'LL'
        aux.loc[aux['clusters' + month + year] == 4, ['clusters' + month + year]] = 'HL'
        cluster = pd.concat((ntl_scale_NTL2, aux), axis=1)
        cluster = pd.get_dummies(cluster)

        olsmodelstring_spatial = 'CNTL' + month + year + ' ~ ' + 'area_hr' + month + year + ' + '  + \
                                 'area_nr' + month + year + ' + ' + 'area_bg' + month + year + ' + ' + \
                                 'clusters' + month + year + '_HH'  + ' + ' + \
                                 'clusters' + month + year + '_NS' + ' + ' + \
                                 'area_lr' + month + year

        print('Spatial Multiple Linear Regression for disaggregating nightlight'  + month + '-' + year + ':')
        model_NTL_spatial = ols(olsmodelstring_spatial,cluster).fit()
        print(model_NTL_spatial.summary())
        print("\nRetrieving manually the parameter estimates:")
        print(model_NTL_spatial._results.params)

        y, X = dmatrices(olsmodelstring_spatial,data=cluster, return_type='dataframe')
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['variable'] = X.columns
        print(vif)

        model_NTL_spatial.save(results + 'ols_ntl_spatial.pickle', remove_data=False)

        NTL_clip.reset_index(inplace=True)
        cluster = cluster.merge(NTL_clip[['ntl_clip_id']], left_on=cluster.index, right_on=NTL_clip.index.array,
                                how='left')
        cluster.drop('key_0', axis=1, inplace=True)
        cluster.set_index('ntl_clip_id', inplace=True)
        intersect2 = intersect2.merge(
            cluster.loc[:, ['clusters' + month + year + '_HH', 'clusters' + month + year + '_HL', 'clusters' + month + year + '_LL', 'clusters' + month + year + '_NS', 'clusters' + month + year + '_LH']],
            left_on=intersect2.index, right_on=cluster.index, how='left')
        intersect2.rename({'key_0': 'ntl_clip_id'}, inplace=True, axis=1)
        try:
            intersect2.drop('level_0', inplace=True, axis=1)
        except:
            print('done!')
        cluster.reset_index(inplace=True)

        model_NTL_pred = cluster['area_bg' + month + year] * model_NTL.params[3] + cluster['area_hr' +  month + year] * \
                         model_NTL.params[1] + \
                         cluster['area_nr' +  month + year] * model_NTL.params[2] + cluster['area_lr' +  month + year] * \
                         model_NTL.params[4]
        model_NTL_pred = pd.DataFrame(model_NTL_pred, columns=['ntlpred' +  month + year])
        Predictions = pd.concat((ntl_scale_NTL2, model_NTL_pred), axis=1)

        model_NTL_spatial_pred = cluster['area_bg' + month + year] * model_NTL_spatial.params[3] + cluster[
            'area_hr' + month + year] * model_NTL_spatial.params[1] + \
                                 cluster['area_nr' + month + year] * model_NTL_spatial.params[2] + cluster[
                                     'clusters' + month + year + '_HH'] * model_NTL_spatial.params[4] + \
                                 cluster['clusters' + month + year + '_NS'] * model_NTL_spatial.params[5] + \
                                 cluster['area_lr' + month + year] * model_NTL_spatial.params[6]
        model_NTL_spatial_pred = pd.DataFrame(model_NTL_spatial_pred, columns=['ntlpred' + month + year])
        Predictions_spatial = pd.concat((ntl_scale_NTL2, model_NTL_spatial_pred), axis=1)

        sns.set(rc={'figure.figsize': (5, 10)}, style="whitegrid")
        f, axes = plt.subplots(2, 1)
        f.subplots_adjust(hspace=.5)
        sns.scatterplot(x=Predictions['ntlpred' + month + year], y=Predictions['CNTL' + month + year], ax=axes[0],
                        color='black')
        axes[0].set(xlabel='Nightlight emmission ' + month + year + ' (OLS model)')
        axes[0].set(ylabel='Nightlight emmission ' + month + year)
        sns.scatterplot(x=Predictions_spatial['ntlpred' + month + year], y=Predictions['CNTL' + month + year], ax=axes[1],
                        color='black')
        axes[1].set(xlabel='Nightlight emmission ' + month + year + ' (Spatial model)')
        axes[1].set(ylabel='Nightlight emmission ' + month + year)

    elif int(year) == 2017 and month == 'Jun':  # LH is insignificant
        olsmodelstring = 'CNTL' + month + year + ' ~ ' + 'area_hr' + month + year + ' + '  + \
                        'area_nr' + month + year + ' + ' + 'area_bg' + month + year + ' + ' + 'area_lr' + month + year
        model_NTL = ols(olsmodelstring,ntl_scale_NTL2).fit()
        # print(mdl + ': ' + olsmodelstring + "\n")
        # print(model_NTL.summary())
        # print("\nRetrieving manually the parameter estimates:")
        print(model_NTL._results.params)
        model_NTL.save(results + 'ols_ntl.pickle', remove_data=False)

        y, X = dmatrices(olsmodelstring,data=ntl_scale_NTL2, return_type='dataframe')
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['variable'] = X.columns
        print(vif)

        ntlresid = pd.concat((ntl_scale_NTL2, model_NTL.resid), axis=1)
        ntlresid.rename({0: 'ntlresid' + month + year}, axis=1, inplace=True)
        W = Queen.from_dataframe(ntlresid)
        W.transform = 'r'
        moran_ntl = Moran(ntlresid['ntlresid' + month + year], W)
        print('moran_ntl' + month + year + ': ' + str(moran_ntl.I))
        moran_loc = Moran_Local(ntlresid['ntlresid' + month + year], W)
        p = lisa_cluster(moran_loc, ntlresid, p=0.05, figsize=(9, 9))
        plt.title('Cluster Map of Nightlight Residuals(' + month +  '-' + year + ')', size=20)
        plt.show()
        plt.savefig(
            'C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Results/Cluster Map of Nightlight Residuals_' + month + '-' + year + '.png',
            dpi=500, bbox_inches='tight')

        # 1 HH, 2 LH, 3 LL, 4 HL
        cluster = _viz_utils.moran_hot_cold_spots(moran_loc, p=0.05)
        aux = pd.DataFrame(cluster)
        aux.rename({0: 'clusters' + month + year}, inplace=True, axis=1)
        aux.loc[aux['clusters' + month + year] == 0, ['clusters' + month + year]] = 'NS'
        aux.loc[aux['clusters' + month + year] == 1, ['clusters' + month + year]] = 'HH'
        aux.loc[aux['clusters' + month + year] == 2, ['clusters' + month + year]] = 'LH'
        aux.loc[aux['clusters' + month + year] == 3, ['clusters' + month + year]] = 'LL'
        aux.loc[aux['clusters' + month + year] == 4, ['clusters' + month + year]] = 'HL'
        cluster = pd.concat((ntl_scale_NTL2, aux), axis=1)
        cluster = pd.get_dummies(cluster)

        olsmodelstring_spatial = 'CNTL' + month + year + ' ~ ' + 'area_hr' + month + year + ' + '  + \
                                 'area_nr' + month + year + ' + ' + 'area_bg' + month + year + ' + ' + \
                                 'clusters' + month + year + '_HH'  + ' + '  + 'clusters' + month + year + '_HL'  + ' + ' + \
                                 'clusters' + month + year + '_NS' + ' + '  + \
                                 'area_lr' + month + year

        print('Spatial Multiple Linear Regression for disaggregating nightlight'  + month + '-' + year + ':')
        model_NTL_spatial = ols(olsmodelstring_spatial,cluster).fit()
        print(model_NTL_spatial.summary())
        print("\nRetrieving manually the parameter estimates:")
        print(model_NTL_spatial._results.params)

        y, X = dmatrices(olsmodelstring_spatial,data=cluster, return_type='dataframe')
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['variable'] = X.columns
        print(vif)

        model_NTL_spatial.save(results + 'ols_ntl_spatial.pickle', remove_data=False)

        NTL_clip.reset_index(inplace=True)
        cluster = cluster.merge(NTL_clip[['ntl_clip_id']], left_on=cluster.index, right_on=NTL_clip.index.array,
                                how='left')
        cluster.drop('key_0', axis=1, inplace=True)
        cluster.set_index('ntl_clip_id', inplace=True)
        intersect2 = intersect2.merge(
            cluster.loc[:, ['clusters' + month + year + '_HH', 'clusters' + month + year + '_HL', 'clusters' + month + year + '_LL', 'clusters' + month + year + '_NS', 'clusters' + month + year + '_LH']],
            left_on=intersect2.index, right_on=cluster.index, how='left')
        intersect2.rename({'key_0': 'ntl_clip_id'}, inplace=True, axis=1)
        try:
            intersect2.drop('level_0', inplace=True, axis=1)
        except:
            print('done!')
        cluster.reset_index(inplace=True)

        model_NTL_pred = cluster['area_bg' + month + year] * model_NTL.params[3] + cluster['area_hr' +  month + year] * \
                         model_NTL.params[1] + \
                         cluster['area_nr' +  month + year] * model_NTL.params[2] + cluster['area_lr' +  month + year] * \
                         model_NTL.params[4]
        model_NTL_pred = pd.DataFrame(model_NTL_pred, columns=['ntlpred' +  month + year])
        Predictions = pd.concat((ntl_scale_NTL2, model_NTL_pred), axis=1)

        model_NTL_spatial_pred = cluster['area_bg' + month + year] * model_NTL_spatial.params[3] + cluster[
            'area_hr' + month + year] * model_NTL_spatial.params[1] + \
                                 cluster['area_nr' + month + year] * model_NTL_spatial.params[2] + cluster[
                                     'clusters' + month + year + '_HH'] * model_NTL_spatial.params[4] + \
                                 cluster['clusters' + month + year + '_HL'] * model_NTL_spatial.params[5] + \
                                 cluster['clusters' + month + year + '_NS'] * model_NTL_spatial.params[6] + \
                                 cluster['area_lr' + month + year] * model_NTL_spatial.params[7]
        model_NTL_spatial_pred = pd.DataFrame(model_NTL_spatial_pred, columns=['ntlpred' + month + year])
        Predictions_spatial = pd.concat((ntl_scale_NTL2, model_NTL_spatial_pred), axis=1)

        sns.set(rc={'figure.figsize': (5, 10)}, style="whitegrid")
        f, axes = plt.subplots(2, 1)
        f.subplots_adjust(hspace=.5)
        sns.scatterplot(x=Predictions['ntlpred' + month + year], y=Predictions['CNTL' + month + year], ax=axes[0],
                        color='black')
        axes[0].set(xlabel='Nightlight emmission ' + month + year + ' (OLS model)')
        axes[0].set(ylabel='Nightlight emmission ' + month + year)
        sns.scatterplot(x=Predictions_spatial['ntlpred' + month + year], y=Predictions['CNTL' + month + year], ax=axes[1],
                        color='black')
        axes[1].set(xlabel='Nightlight emmission ' + month + year + ' (Spatial model)')
        axes[1].set(ylabel='Nightlight emmission ' + month + year)

    elif int(year) == 2017 and month == 'Jul':  # LH is insignificant
        olsmodelstring = 'CNTL' + month + year + ' ~ ' + 'area_hr' + month + year + ' + '  + \
                        'area_nr' + month + year + ' + ' + 'area_bg' + month + year + ' + ' + 'area_lr' + month + year
        model_NTL = ols(olsmodelstring,ntl_scale_NTL2).fit()
        # print(mdl + ': ' + olsmodelstring + "\n")
        # print(model_NTL.summary())
        # print("\nRetrieving manually the parameter estimates:")
        print(model_NTL._results.params)
        model_NTL.save(results + 'ols_ntl.pickle', remove_data=False)

        y, X = dmatrices(olsmodelstring,data=ntl_scale_NTL2, return_type='dataframe')
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['variable'] = X.columns
        print(vif)

        ntlresid = pd.concat((ntl_scale_NTL2, model_NTL.resid), axis=1)
        ntlresid.rename({0: 'ntlresid' + month + year}, axis=1, inplace=True)
        W = Queen.from_dataframe(ntlresid)
        W.transform = 'r'
        moran_ntl = Moran(ntlresid['ntlresid' + month + year], W)
        print('moran_ntl' + month + year + ': ' + str(moran_ntl.I))
        moran_loc = Moran_Local(ntlresid['ntlresid' + month + year], W)
        p = lisa_cluster(moran_loc, ntlresid, p=0.05, figsize=(9, 9))
        plt.title('Cluster Map of Nightlight Residuals(' + month +  '-' + year + ')', size=20)
        plt.show()
        plt.savefig(
            'C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Results/Cluster Map of Nightlight Residuals_' + month + '-' + year + '.png',
            dpi=500, bbox_inches='tight')

        # 1 HH, 2 LH, 3 LL, 4 HL
        cluster = _viz_utils.moran_hot_cold_spots(moran_loc, p=0.05)
        aux = pd.DataFrame(cluster)
        aux.rename({0: 'clusters' + month + year}, inplace=True, axis=1)
        aux.loc[aux['clusters' + month + year] == 0, ['clusters' + month + year]] = 'NS'
        aux.loc[aux['clusters' + month + year] == 1, ['clusters' + month + year]] = 'HH'
        aux.loc[aux['clusters' + month + year] == 2, ['clusters' + month + year]] = 'LH'
        aux.loc[aux['clusters' + month + year] == 3, ['clusters' + month + year]] = 'LL'
        aux.loc[aux['clusters' + month + year] == 4, ['clusters' + month + year]] = 'HL'
        cluster = pd.concat((ntl_scale_NTL2, aux), axis=1)
        cluster = pd.get_dummies(cluster)

        olsmodelstring_spatial = 'CNTL' + month + year + ' ~ ' + 'area_hr' + month + year + ' + '  + \
                                 'area_nr' + month + year + ' + ' + 'area_bg' + month + year + ' + ' + \
                                 'clusters' + month + year + '_HH'  + ' + '  + 'clusters' + month + year + '_HL'  + ' + ' + \
                                 'clusters' + month + year + '_NS' + ' + '  + \
                                 'area_lr' + month + year

        print('Spatial Multiple Linear Regression for disaggregating nightlight'  + month + '-' + year + ':')
        model_NTL_spatial = ols(olsmodelstring_spatial,cluster).fit()
        print(model_NTL_spatial.summary())
        print("\nRetrieving manually the parameter estimates:")
        print(model_NTL_spatial._results.params)

        y, X = dmatrices(olsmodelstring_spatial,data=cluster, return_type='dataframe')
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['variable'] = X.columns
        print(vif)

        model_NTL_spatial.save(results + 'ols_ntl_spatial.pickle', remove_data=False)

        NTL_clip.reset_index(inplace=True)
        cluster = cluster.merge(NTL_clip[['ntl_clip_id']], left_on=cluster.index, right_on=NTL_clip.index.array,
                                how='left')
        cluster.drop('key_0', axis=1, inplace=True)
        cluster.set_index('ntl_clip_id', inplace=True)
        intersect2 = intersect2.merge(
            cluster.loc[:, ['clusters' + month + year + '_HH', 'clusters' + month + year + '_HL', 'clusters' + month + year + '_LL', 'clusters' + month + year + '_NS', 'clusters' + month + year + '_LH']],
            left_on=intersect2.index, right_on=cluster.index, how='left')
        intersect2.rename({'key_0': 'ntl_clip_id'}, inplace=True, axis=1)
        try:
            intersect2.drop('level_0', inplace=True, axis=1)
        except:
            print('done!')
        cluster.reset_index(inplace=True)

        model_NTL_pred = cluster['area_bg' + month + year] * model_NTL.params[3] + cluster['area_hr' +  month + year] * \
                         model_NTL.params[1] + \
                         cluster['area_nr' +  month + year] * model_NTL.params[2] + cluster['area_lr' +  month + year] * \
                         model_NTL.params[4]
        model_NTL_pred = pd.DataFrame(model_NTL_pred, columns=['ntlpred' +  month + year])
        Predictions = pd.concat((ntl_scale_NTL2, model_NTL_pred), axis=1)

        model_NTL_spatial_pred = cluster['area_bg' + month + year] * model_NTL_spatial.params[3] + cluster[
            'area_hr' + month + year] * model_NTL_spatial.params[1] + \
                                 cluster['area_nr' + month + year] * model_NTL_spatial.params[2] + cluster[
                                     'clusters' + month + year + '_HH'] * model_NTL_spatial.params[4] + \
                                 cluster['clusters' + month + year + '_HL'] * model_NTL_spatial.params[5] + \
                                 cluster['clusters' + month + year + '_NS'] * model_NTL_spatial.params[6] + \
                                 cluster['area_lr' + month + year] * model_NTL_spatial.params[7]
        model_NTL_spatial_pred = pd.DataFrame(model_NTL_spatial_pred, columns=['ntlpred' + month + year])
        Predictions_spatial = pd.concat((ntl_scale_NTL2, model_NTL_spatial_pred), axis=1)

        sns.set(rc={'figure.figsize': (5, 10)}, style="whitegrid")
        f, axes = plt.subplots(2, 1)
        f.subplots_adjust(hspace=.5)
        sns.scatterplot(x=Predictions['ntlpred' + month + year], y=Predictions['CNTL' + month + year], ax=axes[0],
                        color='black')
        axes[0].set(xlabel='Nightlight emmission ' + month + year + ' (OLS model)')
        axes[0].set(ylabel='Nightlight emmission ' + month + year)
        sns.scatterplot(x=Predictions_spatial['ntlpred' + month + year], y=Predictions['CNTL' + month + year], ax=axes[1],
                        color='black')
        axes[1].set(xlabel='Nightlight emmission ' + month + year + ' (Spatial model)')
        axes[1].set(ylabel='Nightlight emmission ' + month + year)

    from statsmodels.regression.linear_model import OLSResults
    ols_ntl_spatial = OLSResults.load(results + 'ols_ntl_spatial.pickle')
    if int(year) == 2014 and month == 'Jun':  # All significant

        intersect2['coef_ntl' + month + year] = np.nan
        intersect2['spatial_ntl' + month + year] = np.nan

        mask = intersect2['landuse' + year] == 1
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[3]
        mask = intersect2['landuse' + year] == 2
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[8]
        mask = intersect2['landuse' + year] == 3
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[1]
        mask = intersect2['landuse' + year] == 4
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[2]

        mask = (intersect2['clusters' + month + year + '_HH'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[4]
        mask = (intersect2['clusters' + month + year + '_HL'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[5]
        mask = (intersect2['clusters' + month + year + '_NS'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[6]
        mask = (intersect2['clusters' + month + year + '_LH'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[7]
        mask = (intersect2['clusters' + month + year + '_LL'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = 0

    elif int(year) == 2014 and month == 'Jul': # lr is not significant

        intersect2['coef_ntl' + month + year] = np.nan
        intersect2['spatial_ntl' + month + year] = np.nan

        mask = intersect2['landuse' + year] == 1
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[3]
        mask = intersect2['landuse' + year] == 2
        intersect2.loc[mask, ['coef_ntl' + month + year]] = 0
        mask = intersect2['landuse' + year] == 3
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[1]
        mask = intersect2['landuse' + year] == 4
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[2]

        mask = (intersect2['clusters' + month + year + '_HH'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[4]
        mask = (intersect2['clusters' + month + year + '_HL'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[5]
        mask = (intersect2['clusters' + month + year + '_NS'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[6]
        mask = (intersect2['clusters' + month + year + '_LH'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[7]
        mask = (intersect2['clusters' + month + year + '_LL'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = 0

    elif int(year) == 2016 and month == 'Oct':  # All significant

        intersect2['coef_ntl' + month + year] = np.nan
        intersect2['spatial_ntl' + month + year] = np.nan

        mask = intersect2['landuse' + year] == 1
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[3]
        mask = intersect2['landuse' + year] == 2
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[8]
        mask = intersect2['landuse' + year] == 3
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[1]
        mask = intersect2['landuse' + year] == 4
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[2]

        mask = (intersect2['clusters' + month + year + '_HH'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[4]
        mask = (intersect2['clusters' + month + year + '_HL'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[5]
        mask = (intersect2['clusters' + month + year + '_NS'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[6]
        mask = (intersect2['clusters' + month + year + '_LH'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[7]
        mask = (intersect2['clusters' + month + year + '_LL'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = 0

    elif int(year) == 2016 and month == 'Nov':  # lr is insignificant

        intersect2['coef_ntl' + month + year] = np.nan
        intersect2['spatial_ntl' + month + year] = np.nan

        mask = intersect2['landuse' + year] == 1
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[3]
        mask = intersect2['landuse' + year] == 2
        intersect2.loc[mask, ['coef_ntl' + month + year]] = 0
        mask = intersect2['landuse' + year] == 3
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[1]
        mask = intersect2['landuse' + year] == 4
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[2]

        mask = (intersect2['clusters' + month + year + '_HH'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[4]
        mask = (intersect2['clusters' +  month + year + '_HL'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[5]
        mask = (intersect2['clusters' +  month + year + '_NS'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[6]
        mask = (intersect2['clusters' +  month + year + '_LH'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[7]
        mask = (intersect2['clusters' + month + year + '_LL'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = 0

    elif int(year) == 2017 and month == 'Jan':  # HL and LH are not significant

        intersect2['coef_ntl' + month + year] = np.nan
        intersect2['spatial_ntl' + month + year] = np.nan

        mask = intersect2['landuse' + year] == 1
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[3]
        mask = intersect2['landuse' + year] == 2
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[6]
        mask = intersect2['landuse' + year] == 3
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[1]
        mask = intersect2['landuse' + year] == 4
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[2]

        mask = (intersect2['clusters' + month + year + '_HH'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[4]
        mask = (intersect2['clusters' + month + year + '_HL'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = 0
        mask = (intersect2['clusters' + month + year + '_NS'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[5]
        mask = (intersect2['clusters' + month + year + '_LH'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = 0
        mask = (intersect2['clusters' + month + year + '_LL'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = 0

    elif int(year) == 2017 and month == 'Feb':  # LH is insignificant

        intersect2['coef_ntl' + month + year] = np.nan
        intersect2['spatial_ntl' + month + year] = np.nan

        mask = intersect2['landuse' + year] == 1
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[3]
        mask = intersect2['landuse' + year] == 2
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[7]
        mask = intersect2['landuse' + year] == 3
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[1]
        mask = intersect2['landuse' + year] == 4
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[2]

        mask = (intersect2['clusters' + month + year + '_HH'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[4]
        mask = (intersect2['clusters' + month + year + '_HL'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[5]
        mask = (intersect2['clusters' + month + year + '_NS'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[6]
        mask = (intersect2['clusters' + month + year + '_LH'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = 0
        mask = (intersect2['clusters' + month + year + '_LL'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = 0

    elif int(year) == 2017 and month == 'Mar':  # HL and LH are not significant

        intersect2['coef_ntl' + month + year] = np.nan
        intersect2['spatial_ntl' + month + year] = np.nan

        mask = intersect2['landuse' + year] == 1
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[3]
        mask = intersect2['landuse' + year] == 2
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[6]
        mask = intersect2['landuse' + year] == 3
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[1]
        mask = intersect2['landuse' + year] == 4
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[2]

        mask = (intersect2['clusters' + month + year + '_HH'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[4]
        mask = (intersect2['clusters' + month + year + '_HL'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = 0
        mask = (intersect2['clusters' + month + year + '_NS'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[5]
        mask = (intersect2['clusters' + month + year + '_LH'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = 0
        mask = (intersect2['clusters' + month + year + '_LL'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = 0

    elif int(year) == 2017 and month == 'Jun':  # LH is insignificant

        intersect2['coef_ntl' + month + year] = np.nan
        intersect2['spatial_ntl' + month + year] = np.nan

        mask = intersect2['landuse' + year] == 1
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[3]
        mask = intersect2['landuse' + year] == 2
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[7]
        mask = intersect2['landuse' + year] == 3
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[1]
        mask = intersect2['landuse' + year] == 4
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[2]

        mask = (intersect2['clusters' + month + year + '_HH'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[4]
        mask = (intersect2['clusters' + month + year + '_HL'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[5]
        mask = (intersect2['clusters' + month + year + '_NS'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[6]
        mask = (intersect2['clusters' + month + year + '_LH'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = 0
        mask = (intersect2['clusters' + month + year + '_LL'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = 0

    elif int(year) == 2017 and month == 'Jul':  # LH is insignificant

        intersect2['coef_ntl' + month + year] = np.nan
        intersect2['spatial_ntl' + month + year] = np.nan

        mask = intersect2['landuse' + year] == 1
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[3]
        mask = intersect2['landuse' + year] == 2
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[7]
        mask = intersect2['landuse' + year] == 3
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[1]
        mask = intersect2['landuse' + year] == 4
        intersect2.loc[mask, ['coef_ntl' + month + year]] = model_NTL_spatial.params[2]

        mask = (intersect2['clusters' + month + year + '_HH'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[4]
        mask = (intersect2['clusters' + month + year + '_HL'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[5]
        mask = (intersect2['clusters' + month + year + '_NS'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = model_NTL_spatial.params[6]
        mask = (intersect2['clusters' + month + year + '_LH'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = 0
        mask = (intersect2['clusters' + month + year + '_LL'] == 1)
        intersect2.loc[mask, ['spatial_ntl' + month + year]] = 0

    intersect2['disNTL' + my] = intersect2['landuse_clip_area'] * intersect2['coef_ntl' + my] + intersect2['spatial_ntl' + my]

    intersect2.reset_index(inplace=True)
    # We have to set the negative values to 0
    mask = intersect2['disNTL' + my] < 0
    mask2 = intersect2['disNTL' + my] >= 0
    print('Percentage error caused by removing negative values in nightlight: ',
          abs(intersect2[mask].sum()['disNTL' + my] / intersect2[mask2].sum()['disNTL' + my]) * 100)

    intersect2.loc[mask, ['disNTL' + my]] = 0

    intersect2['disNTL_verify' + my] = intersect2['disNTL' + my].groupby(intersect2.ntl_clip_id).transform(
        'sum')

    intersect2['disNTL_prime' + my] = 0
    mask = intersect2['disNTL_verify' + my] != 0
    intersect2.loc[mask, 'disNTL_prime' + my] = np.array(intersect2.loc[mask, ['disNTL' + my]]) * \
                                                  (np.array(intersect2.loc[mask, ['CNTL' + my]]) / np.array(
                                                      intersect2.loc[mask, ['disNTL_verify' + my]]))

    print(intersect2.groupby('ntl_clip_id').sum().loc[:, ['disNTL_prime' + my]])
    print(intersect2.groupby('ntl_clip_id').max().loc[:, ['CNTL' + my]])

# in the level of census
intersect2.set_index('census_id', inplace=True)
intersect2['countPop'] = intersect2['index'].groupby(intersect2.index).transform('count')
ntl_scale_Pop = intersect2.groupby(['census_id', 'landuse2014']).sum().loc[:,['intersect_area']]
ntl_scale_Pop2 = ntl_scale_Pop.unstack('landuse2014')
ntl_scale_Pop2.columns = ['area_bg', 'area_lr', 'area_hr', 'area_nr']
ntl_scale_Pop2.fillna(0, inplace=True)

ntl_scale_Pop2['target_pop2013'] = intersect2.groupby(intersect2.index).max()['estPop2013']

ntl_scale_Pop2.reset_index(inplace=True)
ntl_scale_Pop2 = census.merge(ntl_scale_Pop2, left_on = census.index.array, right_on = ntl_scale_Pop2.index.array, how='left')
# ntl_scale_Pop2.drop(['key_0', 'index', 'Shape_Leng', 'Shape_Area', 'MAX_popult', 'census_id_x', 'census_area'],
#                 inplace=True, axis=1)
ntl_scale_Pop2.drop(['key_0', 'Shape_Leng', 'Shape_Area', 'MAX_popult', 'census_area', 'census_id_x'],inplace=True, axis=1)
ntl_scale_Pop2.rename({'census_id_y': 'census_id'}, axis=1, inplace=True)
ntl_scale_Pop2['X'] = ntl_scale_Pop2.geometry.centroid.x
ntl_scale_Pop2['Y'] = ntl_scale_Pop2.geometry.centroid.y

y, X = dmatrices(
    "estPop2013 ~ area_hr + area_nr + area_lr + area_bg",
    data=ntl_scale_Pop2, return_type='dataframe')
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['variable'] = X.columns
print(vif)

print('Multiple Linear Regression for disaggregating population with land use')
model_Pop = ols("estPop2013 ~ area_hr + area_nr + area_lr + area_bg", ntl_scale_Pop2).fit()
print(model_Pop.summary())
print("\nRetrieving manually the parameter estimates:")
print(model_Pop._results.params)

model_Pop.save(results + 'ols_census.pickle', remove_data=False)

popresid = pd.concat((ntl_scale_Pop2, model_Pop.resid), axis=1)
popresid.rename({0: 'popresid2013'}, axis=1, inplace=True)
W = Queen.from_dataframe(popresid)
W.transform = 'r'
moran_pop = Moran(popresid.popresid2013, W)
print('moran_pop2013: ' + str(moran_pop.I))
moran_loc = Moran_Local(popresid['popresid2013'], W)
p = lisa_cluster(moran_loc, popresid, p=0.05, figsize=(10, 10))
plt.title('Cluster Map of Population Residuals (2013)', size=20)
plt.show()
plt.savefig(
    'C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Results/Cluster Map of Population Residuals.png',
    dpi=500, bbox_inches='tight')

# 1 HH, 2 LH, 3 LL, 4 HL
cluster = _viz_utils.moran_hot_cold_spots(moran_loc, p=0.05)
aux = pd.DataFrame(cluster)
aux.rename({0: 'clusters2013pop'}, inplace=True, axis=1)
aux.loc[aux['clusters2013pop'] == 0, ['clusters2013pop']] = 'NS'
aux.loc[aux['clusters2013pop'] == 1, ['clusters2013pop']] = 'HH'
aux.loc[aux['clusters2013pop'] == 2, ['clusters2013pop']] = 'LH'
aux.loc[aux['clusters2013pop'] == 3, ['clusters2013pop']] = 'LL'
aux.loc[aux['clusters2013pop'] == 4, ['clusters2013pop']] = 'HL'
cluster = pd.concat((ntl_scale_Pop2, aux), axis=1)
cluster = pd.get_dummies(cluster)

print('Spatial Multiple Linear Regression for disaggregating population 2013:')
model_pop_spatial = ols(
    "estPop2013 ~  area_hr + area_nr + area_bg + area_lr + clusters2013pop_HH + clusters2013pop_HL + clusters2013pop_LH + clusters2013pop_NS",cluster).fit()
print(model_pop_spatial.summary())
print("\nRetrieving manually the parameter estimates:")
print(model_pop_spatial._results.params)

print('Spatial Multiple Linear Regression for disaggregating population 2013:')
model_pop_spatial = ols(
    "estPop2013 ~  area_hr + clusters2013pop_HH + clusters2013pop_HL + clusters2013pop_NS",cluster).fit()
print(model_pop_spatial.summary())
print("\nRetrieving manually the parameter estimates:")
print(model_pop_spatial._results.params)

model_pop_spatial.save(results + 'ols_pop_spatial.pickle', remove_data=False)

try:
    intersect2.drop('level_0', axis=1, inplace=True)
except:
    print('done!')

cluster = cluster.merge(census['census_id'], left_on=cluster.index, right_on=census.index.array, how='left')
cluster.drop(['key_0', 'census_id_x'], axis=1, inplace=True)
cluster.rename({'census_id_y': 'census_id'}, axis=1, inplace=True)

cluster.set_index('census_id', inplace=True)
intersect2 = intersect2.merge(
    cluster.loc[:, ['clusters2013pop_HH', 'clusters2013pop_HL', 'clusters2013pop_LH', 'clusters2013pop_LL', 'clusters2013pop_NS']],
    left_on=intersect2.index, right_on=cluster.index, how='left')
intersect2.rename({'key_0': 'census_id'}, inplace=True, axis=1)
# intersect2.drop('level_0', axis=1, inplace=True)

cluster.reset_index(inplace=True)

model_Pop_pred = cluster['area_hr'] * model_Pop.params[1] + model_Pop.params[0]
model_Pop_pred = pd.DataFrame(model_Pop_pred)
model_Pop_pred.rename({'area_hr':'poppred2013'}, inplace=True, axis=1)
Predictions = pd.concat((ntl_scale_Pop2, model_Pop_pred), axis=1)

model_pop_spatial_pred = cluster['area_hr'] * model_pop_spatial.params[1] + cluster['clusters2013pop_HH'] * \
                         model_pop_spatial.params[2] + \
                         cluster['clusters2013pop_HL'] * model_pop_spatial.params[3] + \
                         cluster['clusters2013pop_NS'] * model_pop_spatial.params[4]

model_pop_spatial_pred = pd.DataFrame(model_pop_spatial_pred, columns=['poppred2013'])
Predictions_spatial = pd.concat((ntl_scale_Pop2, model_pop_spatial_pred), axis=1)

sns.set(rc={'figure.figsize': (10, 15)}, style="whitegrid")
f, axes = plt.subplots(2, 1)
f.subplots_adjust(hspace=.5)
sns.scatterplot(x=Predictions['poppred2013'], y=Predictions['estPop2013'], ax=axes[0], color='black')
axes[0].set(xlabel='Population 2013 (OLS model)')
axes[0].set(ylabel='Population 2013 (Census)')
sns.scatterplot(x=Predictions_spatial['poppred2013'], y=Predictions['estPop2013'], ax=axes[1], color='black')
axes[1].set(xlabel='Population 2013 (Spatial model)')
axes[1].set(ylabel='Population 2013 (Census)')

from statsmodels.regression.linear_model import OLSResults
model_pop_spatial = OLSResults.load(results + 'ols_pop_spatial.pickle')

intersect2['coef_Pop2014'] = np.nan
intersect2['spatial_pop'] = np.nan

mask = intersect2['landuse2014'] == 1
intersect2.loc[mask, ['coef_Pop2014']] = 0 #model_pop_spatial.params[3]
mask = intersect2['landuse2014'] == 2
intersect2.loc[mask, ['coef_Pop2014']] = 0 # model_NTL.params[2]
mask = intersect2['landuse2014'] == 3
intersect2.loc[mask, ['coef_Pop2014']] = model_pop_spatial.params[1]
mask = intersect2['landuse2014'] == 4
intersect2.loc[mask, ['coef_Pop2014']] = 0 # model_pop_spatial.params[2]

mask = (intersect2['clusters2013pop_HH'] == 1) & (intersect2['landuse2014'] == 3)
intersect2.loc[mask, ['spatial_pop']] = model_pop_spatial.params[2]
mask = (intersect2['clusters2013pop_HH'] == 1) & (intersect2['landuse2014'] != 3)
intersect2.loc[mask, ['spatial_pop']] = 0
mask = (intersect2['clusters2013pop_HL'] == 1) & (intersect2['landuse2014'] == 3)
intersect2.loc[mask, ['spatial_pop']] = model_pop_spatial.params[3]
mask = (intersect2['clusters2013pop_HL'] == 1) & (intersect2['landuse2014'] != 3)
intersect2.loc[mask, ['spatial_pop']] = 0
mask = (intersect2['clusters2013pop_LH'] == 1)
intersect2.loc[mask, ['spatial_pop']] = 0
mask = (intersect2['clusters2013pop_NS'] == 1) & (intersect2['landuse2014'] == 3)
intersect2.loc[mask, ['spatial_pop']] = model_pop_spatial.params[4]
mask = (intersect2['clusters2013pop_NS'] == 1) & (intersect2['landuse2014'] != 3)
intersect2.loc[mask, ['spatial_pop']] = 0
mask = (intersect2['clusters2013pop_LL'] == 1)
intersect2.loc[mask, ['spatial_pop']] = 0

intersect2['disPop2013'] = intersect2['landuse_clip_area']*intersect2['coef_Pop2014'] + intersect2['spatial_pop'] #+ \
                        # (intersect2['intercept_Pop2014']/intersect2['countPop'])

intersect2.set_index('census_id', inplace=True)

intersect2['disPop2013_verify'] = intersect2['disPop2013'].groupby(intersect2.index).transform('sum')

intersect2['disPop2013_prime'] = 0
mask = intersect2['disPop2013_verify'] != 0
intersect2.loc[mask, 'disPop2013_prime'] = np.array(intersect2.loc[mask, ['disPop2013']]) * (np.array(intersect2.loc[mask, ['estPop2013']]) / np.array(intersect2.loc[mask, ['disPop2013_verify']]))

# verify: these two must be the same
intersect2.groupby('census_id').sum().loc[:, ['disPop2013_prime']]
intersect2.groupby('census_id').max().loc[:, ['estPop2013']]

# in the level of night light
try:
    NTL_clip.drop(['level_0'], inplace=True, axis=1)

except:
    print('level_0 is not in the columns')
try:
    NTL_clip.set_index('ntl_clip_id', inplace=True)
except:
    print('Already satisfied')
NTL_clip_aux = NTL_clip

intersect2.set_index('ntl_clip_id', inplace=True)

for my in month_year:
    year = my[-4:]
    month = my[:3]

    ntl_prediction = intersect2.groupby(['ntl_clip_id', 'landuse' + year]).sum().loc[:,['intersect_area','disNTL_prime' + month + year]]
    ntl_prediction2 = ntl_prediction.unstack('landuse' + year)
    ntl_prediction2.columns = ['area_bg' + year, 'area_lr' + year, 'area_hr' + year, 'area_nr' + year,
                                       'NTL_bg' + month + year, 'NTL_lr' + month + year, 'NTL_hr' + month + year, 'NTL_nr' + month + year]
    ntl_prediction2['CNTL' + month + year] = intersect2.groupby(intersect2.index).max()['CNTL' + month + year]
    NTL_clip_aux['ntl_clip_id_copy'] = NTL_clip_aux.index
    NTL_clip_aux = NTL_clip_aux.merge(ntl_prediction2, left_on=NTL_clip_aux.index, right_on=ntl_prediction2.index, how='left')
    NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
    NTL_clip_aux.drop('key_0', inplace=True, axis=1)

NTL_clip_aux.fillna(0, inplace=True)
NTL_clip_aux['X'] = NTL_clip_aux.geometry.centroid.x
NTL_clip_aux['Y'] = NTL_clip_aux.geometry.centroid.y

NTL_clip_aux.to_csv(results + 'NTL_Level_monthly_incorrected_median_' + date + '.csv')

# Overall popualtion predictions by different reference months (one population model)
try:
    NTL_clip.set_index('ntl_clip_id', inplace=True)
except:
    print('ntl_clip_id is already index')

aux1 = []

for my in month_year:
    year = my[-4:]
    month = my[:3]
    aux1.append('estpop' + month + year)

df = pd.DataFrame(columns=['lndus', 'lndus_ntl_annual', 'lndus_ntlhr_annual', 'lndus_ntlhrnr_annual'],
                  index=['censuspop2013', 'pred', 'estpopJun2014', 'estpopJul2014', 'estpopOct2016', 'estpopNov2016', 'estpopJan2017',
                         'estpopFeb2017', 'estpopMar2017', 'estpopJun2017', 'estpopJul2017', 'RMSE', 'MAE', 'GWR_R2'])

nn = 0
predpoulations = pd.DataFrame(index=aux1)

# Landuse model + annual ntlhr *****
ntl_scale_NTL2 = pd.read_csv(results + 'NTL_Level_monthly_incorrected_median_' + date + '.csv')
pop2013 = pd.read_csv(results + 'observations_median_' + 'ntl_annual_incorrected_' + date + '.csv')
gwr_model = pd.read_csv(results + 'GWR_median_ntlhr_annual_incorrected_' + date + '.csv')
ntl_scale_NTL2.rename({'ntl_clip_id_copy':'ntl_clip_id'}, axis=1, inplace=True)
ntl_scale_NTL2.set_index('ntl_clip_id', inplace=True)
gwr_model.set_index('ntl_clip_id', inplace=True)

predict_all_years = ntl_scale_NTL2.merge(gwr_model, left_on=ntl_scale_NTL2.index, right_on=gwr_model.index, how='left')
predict_all_years.rename({'key_0':'ntl_clip_id'}, axis=1, inplace=True)

for my in month_year:
    year = my[-4:]
    month = my[:3]
    predict_all_years['estpop' + month + year] = predict_all_years['X.Intercept.'] + \
                                         (predict_all_years['NTL2013_hr'] * predict_all_years['NTL_hr' + month + year])  # + \
    # (predict_all_years['NTL2013_nr'] * predict_all_years['NTL_nr' + year])
    # (predict_all_years['CNTL2013'] * predict_all_years['CNTL' + year])

mask = predict_all_years['pred'] >=0
df.iloc[1, 2] = predict_all_years.loc[mask, ['pred']].sum()[0]
mask = predict_all_years['estpopJun2014'] >=0
df.iloc[2, 2] = predict_all_years.loc[mask, ['estpopJun2014']].sum()[0]
mask = predict_all_years['estpopJul2014'] >=0
df.iloc[3, 2] = predict_all_years.loc[mask, ['estpopJul2014']].sum()[0]
mask = predict_all_years['estpopOct2016'] >=0
df.iloc[4, 2] = predict_all_years.loc[mask, ['estpopOct2016']].sum()[0]
mask = predict_all_years['estpopNov2016'] >=0
df.iloc[5, 2] = predict_all_years.loc[mask, ['estpopNov2016']].sum()[0]
mask = predict_all_years['estpopJan2017'] >=0
df.iloc[6, 2] = predict_all_years.loc[mask, ['estpopJan2017']].sum()[0]
mask = predict_all_years['estpopFeb2017'] >=0
df.iloc[7, 2] = predict_all_years.loc[mask, ['estpopFeb2017']].sum()[0]
mask = predict_all_years['estpopMar2017'] >=0
df.iloc[8, 2] = predict_all_years.loc[mask, ['estpopMar2017']].sum()[0]
mask = predict_all_years['estpopJun2017'] >=0
df.iloc[9, 2] = predict_all_years.loc[mask, ['estpopJun2017']].sum()[0]
mask = predict_all_years['estpopJul2017'] >=0
df.iloc[10, 2] = predict_all_years.loc[mask, ['estpopJul2017']].sum()[0]
df.iloc[-3, 2] = mean_squared_error(pop2013['Pop2013'], predict_all_years.pred, squared=True)
df.iloc[-2:, 2] = mean_absolute_error(pop2013['Pop2013'], predict_all_years.pred)
predict_all_years.to_csv(results + 'predict_all_month_incorrected_median_' + date + '.csv')

predict_all_years.drop('geometry', inplace=True, axis=1)
NTL_clip_aux3 = predict_all_years.set_index('ntl_clip_id')


NTL_clip_aux3_noNeg = NTL_clip_aux3
mask = NTL_clip_aux3_noNeg['pred'] < 0
NTL_clip_aux3_noNeg.loc[mask, ['pred']] = 0

# Here
for my in month_year:
    year = my[-4:]
    month = my[:3]
    mask = NTL_clip_aux3_noNeg['estpop' + month + year] < 0
    NTL_clip_aux3_noNeg.loc[mask, ['estpop' + month + year]] = 0

# Read annual predictions as well:
predict_all_annual = pd.read_csv(results + 'predict_all_years_ntl_annual_incorrected_' + date + '.csv')
predict_all_annual.drop('geometry', inplace=True, axis=1)
predict_all_annual.set_index('ntl_clip_id', inplace=True)
NTL_clip_aux3 = NTL_clip_aux3.merge(predict_all_annual[['estpop2014', 'estpop2015', 'estpop2016','estpop2017', 'estpop2018']],
                                    left_on=NTL_clip_aux3.index, right_on=predict_all_annual.index)
NTL_clip_aux3.rename({'key_0':'ntl_clip_id'}, inplace=True, axis=1)

NTL_clip_aux3_noNeg = NTL_clip_aux3
mask = NTL_clip_aux3_noNeg['pred'] < 0
NTL_clip_aux3_noNeg.loc[mask, ['pred']] = 0

# Here
for year in ['2014','2015','2016','2017','2018']:
    if int(year) >= 2014:
        mask = NTL_clip_aux3_noNeg['estpop' + year] < 0
        NTL_clip_aux3_noNeg.loc[mask, ['estpop' + year]] = 0

NTL_clip_aux3_noNeg['estpopJun2014change'] = NTL_clip_aux3_noNeg['estpopJun2014'] - NTL_clip_aux3_noNeg['pred']
NTL_clip_aux3_noNeg['estpopJul2014change'] = NTL_clip_aux3_noNeg['estpopJul2014'] - NTL_clip_aux3_noNeg['estpopJun2014']
NTL_clip_aux3_noNeg['estpop2015change'] = NTL_clip_aux3_noNeg['estpop2015'] - NTL_clip_aux3_noNeg['estpopJul2014']
NTL_clip_aux3_noNeg['estpopOct2016change'] = NTL_clip_aux3_noNeg['estpopOct2016'] - NTL_clip_aux3_noNeg['estpop2015']
NTL_clip_aux3_noNeg['estpopNov2016change'] = NTL_clip_aux3_noNeg['estpopNov2016'] - NTL_clip_aux3_noNeg['estpopOct2016']
NTL_clip_aux3_noNeg['estpopNov2016change2'] = NTL_clip_aux3_noNeg['estpopNov2016'] - NTL_clip_aux3_noNeg['estpop2015']
NTL_clip_aux3_noNeg['estpopJan2017change'] = NTL_clip_aux3_noNeg['estpopJan2017'] - NTL_clip_aux3_noNeg['estpopNov2016']
NTL_clip_aux3_noNeg['estpopFeb2017change'] = NTL_clip_aux3_noNeg['estpopFeb2017'] - NTL_clip_aux3_noNeg['estpopJan2017']
NTL_clip_aux3_noNeg['estpopMar2017change'] = NTL_clip_aux3_noNeg['estpopMar2017'] - NTL_clip_aux3_noNeg['estpopFeb2017']
NTL_clip_aux3_noNeg['estpopMar2017change2'] = NTL_clip_aux3_noNeg['estpopMar2017'] - NTL_clip_aux3_noNeg['estpopNov2016']
NTL_clip_aux3_noNeg['estpopJun2017change'] = NTL_clip_aux3_noNeg['estpopJun2017'] - NTL_clip_aux3_noNeg['estpopMar2017']
NTL_clip_aux3_noNeg['estpopJul2017change'] = NTL_clip_aux3_noNeg['estpopJul2017'] - NTL_clip_aux3_noNeg['estpopJun2017']
NTL_clip_aux3_noNeg['estpop2018change'] = NTL_clip_aux3_noNeg['estpop2018'] - NTL_clip_aux3_noNeg['estpopJul2017']

# Only positive change
mask = NTL_clip_aux3_noNeg['estpopJun2014change'] > 0
poschange14 = NTL_clip_aux3_noNeg.loc[mask, ['estpopJun2014change']].sum()
mask = NTL_clip_aux3_noNeg['estpopJul2014change'] > 0
poschange15 = NTL_clip_aux3_noNeg.loc[mask, ['estpopJul2014change']].sum()
mask = NTL_clip_aux3_noNeg['estpop2015change'] > 0
poschange16 = NTL_clip_aux3_noNeg.loc[mask, ['estpop2015change']].sum()
mask = NTL_clip_aux3_noNeg['estpopOct2016change'] > 0
poschange17 = NTL_clip_aux3_noNeg.loc[mask, ['estpopOct2016change']].sum()
mask = NTL_clip_aux3_noNeg['estpopNov2016change'] > 0
poschange18 = NTL_clip_aux3_noNeg.loc[mask, ['estpopNov2016change']].sum()
mask = NTL_clip_aux3_noNeg['estpopNov2016change2'] > 0
poschange18 = NTL_clip_aux3_noNeg.loc[mask, ['estpopNov2016change2']].sum()
mask = NTL_clip_aux3_noNeg['estpopJan2017change'] > 0
poschange18 = NTL_clip_aux3_noNeg.loc[mask, ['estpopJan2017change']].sum()
mask = NTL_clip_aux3_noNeg['estpopFeb2017change'] > 0
poschange18 = NTL_clip_aux3_noNeg.loc[mask, ['estpopFeb2017change']].sum()
mask = NTL_clip_aux3_noNeg['estpopMar2017change'] > 0
poschange18 = NTL_clip_aux3_noNeg.loc[mask, ['estpopMar2017change']].sum()
mask = NTL_clip_aux3_noNeg['estpopMar2017change2'] > 0
poschange18 = NTL_clip_aux3_noNeg.loc[mask, ['estpopMar2017change2']].sum()
mask = NTL_clip_aux3_noNeg['estpopJun2017change'] > 0
poschange18 = NTL_clip_aux3_noNeg.loc[mask, ['estpopJun2017change']].sum()
mask = NTL_clip_aux3_noNeg['estpopJul2017change'] > 0
poschange18 = NTL_clip_aux3_noNeg.loc[mask, ['estpopJul2017change']].sum()
mask = NTL_clip_aux3_noNeg['estpop2018change'] > 0
poschange18 = NTL_clip_aux3_noNeg.loc[mask, ['estpop2018change']].sum()

vmin = np.min(np.array(NTL_clip_aux3_noNeg.loc[:, [ 'estpopJun2014',
 'estpopJul2014',
 'estpopOct2016',
 'estpopNov2016',
 'estpopJan2017',
 'estpopFeb2017',
 'estpopMar2017',
 'estpopJun2017',
 'estpopJul2017','estpop2014', 'estpop2015', 'estpop2016', 'estpop2017', 'estpop2018']]))
vmax = np.max(np.array(NTL_clip_aux3_noNeg.loc[:, [ 'estpopJun2014',
 'estpopJul2014',
 'estpopOct2016',
 'estpopNov2016',
 'estpopJan2017',
 'estpopFeb2017',
 'estpopMar2017',
 'estpopJun2017',
 'estpopJul2017','estpop2014', 'estpop2015', 'estpop2016', 'estpop2017', 'estpop2018']]))
vmin=0
vmax=8000

NTL_clip_aux4 = NTL_clip
pop2013.set_index('ntl_clip_id_y', inplace=True)
pop2013.drop('geometry', inplace=True, axis=1)
NTL_clip_aux5 = NTL_clip_aux4.merge(pop2013, left_on=NTL_clip_aux4.index, right_on=pop2013.index)
NTL_clip_aux5.drop('key_0', inplace=True, axis=1)
fig, axs = plt.subplots(1, 1, figsize=(8, 5))
p = NTL_clip_aux5.plot(column='Pop2013', cmap='Spectral_r', linewidth=0.1, ax=axs, edgecolor='white', legend=True)
axs.get_xaxis().set_visible(False)
axs.get_yaxis().set_visible(False)
axs.title.set_text('Census Population (2013)')
fig.savefig('C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Results/model/Disaggregated Census Population_test.png', dpi=500, bbox_inches='tight')


NTL_clip_aux6 = NTL_clip
NTL_clip_aux3_noNeg.set_index('ntl_clip_id', inplace=True)
NTL_clip_aux3_noNeg = gp.GeoDataFrame(NTL_clip_aux3_noNeg.merge(NTL_clip_aux6['geometry'], left_on=NTL_clip_aux3_noNeg.index, right_on=NTL_clip_aux6.index))

fig, axs = plt.subplots(5, 3, figsize=(30, 20))
NTL_clip_aux3_noNeg.plot(column='pred', cmap='Spectral_r', linewidth=0.1, ax=axs[0,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,0].get_xaxis().set_visible(False)
axs[0,0].get_yaxis().set_visible(False)
axs[0,0].title.set_text('Estimated population (2013)')
NTL_clip_aux3_noNeg.plot(column='estpopJun2014', cmap='Spectral_r', linewidth=0.1, ax=axs[0,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,1].get_xaxis().set_visible(False)
axs[0,1].get_yaxis().set_visible(False)
axs[0,1].title.set_text('Estimated population (Jun 2014)')
NTL_clip_aux3_noNeg.plot(column='estpopJul2014', cmap='Spectral_r', linewidth=0.1, ax=axs[0,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,2].get_xaxis().set_visible(False)
axs[0,2].get_yaxis().set_visible(False)
axs[0,2].title.set_text('Estimated population (Jul 2014)')
NTL_clip_aux3_noNeg.plot(column='estpop2015', cmap='Spectral_r', linewidth=0.1, ax=axs[1,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,0].get_xaxis().set_visible(False)
axs[1,0].get_yaxis().set_visible(False)
axs[1,0].title.set_text('Estimated population (2015)')
NTL_clip_aux3_noNeg.plot(column='estpopOct2016', cmap='Spectral_r', linewidth=0.1, ax=axs[1,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,1].get_xaxis().set_visible(False)
axs[1,1].get_yaxis().set_visible(False)
axs[1,1].title.set_text('Estimated population (Oct 2016)')
NTL_clip_aux3_noNeg.plot(column='estpopNov2016', cmap='Spectral_r', linewidth=0.1, ax=axs[1,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,2].get_xaxis().set_visible(False)
axs[1,2].get_yaxis().set_visible(False)
axs[1,2].title.set_text('Estimated population (Nov 2016)')
NTL_clip_aux3_noNeg.plot(column='estpopNov2016', cmap='Spectral_r', linewidth=0.1, ax=axs[2,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[2,0].get_xaxis().set_visible(False)
axs[2,0].get_yaxis().set_visible(False)
axs[2,0].title.set_text('Estimated population (Nov 2016)')
NTL_clip_aux3_noNeg.plot(column='estpopJan2017', cmap='Spectral_r', linewidth=0.1, ax=axs[2,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[2,1].get_xaxis().set_visible(False)
axs[2,1].get_yaxis().set_visible(False)
axs[2,1].title.set_text('Estimated population (Jan 2017)')
NTL_clip_aux3_noNeg.plot(column='estpopFeb2017', cmap='Spectral_r', linewidth=0.1, ax=axs[2,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[2,2].get_xaxis().set_visible(False)
axs[2,2].get_yaxis().set_visible(False)
axs[2,2].title.set_text('Estimated population (Feb 2016)')
NTL_clip_aux3_noNeg.plot(column='estpopMar2017', cmap='Spectral_r', linewidth=0.1, ax=axs[3,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[3,0].get_xaxis().set_visible(False)
axs[3,0].get_yaxis().set_visible(False)
axs[3,0].title.set_text('Estimated population (Mar 2017)')
NTL_clip_aux3_noNeg.plot(column='estpopJun2017', cmap='Spectral_r', linewidth=0.1, ax=axs[3,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[3,1].get_xaxis().set_visible(False)
axs[3,1].get_yaxis().set_visible(False)
axs[3,1].title.set_text('Estimated population (Jun 2017)')
NTL_clip_aux3_noNeg.plot(column='estpopJul2017', cmap='Spectral_r', linewidth=0.1, ax=axs[3,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[3,2].get_xaxis().set_visible(False)
axs[3,2].get_yaxis().set_visible(False)
axs[3,2].title.set_text('Estimated population (Jul 2017)')
NTL_clip_aux3_noNeg.plot(column='estpop2018', cmap='Spectral_r', linewidth=0.1, ax=axs[4,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[4,0].get_xaxis().set_visible(False)
axs[4,0].get_yaxis().set_visible(False)
axs[4,0].title.set_text('Estimated population (2018)')
plt.suptitle("Population (GWR-E)", size=16)
plt.savefig('C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Results/model/Population (Landuse-NTL)_test.png', dpi=500, bbox_inches='tight')

vmin = np.min(np.array(NTL_clip_aux3_noNeg[[ 'estpopJun2014change','estpopJul2014change','estpop2015change','estpopOct2016change', 'estpopNov2016change2',
                                             'estpopNov2016change','estpopJan2017change','estpopFeb2017change','estpopMar2017change', 'estpopMar2017change2',
                                             'estpopJun2017change','estpopJul2017change','estpop2018change']]))

vmax = np.max(np.array(NTL_clip_aux3_noNeg.loc[:, [ 'estpopJun2014change',
 'estpopJul2014change',
 'estpop2015change',
 'estpopOct2016change',
 'estpopNov2016change',
 'estpopNov2016change2',
 'estpopJan2017change',
 'estpopFeb2017change',
 'estpopMar2017change',
 'estpopMar2017change2',
 'estpopJun2017change',
 'estpopJul2017change',
 'estpop2018change',]]))

vmin=-5000
vmax=4000

fig, axs = plt.subplots(5, 3, figsize=(30, 20))
NTL_clip_aux3_noNeg.plot(column='pred', cmap='Spectral_r', linewidth=0.1, ax=axs[0,0], edgecolor='white', legend=True)
axs[0,0].get_xaxis().set_visible(False)
axs[0,0].get_yaxis().set_visible(False)
axs[0,0].title.set_text('Estimated population (2013)')
NTL_clip_aux3_noNeg.plot(column='estpopJun2014change', cmap='Spectral_r', linewidth=0.1, ax=axs[0,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,1].get_xaxis().set_visible(False)
axs[0,1].get_yaxis().set_visible(False)
axs[0,1].title.set_text('Estimated population change (Jun 2014)')
NTL_clip_aux3_noNeg.plot(column='estpopJul2014change', cmap='Spectral_r', linewidth=0.1, ax=axs[0,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,2].get_xaxis().set_visible(False)
axs[0,2].get_yaxis().set_visible(False)
axs[0,2].title.set_text('Estimated population change (Jul 2014)')
NTL_clip_aux3_noNeg.plot(column='estpop2015change', cmap='Spectral_r', linewidth=0.1, ax=axs[1,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,0].get_xaxis().set_visible(False)
axs[1,0].get_yaxis().set_visible(False)
axs[1,0].title.set_text('Estimated population change (2015)')
NTL_clip_aux3_noNeg.plot(column='estpopOct2016change', cmap='Spectral_r', linewidth=0.1, ax=axs[1,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,1].get_xaxis().set_visible(False)
axs[1,1].get_yaxis().set_visible(False)
axs[1,1].title.set_text('Estimated population change (Oct 2016)')
NTL_clip_aux3_noNeg.plot(column='estpopNov2016change', cmap='Spectral_r', linewidth=0.1, ax=axs[1,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,2].get_xaxis().set_visible(False)
axs[1,2].get_yaxis().set_visible(False)
axs[1,2].title.set_text('Estimated population change (Nov 2016)')
NTL_clip_aux3_noNeg.plot(column='estpopNov2016change', cmap='Spectral_r', linewidth=0.1, ax=axs[2,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[2,0].get_xaxis().set_visible(False)
axs[2,0].get_yaxis().set_visible(False)
axs[2,0].title.set_text('Estimated population change (Nov 2016)')
NTL_clip_aux3_noNeg.plot(column='estpopJan2017change', cmap='Spectral_r', linewidth=0.1, ax=axs[2,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[2,1].get_xaxis().set_visible(False)
axs[2,1].get_yaxis().set_visible(False)
axs[2,1].title.set_text('Estimated population change (Jan 2017)')
NTL_clip_aux3_noNeg.plot(column='estpopFeb2017change', cmap='Spectral_r', linewidth=0.1, ax=axs[2,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[2,2].get_xaxis().set_visible(False)
axs[2,2].get_yaxis().set_visible(False)
axs[2,2].title.set_text('Estimated population change (Feb 2016)')
NTL_clip_aux3_noNeg.plot(column='estpopMar2017change', cmap='Spectral_r', linewidth=0.1, ax=axs[3,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[3,0].get_xaxis().set_visible(False)
axs[3,0].get_yaxis().set_visible(False)
axs[3,0].title.set_text('Estimated population change (Mar 2017)')
NTL_clip_aux3_noNeg.plot(column='estpopJun2017change', cmap='Spectral_r', linewidth=0.1, ax=axs[3,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[3,1].get_xaxis().set_visible(False)
axs[3,1].get_yaxis().set_visible(False)
axs[3,1].title.set_text('Estimated population change (Jun 2017)')
NTL_clip_aux3_noNeg.plot(column='estpopJul2017change', cmap='Spectral_r', linewidth=0.1, ax=axs[3,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[3,2].get_xaxis().set_visible(False)
axs[3,2].get_yaxis().set_visible(False)
axs[3,2].title.set_text('Estimated population change (Jul 2017)')
NTL_clip_aux3_noNeg.plot(column='estpop2018change', cmap='Spectral_r', linewidth=0.1, ax=axs[4,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[4,0].get_xaxis().set_visible(False)
axs[4,0].get_yaxis().set_visible(False)
axs[4,0].title.set_text('Estimated population change (2018)')

plt.suptitle("Population Change (GWR-E)", size=16)
plt.savefig('C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Results/model/Population Change (Landuse-NTL)_test.png', dpi=500, bbox_inches='tight')

fig, axs = plt.subplots(1, 1, figsize=(8, 5))
NTL_clip_aux3_noNeg.plot(column='estpopNov2016change2', cmap='Spectral_r', linewidth=0.1, ax=axs, edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs.get_xaxis().set_visible(False)
axs.get_yaxis().set_visible(False)
axs.title.set_text('Estimated population change (Nov 2016)')
fig.savefig('C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Results/model/Disaggregated Census Population_Nov2016_2015.png', dpi=500, bbox_inches='tight')

fig, axs = plt.subplots(1, 1, figsize=(8, 5))
NTL_clip_aux3_noNeg.plot(column='estpopMar2017change2', cmap='Spectral_r', linewidth=0.1, ax=axs, edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs.get_xaxis().set_visible(False)
axs.get_yaxis().set_visible(False)
axs.title.set_text('Estimated population change (Mar 2017)')
fig.savefig('C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Results/model/Disaggregated Census Population_Mar2017_Nov2016.png', dpi=500, bbox_inches='tight')

fig, axs = plt.subplots(1, 1, figsize=(8, 5))
NTL_clip_aux3_noNeg.plot(column='estpopJun2017change', cmap='Spectral_r', linewidth=0.1, ax=axs, edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs.get_xaxis().set_visible(False)
axs.get_yaxis().set_visible(False)
axs.title.set_text('Estimated population change (Jun 2017)')
fig.savefig('C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Results/model/Disaggregated Census Population_Mar2017_Jun,_2017.png', dpi=500, bbox_inches='tight')





# Scatter plot of predictions
LR = pd.read_csv('C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Sources/Results/LR_median_ntlhr_annual_incorrected_03292020.csv')
try:
    LR.set_index('ntl.data1$ntl_clip_id_y', inplace=True)
except:
    print('Already done!')
try:
    ntl_scale_NTL2.set_index('ntl_clip_id_y', inplace=True)
except:
    print('Already done!')
try:
    Predictions_spatial.set_index('ntl_clip_id_y', inplace=True)
except:
    print('Already done!')

LrpopLanduseNTL = pd.concat((ntl_scale_NTL2.loc[:, ['Pop2013']], LR['pred']), axis=1)
sLrpopLanduseNTL = pd.concat((ntl_scale_NTL2.loc[:, ['Pop2013']], Predictions_spatial['popntlpred2013']), axis=1)
gwrpopLanduseNTL = pd.concat((ntl_scale_NTL2.loc[:, ['Pop2013']], gwr_model['pred']), axis=1)
fig, axs = plt.subplots(2, 2, figsize=(14, 14))
p = sns.scatterplot(data= LrpopLanduseNTL, x='pred', y='Pop2013', ax=axs[0,0])
axs[0,0].title.set_text('MLR-E')
axs[0,0].set(xlabel="Predicted Population \n (a) ")
axs[0,0].set(ylabel="Census Population")
p = sns.scatterplot(data= sLrpopLanduseNTL, x='popntlpred2013', y='Pop2013', ax=axs[0,1])
axs[0,1].title.set_text('SMLR-E')
axs[0,1].set(xlabel="Predicted Population \n (b) ")
axs[0,1].set(ylabel="Census Population")
p = sns.scatterplot(data= gwrpopLanduse, x='pred', y='Pop2013', ax=axs[1,0])
axs[1,0].title.set_text('GWR-A')
axs[1,0].set(xlabel="Predicted Population \n (c) ")
axs[1,0].set(ylabel="Census Population")
p = sns.scatterplot(data= gwrpopLanduseNTL, x='pred', y='Pop2013', ax=axs[1,1])
axs[1,1].title.set_text('GWR-E')
axs[1,1].set(xlabel="Predicted Population \n (d) ")
axs[1,1].set(ylabel="Census Population")
fig.savefig('C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Results/model/GWR-LR-Predictions.png', dpi=500, bbox_inches='tight')

import sklearn.metrics as metrics
mae_LrpopLanduseNTL = metrics.mean_absolute_error(LrpopLanduseNTL.Pop2013, LrpopLanduseNTL.pred)
rmse_LrpopLanduseNTL = np.sqrt(metrics.mean_squared_error(LrpopLanduseNTL.Pop2013, LrpopLanduseNTL.pred))

mae_sLrpopLanduseNTL = metrics.mean_absolute_error(sLrpopLanduseNTL.Pop2013, sLrpopLanduseNTL.popntlpred2013)
rmse_sLrpopLanduseNTL = np.sqrt(metrics.mean_squared_error(sLrpopLanduseNTL.Pop2013, sLrpopLanduseNTL.popntlpred2013))

mae_gwrpopLanduse = metrics.mean_absolute_error(gwrpopLanduse.Pop2013, gwrpopLanduse.pred)
rmse_gwrpopLanduse = np.sqrt(metrics.mean_squared_error(gwrpopLanduse.Pop2013, gwrpopLanduse.pred))

mae_gwrpopLanduseNTL= metrics.mean_absolute_error(gwrpopLanduseNTL.Pop2013, gwrpopLanduseNTL.pred)
rmse_gwrpopLanduseNTL = np.sqrt(metrics.mean_squared_error(gwrpopLanduseNTL.Pop2013, gwrpopLanduseNTL.pred))

mask = gwrpopLanduse['pred'] > 0
positive = gwrpopLanduse.loc[mask,['pred']].sum()
all = gwrpopLanduse['pred'].sum()
((positive - all)/positive)*100

mask = gwrpopLanduseNTL['pred'] > 0
positive = gwrpopLanduseNTL.loc[mask,['pred']].sum()
all = gwrpopLanduseNTL['pred'].sum()
((positive - all)/positive)*100

# 2017 change
mask = NTL_clip_aux3_noNeg['estpop2017change'] > 0
NTL_clip_aux3_noNeg.loc[mask,['estpop2017change']].sum()
mask = NTL_clip_aux3_noNeg['estpop2017change'] < 0
NTL_clip_aux3_noNeg.loc[mask,['estpop2017change']].sum()

# 2018 change
mask = NTL_clip_aux3_noNeg['estpop2018change'] > 0
NTL_clip_aux3_noNeg.loc[mask,['estpop2018change']].sum()
mask = NTL_clip_aux3_noNeg['estpop2018change'] < 0
NTL_clip_aux3_noNeg.loc[mask,['estpop2018change']].sum()
