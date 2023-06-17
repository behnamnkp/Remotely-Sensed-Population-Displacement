import numpy as np
import matplotlib.pyplot as plt
from matplotlib import *
import pandas as pd
import rtree
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
date = '03292020'

# Choose the year of the analysis:
years = ['2013', '2014', '2015', '2016', '2017', '2018'] # 2014-15-16-17-18 is available now

# arcpy.env.workspace = image_path
# # resample landuse
# for year in years:
#     if int(year) < 2014:
#         print('landue does not exist for ' + year)
#     else:
#         inputraster = 'label' + year +'.tif'
#         outpuraster = 'labelrsm' + year +'.tif'
#         arcpy.Resample_management(inputraster, outpuraster, "50 50", "Majority")
#
# convert to point
# for year in years:
#     if int(year) < 2014:
#         arcpy.RasterToPoint_conversion(viirs_path + 'ntl_corrected_annualByMonth' + year + '.tif', temp + 'ntl_corrected_med_annualByMonth' + year, "VALUE")
#     else:
#         arcpy.RasterToPoint_conversion(viirs_path + 'ntl_corrected_annualByMonth' + year + '.tif', temp + 'ntl_corrected_med_annualByMonth' + year, "VALUE")
#         # arcpy.RasterToPoint_conversion(image_path + 'labelrsm' + year + '.tif', temp + 'label' + year, "VALUE")


# models = ['nontl', 'ntlmed', 'ntl_corrected_med_annualByMonth', 'ntl_corrected_med_monthly', 'ntlMonthlyIncorrected_']
models = ['ntlmed']

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
            image = gp.read_file(temp + 'ntlmed' + year + '.shp')
            NTL = gp.sjoin(NTL, image, how="inner", op='intersects')
            NTL.rename({'grid_code': 'NTL' + year}, inplace=True, axis=1)
            NTL.drop('pointid', inplace=True, axis=1)
            NTL.drop('index_right', inplace=True, axis=1)
    else:
        for year in years:
            image = gp.read_file(temp + mdl + year + '.shp')
            NTL = gp.sjoin(NTL, image, how="inner", op='intersects')
            NTL.rename({'grid_code': 'NTL' + year}, inplace=True, axis=1)
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
        intersect2['CNTL' + year] = (intersect2['ntl_clip_area'] /
                                                intersect2['ntl_area'])*intersect2['NTL' + year]

    # # target NTL Area over NTL
    # intersect2['AONTL'] = intersect2['intersect_area'] / intersect2['ntl_clip_area']
    #
    # # Target NTL
    # for year in years:
    #     intersect2['TNTL' + year] = intersect2['AONTL'] * intersect2['CNTL' + year]

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

        # in the level of night light
        try:
            intersect2.set_index('ntl_clip_id', inplace=True)
        except:
            print('It is already the index')

        intersect2['countNTL'] = intersect2['index'].groupby(intersect2.index).transform('count')
        if int(year) < 2014:
            ntl_scale_NTL = intersect2.groupby(['ntl_clip_id', 'landuse2014']).sum().loc[:,['intersect_area']]
            ntl_scale_NTL2 = ntl_scale_NTL.unstack('landuse2014')
            ntl_scale_NTL2.columns = ['area_bg' + year, 'area_lr' + year, 'area_hr' + year, 'area_nr' + year]
            ntl_scale_NTL2.fillna(0, inplace=True)

        else:
            ntl_scale_NTL = intersect2.groupby(['ntl_clip_id', 'landuse' + year]).sum().loc[:, ['intersect_area']]
            ntl_scale_NTL2 = ntl_scale_NTL.unstack('landuse' + year)
            ntl_scale_NTL2.columns = ['area_bg' + year, 'area_lr' + year, 'area_hr' + year, 'area_nr' + year]
            ntl_scale_NTL2.fillna(0, inplace=True)

        ntl_scale_NTL2['CNTL' + year] = intersect2.groupby(intersect2.index).max()['CNTL' + year]

        # ntl_scale_NTL2.reset_index(inplace=True)
        # NTL_clip.reset_index(inplace=True)
        NTL_clip.set_index('ntl_clip_id', inplace=True)
        ntl_scale_NTL2 = NTL_clip.merge(ntl_scale_NTL2, left_on = NTL_clip.index, right_on = ntl_scale_NTL2.index, how='left')
        # ntl_scale_NTL2.drop(['key_0', 'index', 'Shape_Leng', 'Shape_Area', 'ntl_area', 'NTL2014', 'NTL2015', 'ntl_id',
        #                 'NTL2016' , 'NTL2017' ,'NTL2018' , 'NTL2013', 'ntl_clip_id_x', 'ntl_clip_area'],
        #                 inplace=True, axis=1)
        ntl_scale_NTL2.drop(['key_0', 'Shape_Leng', 'Shape_Area', 'ntl_area', 'NTL2014', 'NTL2015', 'ntl_id',
                        'NTL2016' , 'NTL2017' ,'NTL2018' , 'NTL2013', 'ntl_clip_area'],
                        inplace=True, axis=1)
        try:
            ntl_scale_NTL2.drop(['level_0'], inplace=True, axis=1)
            NTL_clip.drop(['level_0'], inplace=True, axis=1)
        except:
            print('level_0 is not in the columns')

        ntl_scale_NTL2['X'] = ntl_scale_NTL2.geometry.centroid.x
        ntl_scale_NTL2['Y'] = ntl_scale_NTL2.geometry.centroid.y

        if int(year) == 2013:
            print('Multiple Linear Regression for disaggregating nightlight 2013:')
            model_NTL = ols("CNTL2013 ~  area_hr2013 + area_nr2013 + area_bg2013 + area_lr2013",ntl_scale_NTL2).fit()
            print(model_NTL.summary())
            print("\nRetrieving manually the parameter estimates:")
            print(model_NTL._results.params)

            model_NTL.save(results + 'ols_ntl.pickle', remove_data=False)

            y, X = dmatrices('CNTL2013 ~ area_hr2013 + area_nr2013 + area_bg2013 + area_lr2013', data=ntl_scale_NTL2, return_type='dataframe')
            vif = pd.DataFrame()
            vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            vif['variable'] = X.columns
            print(vif)

            ntlresid = pd.concat((ntl_scale_NTL2,model_NTL.resid), axis=1)
            ntlresid.rename({0:'ntlresid' + year}, axis=1, inplace=True)
            W = Queen.from_dataframe(ntlresid)
            W.transform = 'r'
            moran_ntl = Moran(ntlresid['ntlresid'+ year], W)
            print('moran_ntl' + year + ': ' + str(moran_ntl.I))
            moran_loc = Moran_Local(ntlresid['ntlresid'+ year], W)
            p = lisa_cluster(moran_loc, ntlresid, p=0.05, figsize=(9, 9))
            plt.title('Cluster Map of Nightlight Residuals (2013)', size=20)
            plt.show()
            plt.savefig(
                'C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Results/Cluster Map of Nightlight Residuals.png',
                dpi=500, bbox_inches='tight')

            #1 HH, 2 LH, 3 LL, 4 HL
            cluster = _viz_utils.moran_hot_cold_spots(moran_loc, p=0.05)
            aux = pd.DataFrame(cluster)
            aux.rename({0:'clusters' + year}, inplace=True, axis=1)
            aux.loc[aux['clusters' + year] == 0, ['clusters' + year]] = 'NS'
            aux.loc[aux['clusters' + year]==1,['clusters' + year]] = 'HH'
            aux.loc[aux['clusters' + year]==2,['clusters' + year]] = 'LH'
            aux.loc[aux['clusters' + year]==3,['clusters' + year]] = 'LL'
            aux.loc[aux['clusters' + year]==4,['clusters' + year]] = 'HL'
            cluster = pd.concat((ntl_scale_NTL2, aux), axis=1)
            cluster = pd.get_dummies(cluster)

            print('Spatial Multiple Linear Regression for disaggregating nightlight 2013:')
            model_NTL_spatial = ols("CNTL2013 ~  area_hr2013 + area_nr2013 + area_bg2013 + clusters2013_HH + clusters2013_HL + clusters2013_NS + area_lr2013",cluster).fit()
            print(model_NTL_spatial.summary())
            print("\nRetrieving manually the parameter estimates:")
            print(model_NTL_spatial._results.params)

            y, X = dmatrices("CNTL2013 ~  area_hr2013 + area_nr2013 + area_bg2013 + clusters2013_HH + clusters2013_HL + clusters2013_NS",
                             data=cluster, return_type='dataframe')
            vif = pd.DataFrame()
            vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            vif['variable'] = X.columns
            print(vif)

            model_NTL_spatial.save(results + 'ols_ntl_spatial.pickle', remove_data=False)

            NTL_clip.reset_index(inplace=True)
            cluster = cluster.merge(NTL_clip[['ntl_clip_id']], left_on=cluster.index, right_on=NTL_clip.index.array, how='left')
            cluster.drop('key_0', axis=1, inplace=True)
            cluster.set_index('ntl_clip_id', inplace=True)
            intersect2 = intersect2.merge(cluster.loc[:, ['clusters2013_HH','clusters2013_HL', 'clusters2013_LL','clusters2013_NS']],
                                          left_on= intersect2.index, right_on= cluster.index,how='left')
            intersect2.rename({'key_0':'ntl_clip_id'}, inplace=True, axis=1)
            try:
                intersect2.drop('level_0', inplace=True, axis=1)
            except:
                print('done!')
            cluster.reset_index(inplace=True)

            model_NTL_pred = cluster['area_bg'+ year]*model_NTL.params[3] + cluster['area_hr'+ year]*model_NTL.params[1] + \
                  cluster['area_nr'+ year]*model_NTL.params[2] + cluster['area_lr'+ year]*model_NTL.params[4]
            model_NTL_pred = pd.DataFrame(model_NTL_pred, columns=['ntlpred' + year])
            Predictions = pd.concat((ntl_scale_NTL2, model_NTL_pred), axis=1)

            model_NTL_spatial_pred = cluster['area_bg'+ year]*model_NTL_spatial.params[3] + cluster['area_hr'+ year]*model_NTL_spatial.params[1] + \
                  cluster['area_nr'+ year]*model_NTL_spatial.params[2] + cluster['clusters2013_HH']*model_NTL_spatial.params[4] + \
                  cluster['clusters2013_HL']*model_NTL_spatial.params[5]  +\
                                     cluster['clusters2013_NS']*model_NTL_spatial.params[6] + cluster['area_lr'+ year]*model_NTL_spatial.params[7]
            model_NTL_spatial_pred = pd.DataFrame(model_NTL_spatial_pred, columns=['ntlpred' + year])
            Predictions_spatial = pd.concat((ntl_scale_NTL2, model_NTL_spatial_pred), axis=1)

            sns.set(rc={'figure.figsize': (5, 10)}, style="whitegrid")
            f, axes = plt.subplots(2, 1)
            f.subplots_adjust(hspace=.5)
            sns.scatterplot(x=Predictions['ntlpred'+year], y=Predictions['CNTL'+year], ax=axes[0], color='black')
            axes[0].set(xlabel='Nightlight emmission '+ year + ' (OLS model)')
            axes[0].set(ylabel='Nightlight emmission '+year)
            sns.scatterplot(x=Predictions_spatial['ntlpred'+year], y=Predictions['CNTL'+year], ax=axes[1], color='black')
            axes[1].set(xlabel='Nightlight emmission '+ year + ' (Spatial model)')
            axes[1].set(ylabel='Nightlight emmission '+year)

        elif int(year) == 2014:
            print('Multiple Linear Regression for disaggregating nightlight 2014:')
            model_NTL = ols("CNTL2014 ~  area_hr2014 + area_nr2014 + area_bg2014 + area_lr2014",ntl_scale_NTL2).fit()
            print(model_NTL.summary())
            print("\nRetrieving manually the parameter estimates:")
            print(model_NTL._results.params)

            model_NTL.save(results + 'ols_ntl.pickle', remove_data=False)

            y, X = dmatrices('CNTL2014 ~ area_hr2014 + area_lr2014 + area_nr2014 + area_bg2014 + area_lr2014', data=ntl_scale_NTL2, return_type='dataframe')
            vif = pd.DataFrame()
            vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            vif['variable'] = X.columns
            print(vif)

            ntlresid = pd.concat((ntl_scale_NTL2,model_NTL.resid), axis=1)
            ntlresid.rename({0:'ntlresid' + year}, axis=1, inplace=True)
            W = Queen.from_dataframe(ntlresid)
            W.transform = 'r'
            moran_ntl = Moran(ntlresid['ntlresid'+ year], W)
            print('moran_ntl' + year + ': ' + str(moran_ntl.I))
            moran_loc = Moran_Local(ntlresid['ntlresid'+ year], W)
            p = lisa_cluster(moran_loc, ntlresid, p=0.05, figsize=(9, 9))
            plt.title('Local Autocorrelation for nightlight estimation residuals ' + year)
            plt.show()

            #1 HH, 2 LH, 3 LL, 4 HL
            cluster = _viz_utils.moran_hot_cold_spots(moran_loc, p=0.05)
            aux = pd.DataFrame(cluster)
            aux.rename({0:'clusters' + year}, inplace=True, axis=1)
            aux.loc[aux['clusters' + year] == 0, ['clusters' + year]] = 'NS'
            aux.loc[aux['clusters' + year]==1,['clusters' + year]] = 'HH'
            aux.loc[aux['clusters' + year]==2,['clusters' + year]] = 'LH'
            aux.loc[aux['clusters' + year]==3,['clusters' + year]] = 'LL'
            aux.loc[aux['clusters' + year]==4,['clusters' + year]] = 'HL'
            cluster = pd.concat((ntl_scale_NTL2, aux), axis=1)
            cluster = pd.get_dummies(cluster)

            print('Spatial Multiple Linear Regression for disaggregating nightlight 2014:')
            model_NTL_spatial = ols("CNTL2014 ~  area_hr2014 + area_nr2014 + area_bg2014 + clusters2014_HH + clusters2014_HL + clusters2014_LH + clusters2014_NS + area_lr2014"
                                    ,cluster).fit()
            print(model_NTL_spatial.summary())
            print("\nRetrieving manually the parameter estimates:")
            print(model_NTL_spatial._results.params)

            y, X = dmatrices("CNTL2014 ~  area_hr2014 + area_nr2014 + area_bg2014 + area_lr2014 + clusters2014_HH + clusters2014_HL + clusters2014_LH + clusters2014_NS",
                             data=cluster, return_type='dataframe')
            vif = pd.DataFrame()
            vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            vif['variable'] = X.columns
            print(vif)

            model_NTL_spatial.save(results + 'ols_ntl_spatial.pickle', remove_data=False)

            NTL_clip.reset_index(inplace=True)
            cluster = cluster.merge(NTL_clip['ntl_clip_id'], left_on=cluster.index, right_on=NTL_clip.index.array, how='left')
            cluster.drop('key_0', axis=1, inplace=True)
            cluster.set_index('ntl_clip_id', inplace=True)
            intersect2 = intersect2.merge(cluster.loc[:, ['clusters2014_HH','clusters2014_HL', 'clusters2014_LH', 'clusters2014_LL','clusters2014_NS']],
                                          left_on= intersect2.index, right_on= cluster.index,how='left')
            intersect2.rename({'key_0': 'ntl_clip_id'}, inplace=True, axis=1)
            try:
                intersect2.drop('level_0', inplace=True, axis=1)
            except:
                print('done!')
            cluster.reset_index(inplace=True)

            model_NTL_pred = cluster['area_bg'+ year]*model_NTL.params[3] + cluster['area_hr'+ year]*model_NTL.params[1] + \
                  cluster['area_nr'+ year]*model_NTL.params[2] + cluster['area_lr'+ year]*model_NTL.params[4]
            model_NTL_pred = pd.DataFrame(model_NTL_pred, columns=['ntlpred' + year])
            Predictions = pd.concat((ntl_scale_NTL2, model_NTL_pred), axis=1)

            model_NTL_spatial_pred = cluster['area_bg'+ year]*model_NTL_spatial.params[3] + cluster['area_hr'+ year]*model_NTL_spatial.params[1] + \
                  cluster['area_nr'+ year]*model_NTL_spatial.params[2] + cluster['clusters2014_HH']*model_NTL_spatial.params[4] + \
                  cluster['clusters2014_HL']*model_NTL_spatial.params[5] + cluster['clusters2014_LH']*model_NTL_spatial.params[6] +\
                                     cluster['clusters2014_NS']*model_NTL_spatial.params[7] + cluster['area_lr' + year]*model_NTL_spatial.params[8]

            model_NTL_spatial_pred = pd.DataFrame(model_NTL_spatial_pred, columns=['ntlpred' + year])
            Predictions_spatial = pd.concat((ntl_scale_NTL2, model_NTL_spatial_pred), axis=1)

            sns.set(rc={'figure.figsize': (5, 10)}, style="whitegrid")
            f, axes = plt.subplots(2, 1)
            f.subplots_adjust(hspace=.5)
            sns.scatterplot(x=Predictions['ntlpred'+year], y=Predictions['CNTL'+year], ax=axes[0], color='red')
            axes[0].set(xlabel='ntlpred'+year)
            axes[0].set(ylabel='CNTL'+year)
            sns.scatterplot(x=Predictions_spatial['ntlpred'+year], y=Predictions['CNTL'+year], ax=axes[1], color='red')
            axes[1].set(xlabel='ntlpred Spatial'+ year)
            axes[1].set(ylabel='CNTL'+year)

        elif int(year) == 2015:
            print('Multiple Linear Regression for disaggregating nightlight 2015:')
            model_NTL = ols("CNTL2015 ~  area_hr2015 + area_nr2015 + area_bg2015",ntl_scale_NTL2).fit()
            print(model_NTL.summary())
            print("\nRetrieving manually the parameter estimates:")
            print(model_NTL._results.params)

            model_NTL.save(results + 'ols_ntl.pickle', remove_data=False)

            y, X = dmatrices('CNTL2015 ~ area_hr2015 + area_lr2015 + area_nr2015 + area_bg2015', data=ntl_scale_NTL2, return_type='dataframe')
            vif = pd.DataFrame()
            vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            vif['variable'] = X.columns
            print(vif)

            ntlresid = pd.concat((ntl_scale_NTL2,model_NTL.resid), axis=1)
            ntlresid.rename({0:'ntlresid' + year}, axis=1, inplace=True)
            W = Queen.from_dataframe(ntlresid)
            W.transform = 'r'
            moran_ntl = Moran(ntlresid['ntlresid'+ year], W)
            print('moran_ntl' + year + ': ' + str(moran_ntl.I))
            moran_loc = Moran_Local(ntlresid['ntlresid'+ year], W)
            p = lisa_cluster(moran_loc, ntlresid, p=0.05, figsize=(9, 9))
            plt.title('Local Autocorrelation for nightlight estimation residuals ' + year)
            plt.show()

            #1 HH, 2 LH, 3 LL, 4 HL
            cluster = _viz_utils.moran_hot_cold_spots(moran_loc, p=0.05)
            aux = pd.DataFrame(cluster)
            aux.rename({0:'clusters' + year}, inplace=True, axis=1)
            aux.loc[aux['clusters' + year] == 0, ['clusters' + year]] = 'NS'
            aux.loc[aux['clusters' + year]==1,['clusters' + year]] = 'HH'
            aux.loc[aux['clusters' + year]==2,['clusters' + year]] = 'LH'
            aux.loc[aux['clusters' + year]==3,['clusters' + year]] = 'LL'
            aux.loc[aux['clusters' + year]==4,['clusters' + year]] = 'HL'
            cluster = pd.concat((ntl_scale_NTL2, aux), axis=1)
            cluster = pd.get_dummies(cluster)

            print('Spatial Multiple Linear Regression for disaggregating nightlight 2015:')
            model_NTL_spatial = ols("CNTL2015 ~  area_hr2015 + area_nr2015 + area_bg2015 + clusters2015_HH + clusters2015_HL + clusters2015_LH + clusters2015_NS + area_lr2015",cluster).fit()
            print(model_NTL_spatial.summary())
            print("\nRetrieving manually the parameter estimates:")
            print(model_NTL_spatial._results.params)

            y, X = dmatrices("CNTL2015 ~  area_hr2015 + area_lr2015 + area_nr2015 + area_bg2015 + clusters2015_HH + clusters2015_HL + clusters2015_LH + clusters2015_NS",
                             data=cluster, return_type='dataframe')
            vif = pd.DataFrame()
            vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            vif['variable'] = X.columns
            print(vif)

            model_NTL_spatial.save(results + 'ols_ntl_spatial.pickle', remove_data=False)

            NTL_clip.reset_index(inplace=True)
            cluster = cluster.merge(NTL_clip['ntl_clip_id'], left_on=cluster.index, right_on=NTL_clip.index.array, how='left')
            cluster.drop('key_0', axis=1, inplace=True)
            cluster.set_index('ntl_clip_id', inplace=True)
            intersect2 = intersect2.merge(cluster.loc[:, ['clusters2015_HH','clusters2015_HL', 'clusters2015_LH', 'clusters2015_LL','clusters2015_NS']],
                                          left_on= intersect2.index, right_on= cluster.index,how='left')
            intersect2.rename({'key_0': 'ntl_clip_id'}, inplace=True, axis=1)
            try:
                intersect2.drop('level_0', inplace=True, axis=1)
            except:
                print('done!')
            cluster.reset_index(inplace=True)

            model_NTL_pred = cluster['area_bg'+ year]*model_NTL.params[3] + cluster['area_hr'+ year]*model_NTL.params[1] + \
                  cluster['area_nr'+ year]*model_NTL.params[2]
            model_NTL_pred = pd.DataFrame(model_NTL_pred, columns=['ntlpred' + year])
            Predictions = pd.concat((ntl_scale_NTL2, model_NTL_pred), axis=1)

            model_NTL_spatial_pred = cluster['area_bg'+ year]*model_NTL_spatial.params[3] + cluster['area_hr'+ year]*model_NTL_spatial.params[1] + \
                  cluster['area_nr'+ year]*model_NTL_spatial.params[2] + cluster['clusters2015_HH']*model_NTL_spatial.params[4] + \
                  cluster['clusters2015_HL']*model_NTL_spatial.params[5] + cluster['clusters2015_LH']*model_NTL_spatial.params[6] +\
                                     cluster['clusters2015_NS']*model_NTL_spatial.params[7]
            model_NTL_spatial_pred = pd.DataFrame(model_NTL_spatial_pred, columns=['ntlpred' + year])
            Predictions_spatial = pd.concat((ntl_scale_NTL2, model_NTL_spatial_pred), axis=1)

            sns.set(rc={'figure.figsize': (5, 10)}, style="whitegrid")
            f, axes = plt.subplots(2, 1)
            f.subplots_adjust(hspace=.5)
            sns.scatterplot(x=Predictions['ntlpred'+year], y=Predictions['CNTL'+year], ax=axes[0], color='red')
            axes[0].set(xlabel='ntlpred'+year)
            axes[0].set(ylabel='CNTL'+year)
            sns.scatterplot(x=Predictions_spatial['ntlpred'+year], y=Predictions['CNTL'+year], ax=axes[1], color='red')
            axes[1].set(xlabel='ntlpred Spatial'+ year)
            axes[1].set(ylabel='CNTL'+year)

        elif int(year) == 2016:
            print('Multiple Linear Regression for disaggregating nightlight 2016:')
            model_NTL = ols("CNTL2016 ~  area_hr2016 + area_nr2016 + area_bg2016 + area_lr2016",ntl_scale_NTL2).fit()
            print(model_NTL.summary())
            print("\nRetrieving manually the parameter estimates:")
            print(model_NTL._results.params)

            model_NTL.save(results + 'ols_ntl.pickle', remove_data=False)

            y, X = dmatrices('CNTL2016 ~ area_hr2016 + area_lr2016 + area_nr2016 + area_bg2016', data=ntl_scale_NTL2, return_type='dataframe')
            vif = pd.DataFrame()
            vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            vif['variable'] = X.columns
            print(vif)

            ntlresid = pd.concat((ntl_scale_NTL2,model_NTL.resid), axis=1)
            ntlresid.rename({0:'ntlresid' + year}, axis=1, inplace=True)
            W = Queen.from_dataframe(ntlresid)
            W.transform = 'r'
            moran_ntl = Moran(ntlresid['ntlresid'+ year], W)
            print('moran_ntl' + year + ': ' + str(moran_ntl.I))
            moran_loc = Moran_Local(ntlresid['ntlresid'+ year], W)
            p = lisa_cluster(moran_loc, ntlresid, p=0.05, figsize=(9, 9))
            plt.title('Local Autocorrelation for nightlight estimation residuals ' + year)
            plt.show()

            #1 HH, 2 LH, 3 LL, 4 HL
            cluster = _viz_utils.moran_hot_cold_spots(moran_loc, p=0.05)
            aux = pd.DataFrame(cluster)
            aux.rename({0:'clusters' + year}, inplace=True, axis=1)
            aux.loc[aux['clusters' + year] == 0, ['clusters' + year]] = 'NS'
            aux.loc[aux['clusters' + year]==1,['clusters' + year]] = 'HH'
            aux.loc[aux['clusters' + year]==2,['clusters' + year]] = 'LH'
            aux.loc[aux['clusters' + year]==3,['clusters' + year]] = 'LL'
            aux.loc[aux['clusters' + year]==4,['clusters' + year]] = 'HL'
            cluster = pd.concat((ntl_scale_NTL2, aux), axis=1)
            cluster = pd.get_dummies(cluster)

            print('Spatial Multiple Linear Regression for disaggregating nightlight 2016:')
            model_NTL_spatial = ols("CNTL2016 ~  area_hr2016 + area_nr2016 + area_bg2016 + clusters2016_HH + clusters2016_HL + clusters2016_LH + clusters2016_NS + area_lr2016",cluster).fit()
            print(model_NTL_spatial.summary())
            print("\nRetrieving manually the parameter estimates:")
            print(model_NTL_spatial._results.params)

            y, X = dmatrices("CNTL2016 ~  area_hr2016 + area_nr2016 + area_bg2016 + clusters2016_HH + clusters2016_HL + clusters2016_LH + clusters2016_NS + area_lr2016",
                             data=cluster, return_type='dataframe')
            vif = pd.DataFrame()
            vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            vif['variable'] = X.columns
            print(vif)

            model_NTL_spatial.save(results + 'ols_ntl_spatial.pickle', remove_data=False)

            NTL_clip.reset_index(inplace=True)
            cluster = cluster.merge(NTL_clip['ntl_clip_id'], left_on=cluster.index, right_on=NTL_clip.index.array, how='left')
            cluster.drop('key_0', axis=1, inplace=True)
            cluster.set_index('ntl_clip_id', inplace=True)
            intersect2 = intersect2.merge(cluster.loc[:, ['clusters2016_HH','clusters2016_HL', 'clusters2016_LH', 'clusters2016_LL','clusters2016_NS']],
                                          left_on= intersect2.index, right_on= cluster.index,how='left')
            intersect2.rename({'key_0': 'ntl_clip_id'}, inplace=True, axis=1)
            try:
                intersect2.drop('level_0', inplace=True, axis=1)
            except:
                print('done!')
            cluster.reset_index(inplace=True)

            model_NTL_pred = cluster['area_bg'+ year]*model_NTL.params[3] + cluster['area_hr'+ year]*model_NTL.params[1] + \
                  cluster['area_nr'+ year]*model_NTL.params[2] + cluster['area_lr'+ year]*model_NTL.params[4]
            model_NTL_pred = pd.DataFrame(model_NTL_pred, columns=['ntlpred' + year])
            Predictions = pd.concat((ntl_scale_NTL2, model_NTL_pred), axis=1)

            model_NTL_spatial_pred = cluster['area_bg'+ year]*model_NTL_spatial.params[3] + cluster['area_hr'+ year]*model_NTL_spatial.params[1] + \
                  cluster['area_nr'+ year]*model_NTL_spatial.params[2] + cluster['clusters2016_HH']*model_NTL_spatial.params[4] + \
                  cluster['clusters2016_HL']*model_NTL_spatial.params[5] + cluster['clusters2016_LH']*model_NTL_spatial.params[6] +\
                                     cluster['clusters2016_NS']*model_NTL_spatial.params[7] + cluster['area_lr' + year]*model_NTL_spatial.params[8]
            model_NTL_spatial_pred = pd.DataFrame(model_NTL_spatial_pred, columns=['ntlpred' + year])
            Predictions_spatial = pd.concat((ntl_scale_NTL2, model_NTL_spatial_pred), axis=1)

            sns.set(rc={'figure.figsize': (5, 10)}, style="whitegrid")
            f, axes = plt.subplots(2, 1)
            f.subplots_adjust(hspace=.5)
            sns.scatterplot(x=Predictions['ntlpred'+year], y=Predictions['CNTL'+year], ax=axes[0], color='red')
            axes[0].set(xlabel='ntlpred'+year)
            axes[0].set(ylabel='CNTL'+year)
            sns.scatterplot(x=Predictions_spatial['ntlpred'+year], y=Predictions['CNTL'+year], ax=axes[1], color='red')
            axes[1].set(xlabel='ntlpred Spatial'+ year)
            axes[1].set(ylabel='CNTL'+year)

        elif int(year) == 2017:
            print('Multiple Linear Regression for disaggregating nightlight 2017:')
            model_NTL = ols("CNTL2017 ~ area_hr2017 + area_nr2017 + area_bg2017 + area_lr2017",ntl_scale_NTL2).fit()
            print(model_NTL.summary())
            print("\nRetrieving manually the parameter estimates:")
            print(model_NTL._results.params)

            model_NTL.save(results + 'ols_ntl.pickle', remove_data=False)

            y, X = dmatrices('CNTL2017 ~ area_hr2017 + area_lr2017 + area_nr2017 + area_bg2017', data=ntl_scale_NTL2, return_type='dataframe')
            vif = pd.DataFrame()
            vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            vif['variable'] = X.columns
            print(vif)

            ntlresid = pd.concat((ntl_scale_NTL2,model_NTL.resid), axis=1)
            ntlresid.rename({0:'ntlresid' + year}, axis=1, inplace=True)
            W = Queen.from_dataframe(ntlresid)
            W.transform = 'r'
            moran_ntl = Moran(ntlresid['ntlresid'+ year], W)
            print('moran_ntl' + year + ': ' + str(moran_ntl.I))
            moran_loc = Moran_Local(ntlresid['ntlresid'+ year], W)
            p = lisa_cluster(moran_loc, ntlresid, p=0.05, figsize=(9, 9))
            plt.title('Local Autocorrelation for nightlight estimation residuals ' + year)
            plt.show()

            #1 HH, 2 LH, 3 LL, 4 HL
            cluster = _viz_utils.moran_hot_cold_spots(moran_loc, p=0.05)
            aux = pd.DataFrame(cluster)
            aux.rename({0:'clusters' + year}, inplace=True, axis=1)
            aux.loc[aux['clusters' + year] == 0, ['clusters' + year]] = 'NS'
            aux.loc[aux['clusters' + year]==1,['clusters' + year]] = 'HH'
            aux.loc[aux['clusters' + year]==2,['clusters' + year]] = 'LH'
            aux.loc[aux['clusters' + year]==3,['clusters' + year]] = 'LL'
            aux.loc[aux['clusters' + year]==4,['clusters' + year]] = 'HL'
            cluster = pd.concat((ntl_scale_NTL2, aux), axis=1)
            cluster = pd.get_dummies(cluster)

            print('Spatial Multiple Linear Regression for disaggregating nightlight 2017:')
            model_NTL_spatial = ols("CNTL2017 ~  area_hr2017 + area_nr2017 + area_bg2017 + clusters2017_HH + clusters2017_LH + clusters2017_NS + area_lr2017",cluster).fit()
            print(model_NTL_spatial.summary())
            print("\nRetrieving manually the parameter estimates:")
            print(model_NTL_spatial._results.params)

            y, X = dmatrices("CNTL2017 ~  area_hr2017 + area_nr2017 + area_bg2017 + clusters2017_HH + clusters2017_LH + clusters2017_NS",
                             data=cluster, return_type='dataframe')
            vif = pd.DataFrame()
            vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            vif['variable'] = X.columns
            print(vif)

            model_NTL_spatial.save(results + 'ols_ntl_spatial.pickle', remove_data=False)

            NTL_clip.reset_index(inplace=True)
            cluster = cluster.merge(NTL_clip['ntl_clip_id'], left_on=cluster.index, right_on=NTL_clip.index.array, how='left')
            cluster.drop('key_0', axis=1, inplace=True)
            cluster.set_index('ntl_clip_id', inplace=True)
            intersect2 = intersect2.merge(cluster.loc[:, ['clusters2017_HH','clusters2017_LH', 'clusters2017_LL','clusters2017_NS']],
                                          left_on= intersect2.index, right_on= cluster.index,how='left')
            intersect2.rename({'key_0': 'ntl_clip_id'}, inplace=True, axis=1)
            try:
                intersect2.drop('level_0', inplace=True, axis=1)
            except:
                print('done!')
            cluster.reset_index(inplace=True)

            model_NTL_pred = cluster['area_bg'+ year]*model_NTL.params[3] + cluster['area_hr'+ year]*model_NTL.params[1] + \
                  cluster['area_nr'+ year]*model_NTL.params[2] + cluster['area_lr'+ year]*model_NTL.params[4]
            model_NTL_pred = pd.DataFrame(model_NTL_pred, columns=['ntlpred' + year])
            Predictions = pd.concat((ntl_scale_NTL2, model_NTL_pred), axis=1)

            model_NTL_spatial_pred = cluster['area_bg'+ year]*model_NTL_spatial.params[3] + cluster['area_hr'+ year]*model_NTL_spatial.params[1] + \
                  cluster['area_nr'+ year]*model_NTL_spatial.params[2] + cluster['clusters2017_HH']*model_NTL_spatial.params[4] + \
                  cluster['clusters2017_LH']*model_NTL_spatial.params[5] + cluster['clusters2017_NS']*model_NTL_spatial.params[6]\
                                     + cluster['area_lr' + year] * model_NTL_spatial.params[7]
            model_NTL_spatial_pred = pd.DataFrame(model_NTL_spatial_pred, columns=['ntlpred' + year])
            Predictions_spatial = pd.concat((ntl_scale_NTL2, model_NTL_spatial_pred), axis=1)

            sns.set(rc={'figure.figsize': (5, 10)}, style="whitegrid")
            f, axes = plt.subplots(2, 1)
            f.subplots_adjust(hspace=.5)
            sns.scatterplot(x=Predictions['ntlpred'+year], y=Predictions['CNTL'+year], ax=axes[0], color='red')
            axes[0].set(xlabel='ntlpred'+year)
            axes[0].set(ylabel='CNTL'+year)
            sns.scatterplot(x=Predictions_spatial['ntlpred'+year], y=Predictions['CNTL'+year], ax=axes[1], color='red')
            axes[1].set(xlabel='ntlpred Spatial'+ year)
            axes[1].set(ylabel='CNTL'+year)

        else:

            print('Multiple Linear Regression for disaggregating nightlight 2018:')
            model_NTL = ols("CNTL2018 ~  area_hr2018 + area_nr2018 + area_bg2018 + area_lr2018",ntl_scale_NTL2).fit()
            print(model_NTL.summary())
            print("\nRetrieving manually the parameter estimates:")
            print(model_NTL._results.params)

            model_NTL.save(results + 'ols_ntl.pickle', remove_data=False)

            y, X = dmatrices('CNTL2018 ~ area_hr2018 + area_lr2018 + area_nr2018 + area_bg2018', data=ntl_scale_NTL2, return_type='dataframe')
            vif = pd.DataFrame()
            vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            vif['variable'] = X.columns
            print(vif)

            ntlresid = pd.concat((ntl_scale_NTL2,model_NTL.resid), axis=1)
            ntlresid.rename({0:'ntlresid' + year}, axis=1, inplace=True)
            W = Queen.from_dataframe(ntlresid)
            W.transform = 'r'
            moran_ntl = Moran(ntlresid['ntlresid'+ year], W)
            print('moran_ntl' + year + ': ' + str(moran_ntl.I))
            moran_loc = Moran_Local(ntlresid['ntlresid'+ year], W)
            p = lisa_cluster(moran_loc, ntlresid, p=0.05, figsize=(9, 9))
            plt.title('Local Autocorrelation for nightlight estimation residuals ' + year)
            plt.show()

            #1 HH, 2 LH, 3 LL, 4 HL
            cluster = _viz_utils.moran_hot_cold_spots(moran_loc, p=0.05)
            aux = pd.DataFrame(cluster)
            aux.rename({0:'clusters' + year}, inplace=True, axis=1)
            aux.loc[aux['clusters' + year] == 0, ['clusters' + year]] = 'NS'
            aux.loc[aux['clusters' + year]==1,['clusters' + year]] = 'HH'
            aux.loc[aux['clusters' + year]==2,['clusters' + year]] = 'LH'
            aux.loc[aux['clusters' + year]==3,['clusters' + year]] = 'LL'
            aux.loc[aux['clusters' + year]==4,['clusters' + year]] = 'HL'
            cluster = pd.concat((ntl_scale_NTL2, aux), axis=1)
            cluster = pd.get_dummies(cluster)

            print('Spatial Multiple Linear Regression for disaggregating nightlight 2018:')
            model_NTL_spatial = ols("CNTL2018 ~  area_hr2018 + area_nr2018 + area_bg2018 + clusters2018_HH + clusters2018_HL + clusters2018_NS + area_lr2018",cluster).fit()
            print(model_NTL_spatial.summary())
            print("\nRetrieving manually the parameter estimates:")
            print(model_NTL_spatial._results.params)

            y, X = dmatrices("CNTL2018 ~  area_hr2018 + area_nr2018 + area_bg2018 + clusters2018_HH + clusters2018_HL + clusters2018_NS",
                             data=cluster, return_type='dataframe')
            vif = pd.DataFrame()
            vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            vif['variable'] = X.columns
            print(vif)

            model_NTL_spatial.save(results + 'ols_ntl_spatial.pickle', remove_data=False)

            NTL_clip.reset_index(inplace=True)
            cluster = cluster.merge(NTL_clip['ntl_clip_id'], left_on=cluster.index, right_on=NTL_clip.index.array, how='left')
            cluster.drop('key_0', axis=1, inplace=True)
            cluster.set_index('ntl_clip_id', inplace=True)
            intersect2 = intersect2.merge(cluster.loc[:, ['clusters2018_HH','clusters2018_HL', 'clusters2018_LL','clusters2018_NS']],
                                          left_on= intersect2.index, right_on= cluster.index,how='left')
            intersect2.rename({'key_0': 'ntl_clip_id'}, inplace=True, axis=1)
            try:
                intersect2.drop('level_0', inplace=True, axis=1)
            except:
                print('done!')
            cluster.reset_index(inplace=True)

            model_NTL_pred = cluster['area_bg'+ year]*model_NTL.params[3] + cluster['area_hr'+ year]*model_NTL.params[1] + \
                  cluster['area_nr'+ year]*model_NTL.params[2] + cluster['area_lr'+ year]*model_NTL.params[4]
            model_NTL_pred = pd.DataFrame(model_NTL_pred, columns=['ntlpred' + year])
            Predictions = pd.concat((ntl_scale_NTL2, model_NTL_pred), axis=1)

            model_NTL_spatial_pred = cluster['area_bg'+ year]*model_NTL_spatial.params[3] + cluster['area_hr'+ year]*model_NTL_spatial.params[1] + \
                  cluster['area_nr'+ year]*model_NTL_spatial.params[2] + cluster['clusters2018_HH']*model_NTL_spatial.params[4] + \
                  cluster['clusters2018_HL']*model_NTL_spatial.params[5] + cluster['clusters2018_NS']*model_NTL_spatial.params[6] + \
                                     cluster['area_lr' + year] * model_NTL_spatial.params[7]

            model_NTL_spatial_pred = pd.DataFrame(model_NTL_spatial_pred, columns=['ntlpred' + year])
            Predictions_spatial = pd.concat((ntl_scale_NTL2, model_NTL_spatial_pred), axis=1)

            sns.set(rc={'figure.figsize': (5, 10)}, style="whitegrid")
            f, axes = plt.subplots(2, 1)
            f.subplots_adjust(hspace=.5)
            sns.scatterplot(x=Predictions['ntlpred'+year], y=Predictions['CNTL'+year], ax=axes[0], color='red')
            axes[0].set(xlabel='ntlpred'+year)
            axes[0].set(ylabel='CNTL'+year)
            sns.scatterplot(x=Predictions_spatial['ntlpred'+year], y=Predictions['CNTL'+year], ax=axes[1], color='red')
            axes[1].set(xlabel='ntlpred Spatial'+ year)
            axes[1].set(ylabel='CNTL'+year)

        from statsmodels.regression.linear_model import OLSResults
        ols_ntl_spatial = OLSResults.load(results + 'ols_ntl_spatial.pickle')

        if int(year) == 2013:

            intersect2['coef_ntl2013'] = np.nan
            intersect2['spatial_ntl' + year] = np.nan

            mask = intersect2['landuse2014'] == 1
            intersect2.loc[mask, ['coef_ntl2013']] = model_NTL_spatial.params[3]
            mask = intersect2['landuse2014'] == 2
            intersect2.loc[mask, ['coef_ntl2013']] = model_NTL_spatial.params[7]
            mask = intersect2['landuse2014'] == 3
            intersect2.loc[mask, ['coef_ntl2013']] = model_NTL_spatial.params[1]
            mask = intersect2['landuse2014'] == 4
            intersect2.loc[mask, ['coef_ntl2013']] = model_NTL_spatial.params[2]

            mask = (intersect2['clusters2013_HH'] == 1)
            intersect2.loc[mask, ['spatial_ntl'+year]] = model_NTL_spatial.params[4]
            mask = (intersect2['clusters2013_HL'] == 1)
            intersect2.loc[mask, ['spatial_ntl'+year]] = model_NTL_spatial.params[5]
            mask = (intersect2['clusters2013_NS'] == 1)
            intersect2.loc[mask, ['spatial_ntl'+year]] = model_NTL_spatial.params[6]
            mask = (intersect2['clusters2013_LL'] == 1)
            intersect2.loc[mask, ['spatial_ntl'+year]] = 0

        elif int(year) == 2014:

            intersect2['coef_ntl' + year] = np.nan
            intersect2['spatial_ntl' + year] = np.nan

            mask = intersect2['landuse' + year] == 1
            intersect2.loc[mask, ['coef_ntl' + year]] = model_NTL_spatial.params[3]
            mask = intersect2['landuse' + year] == 2
            intersect2.loc[mask, ['coef_ntl' + year]] = model_NTL_spatial.params[8]
            mask = intersect2['landuse' + year] == 3
            intersect2.loc[mask, ['coef_ntl' + year]] = model_NTL_spatial.params[1]
            mask = intersect2['landuse' + year] == 4
            intersect2.loc[mask, ['coef_ntl' + year]] = model_NTL_spatial.params[2]

            mask = (intersect2['clusters2014_HH'] == 1)
            intersect2.loc[mask, ['spatial_ntl' + year]] = model_NTL_spatial.params[4]
            mask = (intersect2['clusters2014_HL'] == 1)
            intersect2.loc[mask, ['spatial_ntl' + year]] = model_NTL_spatial.params[5]
            mask = (intersect2['clusters2014_LH'] == 1)
            intersect2.loc[mask, ['spatial_ntl' + year]] = model_NTL_spatial.params[6]
            mask = (intersect2['clusters2014_NS'] == 1)
            intersect2.loc[mask, ['spatial_ntl' + year]] = model_NTL_spatial.params[7]
            mask = (intersect2['clusters2014_LL'] == 1)
            intersect2.loc[mask, ['spatial_ntl'+year]] = 0

        elif int(year) == 2015:

            intersect2['coef_ntl' + year] = np.nan
            intersect2['spatial_ntl' + year] = np.nan

            mask = intersect2['landuse' + year] == 1
            intersect2.loc[mask, ['coef_ntl' + year]] = model_NTL_spatial.params[3]
            mask = intersect2['landuse' + year] == 2
            intersect2.loc[mask, ['coef_ntl' + year]] = model_NTL_spatial.params[8]
            mask = intersect2['landuse' + year] == 3
            intersect2.loc[mask, ['coef_ntl' + year]] = model_NTL_spatial.params[1]
            mask = intersect2['landuse' + year] == 4
            intersect2.loc[mask, ['coef_ntl' + year]] = model_NTL_spatial.params[2]

            mask = (intersect2['clusters2015_HH'] == 1)
            intersect2.loc[mask, ['spatial_ntl' + year]] = model_NTL_spatial.params[4]
            mask = (intersect2['clusters2015_HL'] == 1)
            intersect2.loc[mask, ['spatial_ntl' + year]] = model_NTL_spatial.params[5]
            mask = (intersect2['clusters2015_LH'] == 1)
            intersect2.loc[mask, ['spatial_ntl' + year]] = model_NTL_spatial.params[6]
            mask = (intersect2['clusters2015_NS'] == 1)
            intersect2.loc[mask, ['spatial_ntl' + year]] = model_NTL_spatial.params[7]
            mask = (intersect2['clusters2015_LL'] == 1)
            intersect2.loc[mask, ['spatial_ntl' + year]] = 0

        elif int(year) == 2016:

            intersect2['coef_ntl' + year] = np.nan
            intersect2['spatial_ntl' + year] = np.nan

            mask = intersect2['landuse' + year] == 1
            intersect2.loc[mask, ['coef_ntl' + year]] = model_NTL_spatial.params[3]
            mask = intersect2['landuse' + year] == 2
            intersect2.loc[mask, ['coef_ntl' + year]] = model_NTL_spatial.params[8]
            mask = intersect2['landuse' + year] == 3
            intersect2.loc[mask, ['coef_ntl' + year]] = model_NTL_spatial.params[1]
            mask = intersect2['landuse' + year] == 4
            intersect2.loc[mask, ['coef_ntl' + year]] = model_NTL_spatial.params[2]

            mask = (intersect2['clusters2016_HH'] == 1)
            intersect2.loc[mask, ['spatial_ntl' + year]] = model_NTL_spatial.params[4]
            mask = (intersect2['clusters2016_HL'] == 1)
            intersect2.loc[mask, ['spatial_ntl' + year]] = model_NTL_spatial.params[5]
            mask = (intersect2['clusters2016_LH'] == 1)
            intersect2.loc[mask, ['spatial_ntl' + year]] = model_NTL_spatial.params[6]
            mask = (intersect2['clusters2016_NS'] == 1)
            intersect2.loc[mask, ['spatial_ntl' + year]] = model_NTL_spatial.params[7]
            mask = (intersect2['clusters2016_LL'] == 1)
            intersect2.loc[mask, ['spatial_ntl' + year]] = 0

        elif int(year) == 2017:

            intersect2['coef_ntl' + year] = np.nan
            intersect2['spatial_ntl' + year] = np.nan

            mask = intersect2['landuse' + year] == 1
            intersect2.loc[mask, ['coef_ntl' + year]] = model_NTL_spatial.params[3]
            mask = intersect2['landuse' + year] == 2
            intersect2.loc[mask, ['coef_ntl' + year]] = model_NTL_spatial.params[7]
            mask = intersect2['landuse' + year] == 3
            intersect2.loc[mask, ['coef_ntl' + year]] = model_NTL_spatial.params[1]
            mask = intersect2['landuse' + year] == 4
            intersect2.loc[mask, ['coef_ntl' + year]] = model_NTL_spatial.params[2]

            mask = (intersect2['clusters2017_HH'] == 1)
            intersect2.loc[mask, ['spatial_ntl' + year]] = model_NTL_spatial.params[4]
            mask = (intersect2['clusters2017_LH'] == 1)
            intersect2.loc[mask, ['spatial_ntl' + year]] = model_NTL_spatial.params[5]
            mask = (intersect2['clusters2017_NS'] == 1)
            intersect2.loc[mask, ['spatial_ntl' + year]] = model_NTL_spatial.params[6]
            mask = (intersect2['clusters2017_LL'] == 1)
            intersect2.loc[mask, ['spatial_ntl' + year]] = 0

        else:

            intersect2['coef_ntl' + year] = np.nan
            intersect2['spatial_ntl' + year] = np.nan

            mask = intersect2['landuse' + year] == 1
            intersect2.loc[mask, ['coef_ntl' + year]] = model_NTL_spatial.params[3]
            mask = intersect2['landuse' + year] == 2
            intersect2.loc[mask, ['coef_ntl' + year]] = model_NTL_spatial.params[7]
            mask = intersect2['landuse' + year] == 3
            intersect2.loc[mask, ['coef_ntl' + year]] = model_NTL_spatial.params[1]
            mask = intersect2['landuse' + year] == 4
            intersect2.loc[mask, ['coef_ntl' + year]] = model_NTL_spatial.params[2]

            mask = (intersect2['clusters2018_HH'] == 1)
            intersect2.loc[mask, ['spatial_ntl' + year]] = model_NTL_spatial.params[4]
            mask = (intersect2['clusters2018_HL'] == 1)
            intersect2.loc[mask, ['spatial_ntl' + year]] = model_NTL_spatial.params[5]
            mask = (intersect2['clusters2018_NS'] == 1)
            intersect2.loc[mask, ['spatial_ntl' + year]] = model_NTL_spatial.params[6]
            mask = (intersect2['clusters2018_LL'] == 1)
            intersect2.loc[mask, ['spatial_ntl'+ year]] = 0

        intersect2['disNTL' + year] = intersect2['landuse_clip_area']*intersect2['coef_ntl' + year] + intersect2['spatial_ntl' + year] #+ \
                               # (intersect2['intercept_ntl' + year]/intersect2['countNTL'])
        intersect2.reset_index(inplace=True)
        # We have to set the negative values to 0

        mask = intersect2['disNTL' + year] < 0
        mask2 = intersect2['disNTL' + year] >= 0
        print('Percentage error caused by removing negative values in nightlight: ',
              abs(intersect2[mask].sum()['disNTL' + year] / intersect2[mask2].sum()['disNTL' + year])*100)

        intersect2.loc[mask, ['disNTL' + year]] = 0

        intersect2['disNTL_verify' + year] = intersect2['disNTL' + year].groupby(intersect2.ntl_clip_id).transform('sum')

        intersect2['disNTL_prime' + year] = 0
        mask = intersect2['disNTL_verify' + year] != 0
        intersect2.loc[mask, 'disNTL_prime' + year] = np.array(intersect2.loc[mask, ['disNTL' + year]]) * \
                                                      (np.array(intersect2.loc[mask, ['CNTL' + year]]) / np.array(intersect2.loc[mask, ['disNTL_verify' + year]]))

        print(intersect2.groupby('ntl_clip_id').sum().loc[:, ['disNTL_prime' + year]])
        print(intersect2.groupby('ntl_clip_id').max().loc[:, ['CNTL' + year]])

    # in the level of census
    intersect2.set_index('census_id', inplace=True)
    intersect2['countPop'] = intersect2['index'].groupby(intersect2.index).transform('count')
    ntl_scale_Pop = intersect2.groupby(['census_id', 'landuse2014']).sum().loc[:,['intersect_area', 'disNTL_prime2013']]
    ntl_scale_Pop2 = ntl_scale_Pop.unstack('landuse2014')
    ntl_scale_Pop2.columns = ['area_bg', 'area_lr', 'area_hr', 'area_nr', 'disNTL2013_bg', 'disNTL2013_lr', 'disNTL2013_hr', 'disNTL2013_nr']
    ntl_scale_Pop2.fillna(0, inplace=True)

    ntl_scale_Pop2['target_pop2013'] = intersect2.groupby(intersect2.index).max()['estPop2013']

    ntl_scale_Pop2.reset_index(inplace=True)
    ntl_scale_Pop2 = census.merge(ntl_scale_Pop2, left_on = census.index.array, right_on = ntl_scale_Pop2.index.array, how='left')
    # ntl_scale_Pop2.drop(['key_0', 'index', 'Shape_Leng', 'Shape_Area', 'MAX_popult', 'census_id_x', 'census_area'],
    #                 inplace=True, axis=1)
    ntl_scale_Pop2.drop(['key_0', 'Shape_Leng', 'Shape_Area', 'MAX_popult', 'census_area'],
                    inplace=True, axis=1)

    ntl_scale_Pop2.rename({'census_id_y':'census_id'}, axis=1)
    ntl_scale_Pop2['X'] = ntl_scale_Pop2.geometry.centroid.x
    ntl_scale_Pop2['Y'] = ntl_scale_Pop2.geometry.centroid.y
    ntl_scale_Pop2['Census_area'] = ntl_scale_Pop2.geometry.area
    ntl_scale_Pop2['disNTL2013_hr_den'] = ntl_scale_Pop2['disNTL2013_hr'] / ntl_scale_Pop2['Census_area']
    ntl_scale_Pop2['disNTL2013_nr_den'] = ntl_scale_Pop2['disNTL2013_nr'] / ntl_scale_Pop2['Census_area']
    ntl_scale_Pop2['disNTL2013_bg_den'] = ntl_scale_Pop2['disNTL2013_bg'] / ntl_scale_Pop2['Census_area']
    ntl_scale_Pop2['disNTL2013_lr_den'] = ntl_scale_Pop2['disNTL2013_lr'] / ntl_scale_Pop2['Census_area']
    ntl_scale_Pop2['area_hr_den'] = ntl_scale_Pop2['area_hr'] / ntl_scale_Pop2['Census_area']
    ntl_scale_Pop2['area_lr_den'] = ntl_scale_Pop2['area_lr'] / ntl_scale_Pop2['Census_area']
    ntl_scale_Pop2['area_nr_den'] = ntl_scale_Pop2['area_nr'] / ntl_scale_Pop2['Census_area']
    ntl_scale_Pop2['area_bg_den'] = ntl_scale_Pop2['area_bg'] / ntl_scale_Pop2['Census_area']
    ntl_scale_Pop2['area_lr2'] = ntl_scale_Pop2['area_lr']*ntl_scale_Pop2['area_lr']
    ntl_scale_Pop2['area_lraux'] = ntl_scale_Pop2['area_lr']
    ntl_scale_Pop2['area_hraux'] = ntl_scale_Pop2['area_hr']
    ntl_scale_Pop2['area_nraux'] = ntl_scale_Pop2['area_nr']
    ntl_scale_Pop2['area_bgaux'] = ntl_scale_Pop2['area_bg']

    ntl_scale_Pop2.loc[ntl_scale_Pop2['area_lraux'] == 0,['area_lraux']] = 1
    ntl_scale_Pop2.loc[ntl_scale_Pop2['area_hraux'] == 0,['area_hraux']] = 1
    ntl_scale_Pop2.loc[ntl_scale_Pop2['area_nraux'] == 0,['area_nraux']] = 1
    ntl_scale_Pop2.loc[ntl_scale_Pop2['area_bgaux'] == 0,['area_bgaux']] = 1
    # ntl_scale_Pop2.loc[ntl_scale_Pop2['pop'] == 0, ['pop']] = 1
    # ntl_scale_Pop2['area_lrauxlog'] = np.log(ntl_scale_Pop2['area_lraux'])
    # ntl_scale_Pop2['area_hrauxlog'] = np.log(ntl_scale_Pop2['area_hraux'])
    # ntl_scale_Pop2['area_nrauxlog'] = np.log(ntl_scale_Pop2['area_nraux'])
    # ntl_scale_Pop2['area_bgauxlog'] = np.log(ntl_scale_Pop2['area_bgaux'])
    # ntl_scale_Pop2['poplog'] = np.log(ntl_scale_Pop2['pop'])
    ntl_scale_Pop2['estPop2013_den'] = ntl_scale_Pop2['estPop2013'] / ntl_scale_Pop2['Census_area']

    y, X = dmatrices(
        "estPop2013 ~ disNTL2013_hr + disNTL2013_nr + disNTL2013_lr + disNTL2013_bg",
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

    cluster = cluster.merge(census['census_id'], left_on=cluster.index, right_on=census.index.array, how='left')
    cluster.drop('key_0', axis=1, inplace=True)
    cluster.set_index('census_id', inplace=True)
    intersect2 = intersect2.merge(
        cluster.loc[:, ['clusters2013pop_HH', 'clusters2013pop_HL', 'clusters2013pop_LH', 'clusters2013pop_LL', 'clusters2013pop_NS']],
        left_on=intersect2.index, right_on=cluster.index, how='left')
    intersect2.rename({'key_0': 'census_id'}, inplace=True, axis=1)
    intersect2.drop('level_0', axis=1, inplace=True)

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
    intersect2.set_index('ntl_clip_id', inplace=True)
    ntl_scale_NTL = intersect2.groupby(['ntl_clip_id', 'landuse2014']).sum().loc[:,['intersect_area','disNTL_prime2013', 'disPop2013_prime']]
    ntl_scale_NTL2 = ntl_scale_NTL.unstack('landuse2014')
    ntl_scale_NTL2.columns = ['area_bg', 'area_lr', 'area_hr', 'area_nr','NTL2013_bg', 'NTL2013_lr', 'NTL2013_hr', 'NTL2013_nr',
                            'Pop2013_bg', 'Pop2013_lr', 'Pop2013_hr', 'Pop2013_nr']
    ntl_scale_NTL2.fillna(0, inplace=True)
    ntl_scale_NTL2['NTL2013'] = ntl_scale_NTL2['NTL2013_bg'] + ntl_scale_NTL2['NTL2013_lr'] + \
                                ntl_scale_NTL2['NTL2013_hr'] + ntl_scale_NTL2['NTL2013_nr']
    ntl_scale_NTL2['Pop2013'] = ntl_scale_NTL2['Pop2013_bg'] + ntl_scale_NTL2['Pop2013_lr'] + \
                                ntl_scale_NTL2['Pop2013_hr'] + ntl_scale_NTL2['Pop2013_nr']

    # print('NTL2013_lr sum of the values: ', ntl_scale_NTL2['NTL2013_lr'].sum())
    ntl_scale_NTL2['CNTL2013'] = intersect2.groupby(intersect2.index).max()['CNTL2013']

    ntl_scale_NTL2.reset_index(inplace=True)
    NTL_clip.reset_index(inplace=True)
    ntl_scale_NTL2 = NTL_clip.merge(ntl_scale_NTL2, left_on = NTL_clip.index.array, right_on = ntl_scale_NTL2.index.array, how='left')
    ntl_scale_NTL2.drop(['key_0', 'index', 'Shape_Leng', 'Shape_Area', 'ntl_id','ntl_area', 'NTL2013_x', 'NTL2014',
                         'NTL2015', 'NTL2016','NTL2017', 'NTL2018', 'ntl_clip_id_x', 'ntl_clip_area'],
                        inplace=True, axis=1)
    try:
        ntl_scale_NTL2.drop(['level_0'], inplace=True, axis=1)
        NTL_clip.drop(['level_0'], inplace=True, axis=1)
    except:
        print('level_0 is not in the columns')
    ntl_scale_NTL2['X'] = ntl_scale_NTL2.geometry.centroid.x
    ntl_scale_NTL2['Y'] = ntl_scale_NTL2.geometry.centroid.y

    if mdl == 'nontl':
        print('Multiple Linear Regression: pop_lndus:')
        model_NTL_Pop = ols("Pop2013 ~  area_hr",ntl_scale_NTL2).fit()
        print(model_NTL_Pop.summary())
        print("\nRetrieving manually the parameter estimates:")
        print(model_NTL_Pop._results.params)

        model_NTL_Pop.save(results + 'ols_ntl_pop.pickle', remove_data=False)
        from statsmodels.regression.linear_model import OLSResults
        model_NTL_Pop = OLSResults.load(results + 'ols_ntl_pop.pickle')

        ntl_scale_NTL2.to_csv(results + 'observations_' + 'lndus_' + date + '.csv')
    elif mdl == 'ntlmed':

        print('Multiple Linear Regression: pop_medntl_annual_incorrected')
        model_NTL_Pop = ols("Pop2013 ~  NTL2013_hr",ntl_scale_NTL2).fit()
        print(model_NTL_Pop.summary())
        print("\nRetrieving manually the parameter estimates:")
        print(model_NTL_Pop._results.params)

        y, X = dmatrices(
            "Pop2013 ~  NTL2013_hr + NTL2013_nr + NTL2013_bg + NTL2013_lr",
            data=ntl_scale_NTL2, return_type='dataframe')
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif['variable'] = X.columns
        print(vif)

        popntlresid = pd.concat((ntl_scale_NTL2, model_NTL_Pop.resid), axis=1)
        popntlresid.rename({0: 'popntlresid2013'}, axis=1, inplace=True)
        W = Queen.from_dataframe(popntlresid)
        W.transform = 'r'
        moran_popntl = Moran(popntlresid.popntlresid2013, W)
        print('moran_popntl2013: ' + str(moran_popntl.I))
        moran_loc = Moran_Local(popntlresid['popntlresid2013'], W)
        p = lisa_cluster(moran_loc, popntlresid, p=0.05, figsize=(9, 9))
        plt.title('Local Autocorrelation for population estimation residuals 2013')
        plt.show()

        # 1 HH, 2 LH, 3 LL, 4 HL
        cluster = _viz_utils.moran_hot_cold_spots(moran_loc, p=0.05)
        aux = pd.DataFrame(cluster)
        aux.rename({0: 'clusters2013popntl'}, inplace=True, axis=1)
        aux.loc[aux['clusters2013popntl'] == 0, ['clusters2013popntl']] = 'NS'
        aux.loc[aux['clusters2013popntl'] == 1, ['clusters2013popntl']] = 'HH'
        aux.loc[aux['clusters2013popntl'] == 2, ['clusters2013popntl']] = 'LH'
        aux.loc[aux['clusters2013popntl'] == 3, ['clusters2013popntl']] = 'LL'
        aux.loc[aux['clusters2013popntl'] == 4, ['clusters2013popntl']] = 'HL'
        cluster = pd.concat((ntl_scale_NTL2, aux), axis=1)
        cluster = pd.get_dummies(cluster)

        print('Spatial Multiple Linear Regression for disaggregating population 2013:')
        model_pop_ntl_spatial = ols(
            "Pop2013 ~  NTL2013_hr + clusters2013popntl_HH + clusters2013popntl_HL + clusters2013popntl_LH + clusters2013popntl_NS",
            cluster).fit()
        print(model_pop_ntl_spatial.summary())
        print("\nRetrieving manually the parameter estimates:")
        print(model_pop_ntl_spatial._results.params)

        model_NTL_Pop_pred = model_NTL_Pop.predict()
        model_NTL_Pop_pred = pd.DataFrame(model_NTL_Pop_pred, columns=['popntlpred2013'])
        Predictions = pd.concat((ntl_scale_NTL2, model_NTL_Pop_pred), axis=1)

        model_pop_ntl_spatial_pred = cluster['NTL2013_hr'] * model_pop_ntl_spatial.params[1] +\
                                     cluster['clusters2013popntl_HH'] * model_pop_ntl_spatial.params[2] + \
                                     cluster['clusters2013popntl_HL'] * model_pop_ntl_spatial.params[3] + \
                                     cluster['clusters2013popntl_LH'] * model_pop_ntl_spatial.params[4] + \
                                     cluster['clusters2013popntl_NS'] * model_pop_ntl_spatial.params[5]

        model_pop_ntl_spatial_pred = pd.DataFrame(model_pop_ntl_spatial_pred, columns=['popntlpred2013'])
        Predictions_spatial = pd.concat((ntl_scale_NTL2, model_pop_ntl_spatial_pred), axis=1)

        sns.set(rc={'figure.figsize': (5, 10)}, style="whitegrid")
        f, axes = plt.subplots(2, 1)
        f.subplots_adjust(hspace=.5)
        sns.scatterplot(x=Predictions['popntlpred2013'], y=Predictions['Pop2013'], ax=axes[0], color='red')
        axes[0].set(xlabel='popntlpred2013')
        axes[0].set(ylabel='Pop2013')
        sns.scatterplot(x=Predictions_spatial['popntlpred2013'], y=Predictions['Pop2013'], ax=axes[1], color='red')
        axes[1].set(xlabel='popntlpred2013 Spatial 2013')
        axes[1].set(ylabel='Pop2013')

        model_pop_ntl_spatial.save(results + 'ols_ntl_pop.pickle', remove_data=False)
        from statsmodels.regression.linear_model import OLSResults
        model_pop_ntl_spatial = OLSResults.load(results + 'ols_ntl_pop.pickle')

        ntl_scale_NTL2.to_csv(results + 'observations_median_' + 'ntl_annual_incorrected_' + date + '.csv')

    elif mdl == 'ntl_corrected_med_annualByMonth':
        print('Multiple Linear Regression: pop_medntl_annual_corrected')
        model_NTL_Pop = ols("Pop2013 ~  area_hr + NTL2013_hr + NTL2013_nr",ntl_scale_NTL2).fit()
        print(model_NTL_Pop.summary())
        print("\nRetrieving manually the parameter estimates:")
        print(model_NTL_Pop._results.params)

        model_NTL_Pop.save(results + 'ols_ntl_pop.pickle', remove_data=False)
        from statsmodels.regression.linear_model import OLSResults
        model_NTL_Pop = OLSResults.load(results + 'ols_ntl_pop.pickle')

        ntl_scale_NTL2.to_csv(results + 'observations_median_' + 'ntl_annual_corrected_' + date + '.csv')
    else:
        print('Multiple Linear Regression: pop_medntl_monthly_corrected')
        model_NTL_Pop = ols("Pop2013 ~  area_hr + NTL2013_nr",ntl_scale_NTL2).fit()
        print(model_NTL_Pop.summary())
        print("\nRetrieving manually the parameter estimates:")
        print(model_NTL_Pop._results.params)

        model_NTL_Pop.save(results + 'ols_ntl_pop.pickle', remove_data=False)
        from statsmodels.regression.linear_model import OLSResults

        ntl_scale_NTL2.to_csv(results + 'observations_median_' + 'ntl_monthly_corrected_' + date + '.csv')

    # Prepare all years:
    try:
        NTL_clip.drop(['level_0'], inplace=True, axis=1)

    except:
        print('level_0 is not in the columns')
    try:
        NTL_clip.set_index('ntl_clip_id', inplace=True)
    except:
        print('Already satisfied')
    NTL_clip_aux = NTL_clip
    # for all years

    ntl_prediction = intersect2.groupby(['ntl_clip_id', 'landuse2014']).sum().loc[:,['intersect_area','disNTL_prime2014']]
    ntl_prediction2 = ntl_prediction.unstack('landuse2014')
    ntl_prediction2.columns = ['area_bg2014', 'area_lr2014', 'area_hr2014', 'area_nr2014',
                                       'NTL_bg2014', 'NTL_lr2014', 'NTL_hr2014', 'NTL_nr2014']
    ntl_prediction2['CNTL2014'] = intersect2.groupby(intersect2.index).max()['CNTL2014']
    NTL_clip_aux['ntl_clip_id_copy'] = NTL_clip_aux.index
    NTL_clip_aux = NTL_clip_aux.merge(ntl_prediction2, left_on=NTL_clip_aux.index, right_on=ntl_prediction2.index, how='left')
    NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
    NTL_clip_aux.drop('key_0', inplace=True, axis=1)

    ntl_prediction = intersect2.groupby(['ntl_clip_id', 'landuse2015']).sum().loc[:,['intersect_area','disNTL_prime2015']]
    ntl_prediction2 = ntl_prediction.unstack('landuse2015')
    ntl_prediction2.columns = ['area_bg2015', 'area_lr2015', 'area_hr2015', 'area_nr2015',
                                       'NTL_bg2015', 'NTL_lr2015', 'NTL_hr2015', 'NTL_nr2015']
    ntl_prediction2['CNTL2015'] = intersect2.groupby(intersect2.index).max()['CNTL2015']
    NTL_clip_aux['ntl_clip_id_copy'] = NTL_clip_aux.index
    NTL_clip_aux = NTL_clip_aux.merge(ntl_prediction2, left_on=NTL_clip_aux.index, right_on=ntl_prediction2.index, how='left')
    NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
    NTL_clip_aux.drop('key_0', inplace=True, axis=1)

    ntl_prediction = intersect2.groupby(['ntl_clip_id', 'landuse2016']).sum().loc[:,['intersect_area','disNTL_prime2016']]
    ntl_prediction2 = ntl_prediction.unstack('landuse2016')
    ntl_prediction2.columns = ['area_bg2016', 'area_lr2016', 'area_hr2016', 'area_nr2016',
                                       'NTL_bg2016', 'NTL_lr2016', 'NTL_hr2016', 'NTL_nr2016']
    ntl_prediction2['CNTL2016'] = intersect2.groupby(intersect2.index).max()['CNTL2016']
    NTL_clip_aux['ntl_clip_id_copy'] = NTL_clip_aux.index
    NTL_clip_aux = NTL_clip_aux.merge(ntl_prediction2, left_on=NTL_clip_aux.index, right_on=ntl_prediction2.index, how='left')
    NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
    NTL_clip_aux.drop('key_0', inplace=True, axis=1)

    ntl_prediction = intersect2.groupby(['ntl_clip_id', 'landuse2017']).sum().loc[:,['intersect_area','disNTL_prime2017']]
    ntl_prediction2 = ntl_prediction.unstack('landuse2017')
    ntl_prediction2.columns = ['area_bg2017', 'area_lr2017', 'area_hr2017', 'area_nr2017',
                                       'NTL_bg2017', 'NTL_lr2017', 'NTL_hr2017', 'NTL_nr2017']
    ntl_prediction2['CNTL2017'] = intersect2.groupby(intersect2.index).max()['CNTL2017']
    NTL_clip_aux['ntl_clip_id_copy'] = NTL_clip_aux.index
    NTL_clip_aux = NTL_clip_aux.merge(ntl_prediction2, left_on=NTL_clip_aux.index, right_on=ntl_prediction2.index, how='left')
    NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
    NTL_clip_aux.drop('key_0', inplace=True, axis=1)

    ntl_prediction = intersect2.groupby(['ntl_clip_id', 'landuse2018']).sum().loc[:,['intersect_area','disNTL_prime2018']]
    ntl_prediction2 = ntl_prediction.unstack('landuse2018')
    ntl_prediction2.columns = ['area_bg2018', 'area_lr2018', 'area_hr2018', 'area_nr2018',
                                       'NTL_bg2018', 'NTL_lr2018', 'NTL_hr2018', 'NTL_nr2018']
    ntl_prediction2['CNTL2018'] = intersect2.groupby(intersect2.index).max()['CNTL2018']
    NTL_clip_aux['ntl_clip_id_copy'] = NTL_clip_aux.index
    NTL_clip_aux = NTL_clip_aux.merge(ntl_prediction2, left_on=NTL_clip_aux.index, right_on=ntl_prediction2.index, how='left')
    NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
    NTL_clip_aux.drop('key_0', inplace=True, axis=1)

    NTL_clip_aux.fillna(0, inplace=True)
    NTL_clip_aux['X'] = NTL_clip_aux.geometry.centroid.x
    NTL_clip_aux['Y'] = NTL_clip_aux.geometry.centroid.y

    if mdl == 'nontl':
        NTL_clip_aux.to_csv(results + 'NTL_Level_All_years_' + 'lndus_' + date + '.csv')
    elif mdl == 'ntlmed':
        NTL_clip_aux.to_csv(results + 'NTL_Level_All_years_median_' + 'ntl_annual_incorrected_' + date + '.csv')
    elif mdl == 'ntl_corrected_med_annualByMonth':
        NTL_clip_aux.to_csv(results + 'NTL_Level_All_years_median_' + 'ntl_annual_corrected_' + date + '.csv')
    else:
        NTL_clip_aux.to_csv(results + 'NTL_Level_All_years_median_' + 'ntl_monthly_corrected_' + date + '.csv')

# In the level of land use
# Predict values for all years (when not disaggregating nithglight)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import *
df = pd.DataFrame(columns=['lndus', 'lndus_ntl_annual', 'lndus_ntlhr_annual', 'lndus_ntlhrnr_annual'],
                  index=['censuspop2013', 'pred', 'estpop2014', 'estpop2015', 'estpop2016', 'estpop2017', 'estpop2018', 'RMSE', 'MAE', 'GWR_R2'])
try:
    NTL_clip.set_index('ntl_clip_id', inplace=True)
except:
    print('ntl_clip_id is already index')
# Landuse model
ntl_scale_NTL2 = pd.read_csv(results + 'observations_median_' + 'ntl_annual_incorrected_' + date + '.csv')
NTL_clip_aux = pd.read_csv(results + 'NTL_Level_All_years_median_ntl_annual_incorrected_' + date + '.csv')
gwr_model = pd.read_csv(results + 'GWR_lndus_' + date + '.csv')
NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
gwr_model.set_index('ntl_clip_id', inplace=True)
NTL_clip_aux['ntl_clip_id'] = NTL_clip_aux.index
predict_all_years = NTL_clip_aux.merge(gwr_model, left_on=NTL_clip_aux.index, right_on=gwr_model.index, how='left')
predict_all_years.rename({})

for year in years:
    if int(year) >= 2014:
        predict_all_years['estpop' + year] = predict_all_years['X.Intercept.'] + \
                                             (predict_all_years['area_hr'] * predict_all_years['area_hr' + year])
predict_all_years.drop(['key_0'],inplace=True, axis=1)
mask = predict_all_years['pred'] >=0
df.iloc[1, 0] = predict_all_years.loc[mask, ['pred']].sum()[0]
mask = predict_all_years['estpop2014'] >=0
df.iloc[2, 0] = predict_all_years.loc[mask, ['estpop2014']].sum()[0]
mask = predict_all_years['estpop2015'] >=0
df.iloc[3, 0] = predict_all_years.loc[mask, ['estpop2015']].sum()[0]
mask = predict_all_years['estpop2016'] >=0
df.iloc[4, 0] = predict_all_years.loc[mask, ['estpop2016']].sum()[0]
mask = predict_all_years['estpop2017'] >=0
df.iloc[5, 0] = predict_all_years.loc[mask, ['estpop2017']].sum()[0]
mask = predict_all_years['estpop2018'] >=0
df.iloc[6, 0] = predict_all_years.loc[mask, ['estpop2018']].sum()[0]

df.iloc[-3, 0] = mean_squared_error(ntl_scale_NTL2.Pop2013, predict_all_years.pred, squared=True)
df.iloc[-2:, 0] = mean_absolute_error(ntl_scale_NTL2.Pop2013, predict_all_years.pred)

predict_all_years.to_csv(results + 'predict_all_years_lndus_' + date + '.csv')
predict_all_years.drop('geometry', inplace=True, axis=1)
NTL_clip_aux2 = NTL_clip
predict_all_years.set_index('ntl_clip_id', inplace=True)
NTL_clip_aux3 = NTL_clip_aux2.merge(predict_all_years, left_on=NTL_clip_aux2.index, right_on=predict_all_years.index)
NTL_clip_aux3.drop('key_0', inplace=True, axis=1)

NTL_clip_aux3_noNeg = NTL_clip_aux3
mask = NTL_clip_aux3_noNeg['pred'] < 0
NTL_clip_aux3_noNeg.loc[mask, ['pred']] = 0

for year in years:
    if int(year) >= 2014:
        mask = NTL_clip_aux3_noNeg['estpop' + year] < 0
        NTL_clip_aux3_noNeg.loc[mask, ['estpop' + year]] = 0

NTL_clip_aux3_noNeg['estpop2014change'] = NTL_clip_aux3_noNeg['estpop2014'] - NTL_clip_aux3_noNeg['pred']
NTL_clip_aux3_noNeg['estpop2015change'] = NTL_clip_aux3_noNeg['estpop2015'] - NTL_clip_aux3_noNeg['estpop2014']
NTL_clip_aux3_noNeg['estpop2016change'] = NTL_clip_aux3_noNeg['estpop2016'] - NTL_clip_aux3_noNeg['estpop2015']
NTL_clip_aux3_noNeg['estpop2017change'] = NTL_clip_aux3_noNeg['estpop2017'] - NTL_clip_aux3_noNeg['estpop2016']
NTL_clip_aux3_noNeg['estpop2018change'] = NTL_clip_aux3_noNeg['estpop2018'] - NTL_clip_aux3_noNeg['estpop2017']

# Only positive change
mask = NTL_clip_aux3_noNeg['estpop2014change'] > 0
poschange14 = NTL_clip_aux3_noNeg.loc[mask, ['estpop2014change']].sum()
mask = NTL_clip_aux3_noNeg['estpop2015change'] > 0
poschange15 = NTL_clip_aux3_noNeg.loc[mask, ['estpop2015change']].sum()
mask = NTL_clip_aux3_noNeg['estpop2016change'] > 0
poschange16 = NTL_clip_aux3_noNeg.loc[mask, ['estpop2016change']].sum()
mask = NTL_clip_aux3_noNeg['estpop2017change'] > 0
poschange17 = NTL_clip_aux3_noNeg.loc[mask, ['estpop2017change']].sum()
mask = NTL_clip_aux3_noNeg['estpop2018change'] > 0
poschange18 = NTL_clip_aux3_noNeg.loc[mask, ['estpop2018change']].sum()

vmin = np.min(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014', 'estpop2015', 'estpop2016', 'estpop2017', 'estpop2018']]))
vmax = np.max(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014', 'estpop2015', 'estpop2016', 'estpop2017', 'estpop2018']]))
# vmin=0
# vmax=8000

NTL_clip_aux4 = NTL_clip
ntl_scale_NTL2.set_index('ntl_clip_id_y', inplace=True)
ntl_scale_NTL2.drop('geometry', inplace=True, axis=1)
NTL_clip_aux5 = NTL_clip_aux4.merge(ntl_scale_NTL2, left_on=NTL_clip_aux4.index, right_on=ntl_scale_NTL2.index)
NTL_clip_aux5.drop('key_0', inplace=True, axis=1)
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
NTL_clip_aux5.plot(column='Pop2013', cmap='Spectral_r', linewidth=0.1, ax=axs, edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs.get_xaxis().set_visible(False)
axs.get_yaxis().set_visible(False)
axs.title.set_text('2013 Population')

fig, axs = plt.subplots(2, 3, figsize=(20, 10))
NTL_clip_aux3_noNeg.plot(column='pred', cmap='Spectral_r', linewidth=0.1, ax=axs[0,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,0].get_xaxis().set_visible(False)
axs[0,0].get_yaxis().set_visible(False)
axs[0,0].title.set_text('Estimated population (2013)')
NTL_clip_aux3_noNeg.plot(column='estpop2014', cmap='Spectral_r', linewidth=0.1, ax=axs[0,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,1].get_xaxis().set_visible(False)
axs[0,1].get_yaxis().set_visible(False)
axs[0,1].title.set_text('Estimated population (2014)')
NTL_clip_aux3_noNeg.plot(column='estpop2015', cmap='Spectral_r', linewidth=0.1, ax=axs[0,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,2].get_xaxis().set_visible(False)
axs[0,2].get_yaxis().set_visible(False)
axs[0,2].title.set_text('Estimated population (2015)')
NTL_clip_aux3_noNeg.plot(column='estpop2016', cmap='Spectral_r', linewidth=0.1, ax=axs[1,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,0].get_xaxis().set_visible(False)
axs[1,0].get_yaxis().set_visible(False)
axs[1,0].title.set_text('Estimated population (2016)')
NTL_clip_aux3_noNeg.plot(column='estpop2017', cmap='Spectral_r', linewidth=0.1, ax=axs[1,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,1].get_xaxis().set_visible(False)
axs[1,1].get_yaxis().set_visible(False)
axs[1,1].title.set_text('Estimated population (2017)')
NTL_clip_aux3_noNeg.plot(column='estpop2018', cmap='Spectral_r', linewidth=0.1, ax=axs[1,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,2].get_xaxis().set_visible(False)
axs[1,2].get_yaxis().set_visible(False)
axs[1,2].title.set_text('Estimated population (2018)')
plt.suptitle("Population (Landuse Model)", size=16)
plt.savefig('C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Results/model/Population (Landuse).png', dpi=500, bbox_inches='tight')

vmin = np.min(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014change', 'estpop2015change', 'estpop2016change', 'estpop2017change', 'estpop2018change']]))
vmax = np.max(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014change', 'estpop2015change', 'estpop2016change', 'estpop2017change', 'estpop2018change']]))
# vmin=-6000
# vmax=3000

fig, axs = plt.subplots(2, 3, figsize=(20, 10))
NTL_clip_aux3_noNeg.plot(column='pred', cmap='Spectral_r', linewidth=0.1, ax=axs[0,0], edgecolor='white', legend=True)
axs[0,0].get_xaxis().set_visible(False)
axs[0,0].get_yaxis().set_visible(False)
axs[0,0].title.set_text('Estimated population change (2013)')
NTL_clip_aux3_noNeg.plot(column='estpop2014change', cmap='Spectral_r', linewidth=0.1, ax=axs[0,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,1].get_xaxis().set_visible(False)
axs[0,1].get_yaxis().set_visible(False)
axs[0,1].title.set_text('Estimated population change (2014)')
NTL_clip_aux3_noNeg.plot(column='estpop2015change', cmap='Spectral_r', linewidth=0.1, ax=axs[0,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,2].get_xaxis().set_visible(False)
axs[0,2].get_yaxis().set_visible(False)
axs[0,2].title.set_text('Estimated population change (2015)')
NTL_clip_aux3_noNeg.plot(column='estpop2016change', cmap='Spectral_r', linewidth=0.1, ax=axs[1,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,0].get_xaxis().set_visible(False)
axs[1,0].get_yaxis().set_visible(False)
axs[1,0].title.set_text('Estimated population change (2016)')
NTL_clip_aux3_noNeg.plot(column='estpop2017change', cmap='Spectral_r', linewidth=0.1, ax=axs[1,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,1].get_xaxis().set_visible(False)
axs[1,1].get_yaxis().set_visible(False)
axs[1,1].title.set_text('Estimated population change (2017)')
NTL_clip_aux3_noNeg.plot(column='estpop2018change', cmap='Spectral_r', linewidth=0.1, ax=axs[1,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,2].get_xaxis().set_visible(False)
axs[1,2].get_yaxis().set_visible(False)
axs[1,2].title.set_text('Estimated population change (2018)')
plt.suptitle("Population Change (Landuse Model)", size=16)
plt.savefig('C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Results/Model/Population Change (Landuse).png', dpi=500, bbox_inches='tight')

# Scatter plot of predictions
try:
    ntl_scale_NTL2.set_index('ntl_clip_id_y', inplace=True)
except:
    print('Already done!')
gwrpopLanduse = pd.concat((ntl_scale_NTL2.loc[:, ['Pop2013']], gwr_model['pred']), axis=1)


# # Landuse model + annual ntl *****
# ntl_scale_NTL2 = pd.read_csv(results + 'observations_median_' + 'ntl_annual_incorrected_' + date + '.csv')
# NTL_clip_aux = pd.read_csv(results + 'NTL_Level_All_years_median_ntl_annual_incorrected_' + date + '.csv')
# gwr_model = pd.read_csv(results + 'GWR_median_ntl_annual_incorrected_' + date + '.csv')
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
#                                              # (predict_all_years['CNTL2013'] * predict_all_years['CNTL' + year])
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
# predict_all_years.to_csv(results + 'predict_all_years_ntl_annual_incorrected_' + date + '.csv')
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
# # Only positive change
# mask = NTL_clip_aux3_noNeg['estpop2014change'] > 0
# poschange14 = NTL_clip_aux3_noNeg.loc[mask, ['estpop2014change']].sum()
# mask = NTL_clip_aux3_noNeg['estpop2015change'] > 0
# poschange15 = NTL_clip_aux3_noNeg.loc[mask, ['estpop2015change']].sum()
# mask = NTL_clip_aux3_noNeg['estpop2016change'] > 0
# poschange16 = NTL_clip_aux3_noNeg.loc[mask, ['estpop2016change']].sum()
# mask = NTL_clip_aux3_noNeg['estpop2017change'] > 0
# poschange17 = NTL_clip_aux3_noNeg.loc[mask, ['estpop2017change']].sum()
# mask = NTL_clip_aux3_noNeg['estpop2018change'] > 0
# poschange18 = NTL_clip_aux3_noNeg.loc[mask, ['estpop2018change']].sum()
#
# vmin = np.min(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014', 'estpop2015', 'estpop2016', 'estpop2017', 'estpop2018']]))
# vmax = np.max(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014', 'estpop2015', 'estpop2016', 'estpop2017', 'estpop2018']]))
# # vmin=0
# # vmax=8000
#
# NTL_clip_aux4 = NTL_clip
# ntl_scale_NTL2.set_index('ntl_clip_id_y', inplace=True)
# ntl_scale_NTL2.drop('geometry', inplace=True, axis=1)
# NTL_clip_aux5 = NTL_clip_aux4.merge(ntl_scale_NTL2, left_on=NTL_clip_aux4.index, right_on=ntl_scale_NTL2.index)
# NTL_clip_aux5.drop('key_0', inplace=True, axis=1)
# fig, axs = plt.subplots(1, 1, figsize=(5, 5))
# NTL_clip_aux5.plot(column='Pop2013', cmap='Spectral_r', linewidth=0.1, ax=axs, edgecolor='white', legend=True, vmin=0, vmax=7000)
# axs.get_xaxis().set_visible(False)
# axs.get_yaxis().set_visible(False)
# axs.title.set_text('2013 Population')
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
# plt.suptitle("Population (Landuse-NTL1)", size=16)
# plt.savefig('G:/backupC27152020/Population_Displacement_Final/Results/Model/Population (Landuse-NTL1).png', dpi=500, bbox_inches='tight')
#
# vmin = np.min(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014change', 'estpop2015change', 'estpop2016change', 'estpop2017change', 'estpop2018change']]))
# vmax = np.max(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014change', 'estpop2015change', 'estpop2016change', 'estpop2017change', 'estpop2018change']]))
# # vmin=-6000
# # vmax=3000
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
# plt.suptitle("Population Change (Landuse-NTL1)", size=16)
# plt.savefig('G:/backupC27152020/Population_Displacement_Final/Results/Model/Population Change (Landuse-NTL1).png', dpi=500, bbox_inches='tight')

# Landuse model + annual ntlhr *****
ntl_scale_NTL2 = pd.read_csv(results + 'observations_median_' + 'ntl_annual_incorrected_' + date + '.csv')
NTL_clip_aux = pd.read_csv(results + 'NTL_Level_All_years_median_ntl_annual_incorrected_' + date + '.csv')
gwr_model = pd.read_csv(results + 'GWR_median_ntlhr_annual_incorrected_' + date + '.csv')
NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
gwr_model.set_index('ntl_clip_id', inplace=True)
NTL_clip_aux['ntl_clip_id'] = NTL_clip_aux.index
predict_all_years = NTL_clip_aux.merge(gwr_model, left_on=NTL_clip_aux.index, right_on=gwr_model.index, how='left')
predict_all_years.rename({})
for year in years:
    if int(year) >= 2014:
        predict_all_years['estpop' + year] = predict_all_years['X.Intercept.'] + \
                                             (predict_all_years['NTL2013_hr'] * predict_all_years['NTL_hr' + year]) #+ \
                                             # (predict_all_years['NTL2013_nr'] * predict_all_years['NTL_nr' + year])
                                             # (predict_all_years['CNTL2013'] * predict_all_years['CNTL' + year])

predict_all_years.drop(['key_0'],inplace=True, axis=1)
mask = predict_all_years['pred'] >=0
df.iloc[1, 2] = predict_all_years.loc[mask, ['pred']].sum()[0]
mask = predict_all_years['estpop2014'] >=0
df.iloc[2, 2] = predict_all_years.loc[mask, ['estpop2014']].sum()[0]
mask = predict_all_years['estpop2015'] >=0
df.iloc[3, 2] = predict_all_years.loc[mask, ['estpop2015']].sum()[0]
mask = predict_all_years['estpop2016'] >=0
df.iloc[4, 2] = predict_all_years.loc[mask, ['estpop2016']].sum()[0]
mask = predict_all_years['estpop2017'] >=0
df.iloc[5, 2] = predict_all_years.loc[mask, ['estpop2017']].sum()[0]
mask = predict_all_years['estpop2018'] >=0
df.iloc[6, 2] = predict_all_years.loc[mask, ['estpop2018']].sum()[0]
df.iloc[-3, 2] = mean_squared_error(ntl_scale_NTL2.Pop2013, predict_all_years.pred, squared=True)
df.iloc[-2:, 2] = mean_absolute_error(ntl_scale_NTL2.Pop2013, predict_all_years.pred)
predict_all_years.to_csv(results + 'predict_all_years_ntl_annual_incorrected_' + date + '.csv')

predict_all_years.drop('geometry', inplace=True, axis=1)
NTL_clip_aux2 = NTL_clip
predict_all_years.set_index('ntl_clip_id', inplace=True)
NTL_clip_aux3 = NTL_clip_aux2.merge(predict_all_years, left_on=NTL_clip_aux2.index, right_on=predict_all_years.index)
NTL_clip_aux3.drop('key_0', inplace=True, axis=1)

NTL_clip_aux3_noNeg = NTL_clip_aux3
mask = NTL_clip_aux3_noNeg['pred'] < 0
NTL_clip_aux3_noNeg.loc[mask, ['pred']] = 0

for year in years:
    if int(year) >= 2014:
        mask = NTL_clip_aux3_noNeg['estpop' + year] < 0
        NTL_clip_aux3_noNeg.loc[mask, ['estpop' + year]] = 0

NTL_clip_aux3_noNeg['estpop2014change'] = NTL_clip_aux3_noNeg['estpop2014'] - NTL_clip_aux3_noNeg['pred']
NTL_clip_aux3_noNeg['estpop2015change'] = NTL_clip_aux3_noNeg['estpop2015'] - NTL_clip_aux3_noNeg['estpop2014']
NTL_clip_aux3_noNeg['estpop2016change'] = NTL_clip_aux3_noNeg['estpop2016'] - NTL_clip_aux3_noNeg['estpop2015']
NTL_clip_aux3_noNeg['estpop2017change'] = NTL_clip_aux3_noNeg['estpop2017'] - NTL_clip_aux3_noNeg['estpop2016']
NTL_clip_aux3_noNeg['estpop2018change'] = NTL_clip_aux3_noNeg['estpop2018'] - NTL_clip_aux3_noNeg['estpop2017']

# Only positive change
mask = NTL_clip_aux3_noNeg['estpop2014change'] > 0
poschange14 = NTL_clip_aux3_noNeg.loc[mask, ['estpop2014change']].sum()
mask = NTL_clip_aux3_noNeg['estpop2015change'] > 0
poschange15 = NTL_clip_aux3_noNeg.loc[mask, ['estpop2015change']].sum()
mask = NTL_clip_aux3_noNeg['estpop2016change'] > 0
poschange16 = NTL_clip_aux3_noNeg.loc[mask, ['estpop2016change']].sum()
mask = NTL_clip_aux3_noNeg['estpop2017change'] > 0
poschange17 = NTL_clip_aux3_noNeg.loc[mask, ['estpop2017change']].sum()
mask = NTL_clip_aux3_noNeg['estpop2018change'] > 0
poschange18 = NTL_clip_aux3_noNeg.loc[mask, ['estpop2018change']].sum()

vmin = np.min(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014', 'estpop2015', 'estpop2016', 'estpop2017', 'estpop2018']]))
vmax = np.max(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014', 'estpop2015', 'estpop2016', 'estpop2017', 'estpop2018']]))
vmin=0
vmax=8000

NTL_clip_aux4 = NTL_clip
ntl_scale_NTL2.set_index('ntl_clip_id_y', inplace=True)
ntl_scale_NTL2.drop('geometry', inplace=True, axis=1)
NTL_clip_aux5 = NTL_clip_aux4.merge(ntl_scale_NTL2, left_on=NTL_clip_aux4.index, right_on=ntl_scale_NTL2.index)
NTL_clip_aux5.drop('key_0', inplace=True, axis=1)
fig, axs = plt.subplots(1, 1, figsize=(8, 5))
p = NTL_clip_aux5.plot(column='Pop2013', cmap='Spectral_r', linewidth=0.1, ax=axs, edgecolor='white', legend=True)
axs.get_xaxis().set_visible(False)
axs.get_yaxis().set_visible(False)
axs.title.set_text('Census Population (2013)')
fig.savefig('C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Results/model/Disaggregated Census Population.png', dpi=500, bbox_inches='tight')

fig, axs = plt.subplots(2, 3, figsize=(20, 10))
NTL_clip_aux3_noNeg.plot(column='pred', cmap='Spectral_r', linewidth=0.1, ax=axs[0,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,0].get_xaxis().set_visible(False)
axs[0,0].get_yaxis().set_visible(False)
axs[0,0].title.set_text('Estimated population (2013)')
NTL_clip_aux3_noNeg.plot(column='estpop2014', cmap='Spectral_r', linewidth=0.1, ax=axs[0,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,1].get_xaxis().set_visible(False)
axs[0,1].get_yaxis().set_visible(False)
axs[0,1].title.set_text('Estimated population (2014)')
NTL_clip_aux3_noNeg.plot(column='estpop2015', cmap='Spectral_r', linewidth=0.1, ax=axs[0,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,2].get_xaxis().set_visible(False)
axs[0,2].get_yaxis().set_visible(False)
axs[0,2].title.set_text('Estimated population (2015)')
NTL_clip_aux3_noNeg.plot(column='estpop2016', cmap='Spectral_r', linewidth=0.1, ax=axs[1,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,0].get_xaxis().set_visible(False)
axs[1,0].get_yaxis().set_visible(False)
axs[1,0].title.set_text('Estimated population (2016)')
NTL_clip_aux3_noNeg.plot(column='estpop2017', cmap='Spectral_r', linewidth=0.1, ax=axs[1,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,1].get_xaxis().set_visible(False)
axs[1,1].get_yaxis().set_visible(False)
axs[1,1].title.set_text('Estimated population (2017)')
NTL_clip_aux3_noNeg.plot(column='estpop2018', cmap='Spectral_r', linewidth=0.1, ax=axs[1,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,2].get_xaxis().set_visible(False)
axs[1,2].get_yaxis().set_visible(False)
axs[1,2].title.set_text('Estimated population (2018)')
plt.suptitle("Population (GWR-E)", size=16)
plt.savefig('C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Results/model/Population (Landuse-NTL).png', dpi=500, bbox_inches='tight')

vmin = np.min(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014change', 'estpop2015change', 'estpop2016change', 'estpop2017change', 'estpop2018change']]))
vmax = np.max(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014change', 'estpop2015change', 'estpop2016change', 'estpop2017change', 'estpop2018change']]))
vmin=-5000
vmax=4000

fig, axs = plt.subplots(2, 3, figsize=(20, 10))
p = NTL_clip_aux3_noNeg.plot(column='pred', cmap='Spectral_r', linewidth=0.1, ax=axs[0,0], edgecolor='white', legend=True)
axs[0,0].get_xaxis().set_visible(False)
axs[0,0].get_yaxis().set_visible(False)
axs[0,0].title.set_text('Estimated population change (2013)')
NTL_clip_aux3_noNeg.plot(column='estpop2014change', cmap='Spectral_r', linewidth=0.1, ax=axs[0,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,1].get_xaxis().set_visible(False)
axs[0,1].get_yaxis().set_visible(False)
axs[0,1].title.set_text('Estimated population change (2014)')
NTL_clip_aux3_noNeg.plot(column='estpop2015change', cmap='Spectral_r', linewidth=0.1, ax=axs[0,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,2].get_xaxis().set_visible(False)
axs[0,2].get_yaxis().set_visible(False)
axs[0,2].title.set_text('Estimated population change (2015)')
NTL_clip_aux3_noNeg.plot(column='estpop2016change', cmap='Spectral_r', linewidth=0.1, ax=axs[1,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,0].get_xaxis().set_visible(False)
axs[1,0].get_yaxis().set_visible(False)
axs[1,0].title.set_text('Estimated population change (2016)')
NTL_clip_aux3_noNeg.plot(column='estpop2017change', cmap='Spectral_r', linewidth=0.1, ax=axs[1,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,1].get_xaxis().set_visible(False)
axs[1,1].get_yaxis().set_visible(False)
axs[1,1].title.set_text('Estimated population change (2017)')
NTL_clip_aux3_noNeg.plot(column='estpop2018change', cmap='Spectral_r', linewidth=0.1, ax=axs[1,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,2].get_xaxis().set_visible(False)
axs[1,2].get_yaxis().set_visible(False)
axs[1,2].title.set_text('Estimated population change (2018)')
plt.suptitle("Population Change (GWR-E)", size=16)
plt.savefig('C:/Users/bzn5190/Dropbox (UNC Charlotte)/Population Displacement - paper and resource/Results/model/Population Change (Landuse-NTL).png', dpi=500, bbox_inches='tight')

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

#
# # Landuse model + annual ntlhr *****
# ntl_scale_NTL2 = pd.read_csv(results + 'observations_median_' + 'ntl_annual_incorrected_' + date + '.csv')
# NTL_clip_aux = pd.read_csv(results + 'NTL_Level_All_years_median_ntl_annual_incorrected_' + date + '.csv')
# gwr_model = pd.read_csv(results + 'GWR_median_ntlhrnr_annual_incorrected_' + date + '.csv')
# NTL_clip_aux.set_index('ntl_clip_id_copy', inplace=True)
# gwr_model.set_index('ntl_clip_id', inplace=True)
# NTL_clip_aux['ntl_clip_id'] = NTL_clip_aux.index
# predict_all_years = NTL_clip_aux.merge(gwr_model, left_on=NTL_clip_aux.index, right_on=gwr_model.index, how='left')
# predict_all_years.rename({})
# for year in years:
#     if int(year) >= 2014:
#         predict_all_years['estpop' + year] = predict_all_years['X.Intercept.'] + \
#                                              (predict_all_years['NTL2013_hr'] * predict_all_years['NTL_hr' + year]) + \
#                                              (predict_all_years['NTL2013_nr'] * predict_all_years['NTL_nr' + year])
#                                              # (predict_all_years['CNTL2013'] * predict_all_years['CNTL' + year])
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
# predict_all_years.to_csv(results + 'predict_all_years_ntl_annual_incorrected_' + date + '.csv')
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
# vmin=0
# vmax=8000
#
# NTL_clip_aux4 = NTL_clip
# ntl_scale_NTL2.set_index('ntl_clip_id_y', inplace=True)
# ntl_scale_NTL2.drop('geometry', inplace=True, axis=1)
# NTL_clip_aux5 = NTL_clip_aux4.merge(ntl_scale_NTL2, left_on=NTL_clip_aux4.index, right_on=ntl_scale_NTL2.index)
# NTL_clip_aux5.drop('key_0', inplace=True, axis=1)
# fig, axs = plt.subplots(1, 1, figsize=(5, 5))
# NTL_clip_aux5.plot(column='Pop2013', cmap='Spectral_r', linewidth=0.1, ax=axs, edgecolor='white', legend=True, vmin=0, vmax=7000)
# axs.get_xaxis().set_visible(False)
# axs.get_yaxis().set_visible(False)
# axs.title.set_text('2013 Population')
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
# plt.suptitle("Population (NTL_HrNr)", size=16)
#
# vmin = np.min(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014change', 'estpop2015change', 'estpop2016change', 'estpop2017change', 'estpop2018change']]))
# vmax = np.max(np.array(NTL_clip_aux3_noNeg.loc[:, ['estpop2014change', 'estpop2015change', 'estpop2016change', 'estpop2017change', 'estpop2018change']]))
# vmin=-6000
# vmax=3000
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
# plt.suptitle("Population Change (NTL_HrNr)", size=16)

df.iloc[0, :] = ntl_scale_NTL2.Pop2013.sum()
df.iloc[-1, :] = [0.9829488, 0.9660752 , 0.9845 , 0.9582442]
df.rename({'pred':'estpop2013'}, inplace=True, axis=0)
df.reset_index(inplace=True)

sns.set(rc={'figure.figsize': (20, 11)}, style="whitegrid")
f, axes = plt.subplots(2, 2)
f.subplots_adjust(hspace=.5)
sns.barplot(x=df.columns[1:], y=list(df.iloc[1, 1:]-df.iloc[0, 1:]), ax=axes[0, 0], color='red')
axes[0, 0].set(xlabel='model')
axes[0, 0].set(ylabel='Difference in Overall Population')
axes[0, 0].set(title='Predicted Population of 2013 - Census Population of 2013')
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation = 20)

sns.barplot(x=df.columns[1:], y=list(df.iloc[-2, 1:]), ax=axes[0, 1], color='green')
axes[0, 1].set(xlabel='Model')
axes[0, 1].set(ylabel='Mean Absolute Error')
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation = 20)

sns.barplot(x=df.columns[1:], y=list(df.iloc[-1, 1:]), ax=axes[1, 0], color='blue')
axes[1, 0].set(xlabel='Model')
axes[1, 0].set(ylabel='GWR_R2')
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation = 20)

sns.lineplot(x=range(2013, 2019), y=list(df.iloc[1:-3, 1]), ax=axes[1, 1], color='green')
axes[1, 1].set(xlabel='Year')
axes[1, 1].set(ylabel='Population')

sns.lineplot(x=range(2013, 2019), y=list(df.iloc[1:-3, 2]), ax=axes[1, 1], color='orange')
axes[1, 1].set(xlabel='Year')
axes[1, 1].set(ylabel='Population')

sns.lineplot(x=range(2013, 2019), y=list(df.iloc[1:-3, 3]), ax=axes[1, 1], color='blue')
axes[1, 1].set(xlabel='Year')
axes[1, 1].set(ylabel='Population')

sns.lineplot(x=range(2013, 2019), y=list(df.iloc[1:-3, 4]), ax=axes[1, 1], color='red')
axes[1, 1].set(xlabel='Year')
axes[1, 1].set(ylabel='Population')
axes[1, 1].legend(df.columns[1:], loc='lower left')
axes[1, 1].set(title='Predictions over 5 years')


totalpop = df.iloc[1:-3, [1,3]]
# totalpop.columns = ['Landuse', 'Landuse-NTL1', 'Landuse-NTL2']
totalpop.columns = ['GWR-A', 'GWR-E']
totalpop.index = ['2013', '2014','2015','2016','2017','2018']
sns.set(rc={'figure.figsize': (6, 7)})
sns.set_style("whitegrid", {'axes.grid' : False})
ax = totalpop.plot(kind="bar", width=0.8)
plt.xlabel(xlabel="Year", size=15)
plt.ylabel(ylabel = "Estimated Population", size=15)
plt.xticks(size=15, rotation=0)
plt.yticks(size=15)
plt.title('Estimated Population by Year', size=18)
plt.legend(bbox_to_anchor=(1, 1),borderaxespad=0)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.text(x+width/2,
            y+height/2,
            '{:.0f}'.format(height),
            horizontalalignment='center',
            verticalalignment='center',
            color='w',
            weight='bold',
            size=15, rotation=90)
plt.savefig('G:/backupC27152020/Population_Displacement_Final/Results/Model/Total_population.png', dpi=500, bbox_inches='tight')

fig, axs = plt.subplots(1, 1, figsize=(20, 20))
intersect2.plot(column='disPop2013_prime', cmap='Spectral_r', linewidth=0.1, ax=axs, edgecolor='white', legend=True, scheme='quantiles', k=46)
axs.get_xaxis().set_visible(False)
axs.get_yaxis().set_visible(False)
axs.title.set_text('2013 Population Estimation')

intersect2.loc[:, ['geometry','disPop2013_prime']].to_file(temp + 'microresolutionpopulation2013.shp')

fig, axs = plt.subplots(1, 1, figsize=(20, 20))
intersect2.plot(column='disNTL_prime2013', cmap='Spectral_r', linewidth=0.1, ax=axs, edgecolor='white', legend=True, scheme='quantiles', k=50)
axs.get_xaxis().set_visible(False)
axs.get_yaxis().set_visible(False)
axs.title.set_text('2013 Nightlight Estimation')

print('Spatial Multiple Linear Regression for disaggregating population 2013:')
model_pop_ntl_spatial = ols(
    "disPop2013_prime ~  landuse2014 + clusters2013pop_HH + clusters2013pop_HL + clusters2013pop_LH + clusters2013pop_NS",
    intersect2).fit()
print(model_pop_ntl_spatial.summary())
print("\nRetrieving manually the parameter estimates:")
print(model_pop_ntl_spatial._results.params)

# evaluation
import pandas as pd
import geopandas as gp
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as ani

field = 'G:/backupC27152020/Population_Displacement_Final/Resources/Field/'
figures = 'G:/backupC27152020/Population_Displacement_Final/Results/Model/'

# Preprocessing data
df = pd.read_csv(field + 'mosul_processed.csv')

years = ['2014', '2015', '2016', '2017', '2018']
months = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
coord = ['lat', 'long']

# families = df.groupby(['id', 'year', 'month', 'type']).max()['families'].unstack(level=[1, 2])
# columns = []
# for year in years:
#     for month in months:
#         columns.append(year + '_' + month)
#
# columns.remove('2018_11')
# families.columns = columns
# families.reset_index(inplace=True)
# families.set_index('id', inplace=True)
#
# locations = df.groupby(['id', 'year', 'month', 'type']).mean().loc[:, ['lat', 'long']].groupby('id').mean()
# data = families.join(locations, lsuffix='_families', rsuffix='_locations')
#
# mask = data['type'] == 'displacement'
# displacements = data[mask]
# mask = data['type'] == 'return'
# returns = data[mask]
#
# displacements.to_csv(field + 'displacements.csv')
# returns.to_csv(field + 'return.csv')
#

# Now we need to read the data and the neighborhoods we created
aux1 = ['id', 'type']
for year in years:
    if int(year) >= 2014:
        for month in months:
            aux1.append('Ret' + year + '_' + month)
aux1.remove('Ret2018_11')
aux1.append('lat')
aux1.append('long')
aux1.append('geometry')

aux2 = ['id', 'type']
for year in years:
    if int(year) >= 2014:
        for month in months:
            aux2.append('Disp' + year + '_' + month)
aux2.remove('Disp2018_11')
aux2.append('lat')
aux2.append('long')
aux2.append('geometry')

returns = gp.read_file(field + 'return2.shp')
returns.columns = aux1
returnsUnits = gp.read_file(field + 'field_ntl_thiessen.shp')
displacements = gp.read_file(field + 'displacement2.shp')
displacements.columns = aux2
displacementsUnits = gp.read_file(field + 'field_ntl_thiessen_displaced.shp')

ret = returnsUnits.merge(returns, left_on=returnsUnits.NEAR_FID, right_on=returns.index, how='left')
ret.drop(['key_0', 'geometry_y'], inplace=True, axis=1)
ret.rename({'NEAR_FID':'GEOID', 'geometry_x':'geometry'}, inplace=True, axis=1)
ret = gp.GeoDataFrame(ret)
ret['area'] = ret.geometry.area
ret['areaAll'] = ret.groupby('GEOID')['area'].transform('sum')
ret['AreaProp'] = ret['area'] / ret['areaAll']
for item in aux1[2:-3]:
    ret[item + 'Prop'] = ret[item] * ret['AreaProp']

disp = displacementsUnits.merge(displacements, left_on=displacementsUnits.NEAR_FID, right_on=displacements.index, how='left')
disp.drop(['key_0', 'geometry_y'], inplace=True, axis=1)
disp.rename({'NEAR_FID':'GEOID', 'geometry_x':'geometry'}, inplace=True, axis=1)
disp = gp.GeoDataFrame(disp)
disp['area'] = disp.geometry.area
disp['areaAll'] = disp.groupby('GEOID')['area'].transform('sum')
disp['AreaProp'] = disp['area'] / disp['areaAll']
for item in aux2[2:-3]:
    disp[item + 'Prop'] = disp[item] * disp['AreaProp']

# Monthly change Return and Displacement
disp['RetDisp2017_1Prop'] = ret['Ret2017_1Prop'] + disp['Disp2017_1Prop']
disp['RetDisp2017_2Prop'] = ret['Ret2017_2Prop'] + disp['Disp2017_2Prop']
disp['RetDisp2017_3Prop'] = ret['Ret2017_3Prop'] + disp['Disp2017_3Prop']
disp['RetDisp2017_4Prop'] = ret['Ret2017_4Prop'] + disp['Disp2017_4Prop']
disp['RetDisp2017_5Prop'] = ret['Ret2017_5Prop'] + disp['Disp2017_5Prop']
disp['RetDisp2017_6Prop'] = ret['Ret2017_6Prop'] + disp['Disp2017_6Prop']
disp['RetDisp2017_7Prop'] = ret['Ret2017_7Prop'] + disp['Disp2017_7Prop']
disp['RetDisp2017_8Prop'] = ret['Ret2017_8Prop'] + disp['Disp2017_8Prop']
disp['RetDisp2017_9Prop'] = ret['Ret2017_9Prop'] + disp['Disp2017_9Prop']
disp['RetDisp2017_10Prop'] = ret['Ret2017_10Prop'] + disp['Disp2017_10Prop']
disp['RetDisp2017_11Prop'] = ret['Ret2017_11Prop'] + disp['Disp2017_11Prop']
disp['RetDisp2017_12Prop'] = ret['Ret2017_12Prop'] + disp['Disp2017_12Prop']
disp['RetDisp2018_1Prop'] = ret['Ret2018_1Prop'] + disp['Disp2018_1Prop']
disp['RetDisp2018_2Prop'] = ret['Ret2018_2Prop'] + disp['Disp2018_2Prop']
disp['RetDisp2018_3Prop'] = ret['Ret2018_3Prop'] + disp['Disp2018_3Prop']
disp['RetDisp2018_4Prop'] = ret['Ret2018_4Prop'] + disp['Disp2018_4Prop']
disp['RetDisp2018_5Prop'] = ret['Ret2018_5Prop'] + disp['Disp2018_5Prop']
disp['RetDisp2018_6Prop'] = ret['Ret2018_6Prop'] + disp['Disp2018_6Prop']
disp['RetDisp2018_7Prop'] = ret['Ret2018_7Prop'] + disp['Disp2018_7Prop']
disp['RetDisp2018_8Prop'] = ret['Ret2018_8Prop'] + disp['Disp2018_8Prop']
disp['RetDisp2018_9Prop'] = ret['Ret2018_9Prop'] + disp['Disp2018_9Prop']
disp['RetDisp2018_10Prop'] = ret['Ret2018_10Prop'] + disp['Disp2018_10Prop']
disp['RetDisp2018_12Prop'] = ret['Ret2018_12Prop'] + disp['Disp2018_12Prop']

# Monthly change Return
ret['Ret2017_1Prop_Mchange'] = ret['Ret2017_1Prop'] - ret['Ret2016_12Prop']
ret['Ret2017_2Prop_Mchange'] = ret['Ret2017_2Prop'] - ret['Ret2017_1Prop']
ret['Ret2017_3Prop_Mchange'] = ret['Ret2017_3Prop'] - ret['Ret2017_2Prop']
ret['Ret2017_4Prop_Mchange'] = ret['Ret2017_4Prop'] - ret['Ret2017_3Prop']
ret['Ret2017_5Prop_Mchange'] = ret['Ret2017_5Prop'] - ret['Ret2017_4Prop']
ret['Ret2017_6Prop_Mchange'] = ret['Ret2017_6Prop'] - ret['Ret2017_5Prop']
ret['Ret2017_7Prop_Mchange'] = ret['Ret2017_7Prop'] - ret['Ret2017_6Prop']
ret['Ret2017_8Prop_Mchange'] = ret['Ret2017_8Prop'] - ret['Ret2017_7Prop']
ret['Ret2017_9Prop_Mchange'] = ret['Ret2017_9Prop'] - ret['Ret2017_8Prop']
ret['Ret2017_10Prop_Mchange'] = ret['Ret2017_10Prop'] - ret['Ret2017_9Prop']
ret['Ret2017_11Prop_Mchange'] = ret['Ret2017_11Prop'] - ret['Ret2017_10Prop']
ret['Ret2017_12Prop_Mchange'] = ret['Ret2017_12Prop'] - ret['Ret2017_11Prop']
ret['Ret2018_1Prop_Mchange'] = ret['Ret2018_1Prop'] - ret['Ret2017_12Prop']
ret['Ret2018_2Prop_Mchange'] = ret['Ret2018_2Prop'] - ret['Ret2018_1Prop']
ret['Ret2018_3Prop_Mchange'] = ret['Ret2018_3Prop'] - ret['Ret2018_2Prop']
ret['Ret2018_4Prop_Mchange'] = ret['Ret2018_4Prop'] - ret['Ret2018_3Prop']
ret['Ret2018_5Prop_Mchange'] = ret['Ret2018_5Prop'] - ret['Ret2018_4Prop']
ret['Ret2018_6Prop_Mchange'] = ret['Ret2018_6Prop'] - ret['Ret2018_5Prop']
ret['Ret2018_7Prop_Mchange'] = ret['Ret2018_7Prop'] - ret['Ret2018_6Prop']
ret['Ret2018_8Prop_Mchange'] = ret['Ret2018_8Prop'] - ret['Ret2018_7Prop']
ret['Ret2018_9Prop_Mchange'] = ret['Ret2018_9Prop'] - ret['Ret2018_8Prop']
ret['Ret2018_10Prop_Mchange'] = ret['Ret2018_10Prop'] - ret['Ret2018_9Prop']
ret['Ret2018_12Prop_Mchange'] = ret['Ret2018_12Prop'] - ret['Ret2018_10Prop']

# Monthly change Displacement
disp['Disp2017_1Prop_Mchange'] = disp['Disp2017_1Prop'] - disp['Disp2016_12Prop']
disp['Disp2017_2Prop_Mchange'] = disp['Disp2017_2Prop'] - disp['Disp2017_1Prop']
disp['Disp2017_3Prop_Mchange'] = disp['Disp2017_3Prop'] - disp['Disp2017_2Prop']
disp['Disp2017_4Prop_Mchange'] = disp['Disp2017_4Prop'] - disp['Disp2017_3Prop']
disp['Disp2017_5Prop_Mchange'] = disp['Disp2017_5Prop'] - disp['Disp2017_4Prop']
disp['Disp2017_6Prop_Mchange'] = disp['Disp2017_6Prop'] - disp['Disp2017_5Prop']
disp['Disp2017_7Prop_Mchange'] = disp['Disp2017_7Prop'] - disp['Disp2017_6Prop']
disp['Disp2017_8Prop_Mchange'] = disp['Disp2017_8Prop'] - disp['Disp2017_7Prop']
disp['Disp2017_9Prop_Mchange'] = disp['Disp2017_9Prop'] - disp['Disp2017_8Prop']
disp['Disp2017_10Prop_Mchange'] = disp['Disp2017_10Prop'] - disp['Disp2017_9Prop']
disp['Disp2017_11Prop_Mchange'] = disp['Disp2017_11Prop'] - disp['Disp2017_10Prop']
disp['Disp2017_12Prop_Mchange'] = disp['Disp2017_12Prop'] - disp['Disp2017_11Prop']
disp['Disp2018_1Prop_Mchange'] = disp['Disp2018_1Prop'] - disp['Disp2017_12Prop']
disp['Disp2018_2Prop_Mchange'] = disp['Disp2018_2Prop'] - disp['Disp2018_1Prop']
disp['Disp2018_3Prop_Mchange'] = disp['Disp2018_3Prop'] - disp['Disp2018_2Prop']
disp['Disp2018_4Prop_Mchange'] = disp['Disp2018_4Prop'] - disp['Disp2018_3Prop']
disp['Disp2018_5Prop_Mchange'] = disp['Disp2018_5Prop'] - disp['Disp2018_4Prop']
disp['Disp2018_6Prop_Mchange'] = disp['Disp2018_6Prop'] - disp['Disp2018_5Prop']
disp['Disp2018_7Prop_Mchange'] = disp['Disp2018_7Prop'] - disp['Disp2018_6Prop']
disp['Disp2018_8Prop_Mchange'] = disp['Disp2018_8Prop'] - disp['Disp2018_7Prop']
disp['Disp2018_9Prop_Mchange'] = disp['Disp2018_9Prop'] - disp['Disp2018_8Prop']
disp['Disp2018_10Prop_Mchange'] = disp['Disp2018_10Prop'] - disp['Disp2018_9Prop']
disp['Disp2018_12Prop_Mchange'] = disp['Disp2018_12Prop'] - disp['Disp2018_10Prop']

# Monthly change Return and Displacement
disp['Sum2017_1Prop'] = ret['Ret2017_1Prop_Mchange'] + disp['Disp2017_1Prop_Mchange']
disp['Sum2017_2Prop'] = ret['Ret2017_2Prop_Mchange'] + disp['Disp2017_2Prop_Mchange']
disp['Sum2017_3Prop'] = ret['Ret2017_3Prop_Mchange'] + disp['Disp2017_3Prop_Mchange']
disp['Sum2017_4Prop'] = ret['Ret2017_4Prop_Mchange'] + disp['Disp2017_4Prop_Mchange']
disp['Sum2017_5Prop'] = ret['Ret2017_5Prop_Mchange'] + disp['Disp2017_5Prop_Mchange']
disp['Sum2017_6Prop'] = ret['Ret2017_6Prop_Mchange'] + disp['Disp2017_6Prop_Mchange']
disp['Sum2017_7Prop'] = ret['Ret2017_7Prop_Mchange'] + disp['Disp2017_7Prop_Mchange']
disp['Sum2017_8Prop'] = ret['Ret2017_8Prop_Mchange'] + disp['Disp2017_8Prop_Mchange']
disp['Sum2017_9Prop'] = ret['Ret2017_9Prop_Mchange'] + disp['Disp2017_9Prop_Mchange']
disp['Sum2017_10Prop'] = ret['Ret2017_10Prop_Mchange'] + disp['Disp2017_10Prop_Mchange']
disp['Sum2017_11Prop'] = ret['Ret2017_11Prop_Mchange'] + disp['Disp2017_11Prop_Mchange']
disp['Sum2017_12Prop'] = ret['Ret2017_12Prop_Mchange'] + disp['Disp2017_12Prop_Mchange']
disp['Sum2018_1Prop'] = ret['Ret2018_1Prop_Mchange'] + disp['Disp2018_1Prop_Mchange']
disp['Sum2018_2Prop'] = ret['Ret2018_2Prop_Mchange'] + disp['Disp2018_2Prop_Mchange']
disp['Sum2018_3Prop'] = ret['Ret2018_3Prop_Mchange'] + disp['Disp2018_3Prop_Mchange']
disp['Sum2018_4Prop'] = ret['Ret2018_4Prop_Mchange'] + disp['Disp2018_4Prop_Mchange']
disp['Sum2018_5Prop'] = ret['Ret2018_5Prop_Mchange'] + disp['Disp2018_5Prop_Mchange']
disp['Sum2018_6Prop'] = ret['Ret2018_6Prop_Mchange'] + disp['Disp2018_6Prop_Mchange']
disp['Sum2018_7Prop'] = ret['Ret2018_7Prop_Mchange'] + disp['Disp2018_7Prop_Mchange']
disp['Sum2018_8Prop'] = ret['Ret2018_8Prop_Mchange'] + disp['Disp2018_8Prop_Mchange']
disp['Sum2018_9Prop'] = ret['Ret2018_9Prop_Mchange'] + disp['Disp2018_9Prop_Mchange']
disp['Sum2018_10Prop'] = ret['Ret2018_10Prop_Mchange'] + disp['Disp2018_10Prop_Mchange']
disp['Sum2018_12Prop'] = ret['Ret2018_12Prop_Mchange'] + disp['Disp2018_12Prop_Mchange']


# barchart: current returnee and displaced families
overall_retune_by_month = ret.loc[:, [j + 'Prop' for j in ['Ret2017_' + str(i) for i in range(1,13)]] + \
    [j + 'Prop' for j in ['Ret2018_' + str(i) for i in range(1,11)] + ['Ret2018_12']]].sum()
overall_retune_by_month = pd.DataFrame(overall_retune_by_month)
overall_retune_by_month.rename({0:'Current Returnee'}, axis=1, inplace=True)

overall_disp_by_month = disp.loc[:, [j + 'Prop' for j in ['Disp2017_' + str(i) for i in range(1,13)]] + \
    [j + 'Prop' for j in ['Disp2018_' + str(i) for i in range(1,11)] + ['Disp2018_12']]].sum()
overall_disp_by_month = pd.DataFrame(overall_disp_by_month)
overall_disp_by_month.rename({0:'Current Displaced'}, axis=1, inplace=True)

overall_by_month = disp.loc[:, [j + 'Prop' for j in ['RetDisp2017_' + str(i) for i in range(1,13)]] + \
    [j + 'Prop' for j in ['RetDisp2018_' + str(i) for i in range(1,11)] + ['RetDisp2018_12']]].sum()
overall_by_month = pd.DataFrame(overall_by_month)
overall_by_month.rename({0:'Current Returnee-Displaced'}, axis=1, inplace=True)

aux0 = overall_retune_by_month.transpose()
aux0.columns = ['2017_' + str(i) for i in range(1,13)] + ['2018_' + str(i) for i in range(1,11)]+['2018_12']
# aux0.reset_index(inplace=True)
aux = overall_disp_by_month.transpose()
aux.columns = ['2017_' + str(i) for i in range(1,13)] + ['2018_' + str(i) for i in range(1,11)]+['2018_12']
# aux.reset_index(inplace=True)
aux2 = overall_by_month.transpose()
aux2.columns = ['2017_' + str(i) for i in range(1,13)] + ['2018_' + str(i) for i in range(1,11)]+['2018_12']
# aux2.reset_index(inplace=True)

overall = pd.concat((aux0,aux,aux2), axis=0)

sns.set(rc={'figure.figsize': (25, 10)})
sns.set_style("whitegrid", {'axes.grid' : False})
overall.transpose().plot(kind="bar", width=0.8)
plt.xlabel(xlabel="Month", size=20)
plt.ylabel(ylabel = "Number of Families", size=20)
plt.xticks(rotation = 45, size=13)
plt.yticks(size=15)
plt.title('Current Returnee-Displaced Family Population', size=20)
plt.savefig(figures + 'Current_Returnee_Population_barchart.png', dpi=500, bbox_inches='tight')

avg_2017 = overall.iloc[:, 0:12].mean(axis=1)
avg_2018 = overall.iloc[:, 12:].mean(axis=1)


# Maps
average_returnee_2017 = ret.loc[:, [j + 'Prop' for j in ['Ret2017_' + str(i) for i in range(1,13)]]].mean(axis=1)
average_returnee_2017 = pd.concat((ret['geometry'],average_returnee_2017), axis=1)
average_returnee_2017.rename({0:'average_returnee_2017'}, inplace=True,axis=1)

average_returnee_2018 = ret.loc[:, [j + 'Prop' for j in ['Ret2018_' + str(i) for i in range(1,11)] + ['Ret2018_12']]].mean(axis=1)
average_returnee_2018 = pd.concat((ret['geometry'],average_returnee_2018), axis=1)
average_returnee_2018.rename({0:'average_returnee_2018'}, inplace=True,axis=1)

average_returnee_displaced_2017 = disp.loc[:, [j + 'Prop' for j in ['RetDisp2017_' + str(i) for i in range(1,13)]]].mean(axis=1)
average_returnee_displaced_2017 = pd.concat((ret['geometry'],average_returnee_displaced_2017), axis=1)
average_returnee_displaced_2017.rename({0:'average_returnee_displaced_2017'}, inplace=True,axis=1)

average_returnee_displaced_2018 = disp.loc[:, [j + 'Prop' for j in ['RetDisp2018_' + str(i) for i in range(1,11)] + ['RetDisp2018_12']]].mean(axis=1)
average_returnee_displaced_2018 = pd.concat((ret['geometry'],average_returnee_displaced_2018), axis=1)
average_returnee_displaced_2018.rename({0:'average_returnee_displaced_2018'}, inplace=True,axis=1)


# average_returnee_2017.plot(column='average_returnee_2017', cmap='Spectral_r', linewidth=0.1, edgecolor='white', legend=True)
# plt.title('average_returnee_2017')
# average_returnee_2018.plot(column='average_returnee_2018', cmap='Spectral_r', linewidth=0.1, edgecolor='white', legend=True)
# plt.title('average_returnee_2018')

vmin=-5000
vmax=3000

fig, axs = plt.subplots(1, 3, figsize=(20, 4))
# p = NTL_clip_aux3_noNeg.plot(column='estpop2017change', cmap='Spectral_r', linewidth=0.1, ax=axs[0,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
# axs[0,0].get_xaxis().set_visible(False)
# axs[0,0].get_yaxis().set_visible(False)
# axs[0,0].title.set_text('Estimated population change (2017)')
p = NTL_clip_aux3_noNeg.plot(column='estpop2018change', cmap='Spectral_r', linewidth=0.1, ax=axs[0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0].get_xaxis().set_visible(False)
axs[0].get_yaxis().set_visible(False)
axs[0].title.set_text('Estimated population change (GWR-E 2018)')
# average_returnee_2017.plot(column='average_returnee_2017', cmap='Spectral_r', linewidth=0.1, edgecolor='white', legend=True, ax=axs[0,1])
# axs[0,1].get_xaxis().set_visible(False)
# axs[0,1].get_yaxis().set_visible(False)
# axs[0,1].title.set_text('Average Returnee Family Population (2017)')
average_returnee_2018.plot(column='average_returnee_2018', cmap='Spectral_r', linewidth=0.1, edgecolor='white', legend=True, ax=axs[1])
axs[1].get_xaxis().set_visible(False)
axs[1].get_yaxis().set_visible(False)
axs[1].title.set_text('Average Returnee Family Population (2018)')
# average_returnee_displaced_2017.plot(column='average_returnee_displaced_2017', cmap='Spectral_r', linewidth=0.1, edgecolor='white', legend=True, ax=axs[0,2])
# axs[0,2].get_xaxis().set_visible(False)
# axs[0,2].get_yaxis().set_visible(False)
# axs[0,2].title.set_text('Average Returnee-Displaced Family Population (2017)')
average_returnee_displaced_2018.plot(column='average_returnee_displaced_2018', cmap='Spectral_r', linewidth=0.1, edgecolor='white', legend=True, ax=axs[2])
axs[2].get_xaxis().set_visible(False)
axs[2].get_yaxis().set_visible(False)
axs[2].title.set_text('Average Returnee-Displaced Family Population (2018)')
plt.savefig(figures + 'Evaluation-maps.png', dpi=500, bbox_inches='tight')

# for evaluation
returnees = NTL_clip_aux3_noNeg.loc[:, ['estpop2017change', 'estpop2018change', 'coord.x', 'coord.y','geometry']]
mask = returnees['estpop2017change'] < 0
returnees.loc[mask,['estpop2017change']] = 0
returnees = pd.concat((returnees,average_returnee_displaced_2017['average_returnee_displaced_2017']), axis=1)
returnees = pd.concat((returnees,average_returnee_2017['average_returnee_2017']), axis=1)
mask = returnees['estpop2018change'] < 0
returnees.loc[mask,['estpop2018change']] = 0
returnees = pd.concat((returnees,average_returnee_displaced_2018['average_returnee_displaced_2018']), axis=1)
returnees = pd.concat((returnees,average_returnee_2018['average_returnee_2018']), axis=1)
returnees.to_csv(field + 'returnees_cross_correlation.csv')

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
#                                              (predict_all_years['area_lr'] * predict_all_years['area_lr' + year]) + \
#                                              (predict_all_years['area_hr'] * predict_all_years['area_hr' + year]) + \
#                                              (predict_all_years['area_nr'] * predict_all_years['area_nr' + year]) + \
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




