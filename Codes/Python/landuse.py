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
import os
# import arcpy
# from arcpy import env
# from arcpy.sa import *
# from simpledbf import Dbf5
from os import listdir
from os.path import isfile, join
# from simpledbf import Dbf5
import geopandas as gp
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

image_path = 'G:/backupC27152020/Population_Displacement_Final/Resources/VHR/images/'
landuse_path = 'G:/backupC27152020/Population_Displacement_Final/Resources/VHR/landuse/'
viirs_path = 'G:/backupC27152020/Population_Displacement_Final/Resources/VIIRS/VNP46A2/'
geodb_path = 'G:/backupC27152020/Population_Displacement_Final/Resources/poulation_disp.gdb/Data/'
temp = 'G:/backupC27152020/Population_Displacement_Final/Resources/Temp/'
results = 'G:/backupC27152020/Population_Displacement_Final/Resources/Results/'

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
#         arcpy.Resample_management(inputraster,outpuraster,"50 50", "Majority")
#
# # convert to point
# for year in years:
#     if int(year) < 2014:
#         arcpy.RasterToPoint_conversion(viirs_path + 'ntl_corrected_med' + year + '.tif', temp + 'ntl_corrected_med' + year, "VALUE")
#     else:
#         arcpy.RasterToPoint_conversion(viirs_path + 'ntl_corrected_med' + year + '.tif', temp + 'ntl_corrected_med' + year, "VALUE")
#         # arcpy.RasterToPoint_conversion(image_path + 'labelrsm' + year + '.tif', temp + 'label' + year, "VALUE")

NTL = gp.read_file(temp + 'NTL.shp')
NTL['ntl_id'] = NTL.index + 1
NTL['ntl_area'] = NTL.geometry.area
landuse = gp.read_file(temp + 'landuse.shp')
landuse['landuse_id'] = landuse.index + 1
landuse['landuse_area'] = landuse.geometry.area
census = gp.read_file(temp + 'census.shp')
census['census_id'] = census.index + 1
census['census_area'] = census.geometry.area
boundary = gp.read_file(temp + 'CensusBoundary.shp')

# assign the values from point features to areal
for year in years:
    image = gp.read_file(temp + 'ntlmed' + year + '.shp')
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
NTL_clip['ntl_clip_area'] = NTL_clip.geometry.area
landuse_clip = gp.clip(landuse, boundary)
landuse_clip['landuse_clip_id'] = landuse_clip.index + 1
landuse_clip['landuse_clip_area'] = landuse_clip.geometry.area

intersect1 = gp.overlay(census, NTL_clip, how='intersection')
intersect2 = gp.overlay(intersect1, landuse_clip, how='intersection')

intersect2['intersect_id'] = intersect2.index + 1
intersect2['intersect_area'] = intersect2.geometry.area

# aux = intersect2.loc[:, ['landuse2014', 'intersect_area']]
#
# aux['area_lr'] = 0
# aux['area_hr'] = 0
# aux['area_nr'] = 0
# aux['area_bg'] = 0
# mask = aux.loc[:, ['landuse2014']] == 1
# aux[aux.loc[mask, ['area_bg']]] = aux.loc[mask, ['intersect_area']]
#
# mask = aux.loc[:, ['landuse2014']] == 2
# aux.loc[mask, ['area_lr']] = aux.loc[mask, ['intersect_area']]
# mask = aux.loc[:, ['landuse2014']] == 3
# aux.loc[mask, ['area_hr']] = aux.loc[mask, ['intersect_area']]
# mask = aux.loc[:, ['landuse2014']] == 4
# aux.loc[mask, ['area_nr']] = aux.loc[mask, ['intersect_area']]
#
# aux['area_lr_p'] = aux['area_lr'] / (aux['area_lr'] + aux['area_hr'] + aux['area_nr']
#                                                    + aux['area_bg'])
# aux['area_hr_p'] = aux['area_hr'] / (aux['area_lr'] + aux['area_hr'] + aux['area_nr']
#                                                    + aux['area_bg'])
# aux['area_nr_p'] = aux['area_nr'] / (aux['area_lr'] + aux['area_hr'] + aux['area_nr']
#                                                    + aux['area_bg'])
# aux['area_bg_p'] = aux['area_bg'] / (aux['area_lr'] + aux['area_hr'] + aux['area_nr']
#                                                    + aux['area_bg'])
# Calculate NTL by considering boundary of the city
for year in years:
    intersect2['CNTL' + year] = (intersect2['ntl_clip_area'] /
                                             intersect2['ntl_area'])*intersect2['NTL' + year]

# target NTL Area over NTL
intersect2['AONTL'] = intersect2['intersect_area'] / intersect2['ntl_clip_area']
# Target NTL
for year in years:
    intersect2['TNTL' + year] = intersect2['AONTL'] * intersect2['CNTL' + year]

# Calculate residential area
for year in years:
    if int(year) >= 2014:
        intersect2['intersect_area2'] = intersect2['intersect_area']
        mask = ((intersect2['landuse' + year] == 1) | (intersect2['landuse' + year] == 4))
        intersect2.loc[mask, ['intersect_area2']] = 0
        areas = intersect2.groupby(['census_id']).sum().astype('float64')
        # pops = intersect2.groupby(['census_id']).max().astype('float64')
        # mask2 = ((areas['intersect_area2'] == 0) & (pops['estPop2013'] != 0))
        # intersect2 = intersect2.reset_index()
        # intersect2 = intersect2.set_index(['census_id'])
        intersect2 = intersect2.join(areas['intersect_area2'], on=['census_id'], how='left', lsuffix='_caller', rsuffix='_other')
        intersect2['census_res_area' + year] = intersect2['intersect_area2_other']
        intersect2.drop('intersect_area2_other', inplace=True, axis=1)
        # intersect2.rename({'intersect_area2_caller': 'intersect_area2'}, inplace=True, axis=1)

intersect2 = intersect2.reset_index()

# Target unit landuse area / Census unit residential area
for year in years:
    if int(year) >= 2014:

        # AOCRA: Area Over Census Residential Area

        # intersect2['AOCRA' + year] = [(np.array(intersect2.loc[mask, ['intersect_area']]) /
        #                                np.array(intersect2.loc[mask,['census_res_area' + year]]))
        #                               if x != 0 else 0 for x in intersect2['census_res_area' + year]]
        # mask = intersect2['census_res_area' + year] != 0
        # intersect2.loc[mask, ['AOCRA' + year]] = \
        intersect2['AOCRA' + year] = (np.array(intersect2.loc[:, ['intersect_area']]) /
                                      np.array(intersect2.loc[:,['census_res_area' + year]]))

# for year in years:
#     if int(year) >= 2014:
#         mask = (intersect2['AOCRA' + year].isna()) & (intersect2['estPop2013'] != 0)
#
#         intersect2.loc[mask, ['census_area_res' + year]] = 0
#         areas = intersect2.groupby(['census_id', 'landuse' + year]).sum().astype('float64')['census_area_res' + year]
#         intersect2 = intersect2.reset_index()
#         intersect2 = intersect2.set_index(['census_id', 'landuse' + year])
#         intersect2 = intersect2.join(areas, on=['census_id', 'landuse' + year], how='left'
#                                      , lsuffix='_caller', rsuffix='_other')

for year in years:
    if int(year) == 2013:
        intersect2['target_pop' + year] = intersect2['AOCRA2014']*intersect2['estPop2013']

# for year in years:
#     if int(year) == 2013:
#         # we have to use 2014 landuse for 2013
#         mask = (intersect2['landuse2014'] == 1) | (intersect2['landuse2014'] == 4)
#         intersect2.loc[mask,['target_pop' + year]] = 0
#
#         mask = (intersect2['landuse2014'] == 2)
#         intersect2.loc[mask,['target_pop' + year]] = np.array(intersect2.loc[mask, ['AOCRA2014']])*\
#                                                      np.array(intersect2.loc[mask, ['estPop2013']])* (1/2)
#
#         mask = (intersect2['landuse2014'] == 3)
#         intersect2.loc[mask,['target_pop' + year]] = np.array(intersect2.loc[mask, ['AOCRA2014']])*\
#                                                      np.array(intersect2.loc[mask, ['estPop2013']])* (1/2)

# intersect2.to_csv(results + 'intersect2.csv')
# NTL_clip.to_file(temp + 'ntl_clip.shp')
# landuse_clip.to_file(temp + 'landuse_clip.shp')
# intersect2.to_file(temp + 'intersect.shp')

# intersect2 = gp.read_file(temp + 'intersect2.shp')

# in the level of night light
intersect2.set_index('ntl_clip_id', inplace=True)
ntl_scale = intersect2.groupby(['ntl_clip_id', 'landuse2014']).sum().loc[:,['intersect_area','estNTL2013_prime', 'estPop2013_prime']]
ntl_scale2 = ntl_scale.unstack('landuse2014')
ntl_scale2.columns = ['area_bg', 'area_lr', 'area_hr', 'area_nr',
                      'TNTL2013_bg', 'TNTL2013_lr', 'TNTL2013_hr', 'TNTL2013_nr',
                      'target_pop2013_bg', 'target_pop2013_lr', 'target_pop2013_hr', 'target_pop2013_nr']
ntl_scale2.fillna(0, inplace=True)
# NTL_clip.set_index('ntl_clip_id', inplace = True)
# ntl_scale2 = ntl_scale2.join(NTL_clip.loc[:,['estPop2013']], on=ntl_scale2.index, how='left', lsuffix='_caller', rsuffix='_other')
ntl_scale2['target_pop2013_r'] = ntl_scale2['target_pop2013_lr'] + ntl_scale2['target_pop2013_hr']# + ntl_scale2['target_pop2013_nr'] + ntl_scale2['target_pop2013_bg']
ntl_scale2['TNTL2013'] = ntl_scale2['TNTL2013_bg'] + ntl_scale2['TNTL2013_lr'] + ntl_scale2['TNTL2013_hr'] + ntl_scale2['TNTL2013_nr']

ntl_scale2.reset_index(inplace=True)
NTL_clip.reset_index(inplace=True)
ntl_scale2 = NTL_clip.merge(ntl_scale2, left_on = NTL_clip.index, right_on = ntl_scale2.index, how='left')
ntl_scale2.drop(['key_0', 'index', 'Shape_Leng', 'Shape_Area', 'ntl_area', 'NTL2013', 'NTL2014',
                 'NTL2015', 'NTL2016','NTL2017', 'NTL2018', 'ntl_clip_id_x', 'ntl_clip_area'],
                inplace=True, axis=1)
ntl_scale2['X'] = ntl_scale2.geometry.centroid.x
ntl_scale2['Y'] = ntl_scale2.geometry.centroid.y

intersect2['countNTL'] = intersect2['index'].groupby(intersect2.index).transform('count')

intersect2.to_csv(results + 'intersect2.csv')
ntl_scale2.to_csv(results + 'Observation_NTL_Level_2013_Median.csv')

ntl_scale2 = pd.read_csv(results + 'Observation_NTL_Level_2013_Median.csv')
sns.set(rc={'figure.figsize':(11.7,8.27)})
f, axes = plt.subplots(1, 2)
sns.regplot(x = ntl_scale2['TNTL2013'],y = ntl_scale2['target_pop2013_r'], ax=axes[0], color='green',
            order=2, line_kws={"color": "black"})
axes[0].set(xlabel='Night Light', ylabel='Population (Count)')
sns.regplot(x = ntl_scale2['TNTL2013_lr'],y = ntl_scale2['target_pop2013_lr'], ax=axes[1], color='orange')
axes[1].set(xlabel='Night Light (Low-residential)', ylabel='Population (Count)')

sns.set(rc={'figure.figsize':(11.7,8.27)})
f, axes = plt.subplots(2, 4)
sns.histplot(data = ntl_scale2['TNTL2013_hr'], ax=axes[0, 0], color='red')
axes[0, 0].set(xlabel='Night Light (High-residential)', ylabel='Frequency')
sns.histplot(data = ntl_scale2['TNTL2013_lr'], ax=axes[0, 1], color='red')
axes[0, 1].set(xlabel='Night Light (Low-residential)', ylabel='Frequency')
sns.histplot(data = ntl_scale2['TNTL2013_nr'], ax=axes[0, 2], color='red')
axes[0, 2].set(xlabel='Night Light (Non-residential)', ylabel='Frequency')
sns.histplot(data = ntl_scale2['TNTL2013_bg'], ax=axes[0, 3], color='red')
axes[0, 3].set(xlabel='Night Light (Other)', ylabel='Frequency')
sns.histplot(data = ntl_scale2['target_pop2013_hr'], ax=axes[1, 0])
axes[1, 0].set(xlabel='Population (High-residential)', ylabel='Frequency')
sns.histplot(data = ntl_scale2['target_pop2013_lr'], ax=axes[1, 1])
axes[1, 1].set(xlabel='Population (Low-residential)', ylabel='Frequency')

# Correlation analysis
# Population
# Spearman’s correlation coefficient

pd_i = ntl_scale2['target_pop2013_r'] / (ntl_scale2['area_bg'] + ntl_scale2['area_lr'] +
                                         ntl_scale2['area_hr'] + ntl_scale2['area_nr'])
pd_mu = pd_i.mean()
bg_area_mu = ntl_scale2['area_bg'].mean()
lr_area_mu = ntl_scale2['area_lr'].mean()
hr_area_mu = ntl_scale2['area_hr'].mean()
nr_area_mu = ntl_scale2['area_nr'].mean()

ro_bg_p = (((pd_i - pd_mu)*(ntl_scale2['area_bg']-bg_area_mu)).sum()) / \
        (np.sqrt((pd_i - pd_mu).pow(2).sum()*(ntl_scale2['area_bg']-bg_area_mu).pow(2).sum()))

ro_hr_p = (((pd_i - pd_mu)*(ntl_scale2['area_hr']-hr_area_mu)).sum()) / \
        (np.sqrt((pd_i - pd_mu).pow(2).sum()*(ntl_scale2['area_hr']-hr_area_mu).pow(2).sum()))

ro_lr_p = (((pd_i - pd_mu)*(ntl_scale2['area_lr']-lr_area_mu)).sum()) / \
        (np.sqrt((pd_i - pd_mu).pow(2).sum()*(ntl_scale2['area_lr']-lr_area_mu).pow(2).sum()))

ro_nr_p = (((pd_i - pd_mu)*(ntl_scale2['area_nr']-nr_area_mu)).sum()) / \
        (np.sqrt((pd_i - pd_mu).pow(2).sum()*(ntl_scale2['area_nr']-nr_area_mu).pow(2).sum()))

# Nightlight
ntl_i = ntl_scale2['TNTL2013'] / (ntl_scale2['area_bg'] + ntl_scale2['area_lr'] +
                                         ntl_scale2['area_hr'] + ntl_scale2['area_nr'])
ntl_i_mu = ntl_i.mean()
ro_bg_ntl = (((ntl_i - ntl_i_mu)*(ntl_scale2['area_bg']-bg_area_mu)).sum()) / \
        (np.sqrt((ntl_i - ntl_i_mu).pow(2).sum()*(ntl_scale2['area_bg']-bg_area_mu).pow(2).sum()))

ro_hr_ntl = (((ntl_i - ntl_i_mu)*(ntl_scale2['area_hr']-hr_area_mu)).sum()) / \
        (np.sqrt((ntl_i - ntl_i_mu).pow(2).sum()*(ntl_scale2['area_hr']-hr_area_mu).pow(2).sum()))

ro_lr_ntl = (((ntl_i - ntl_i_mu)*(ntl_scale2['area_lr']-lr_area_mu)).sum()) / \
        (np.sqrt((ntl_i - ntl_i_mu).pow(2).sum()*(ntl_scale2['area_lr']-lr_area_mu).pow(2).sum()))

ro_nr_ntl = (((ntl_i - ntl_i_mu)*(ntl_scale2['area_nr']-nr_area_mu)).sum()) / \
        (np.sqrt((ntl_i - ntl_i_mu).pow(2).sum()*(ntl_scale2['area_nr']-nr_area_mu).pow(2).sum()))

ntl_scale2['area_lr_p'] = ntl_scale2['area_lr'] / (ntl_scale2['area_lr'] + ntl_scale2['area_hr'] + ntl_scale2['area_nr']
                                                   + ntl_scale2['area_bg'])
ntl_scale2['area_hr_p'] = ntl_scale2['area_hr'] / (ntl_scale2['area_lr'] + ntl_scale2['area_hr'] + ntl_scale2['area_nr']
                                                   + ntl_scale2['area_bg'])
ntl_scale2['area_nr_p'] = ntl_scale2['area_nr'] / (ntl_scale2['area_lr'] + ntl_scale2['area_hr'] + ntl_scale2['area_nr']
                                                   + ntl_scale2['area_bg'])
ntl_scale2['area_bg_p'] = ntl_scale2['area_bg'] / (ntl_scale2['area_lr'] + ntl_scale2['area_hr'] + ntl_scale2['area_nr']
                                                   + ntl_scale2['area_bg'])
ntl_scale2['TNTL2013_l'] = np.log2(ntl_scale2['TNTL2013'])
ntl_scale2['TNTL2013_2'] = ntl_scale2['TNTL2013']*ntl_scale2['TNTL2013']
# multiple linear regression
# On NTL and landuse area
ntl_scale2['area_hr_2'] = ntl_scale2['area_hr']*ntl_scale2['area_hr']
ntl_scale2['area'] = ntl_scale2['area_hr'] + ntl_scale2['area_nr'] + ntl_scale2['area_lr'] + ntl_scale2['area_bg']

# mask = ntl_scale2['TNTL2013'] > 187818
mask = (ntl_scale2['TNTL2013'] < 100) & (ntl_scale2['target_pop2013_r'] < 4000)
test = ntl_scale2[mask]
#
# model_ntl_landuse_area1 = ols("TNTL2013 ~ area_hr"
#                          ,ntl_scale2).fit()
# print(model_ntl_landuse_area1.summary())
# print("\nRetrieving manually the parameter estimates:")
# print(model_ntl_landuse_area1._results.params)
#
# wgt_lr = (1 / (model_ntl_landuse_area1.params[1:4].sum() + 4))
# wgt_hr = (1 + model_ntl_landuse_area1.params[1])*(1 / (model_ntl_landuse_area1.params[1:4].sum()+4))
# wgt_nr = (1 + model_ntl_landuse_area1.params[2])*(1 / (model_ntl_landuse_area1.params[1:4].sum()+4))
# wgt_bg = (1 + model_ntl_landuse_area1.params[3])*(1 / (model_ntl_landuse_area1.params[1:4].sum()+4))
#

# Extract centroids
w_adaptive = weights.distance.KNN.from_dataframe(NTL_clip, k=50)
full_matrix, ids = w_adaptive.full()

ntl_scale2['area_lr_2'] = ntl_scale2['area_lr']*ntl_scale2['area_lr']
ntl_scale2['target_pop2013_r_density'] = ntl_scale2['target_pop2013_r'] / (ntl_scale2['area'])
ntl_scale2['area_lrarea_bg'] = ntl_scale2['area_bg']*ntl_scale2['area_lr']
# model = spreg .OLS(np.array(ntl_scale2['target_pop2013_r']), \
#                    np.array(ntl_scale2.loc[:, ['area_hr' , 'area_nr' , 'area_lr', 'TNTL2013' , 'TNTL2013_2']]), \
#                   w=w_adaptive, spat_diag=True)

# On landuse area
model_landuse_area1 = ols("TNTL2013 ~  area_hr + area_lr + area_nr + area_bg",ntl_scale2).fit()
print(model_landuse_area1.summary())
print("\nRetrieving manually the parameter estimates:")
print(model_landuse_area1._results.params)

model_landuse_area1.save(results + 'ols_ntl_population.pickle', remove_data=False)
from statsmodels.regression.linear_model import OLSResults
model_landuse_area1 = OLSResults.load(results + 'ols_ntl_population.pickle')

intersect2 = pd.read_csv(results + 'intersect2.csv')

mask = intersect2['landuse2014'] == 1
intersect2.loc[mask, ['coefficient2014']] = model_landuse_area1.params[4]
mask = intersect2['landuse2014'] == 2
intersect2.loc[mask, ['coefficient2014']] = model_landuse_area1.params[2]
mask = intersect2['landuse2014'] == 3
intersect2.loc[mask, ['coefficient2014']] = model_landuse_area1.params[1]
mask = intersect2['landuse2014'] == 4
intersect2.loc[mask, ['coefficient2014']] = model_landuse_area1.params[3]
intersect2['interceptNTL2014'] = model_landuse_area1.params[0]

intersect2['estNTL2013'] = intersect2['landuse_clip_area']*intersect2['coefficient2014'] + \
                           (intersect2['interceptNTL2014']/intersect2['countNTL'])

intersect2['estNTL2013_verify'] = intersect2['estNTL2013'].groupby(intersect2.ntl_clip_id).transform('sum')

intersect2['estNTL2013_prime'] = intersect2['estNTL2013'] * (intersect2['NTL2013'] / intersect2['estNTL2013_verify'])

intersect2.to_csv(results + 'intersect2.csv')

#
# On landuse area
model_landuse_area1 = ols("target_pop2013_r ~  TNTL2013_hr + TNTL2013_lr + TNTL2013_nr + TNTL2013_bg",ntl_scale2).fit()
print(model_landuse_area1.summary())
print("\nRetrieving manually the parameter estimates:")
print(model_landuse_area1._results.params)

sns.set(rc={'figure.figsize':(11.7,8.27)})
f, axes = plt.subplots(1, 1)
sns.regplot(y = ntl_scale2['target_pop2013_lr'],x = ntl_scale2['TNTL2013_lr'], ax=axes, color='orange', order=2, line_kws={"color": "black"})
axes.set(ylabel='pop2013_r', xlabel='NTL')
for k, v in test.iterrows():
    if v['target_pop2013_r_density'] >= 4000 and v['TNTL2013']>= 150:
        axes.annotate(v['ntl_id'],(ntl_scale2.loc[k, ['TNTL2013']], ntl_scale2.loc[k, ['target_pop2013_r_density']]), size=10)
sns.regplot(y = model_landuse_area1.resid,x = ntl_scale2['area_hr'], ax=axes[0, 1], color='green', order=2, line_kws={"color": "black"})
axes[0, 1].set(ylabel='Residuals', xlabel='High-Res (Area)')
sns.regplot(y = model_landuse_area1.resid,x = ntl_scale2['area_nr'], ax=axes[1, 0], color='blue', order=2, line_kws={"color": "black"})
axes[1, 0].set(ylabel='Residuals', xlabel='Non-Res (Area)')
sns.regplot(y = model_landuse_area1.resid,x = ntl_scale2['area_bg'], ax=axes[1, 1], color='red', order=2, line_kws={"color": "black"})
axes[1, 1].set(ylabel='Residuals', xlabel='Background (Area)')

pop = pd.DataFrame()
pop['pop'] = model_landuse_area1.predict(ntl_scale2.loc[:,['area_hr', 'area_nr', 'area_lr', 'TNTL2013', 'TNTL2013_2']])

ntl_scale2.to_csv(results + 'Observation_NTL_Corrected_Level_2013_Median_all.csv')

# on NTL
model_ntl_area1 = ols("target_pop2013_r ~ TNTL2013_lr+ TNTL2013_hr+ TNTL2013_nr + TNTL2013_bg"
                         ,ntl_scale2).fit()
print(model_ntl_area1.summary())
print("\nRetrieving manually the parameter estimates:")
print(model_ntl_area1._results.params)

sns.set(rc={'figure.figsize':(11.7,8.27)})
f, axes = plt.subplots(2, 2)
sns.regplot(x = test['target_pop2013_r'],y = test['TNTL2013_lr'], ax=axes[0, 0], color='orange')
axes[0, 0].set(xlabel='Night Light', ylabel='Low-Res (Area)')
sns.regplot(x = test['target_pop2013_r'],y = test['TNTL2013_hr'], ax=axes[0, 1], color='green')
axes[0, 1].set(xlabel='Night Light', ylabel='High-Res (Area)')
sns.regplot(x = test['target_pop2013_r'],y = test['TNTL2013_nr'], ax=axes[1, 0], color='blue')
axes[1, 0].set(xlabel='Night Light', ylabel='Non-Res (Area)')
sns.regplot(x = test['target_pop2013_r'],y = test['TNTL2013_bg'], ax=axes[1, 1], color='red')
axes[1, 1].set(xlabel='Night Light', ylabel='Background (Area)')

pop = pd.DataFrame()
pop['pop'] = model_ntl_area1.predict(ntl_scale2.loc[:,['TNTL2013_lr', 'TNTL2013_hr', 'TNTL2013_nr']])

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

[m, min, max] = mean_confidence_interval(pop['pop'], confidence=0.95)

mu, sigma = pop['pop'].mean(), pop['pop'].std() # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)
sns.histplot(pop['pop'], color='red')
sns.histplot(s, color='green')

pop2013_estimation = NTL_clip.merge(pop['pop'], left_on=NTL_clip.index, right_on=pop.index, how='left')
pop2013_estimation.plot(column='pop', cmap='OrRd', legend=True, scheme="quantiles", figsize=(15, 10))
plt.title('Population Estimation 2013 (NTL-Level)')
# residuals.to_file(temp + 'pop2013_estimation.shp')

res = pd.DataFrame()
res['residuals'] = model_ntl_area1.resid
residuals = NTL_clip.merge(res['residuals'], left_on=NTL_clip.index, right_on=res.index, how='left')
residuals.plot(column='residuals', cmap='bwr', legend=True, scheme="quantiles", figsize=(15, 10))
plt.title('Population Estimation Residuals 2013 (NTL-Level)')
# residuals.to_file(temp + 'residuals.shp')

# # multiple nonlinear regression
# ntl_scale2['TNTL2013_lr_pow2'] = ntl_scale2['TNTL2013_lr']*ntl_scale2['TNTL2013_lr']
# ntl_scale2['TNTL2013_hr_pow2'] = ntl_scale2['TNTL2013_hr']*ntl_scale2['TNTL2013_hr']
#
# model_landuse_area2 = ols("target_pop2013_r ~ TNTL2013_lr + TNTL2013_hr + "
#                           "TNTL2013_hr * TNTL2013_hr_pow2"
#                          ,ntl_scale2).fit()
# print(model_landuse_area2.summary())
# print("\nRetrieving manually the parameter estimates:")
# print(model_landuse_area2._results.params)
#
# pop = pd.DataFrame()
# pop['pop'] = model_landuse_area1.predict(ntl_scale2.loc[:,['TNTL2013_lr', 'TNTL2013_hr', 'TNTL2013_nr', 'TNTL2013_bg',
#                                                            'TNTL2013_lr_pow2', 'TNTL2013_hr_pow2', 'TNTL2013_lr' * 'TNTL2013_lr_pow2',
#                                                            'TNTL2013_hr' * 'TNTL2013_hr_pow2']])
# NTL_clip.reset_index(inplace=True)
# pop2013_estimation = NTL_clip.merge(pop['pop'], left_on=NTL_clip.index, right_on=pop.index, how='left')
# pop2013_estimation.plot(column='pop', cmap='OrRd', legend=True, scheme="quantiles", figsize=(15, 10))
# plt.title('Population Estimation 2013 (NTL-Level)')
# # residuals.to_file(temp + 'pop2013_estimation.shp')
#
# res = pd.DataFrame()
# res['residuals'] = model_landuse_area1.resid
# residuals = NTL_clip.merge(res['residuals'], left_on=NTL_clip.index, right_on=res.index, how='left')
# residuals.plot(column='residuals', cmap='bwr', legend=True, scheme="quantiles", figsize=(15, 10))
# plt.title('Population Estimation Residuals 2013 (NTL-Level)')
# # residuals.to_file(temp + 'residuals.shp')

NTL_clip_aux = NTL_clip
# for all years
for year in years:
    if int(year) >= 2014:
        ntl_prediction = intersect2.groupby(['ntl_clip_id', 'landuse' + year]).sum().loc[:,
                         ['intersect_area', 'TNTL' + year]]
        ntl_prediction2 = ntl_prediction.unstack('landuse' + year)
        ntl_prediction2.columns = ['area_bg' + year, 'area_lr'+ year, 'area_hr'+ year, 'area_nr'+ year,
                                   'TNTL_bg'+ year, 'TNTL_lr'+ year, 'TNTL_hr'+ year, 'TNTL_nr'+ year]
        ntl_prediction2.fillna(0, inplace=True)
        ntl_prediction2.reset_index(inplace=True)
        NTL_clip_aux.reset_index(inplace=True)
        NTL_clip_aux = NTL_clip_aux.merge(ntl_prediction2, left_on=NTL_clip_aux.index, right_on=ntl_prediction2.index, how='left')
        try:
            NTL_clip_aux.drop(['key_0'],inplace=True, axis=1)
        except:
            print('columns were not found!')
        try:
            NTL_clip_aux.drop(['index', 'Shape_Leng', 'Shape_Area', 'ntl_area', 'NTL2013', 'NTL2014',
                               'NTL2015', 'NTL2016', 'NTL2017', 'NTL2018', 'ntl_clip_id_x', 'ntl_clip_area',
                               'ntl_clip_id_y'],inplace=True, axis=1)
        except:
            print('columns were not found!')
        try:
            NTL_clip_aux.drop(['level_0'], inplace=True, axis=1)
        except:
            print('columns were not found!')

NTL_clip_aux['X'] = NTL_clip_aux.geometry.centroid.x
NTL_clip_aux['Y'] = NTL_clip_aux.geometry.centroid.y
NTL_clip_aux.to_csv(results + 'NTL_Level_Predictions.csv')

gwr_model = pd.read_csv(results + 'NTLGWR.csv')
predict_all_years = NTL_clip_aux.merge(gwr_model, left_on=NTL_clip_aux.index, right_on=ntl_prediction2.index, how='left')
predict_all_years.rename({})

for year in years:
    if int(year) >= 2014:
        predict_all_years['estpop' + year] = predict_all_years['X.Intercept.'] + \
                                             (predict_all_years['TNTL2013_bg'] * predict_all_years['TNTL_bg' + year]) + \
                                             (predict_all_years['TNTL2013_lr'] * predict_all_years['TNTL_lr' + year]) + \
                                             (predict_all_years['TNTL2013_hr'] * predict_all_years['TNTL_hr' + year]) + \
                                             (predict_all_years['TNTL2013_nr'] * predict_all_years['TNTL_nr' + year])

predict_all_years.drop(['key_0'],inplace=True, axis=1)
predict_all_years.to_csv(results + 'predict_all_years.csv')

# in the level of census tracts
intersect2.set_index('census_id', inplace=True)
ntl_scale = intersect2.groupby(['census_id', 'landuse2014']).sum().loc[:,['intersect_area','NTL2013', 'target_pop2013']]
ntl_scale2 = ntl_scale.unstack('landuse2014')
ntl_scale2.columns = ['area_bg', 'area_lr', 'area_hr', 'area_nr',
                      'TNTL2013_bg', 'TNTL2013_lr', 'TNTL2013_hr', 'TNTL2013_nr',
                      'target_pop2013_bg', 'target_pop2013_lr', 'target_pop2013_hr', 'target_pop2013_nr']
ntl_scale2.fillna(0, inplace=True)
census.set_index('census_id', inplace = True)
ntl_scale2 = ntl_scale2.join(census.loc[:,['estPop2013']], on=ntl_scale2.index, how='left', lsuffix='_caller', rsuffix='_other')

ntl_scale2['target_pop2013_r'] = ntl_scale2['target_pop2013_lr'] + ntl_scale2['target_pop2013_hr']# + ntl_scale2['target_pop2013_nr'] + ntl_scale2['target_pop2013_bg']
ntl_scale2['estNTL2013_prime'] = ntl_scale2['TNTL2013_bg'] + ntl_scale2['TNTL2013_lr'] + ntl_scale2['TNTL2013_hr'] + ntl_scale2['TNTL2013_nr']
# ntl_scale2.drop(['ntl_id_bg', 'ntl_id_lr', 'ntl_id_hr', 'ntl_id_nr'], inplace=True, axis=1)

intersect2['count'] = intersect2['index'].groupby(intersect2.index).transform('count')
intersect2.to_csv(results + 'intersect2.csv')

ntl_scale2.to_csv(results + 'Observation_Census_Level_2013_Median.csv')



ntl_scale2 = pd.read_csv(results + 'Observation_Census_Level_2013_Median.csv')
# ax = sns.scatterplot(ntl_scale2['TNTL2013'],ntl_scale2['target_pop2013_r'])
# ax.set_title('2013 (baseline year)')

# ax = sns.scatterplot(ntl_scale2['TNTL2013_bg'],ntl_scale2['target_pop2013_bg'])
# ax.set(xlabel='Night Light', ylabel='Population')
#
# ax = sns.scatterplot(ntl_scale2['TNTL2013_hr'],ntl_scale2['target_pop2013_hr'])
# ax.set(xlabel='Night Light', ylabel='Population')
#
# ax = sns.scatterplot(ntl_scale2['TNTL2013_lr'],ntl_scale2['target_pop2013_lr'])
# ax.set(xlabel='Night Light', ylabel='Population')
#
# ax = sns.scatterplot(ntl_scale2['TNTL2013_nr'],ntl_scale2['target_pop2013_nr'])
# ax.set(xlabel='Night Light', ylabel='Population')
#
# labels = ['background', 'high res', 'low res', 'non res']
# ax.legend(labels)
# ax.set_title('2013 (baseline year)')
# plt.show()

census.reset_index(inplace=True)
ntldensity = census.merge(ntl_scale2, left_on=census.index, right_on=ntl_scale2.index, how='left')
ntldensity.plot(column='CNTL2013', cmap='OrRd', legend=True, scheme="quantiles", figsize=(15, 10))
plt.title('Sum of NTL 2013 (Census-Level)')
sns.set(rc={'figure.figsize':(11.7,8.27)})
f, axes = plt.subplots(1, 1)
sns.regplot(x = ntl_scale2['CNTL2013'],y = ntl_scale2['target_pop2013_r'], ax=axes, color='green', order=2, line_kws={"color": "black"})
axes.set(xlabel='Sum of NTL', ylabel='Population (Count)')

sns.set(rc={'figure.figsize':(11.7,8.27)})
f, axes = plt.subplots(1, 1)
sns.histplot(data = ntl_scale2['CNTL2013log'], ax=axes, color='red')
axes.set(xlabel='Night Light (High-residential)', ylabel='Frequency')

sns.set(rc={'figure.figsize':(11.7,8.27)})
f, axes = plt.subplots(1, 2)
sns.regplot(x = ntl_scale2['NTLDensity'],y = ntl_scale2['target_pop2013_hr'], ax=axes[0], color='green')
axes[0].set(xlabel='area_hr', ylabel='Population (Count)')
for k, v in ntl_scale2.iterrows():
    if v['area_hr'] >= 600000 or v['target_pop2013_hr']>= 15000:
        axes[0].annotate(v['census_id'],(ntl_scale2.loc[k, ['area_hr']], ntl_scale2.loc[k, ['target_pop2013_hr']]), size=10)
sns.regplot(x = ntl_scale2['TNTL2013_lr'],y = ntl_scale2['target_pop2013_lr'], ax=axes[1], color='orange')
axes[1].set(xlabel='Night Light (Low-residential)', ylabel='Population (Count)')
for k, v in ntl_scale2.iterrows():
    if v['TNTL2013_lr'] >= 300 or v['target_pop2013_lr']>= 3000:
        axes[1].annotate(v['census_id'],(ntl_scale2.loc[k, ['TNTL2013_lr']], ntl_scale2.loc[k, ['target_pop2013_lr']]), size=10)

sns.set(rc={'figure.figsize':(11.7,8.27)})
f, axes = plt.subplots(2, 4)
sns.histplot(data = ntl_scale2['area_hrlog'], ax=axes[0, 0], color='red')
axes[0, 0].set(xlabel='Night Light (High-residential)', ylabel='Frequency')
sns.histplot(data = ntl_scale2['area_lrlog'], ax=axes[0, 1], color='red')
axes[0, 1].set(xlabel='Night Light (Low-residential)', ylabel='Frequency')
sns.histplot(data = ntl_scale2['area_nrlog'], ax=axes[0, 2], color='red')
axes[0, 2].set(xlabel='Night Light (Non-residential)', ylabel='Frequency')
sns.histplot(data = ntl_scale2['area_bglog'], ax=axes[0, 3], color='red')
axes[0, 3].set(xlabel='Night Light (Other)', ylabel='Frequency')
sns.histplot(data = ntl_scale2['target_pop2013_rlog'], ax=axes[1, 0])
axes[1, 0].set(xlabel='Population (High-residential)', ylabel='Frequency')
sns.histplot(data = ntl_scale2['CNTL2013'], ax=axes[1, 1])
axes[1, 1].set(xlabel='Population (Low-residential)', ylabel='Frequency')
sns.histplot(data = ntl_scale2['CNTL2013'], ax=axes[1, 2])
axes[1, 2].set(xlabel='CNTL2013', ylabel='Frequency')
sns.histplot(data = ntl_scale2['CNTL2013Med'], ax=axes[1, 3])
axes[1, 3].set(xlabel='CNTL2013Med', ylabel='Frequency')

# multiple linear regression
ntl_scale2['Census_NTL2013'] = np.mean(ntl_scale2['TNTL2013_bg'].to_numpy(), ntl_scale2['TNTL2013_lr'].to_numpy(),
                                       ntl_scale2['TNTL2013_hr'].to_numpy(), ntl_scale2['TNTL2013_nr'].to_numpy())

ntl_scale2['CNTL2013_2'] = ntl_scale2['CNTL2013'] * ntl_scale2['CNTL2013']
ntl_scale2['NTLDensity'] = ntl_scale2['TNTL2013'] / \
                                     (ntl_scale2['area_lr'] + ntl_scale2['area_hr'] + ntl_scale2['area_nr'] + ntl_scale2['area_bg'])
ntl_scale2['PopDensity2013'] = ntl_scale2['target_pop2013_r'] / \
                                     (ntl_scale2['area_lr'] + ntl_scale2['area_hr'] + ntl_scale2['area_nr'] + ntl_scale2['area_bg'])
ntl_scale2['CNTL2013_2'] = ntl_scale2['CNTL2013'] * ntl_scale2['CNTL2013']

ntl_scale2['CNTL2013log'] = np.log(ntl_scale2['CNTL2013'])
ntl_scale2['target_pop2013_rlog'] = np.log(ntl_scale2['target_pop2013_r'])
ntl_scale2['area_lrlog'] = np.log1p(ntl_scale2['area_lr'])
ntl_scale2['area_hrlog'] = np.log1p(ntl_scale2['area_hr'])
ntl_scale2['area_nrlog'] = np.log1p(ntl_scale2['area_nr'])
ntl_scale2['area_bglog'] = np.log1p(ntl_scale2['area_bg'])
ntl_scale2['area'] = ntl_scale2['area_bg'] + ntl_scale2['area_hr'] + ntl_scale2['area_lr'] + ntl_scale2['area_nr']
ntl_scale2['target_pop2013_r_density'] = ntl_scale2['target_pop2013_r']/ ntl_scale2['area']

model_landuse_area1 = ols("target_pop2013_r ~ area_hr + area_lr + area_nr + area_bg", ntl_scale2).fit()
print(model_landuse_area1.summary())
print("\nRetrieving manually the parameter estimates:")
print(model_landuse_area1._results.params)

model_landuse_area1.save(results + 'ols_census_population.pickle', remove_data=False)
from statsmodels.regression.linear_model import OLSResults
model_landuse_area1 = OLSResults.load(results + 'ols_census_population.pickle')
# Predict new population in the level of landuse:

mask = intersect2['landuse2014'] == 1
intersect2.loc[mask, ['coefficientPop2014']] = model_landuse_area1.params[4]
mask = intersect2['landuse2014'] == 2
intersect2.loc[mask, ['coefficientPop2014']] = model_landuse_area1.params[2]
mask = intersect2['landuse2014'] == 3
intersect2.loc[mask, ['coefficientPop2014']] = model_landuse_area1.params[1]
mask = intersect2['landuse2014'] == 4
intersect2.loc[mask, ['coefficientPop2014']] = model_landuse_area1.params[3]
intersect2['interceptPop2014'] = model_landuse_area1.params[0]

intersect2['estPop2013lndus'] = intersect2['landuse_clip_area']*intersect2['coefficientPop2014'] + \
                           (intersect2['interceptPop2014']/intersect2['count'])

intersect2['estPop2013_verify'] = intersect2['estPop2013lndus'].groupby(intersect2.index).transform('sum')

intersect2['estPop2013_prime'] = intersect2['estPop2013lndus'] * (intersect2['estPop2013'] / intersect2['estPop2013_verify'])



# rescaled
ntl_scale2['pop13scaled'] = ntl_scale2['pop13dis']*(ntl_scale2['estPop2013']/ntl_scale2['estPop2013'])

sns.set(rc={'figure.figsize':(11.7,8.27)})
f, axes = plt.subplots(1, 2)
sns.regplot(x = ntl_scale2['area_hr'],y = model_landuse_area1.resid, ax=axes[0], color='green')
axes[0].set(xlabel='area_hr', ylabel='Residuals')
for k, v in ntl_scale2.iterrows():
    if v['area_hr'] >= 600000 or v['target_pop2013_hr']>= 15000:
        axes[0].annotate(v['census_id'],(ntl_scale2.loc[k, ['area_hr']], ntl_scale2.loc[k, ['target_pop2013_hr']]), size=10)
sns.regplot(x = ntl_scale2['TNTL2013_lr'],y = ntl_scale2['target_pop2013_lr'], ax=axes[1], color='orange')
axes[1].set(xlabel='Night Light (Low-residential)', ylabel='Population (Count)')
for k, v in ntl_scale2.iterrows():
    if v['TNTL2013_lr'] >= 300 or v['target_pop2013_lr']>= 3000:
        axes[1].annotate(v['census_id'],(ntl_scale2.loc[k, ['TNTL2013_lr']], ntl_scale2.loc[k, ['target_pop2013_lr']]), size=10)

pop2013_estimation = model_landuse_area1.predict(ntl_scale2.loc[:,['area_hr' , 'area_lr' , 'area_nr' , 'area_bg', 'CNTL2013_2', 'CNTL2013']])

sns.set(rc={'figure.figsize':(11.7,8.27)})
f, axes = plt.subplots(1, 1)
sns.regplot(x=ntl_scale2['target_pop2013_r'], y=pop2013_estimation, ax=axes, color='green')
axes.set(xlabel='True value')
axes.set(ylabel='Predicted value')
axes.set(title='target_pop2013_r ~ area_hr + area_lr + area_nr + area_bg + CNTL2013_2 + CNTL2013')

pop2013_estimation = pd.DataFrame(pop2013_estimation)
pop2013_estimation.rename({0:'pop'}, inplace=True, axis=1)

pop2013_estimation = census.merge(pop2013_estimation['pop'], left_on=census.index, right_on=pop2013_estimation.index, how='left')
pop2013_estimation.plot(column='pop', cmap='OrRd', legend=True, scheme="quantiles", figsize=(15, 10))
plt.title('Population Estimation 2013 (Census-Level)')
# residuals.to_file(temp + 'pop2013_estimation.shp')

res = pd.DataFrame()
res['residuals'] = model_landuse_area1.resid
residuals = census.merge(res['residuals'], left_on=census.index, right_on=res.index, how='left')
residuals.plot(column='residuals', cmap='bwr', legend=True, scheme="quantiles", figsize=(15, 10))
plt.title('Population Estimation Residuals 2013 (Census-Level)')
# residuals.to_file(temp + 'residuals.shp')

forgwr = census.loc[:, ['estPop2013', 'geometry', 'census_id']].merge(ntl_scale2, left_on=census.index, right_on=ntl_scale2.index, how='left')
forgwr['x'] = forgwr.geometry.centroid.x
forgwr['y'] = forgwr.geometry.centroid.y

forgwr.to_csv(results + 'Observation_Census_Level_2013_Median_Landuse.csv')

NTL_clip_aux = census
# for all years
for year in years:
    if int(year) >= 2014:
        ntl_prediction = intersect2.groupby(['census_id', 'landuse' + year]).sum().loc[:,
                         ['intersect_area', 'TNTL' + year]]
        ntl_prediction2 = ntl_prediction.unstack('landuse' + year)
        ntl_prediction2.columns = ['area_bg' + year, 'area_lr'+ year, 'area_hr'+ year, 'area_nr'+ year,
                                   'TNTL_bg'+ year, 'TNTL_lr'+ year, 'TNTL_hr'+ year, 'TNTL_nr'+ year]
        ntl_prediction2.fillna(0, inplace=True)

        aux = intersect2.groupby(['census_id']).mean().loc[:, ['CNTL' + year]]
        ntl_prediction2 = ntl_prediction2.merge(aux, left_on=ntl_prediction2.index, right_on=aux.index, how='left')
        ntl_prediction2.rename({'key_0': 'census_id'}, inplace=True, axis=1)
        ntl_prediction2.set_index('census_id', inplace=True)

        ntl_prediction2.reset_index(inplace=True)
        NTL_clip_aux.reset_index(inplace=True)
        NTL_clip_aux = NTL_clip_aux.merge(ntl_prediction2, left_on=NTL_clip_aux.index, right_on=ntl_prediction2.index, how='left')

        try:
            NTL_clip_aux.drop(['key_0'],inplace=True, axis=1)
        except:
            print('columns were not found!')
        try:
            NTL_clip_aux.drop(['index', 'Shape_Leng', 'Shape_Area', 'ntl_area', 'NTL2013', 'NTL2014',
                               'NTL2015', 'NTL2016', 'NTL2017', 'NTL2018', 'ntl_clip_id_x', 'ntl_clip_area',
                               'ntl_clip_id_y'],inplace=True, axis=1)
        except:
            print('columns were not found!')
        try:
            NTL_clip_aux.drop(['level_0'], inplace=True, axis=1)
        except:
            print('columns were not found!')

NTL_clip_aux['x'] = NTL_clip_aux.geometry.centroid.x
NTL_clip_aux['y'] = NTL_clip_aux.geometry.centroid.y
NTL_clip_aux.to_csv(results + 'Census_Level_Predictions.csv')

gwr_model = pd.read_csv(results + 'GWRLanduseNTL.csv')
predict_all_years = NTL_clip_aux.merge(gwr_model, left_on=NTL_clip_aux.index, right_on=ntl_prediction2.index, how='left')
predict_all_years.rename({})

for year in years:
    if int(year) >= 2014:
        predict_all_years['estpop' + year] = predict_all_years['X.Intercept.'] + \
                                             (predict_all_years['area_bg'] * predict_all_years['area_bg' + year]) + \
                                             (predict_all_years['area_lr'] * predict_all_years['area_lr' + year]) + \
                                             (predict_all_years['area_hr'] * predict_all_years['area_hr' + year]) + \
                                             (predict_all_years['area_nr'] * predict_all_years['area_nr' + year])

predict_all_years.drop(['key_0'],inplace=True, axis=1)
predict_all_years.to_csv(results + 'predict_all_yearsLanduseNTL.csv')