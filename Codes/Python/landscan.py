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
import libpysal
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

LS_path = 'G:/backupC27152020/Population_Displacement_Final/Resources/LandScan/'
landuse_path = 'G:/backupC27152020/Population_Displacement_Final/Resources/VHR/landuse/'
viirs_path = 'G:/backupC27152020/Population_Displacement_Final/Resources/VIIRS/VNP46A2/'
geodb_path = 'G:/backupC27152020/Population_Displacement_Final/Resources/poulation_disp.gdb/Data/'
temp = 'G:/backupC27152020/Population_Displacement_Final/Resources/Temp/'
results = 'G:/backupC27152020/Population_Displacement_Final/Resources/Results/'
date = '03292020'

# Choose the year of the analysis:
years = ['2009', '2013'] # 2014-15-16-17-18 is available now
LS = gp.read_file(LS_path + 'landscan.shp')
LS['LS_id'] = LS.index + 1
LS['LS_area'] = LS.area
census = gp.read_file(temp + 'CensusAll.shp')
census['census_id'] = census.index + 1
census['census_area'] = census.area
boundary = gp.read_file(temp + 'CensusBoundaryAll.shp')

for year in years:
    image = gp.read_file(LS_path + 'lspoint' + year + '.shp')
    LS = gp.sjoin(LS, image, how="inner", op='intersects')
    LS.rename({'grid_code': 'LS' + year}, inplace=True, axis=1)
    LS.drop('pointid', inplace=True, axis=1)
    LS.drop('index_right', inplace=True, axis=1)

LS_clip = gp.clip(LS, boundary)
LS_clip['LS_clip_id'] = LS_clip.index + 1
LS_clip['LS_clip_area'] = LS_clip.area

intersect1 = gp.overlay(census, LS_clip, how='intersection')

intersect1['intersect_id'] = intersect1.index + 1
intersect1['intersect_area'] = intersect1.area

for year in years:
    intersect1['CLS' + year] = (intersect1['LS_clip_area'] /
                                 intersect1['LS_area']) * intersect1['LS' + year]

intersect1['LSareaprop'] = intersect1['intersect_area'] / intersect1['LS_clip_area']
intersect1['FinalLS2009'] = intersect1['LSareaprop'] * intersect1['CLS2009']
intersect1['FinalLS2013'] = intersect1['LSareaprop'] * intersect1['CLS2013']


LS_scale = intersect1.groupby(['census_id']).sum().loc[:,['FinalLS2009','FinalLS2013']]
census.set_index('census_id', inplace=True)
LS_scale2 = pd.concat((census, LS_scale),axis=1)

LS_scale2.sum()
LSSum2009 = 3683193
LSSum2013 = 4472701

CensusSum2009 = 1113337
CensusSum2013 = 1377000

LS_scale2['estPop2013'] = ((CensusSum2013-CensusSum2009)/(LSSum2013-LSSum2009))*\
                          (LS_scale2['FinalLS2013'] - LS_scale2['FinalLS2009']) + \
                          LS_scale2['MAX_popult']

LS_scale2.to_file(temp + 'censusAllEstPop2013.shp')