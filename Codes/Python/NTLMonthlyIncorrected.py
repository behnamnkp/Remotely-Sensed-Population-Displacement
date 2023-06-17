# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import gdal, os
import matplotlib.pyplot as plt
from matplotlib import *
import pandas as pd
import os
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
# import arcpy
# from arcpy import env
# from arcpy.sa import *
from simpledbf import Dbf5
from tempfile import TemporaryFile
import scipy.ndimage.filters
import geopandas as gp

# Choose the year of the analysis:
# keep in mind depending on the naming strategy you may have to use different item indexes
years = ['2012','2013', '2014', '2015', '2016', '2017', '2018'] # 2012-13-14-15-16-17-18 is available now
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

temp = 'G:/backupC27152020/Population_Displacement_Final/Resources/Temp/'

# years = ['2012'] # 2012-13-14-15-16-17-18 is available now
#
# #  3) night light annual average and median
# os.chdir('C:/wgetdown/MosulVNP2/extract/brdf_prj_clp/')  #: e.g. 'G:/sigspatial/HDFSLK/'
# # read all file names. Make sure you do not have other file types in the directory
# rasterFiles = os.listdir(os.getcwd())
# for year in years:
#     print('starting the year... ' + year)
#     ntl = np.ones((366, 49, 44))*np.nan
#     day = 1
#     for day in range(1, 366):
#         for item in rasterFiles:
#             if '.tif' in item and '.xml' not in item and '.ovr' not in item and '.cpg' not in item and '.dbf' not in item \
#                     and '.autolock' not in item and '.tfw' not in item:
#                 if 'A' + year in item and int(item[43:46]) == day:
#                     # 48:51>incorrected [54:57]>corrected
#                     # idx = int(item[43:46])# 2013-16 > 54:57 2017-18 > 43:46
#                     arr = arcpy.RasterToNumPyArray(item, nodata_to_value=0).astype('float')
#                     # arr = scipy.ndimage.filters.uniform_filter(arr, size=3, mode='nearest')
#                     ntl[day - 1] = arr
#
#     daynum = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','sep','Oct','Nov','Dec']
#     day=0
#     for item in daynum:
#         print('starting the month... ' + item)
#         if item == 'Jan':
#             ntl0 = ntl[day:day+31, :, :]
#             day = day + 31
#         elif item == 'Feb':
#             ntl0 = ntl[day:day + 29, :, :]
#             day = day + 29
#         elif item == 'Mar':
#             ntl0 = ntl[day:day + 31, :, :]
#             day = day + 31
#         elif item == 'Apr':
#             ntl0 = ntl[day:day + 30, :, :]
#             day = day + 30
#         elif item == 'May':
#             ntl0 = ntl[day:day + 31, :, :]
#             day = day + 31
#         elif item == 'Jun':
#             ntl0 = ntl[day:day + 30, :, :]
#             day = day + 30
#         elif item == 'Jul':
#             ntl0 = ntl[day:day + 31, :, :]
#             day = day + 31
#         elif item == 'Aug':
#             ntl0 = ntl[day:day + 31, :, :]
#             day = day + 31
#         elif item == 'Sep':
#             ntl0 = ntl[day:day + 30, :, :]
#             day = day + 30
#         elif item == 'Oct':
#             ntl0 = ntl[day:day + 31, :, :]
#             day = day + 31
#         elif item == 'Nov':
#             ntl0 = ntl[day:day + 30, :, :]
#             day = day + 30
#         else:
#             ntl0 = ntl[day:day + 31, :, :]
#             day = day + 31
#
#         # this is important to make sure we count the number of used pixels for averaging
#         ntl0_ = np.where(ntl0 == 65535.0, numpy.nan, ntl0)
#         ntl0 = np.where(ntl0 == 65535.0, 0, ntl0)
#
#         counts = np.count_nonzero(~np.isnan(ntl0_), 0)
#         # print('Number of days with ntl value in ' + item + ', ' + year, counts)
#         ntlmed = np.zeros((ntl0_.shape[1], ntl0_.shape[2]))
#         for i in range(0, ntl0.shape[1]):
#             for j in range(0, ntl0.shape[2]):
#                 ntlmed[i, j] = np.nanmedian(ntl0_[:, i, j])
#
#         inRas = arcpy.Raster('C:/wgetdown/MosulVNP2/extract/brdf_prj_clp/'
#                         'clipprj_DNB_BRDF_Corrected_NTLVNP46A2_A2013001_h22v05_001_2020129141119.tif')
#         lowerLeft = arcpy.Point(inRas.extent.XMin, inRas.extent.YMin)
#         cellSize = inRas.meanCellWidth
#         dsc = arcpy.Describe(inRas)
#         sr = dsc.SpatialReference
#
#         myRaster = arcpy.NumPyArrayToRaster(ntlmed, lowerLeft, cellSize)
#         arcpy.DefineProjection_management(myRaster, sr)
#         if item == 'Jan':
#             out_rasterdataset = 'G:/backupC27152020/Population_Displacement_Final/Resources/VIIRS/VNP46A2/ntlMonthlyIncorrected_' + item + year + '.tif'
#         elif item == 'Feb':
#             out_rasterdataset = 'G:/backupC27152020/Population_Displacement_Final/Resources/VIIRS/VNP46A2/ntlMonthlyIncorrected_' + item + year + '.tif'
#         elif item == 'Mar':
#             out_rasterdataset = 'G:/backupC27152020/Population_Displacement_Final/Resources/VIIRS/VNP46A2/ntlMonthlyIncorrected_' + item + year + '.tif'
#         elif item == 'Apr':
#             out_rasterdataset = 'G:/backupC27152020/Population_Displacement_Final/Resources/VIIRS/VNP46A2/ntlMonthlyIncorrected_' + item + year + '.tif'
#         elif item == 'May':
#             out_rasterdataset = 'G:/backupC27152020/Population_Displacement_Final/Resources/VIIRS/VNP46A2/ntlMonthlyIncorrected_' + item + year + '.tif'
#         elif item == 'Jun':
#             out_rasterdataset = 'G:/backupC27152020/Population_Displacement_Final/Resources/VIIRS/VNP46A2/ntlMonthlyIncorrected_' + item + year + '.tif'
#         elif item == 'Jul':
#             out_rasterdataset = 'G:/backupC27152020/Population_Displacement_Final/Resources/VIIRS/VNP46A2/ntlMonthlyIncorrected_' + item + year + '.tif'
#         elif item == 'Aug':
#             out_rasterdataset = 'G:/backupC27152020/Population_Displacement_Final/Resources/VIIRS/VNP46A2/ntlMonthlyIncorrected_' + item + year + '.tif'
#         elif item == 'Sep':
#             out_rasterdataset = 'G:/backupC27152020/Population_Displacement_Final/Resources/VIIRS/VNP46A2/ntlMonthlyIncorrected_' + item + year + '.tif'
#         elif item == 'Oct':
#             out_rasterdataset = 'G:/backupC27152020/Population_Displacement_Final/Resources/VIIRS/VNP46A2/ntlMonthlyIncorrected_' + item + year + '.tif'
#         elif item == 'Nov':
#             out_rasterdataset = 'G:/backupC27152020/Population_Displacement_Final/Resources/VIIRS/VNP46A2/ntlMonthlyIncorrected_' + item + year + '.tif'
#         else:
#             out_rasterdataset = 'G:/backupC27152020/Population_Displacement_Final/Resources/VIIRS/VNP46A2/ntlMonthlyIncorrected_' + item + year + '.tif'
#
#         myRaster.save(out_rasterdataset)


# timeseries
os.chdir('G:/backupC27152020/Population_Displacement_Final/Resources/VIIRS/VNP46A2/')  #: e.g. 'G:/sigspatial/HDFSLK/'
# read all file names. Make sure you do not have other file types in the directory
rasterFiles = os.listdir(os.getcwd())

for year in years:
    for month in months:
        image = gp.read_file(temp + 'ntlMonthlyIncorrected_' + month + year + '.shp')
        NTL = gp.sjoin(NTL, image, how="inner", op='intersects')
        NTL.rename({'grid_code': 'NTL' + month + year}, inplace=True, axis=1)
        NTL.drop('pointid', inplace=True, axis=1)
        NTL.drop('index_right', inplace=True, axis=1)

