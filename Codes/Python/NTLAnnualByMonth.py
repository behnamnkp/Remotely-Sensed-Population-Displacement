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
import seaborn as sns
import arcpy
from arcpy import env
from arcpy.sa import *
from simpledbf import Dbf5
from tempfile import TemporaryFile
import scipy.ndimage.filters
from statsmodels.formula.api import ols


# # 1) reproject night light data
# os.chdir('C:/wgetdown/MosulVNP1/extract/zenith')#: e.g. 'G:/sigspatial/HDFSLK/'
# # read all file names. Make sure you do not have other file types in the directory
# rasterFiles = os.listdir(os.getcwd())
# sr = arcpy.SpatialReference("WGS 1984 UTM Zone 38N")
# for item in rasterFiles:
#     if '.tif' in item and '.xml' not in item and '.ovr' not in item and '.cpg' not in item and '.dbf' not in item :
#         name = 'prj_' + item
#         print(name)
#         arcpy.ProjectRaster_management(in_raster=item, out_raster="C:/wgetdown/MosulVNP1/extract/zenith_prj/" + name,
#                                     out_coor_system=sr)

# #  2) clip night light data
# os.chdir('C:/wgetdown/MosulVNP2/extract/brdf_prj/')#: e.g. 'G:/sigspatial/HDFSLK/'
# # read all file names. Make sure you do not have other file types in the directory
# rasterFiles = os.listdir(os.getcwd())
# for item in rasterFiles:
#     if '.tif' in item and '.xml' not in item and '.ovr' not in item and '.cpg' not in item and '.dbf' not in item :
#         name = 'clip' + item
#         print(name)
#         arcpy.Clip_management(in_raster=item, rectangle="324380.156700 4011347.251500 343193.540500 4031960.943800",
#
#             out_raster="C:/wgetdown/MosulVNP2/extract/brdf_prj_clp/" + name)

# Choose the year of the analysis:
# keep in mind depending on the naming strategy you may have to use different item indexes
years = ['2013', '2014', '2015', '2016', '2017', '2018'] # 2012-13-14-15-16-17-18 is available now
#
# # 1) Azimuth
# os.chdir('C:/wgetdown/MosulVNP1/extract/azimuth_prj_clp/')  #: e.g. 'G:/sigspatial/HDFSLK/'
# # read all file names. Make sure you do not have other file types in the directory
# rasterFiles = os.listdir(os.getcwd())
# for year in years:
#     print('starting the year... ' + year)
#     azimuth = np.zeros((366, 49, 44))
#     for item in rasterFiles:
#         if '.tif' in item and '.xml' not in item and '.ovr' not in item and '.cpg' not in item and '.dbf' not in item \
#                 and '.autolock' not in item and '.tfw' not in item:
#             if 'A' + year in item:
#                 # 48:51>incorrected [54:57]>corrected
#                 idx = int(item[35:38])# 2013-16 > 54:57 2017-18 > 43:46
#                 arr = arcpy.RasterToNumPyArray(item, nodata_to_value=0)
#                 df = arr.astype(str)
#                 for r in range(df.shape[0]):
#                     for c in range(df.shape[1]):
#                         df[r, c] = float(df[r, c][:-2] + '.' + df[r, c][-2:])
#                         # print(df[r, c])
#
#                 azimuth[idx - 1] = df
#     azimuth_ = np.where(azimuth == -32768, numpy.nan, azimuth)
#
#     np.save('C:/wgetdown/MosulVNP1/azimuth_' + year + '.npy', azimuth_)
#
# # 2) Zenith
# os.chdir('C:/wgetdown/MosulVNP1/extract/zenith_prj_clp/')  #: e.g. 'G:/sigspatial/HDFSLK/'
# # read all file names. Make sure you do not have other file types in the directory
# rasterFiles = os.listdir(os.getcwd())
# for year in years:
#     print('starting the year... ' + year)
#     zenith = np.zeros((366, 49, 44))
#     for item in rasterFiles:
#         if '.tif' in item and '.xml' not in item and '.ovr' not in item and '.cpg' not in item and '.dbf' not in item \
#                 and '.autolock' not in item and '.tfw' not in item:
#             if 'A' + year in item:
#                 # 48:51>incorrected [54:57]>corrected
#                 idx = int(item[34:37])# 2013-16 > 54:57 2017-18 > 43:46
#                 arr = arcpy.RasterToNumPyArray(item, nodata_to_value=0)
#                 df = arr.astype(str)
#                 for r in range(df.shape[0]):
#                     for c in range(df.shape[1]):
#                         df[r, c] = float(df[r, c][:-2] + '.' + df[r, c][-2:])
#
#                 zenith[idx - 1] = df
#     zenith_ = np.where(zenith == -32768, numpy.nan, zenith)
#     np.save('C:/wgetdown/MosulVNP1/zenith_' + year + '.npy', zenith_)
#
# #  3) night light annual average and median
# os.chdir('C:/wgetdown/MosulVNP2/extract/brdf_prj_clp/')  #: e.g. 'G:/sigspatial/HDFSLK/'
# # read all file names. Make sure you do not have other file types in the directory
# rasterFiles = os.listdir(os.getcwd())
# for year in years:
#     print('starting the year... ' + year)
#     ntl = np.zeros((366, 49, 44))
#     for item in rasterFiles:
#         if '.tif' in item and '.xml' not in item and '.ovr' not in item and '.cpg' not in item and '.dbf' not in item \
#                 and '.autolock' not in item and '.tfw' not in item:
#             if 'A' + year in item:
#                 # 48:51>incorrected [54:57]>corrected
#                 idx = int(item[43:46])# 2013-16 > 54:57 2017-18 > 43:46
#                 arr = arcpy.RasterToNumPyArray(item, nodata_to_value=0).astype('float')
#                 # arr = scipy.ndimage.filters.uniform_filter(arr, size=3, mode='nearest')
#                 ntl[idx - 1] = arr
#
#     # this is important to make sure we count the number of used pixels for averaging
#     ntl_ = np.where(ntl == 65535.0, numpy.nan, ntl)
#     np.save('C:/wgetdown/MosulVNP2/ntl_' + year + '.npy', ntl_)

for year in years:
    azimuth_ = np.load('C:/wgetdown/MosulVNP1/azimuth_' + year + '.npy')
    zenith_ = np.load('C:/wgetdown/MosulVNP1/zenith_' + year + '.npy')
    ntl_ = np.load('C:/wgetdown/MosulVNP2/ntl_' + year + '.npy')

    ntl_final = ntl_

    for t in range(0, ntl_final.shape[0]-36, 30):
        azimuth_0 = azimuth_[t:t+30, :, :]
        zenith_0 = zenith_[t:t+30, :, :]
        ntl_0 = ntl_[t:t+30, :, :]

        # # for i in range(0, ntl_.shape[0]):
        # #     ntl_[i, :, :] = scipy.ndimage.filters.uniform_filter(ntl_[i, :, :], size=3, mode='nearest')
        # #
        # sns.set(rc={'figure.figsize':(11.7,8.27)})
        # f, axes = plt.subplots(4, 1)
        # sns.lineplot(x = range(0, 30), y = ntl_0[t:t+30, 21, 21], ax=axes[0],color='orange')
        # axes[0].set(xlabel='Day')
        # sns.lineplot(x = range(0, 30), y = zenith_0[t:t+30, 21, 21], ax=axes[0],color='green')
        # axes[0].set(xlabel='Day')
        # axes[0].legend(['NTL','VZA'])
        #
        # sns.lineplot(x = range(0, 30), y = ntl_0[t:t+30, 5, 40], ax=axes[1],color='orange')
        # axes[1].set(xlabel='Day')
        # sns.lineplot(x = range(0, 30), y = zenith_0[t:t+30, 5, 40], ax=axes[1],color='green')
        # axes[1].set(xlabel='Day')
        # axes[1].legend(['NTL','VZA'])
        #
        # sns.lineplot(x = range(0, 30), y = ntl_0[t:t+30, 38, 26], ax=axes[2],color='orange')
        # axes[2].set(xlabel='Day')
        # sns.lineplot(x = range(0, 30), y = zenith_0[t:t+30, 38, 26], ax=axes[2],color='green')
        # axes[2].set(xlabel='Day')
        # axes[2].legend(['NTL','VZA'])
        #
        # sns.lineplot(x = range(0, 30), y = ntl_0[t:t+30, 4, 18], ax=axes[3],color='orange')
        # axes[3].set(xlabel='Day')
        # sns.lineplot(x = range(0, 30), y = zenith_0[t:t+30, 4, 18], ax=axes[3],color='green')
        # axes[3].set(xlabel='Day')
        # axes[3].legend(['NTL','VZA'])
        #
        # sns.set(rc={'figure.figsize':(11.7,8.27)})
        # f, axes = plt.subplots(4, 4)
        # sns.regplot(x=zenith_0[:, 21, 19], y=ntl_0[:, 21, 19], ax=axes[0, 0],color='green', order=2, line_kws={"color": "black"})
        # axes[0, 0].set(xlabel='VZA', ylabel='NTL')
        # sns.regplot(x=zenith_0[:, 5, 5], y=ntl_0[:, 5, 5], ax=axes[0, 1],color='green', order=2, line_kws={"color": "black"})
        # axes[0, 1].set(xlabel='VZA', ylabel='NTL')
        # sns.regplot(x=zenith_0[:, 38, 41], y=ntl_0[:, 38, 41], ax=axes[0, 2],color='green', order=2, line_kws={"color": "black"})
        # axes[0, 2].set(xlabel='VZA', ylabel='NTL')
        # sns.regplot(x=zenith_0[:, 38, 5], y=ntl_0[:, 38, 5], ax=axes[0, 3],color='green', order=2, line_kws={"color": "black"})
        # axes[0, 3].set(xlabel='VZA', ylabel='NTL')
        #
        # sns.regplot(x=zenith_0[:, 5, 41], y=ntl_0[:, 5, 41], ax=axes[1, 0],color='green', order=2, line_kws={"color": "black"})
        # axes[1, 0].set(xlabel='VZA', ylabel='NTL')
        # sns.regplot(x=zenith_0[:, 28, 30], y=ntl_0[:, 28, 30], ax=axes[1, 1],color='green', order=2, line_kws={"color": "black"})
        # axes[1, 1].set(xlabel='VZA', ylabel='NTL')
        # sns.regplot(x=zenith_0[:, 15, 41], y=ntl_0[:, 15, 41], ax=axes[1, 2],color='green', order=2, line_kws={"color": "black"})
        # axes[1, 2].set(xlabel='VZA', ylabel='NTL')
        # sns.regplot(x=zenith_0[:, 43, 43], y=ntl_0[:, 43, 43], ax=axes[1, 3],color='green', order=2, line_kws={"color": "black"})
        # axes[1, 3].set(xlabel='VZA', ylabel='NTL')
        #
        # sns.regplot(x=zenith_0[:, 35, 21], y=ntl_0[:, 35, 21], ax=axes[2, 0],color='green', order=2, line_kws={"color": "black"})
        # axes[2, 0].set(xlabel='VZA', ylabel='NTL')
        # sns.regplot(x=zenith_0[:, 38, 26], y=ntl_0[:, 38, 26], ax=axes[2, 1],color='green', order=2, line_kws={"color": "black"})
        # axes[2, 1].set(xlabel='VZA', ylabel='NTL')
        # sns.regplot(x=zenith_0[:, 22, 3], y=ntl_0[:, 22, 3], ax=axes[2, 2],color='green', order=2, line_kws={"color": "black"})
        # axes[2, 2].set(xlabel='VZA', ylabel='NTL')
        # sns.regplot(x=zenith_0[:, 31, 1], y=ntl_0[:, 31, 1], ax=axes[2, 3],color='green', order=2, line_kws={"color": "black"})
        # axes[2, 3].set(xlabel='VZA', ylabel='NTL')
        #
        # sns.regplot(x=zenith_0[:, 4, 18], y=ntl_0[:, 4, 18], ax=axes[3, 0],color='green', order=2, line_kws={"color": "black"})
        # axes[3, 0].set(xlabel='VZA', ylabel='NTL')
        # sns.regplot(x=zenith_0[:, 25, 27], y=ntl_0[:, 25, 27], ax=axes[3, 1],color='green', order=2, line_kws={"color": "black"})
        # axes[3, 1].set(xlabel='VZA', ylabel='NTL')
        # sns.regplot(x=zenith_0[:, 16, 16], y=ntl_0[:, 16, 16], ax=axes[3, 2],color='green', order=2, line_kws={"color": "black"})
        # axes[3, 2].set(xlabel='VZA', ylabel='NTL')
        # sns.regplot(x=zenith_0[:, 32, 43], y=ntl_0[:, 32, 43], ax=axes[3, 3],color='green', order=2, line_kws={"color": "black"})
        # axes[3, 3].set(xlabel='VZA', ylabel='NTL')

        # zenith-radiance quadratic (ZRQ) model
        zenith_02 = zenith_0[:, :, :].reshape(30, 49*44)
        azimuth_02 = azimuth_0[:, :, :].reshape(30, 49*44)
        ntl_02 = ntl_0[:, :, :].reshape(30, 49*44)

        # depth is the ntl, vza, vaa
        df = np.zeros((30, 49*44, 3))
        df[:,:,0] = ntl_02
        df[:,:,1] = zenith_02
        df[:,:,2] = azimuth_02

        r0 = []
        idx = []
        pred = []
        for i in range(0, df.shape[1]):
            sample = df[:, i, :]
            # sample = sample[~np.isnan(sample).any(axis=1), :]
            sample = pd.DataFrame(sample, columns=['ntl', 'vza', 'vaa'])
            sample['vza2'] = sample['vza'] * sample['vza']
            sample['vaa2'] = sample['vaa'] * sample['vaa']
            sample['vzavaa'] = sample['vza'] * sample['vaa']

            ntl_0vza = ols("ntl ~ vza2 + vza + vaa2+ vaa+ vzavaa", data=sample, missing='drop').fit()
            ypred = ntl_0vza.predict(sample.loc[:, ['vza2', 'vza', 'vaa2', 'vaa', 'vzavaa']])

            r0.append(ntl_0vza.rsquared)
            idx.append(i)
            pred.append(ypred)

            # if ntl_0vza.rsquared > 0.9:
            #     print('index: ', i)
            #     print(ntl_0vza.summary())
            #     print("\nRetrieving manually the parameter estimates:")
            #     print(ntl_0vza._results.params)
            #
            #     sns.set(rc={'figure.figsize': (11.7, 8.27)})
            #     f, axes = plt.subplots(1, 1)
            #     sns.regplot(x=np.array(sample['vza']), y=np.array(sample['ntl']), ax=axes, color='green', order=2,
            #                 line_kws={"color": "black"})
            #     axes.set(xlabel='VZA_' + str(i), ylabel='ntl_0' + str(i))
            #
            #     ypred = ntl_0vza.predict(sample.loc[:, ['vza2', 'vza', 'vaa2', 'vaa', 'vzavaa']])
            #     sns.set(rc={'figure.figsize': (11.7, 11)})
            #     f, axes = plt.subplots(3, 1)
            #     sns.regplot(x=ypred, y=sample['ntl'], ax=axes[0],color='green', line_kws={"color": "black"})
            #     axes[0].set(xlabel='ypred', ylabel='ntl_0' + str(i))
            #     axes[0].legend(['R-squared = ' + str(ntl_0vza.rsquared)])
            #     sns.lineplot(x=range(1, len(sample)+1), y=ypred, ax=axes[1], color='green')
            #     axes[1].set(xlabel='ypred')
            #     sns.lineplot(x=range(1, len(sample)+1), y=sample['ntl'], ax=axes[1], color='orange')
            #     axes[1].set(xlabel='Day')
            #     axes[1].legend(['ypred', 'ntl'])
            #     sns.lineplot(x=range(1, len(sample)+1), y=sample['vza'], ax=axes[2], color='red')
            #     axes[2].set(xlabel='Day')
            #     axes[2].legend(['vza'])

        # sns.set(rc={'figure.figsize': (11.7, 8.27)})
        # f, axes = plt.subplots(1, 1)
        # sns.regplot(x=r0_vza, y=r0_vzavaa, ax=axes, color='green', order=1,line_kws={"color": "black"})
        # axes.set(xlabel='VZA_rsquare' + str(i), ylabel='VZAVAA_rsquare' + str(i))
        #
        # sns.set(rc={'figure.figsize': (11.7, 8.27)})
        # f, axes = plt.subplots(1, 1)
        # sns.histplot(r0)
        # axes.set(xlabel='VZAVAA_rsquare')
        #
        # plt.imshow(np.array(r0).reshape((49,44)))
        # plt.colorbar()

        pred2 = np.array(pred).transpose()
        pred2 = pred2.reshape(30, 49, 44)

        residuals = np.abs(np.subtract(ntl_0,pred2))

        # sns.set(rc={'figure.figsize': (11.7, 8.27)})
        # f, axes = plt.subplots(1, 1)
        # sns.lineplot(x=range(1, pred2.shape[0]+1), y=ntl_0[:, 19, 21], ax=axes, color='green')
        # axes.set(xlabel='ypred')
        # sns.lineplot(x=range(1, pred2.shape[0]+1), y=pred2[:, 19, 21], ax=axes, color='orange')
        # axes.set(xlabel='Day')
        # sns.lineplot(x=range(1, residuals.shape[0]+1), y=residuals[:, 19, 21], ax=axes, color='red')
        # axes.set(xlabel='Day')
        # axes.legend(['ntl_0', 'pred2', 'corrected signal'])
        #
        # # plt.imshow(np.nanmean(residuals, axis=0))
        # # plt.colorbar()
        # plt.imshow(np.nanmedian(residuals, axis=0))
        # plt.colorbar()

        ntl_0corrected = ntl_0
        ntl_0corrected[~np.isnan(ntl_0corrected)] = 1
        ntl_0corrected = np.multiply(ntl_0corrected, residuals)

        ntl_final[t:t + 30, :, :] = ntl_0corrected

    azimuth_ = np.load('C:/wgetdown/MosulVNP1/azimuth_' + year + '.npy')
    zenith_ = np.load('C:/wgetdown/MosulVNP1/zenith_' + year + '.npy')
    ntl_ = np.load('C:/wgetdown/MosulVNP2/ntl_' + year + '.npy')

    for t in range(ntl_final.shape[0] - 36, ntl_final.shape[0], 36):
        azimuth_0 = azimuth_[t:t+36, :, :]
        zenith_0 = zenith_[t:t+36, :, :]
        ntl_0 = ntl_[t:t+36, :, :]

        # zenith-radiance quadratic (ZRQ) model
        zenith_02 = zenith_0[:, :, :].reshape(36, 49*44)
        azimuth_02 = azimuth_0[:, :, :].reshape(36, 49*44)
        ntl_02 = ntl_0[:, :, :].reshape(36, 49*44)

        # depth is the ntl, vza, vaa
        df = np.zeros((36, 49*44, 3))
        df[:,:,0] = ntl_02
        df[:,:,1] = zenith_02
        df[:,:,2] = azimuth_02

        r0 = []
        idx = []
        pred = []
        for i in range(0, df.shape[1]):
            sample = df[:, i, :]
            # sample = sample[~np.isnan(sample).any(axis=1), :]
            sample = pd.DataFrame(sample, columns=['ntl', 'vza', 'vaa'])
            sample['vza2'] = sample['vza'] * sample['vza']
            sample['vaa2'] = sample['vaa'] * sample['vaa']
            sample['vzavaa'] = sample['vza'] * sample['vaa']

            ntl_0vza = ols("ntl ~ vza2 + vza + vaa2+ vaa+ vzavaa", data=sample, missing='drop').fit()
            ypred = ntl_0vza.predict(sample.loc[:, ['vza2', 'vza', 'vaa2', 'vaa', 'vzavaa']])

            r0.append(ntl_0vza.rsquared)
            idx.append(i)
            pred.append(ypred)

        pred2 = np.array(pred).transpose()
        pred2 = pred2.reshape(36, 49, 44)

        residuals = np.abs(np.subtract(ntl_0,pred2))

        ntl_0corrected = ntl_0
        ntl_0corrected[~np.isnan(ntl_0corrected)] = 1
        ntl_0corrected = np.multiply(ntl_0corrected, residuals)

        ntl_final[t:t + 36, :, :] = ntl_0corrected

    np.save(r'G:\backupC27152020\Population_Displacement_Final\Resources\Temp\ntl_corrected_annualByMonth' + year + '.npy', ntl_final)
    # .npy extension is added if not given

years = ['2013', '2014', '2015', '2016', '2017', '2018']
daynum = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','sep','Oct','Nov','Dec']
for year in years:
    ntl_corrected = np.load(r'G:\backupC27152020\Population_Displacement_Final\Resources\Temp\ntl_corrected_annualByMonth' + year + '.npy')
    day=0
    for item in daynum:
        print('starting the month... ' + item)
        if item == 'Jan':
            ntl_corrected0 = ntl_corrected[day:day+31, :, :]
            day = day + 31
        elif item == 'Feb':
            ntl_corrected0 = ntl_corrected[day:day + 29, :, :]
            day = day + 29
        elif item == 'Mar':
            ntl_corrected0 = ntl_corrected[day:day + 31, :, :]
            day = day + 31
        elif item == 'Apr':
            ntl_corrected0 = ntl_corrected[day:day + 30, :, :]
            day = day + 30
        elif item == 'May':
            ntl_corrected0 = ntl_corrected[day:day + 31, :, :]
            day = day + 31
        elif item == 'Jun':
            ntl_corrected0 = ntl_corrected[day:day + 30, :, :]
            day = day + 30
        elif item == 'Jul':
            ntl_corrected0 = ntl_corrected[day:day + 31, :, :]
            day = day + 31
        elif item == 'Aug':
            ntl_corrected0 = ntl_corrected[day:day + 31, :, :]
            day = day + 31
        elif item == 'Sep':
            ntl_corrected0 = ntl_corrected[day:day + 30, :, :]
            day = day + 30
        elif item == 'Oct':
            ntl_corrected0 = ntl_corrected[day:day + 31, :, :]
            day = day + 31
        elif item == 'Nov':
            ntl_corrected0 = ntl_corrected[day:day + 30, :, :]
            day = day + 30
        else:
            ntl_corrected0 = ntl_corrected[day:day + 31, :, :]
            day = day + 31

        ntl_corrected0_ = np.where(ntl_corrected0 == 65535.0, numpy.nan, ntl_corrected0)
        ntl_corrected0 = np.where(ntl_corrected0 == 65535.0, 0, ntl_corrected0)

        counts = np.count_nonzero(~np.isnan(ntl_corrected0_), 0)
        ntlmed = np.zeros((ntl_corrected0_.shape[1], ntl_corrected0_.shape[2]))
        for i in range(0, ntl_corrected0.shape[1]):
            for j in range(0, ntl_corrected0.shape[2]):
                ntlmed[i, j] = np.nanmedian(ntl_corrected0_[:, i, j])

        inRas = arcpy.Raster('C:/wgetdown/MosulVNP2/extract/brdf_prj_clp/'
                            'clipprj_DNB_BRDF_Corrected_NTLVNP46A2_A2013001_h22v05_001_2020129141119.tif')
        lowerLeft = arcpy.Point(inRas.extent.XMin, inRas.extent.YMin)
        cellSize = inRas.meanCellWidth
        dsc = arcpy.Describe(inRas)
        sr = dsc.SpatialReference

        myRaster = arcpy.NumPyArrayToRaster(ntlmed, lowerLeft, cellSize)
        arcpy.DefineProjection_management(myRaster, sr)
        out_rasterdataset = 'G:/backupC27152020/Population_Displacement_Final/Resources/VIIRS/VNP46A2/ntlMonthlyCorrected_' + item + year + '.tif'
        myRaster.save(out_rasterdataset)