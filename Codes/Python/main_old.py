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
import arcpy
from arcpy import env
from arcpy.sa import *
from simpledbf import Dbf5
import geopandas as gp
import pysal as ps
import libpysal
import esda
from esda.moran import Moran
from splot.esda import moran_scatterplot
from splot.esda import plot_moran
from esda.moran import Moran_Local
from splot.esda import plot_local_autocorrelation
from splot.esda import lisa_cluster
from mpl_toolkits.mplot3d import Axes3D
# For statistics. Requires statsmodels 5.0 or more
from statsmodels.formula.api import ols
import statsmodels.api as sm
# Analysis of Variance (ANOVA) on linear models
from statsmodels.stats.anova import anova_lm
import seaborn as sns
from pysal.contrib.viz import mapping as maps

dirname = os.path.dirname('G:/backupC27152020/Population_Displacement_Final/')
# filename = os.path.join(dirname, '')

# Choose the year of the analysis:
years = ['2013','2014','2015','2016','2017','2018'] # 2014-15-16-17-18 is available now

for year in years:
    if int(year) <= 2014:
        lnduse_year = '2014'
    else:
        lnduse_year = year
    # Zonal
    # 1. Nightlight
    # 1.1. NTL
    arcpy.env.workspace = "Resources/poulation_disp.gdb/Data"
    inZoneData = "fishnetNTLMask2"
    zoneField = "Id"
    inValueRaster = 'Resources/VHR/landuse/hr' + lnduse_year + '.tif'
    outTable = 'Resources/Temp/NTL_hr' + year + '.dbf'
    # Execute ZonalStatisticsAsTable
    outZSaT = ZonalStatisticsAsTable(in_zone_data=inZoneData, zone_field=zoneField, in_value_raster=inValueRaster,
                                    out_table=outTable, ignore_nodata=True, statistics_type="SUM")

    inValueRaster = 'Resources/VHR/landuse/lr' + lnduse_year + '.tif'
    outTable = 'Resources/Temp/NTL_lr' + year + '.dbf'
    # Execute ZonalStatisticsAsTable
    outZSaT = ZonalStatisticsAsTable(in_zone_data=inZoneData, zone_field=zoneField, in_value_raster=inValueRaster,
                                     out_table=outTable, ignore_nodata=True, statistics_type="SUM")

    inValueRaster = 'Resources/VHR/landuse/nr' + lnduse_year + '.tif'
    outTable = 'Resources/Temp/NTL_nr' + year + '.dbf'
    # Execute ZonalStatisticsAsTable
    outZSaT = ZonalStatisticsAsTable(in_zone_data=inZoneData, zone_field=zoneField, in_value_raster=inValueRaster,
                                     out_table=outTable, ignore_nodata=True, statistics_type="SUM")

    inValueRaster = 'Resources/VHR/landuse/bg' + lnduse_year + '.tif'
    outTable = 'Resources/Temp/NTL_bg' + year + '.dbf'
    # Execute ZonalStatisticsAsTable
    outZSaT = ZonalStatisticsAsTable(in_zone_data=inZoneData, zone_field=zoneField, in_value_raster=inValueRaster,
                                     out_table=outTable, ignore_nodata=True, statistics_type="SUM")

    # 1.2. NTLDis
    inZoneData = "fishnetNTLDisIntrsct2"
    zoneField = "id_target"
    inValueRaster = 'Resources/VHR/landuse/hr' + lnduse_year + '.tif'
    outTable = 'Resources/Temp/NTLDis_hr' + year + '.dbf'
    # Execute ZonalStatisticsAsTable
    outZSaT = ZonalStatisticsAsTable(in_zone_data=inZoneData, zone_field=zoneField, in_value_raster=inValueRaster,
                                     out_table=outTable, ignore_nodata=True, statistics_type="SUM")

    inValueRaster = 'Resources/VHR/landuse/lr' + lnduse_year + '.tif'
    outTable = 'Resources/Temp/NTLDis_lr' + year + '.dbf'
    # Execute ZonalStatisticsAsTable
    outZSaT = ZonalStatisticsAsTable(in_zone_data=inZoneData, zone_field=zoneField, in_value_raster=inValueRaster,
                                     out_table=outTable, ignore_nodata=True, statistics_type="SUM")

    inValueRaster = 'Resources/VHR/landuse/nr' + lnduse_year + '.tif'
    outTable = 'Resources/Temp/NTLDis_nr' + year + '.dbf'
    # Execute ZonalStatisticsAsTable
    outZSaT = ZonalStatisticsAsTable(in_zone_data=inZoneData, zone_field=zoneField, in_value_raster=inValueRaster,
                                     out_table=outTable, ignore_nodata=True, statistics_type="SUM")

    inValueRaster = 'Resources/VHR/landuse/bg' + lnduse_year + '.tif'
    outTable = 'Resources/Temp/NTLDis_bg' + year + '.dbf'
    # Execute ZonalStatisticsAsTable
    outZSaT = ZonalStatisticsAsTable(in_zone_data=inZoneData, zone_field=zoneField, in_value_raster=inValueRaster,
                                     out_table=outTable, ignore_nodata=True, statistics_type="SUM")

    # 2. Population
    # 2.1. Pop
    inZoneData = "CensusTracts2"
    zoneField = "id"
    inValueRaster = 'Resources/VHR/landuse/hr' + lnduse_year + '.tif'
    outTable = 'Resources/Temp/pop_hr' + year + '.dbf'
    # Execute ZonalStatisticsAsTable
    outZSaT = ZonalStatisticsAsTable(in_zone_data=inZoneData, zone_field=zoneField, in_value_raster=inValueRaster,
                                     out_table=outTable, ignore_nodata=True, statistics_type="SUM")

    inValueRaster = 'Resources/VHR/landuse/lr' + lnduse_year + '.tif'
    outTable = 'Resources/Temp/pop_lr' + year + '.dbf'
    # Execute ZonalStatisticsAsTable
    outZSaT = ZonalStatisticsAsTable(in_zone_data=inZoneData, zone_field=zoneField, in_value_raster=inValueRaster,
                                     out_table=outTable, ignore_nodata=True, statistics_type="SUM")

    # 2.2. PopDis
    inZoneData = "fishnetNTLDisIntrsct2"
    zoneField = "id_target"
    inValueRaster = 'Resources/VHR/landuse/hr' + lnduse_year + '.tif'
    outTable = 'Resources/Temp/popDis_hr' + year + '.dbf'
    # Execute ZonalStatisticsAsTable
    outZSaT = ZonalStatisticsAsTable(in_zone_data=inZoneData, zone_field=zoneField, in_value_raster=inValueRaster,
                                     out_table=outTable, ignore_nodata=True, statistics_type="SUM")

    inValueRaster = 'Resources/VHR/landuse/lr' + lnduse_year + '.tif'
    outTable = 'Resources/Temp/popDis_lr' + year + '.dbf'
    # Execute ZonalStatisticsAsTable
    outZSaT = ZonalStatisticsAsTable(in_zone_data=inZoneData, zone_field=zoneField, in_value_raster=inValueRaster,
                                     out_table=outTable, ignore_nodata=True, statistics_type="SUM")

    # 3.Read fishnets and join data to them
    fishnetNTL = Dbf5('Resources/Temp/fishnetNTLMask2.dbf').to_dataframe()
    fishnetNTLDis = Dbf5('Resources/Temp/fishnetNTLDisIntrsct2.dbf').to_dataframe()
    fishnetPop = Dbf5('Resources/Temp/CensusTracts2.dbf').to_dataframe()
    fishnetPopDis = Dbf5('Resources/Temp/fishnetNTLDisIntrsct2.dbf').to_dataframe()

    NTL_Hres = Dbf5('Resources/Temp/NTL_hr' + year + '.dbf').to_dataframe()
    NTL_Lres = Dbf5('Resources/Temp/NTL_lr' + year + '.dbf').to_dataframe()
    NTL_Nres = Dbf5('Resources/Temp/NTL_nr' + year + '.dbf').to_dataframe()
    NTL_BG = Dbf5('Resources/Temp/NTL_bg' + year + '.dbf').to_dataframe()

    NTLDis_Hres = Dbf5('Resources/Temp/NTLDis_hr' + year + '.dbf').to_dataframe()
    NTLDis_Lres = Dbf5('Resources/Temp/NTLDis_lr' + year + '.dbf').to_dataframe()
    NTLDis_Nres = Dbf5('Resources/Temp/NTLDis_nr' + year + '.dbf').to_dataframe()
    NTLDis_BG = Dbf5('Resources/Temp/NTLDis_bg' + year + '.dbf').to_dataframe()

    Pop_Hres = Dbf5('Resources/Temp/pop_hr' + year + '.dbf').to_dataframe()
    Pop_Lres = Dbf5('Resources/Temp/pop_lr' + year + '.dbf').to_dataframe()

    PopDis_Hres = Dbf5('Resources/Temp/popDis_hr' + year + '.dbf').to_dataframe()
    PopDis_Lres = Dbf5('Resources/Temp/popDis_lr' + year + '.dbf').to_dataframe()

    # 3.1. NTL
    NTL = fishnetNTL.merge(NTL_Hres.loc[:, ['SUM']],
                             left_on=fishnetNTL.loc[:, 'Id'], right_on=NTL_Hres.loc[:, 'Id'], how='left')
    NTL.rename(columns={'SUM': 'NTL_hr' + year}, inplace=True)
    NTL = NTL.drop(columns=['key_0'])

    NTL = NTL.merge(NTL_Lres.loc[:, ['SUM']],
                        left_on=NTL.loc[:, 'Id'], right_on=NTL_Lres.loc[:, 'Id'], how='left')
    NTL.rename(columns={'SUM': 'NTL_lr' + year}, inplace=True)
    NTL = NTL.drop(columns=['key_0'])

    NTL = NTL.merge(NTL_Nres.loc[:, ['SUM']],
                        left_on=NTL.loc[:, 'Id'], right_on=NTL_Nres.loc[:, 'Id'], how='left')
    NTL.rename(columns={'SUM': 'NTL_nr' + year}, inplace=True)
    NTL = NTL.drop(columns=['key_0'])

    NTL = NTL.merge(NTL_BG.loc[:, ['SUM']],
                        left_on=NTL.loc[:, 'Id'], right_on=NTL_BG.loc[:, 'Id'], how='left')
    NTL.rename(columns={'SUM': 'NTL_bg' + year}, inplace=True)
    NTL = NTL.drop(columns=['key_0'])

    NTL.iloc[:, 3:].describe()

    # 3.2 NTLDis
    NTLDis = fishnetNTLDis.merge(NTLDis_Hres.loc[:, ['SUM']],
                                   left_on=fishnetNTLDis.loc[:, 'id_target'],
                                   right_on=NTLDis_Hres.loc[:, 'id_target'],
                                   how='left')
    NTLDis.rename(columns={'SUM': 'NTLDis_hr' + year}, inplace=True)
    NTLDis = NTLDis.drop(columns=['key_0'])

    NTLDis = NTLDis.merge(NTLDis_Lres.loc[:, ['SUM']],
                              left_on=NTLDis.loc[:, 'id_target'], right_on=NTLDis_Lres.loc[:, 'id_target'],
                              how='left')
    NTLDis.rename(columns={'SUM': 'NTLDis_lr' + year}, inplace=True)
    NTLDis = NTLDis.drop(columns=['key_0'])

    NTLDis = NTLDis.merge(NTLDis_Nres.loc[:, ['SUM']],
                              left_on=NTLDis.loc[:, 'id_target'], right_on=NTLDis_Nres.loc[:, 'id_target'],
                              how='left')
    NTLDis.rename(columns={'SUM': 'NTLDis_nr' + year}, inplace=True)
    NTLDis = NTLDis.drop(columns=['key_0'])

    NTLDis = NTLDis.merge(NTLDis_BG.loc[:, ['SUM']],
                              left_on=NTLDis.loc[:, 'id_target'], right_on=NTLDis_BG.loc[:, 'id_target'],
                              how='left')
    NTLDis.rename(columns={'SUM': 'NTLDis_bg' + year}, inplace=True)
    NTLDis = NTLDis.drop(columns=['key_0'])

    NTLDis.iloc[:, 3:].describe()

    # 3.3. Pop
    Pop = fishnetPop.merge(Pop_Hres.loc[:, ['SUM']],
                             left_on=fishnetPop.loc[:, 'id'], right_on=Pop_Hres.loc[:, 'id'], how='left')
    Pop.rename(columns={'SUM': 'pop_hr' + year}, inplace=True)
    Pop = Pop.drop(columns=['key_0'])

    Pop = Pop.merge(Pop_Lres.loc[:, ['SUM']],
                        left_on=Pop.loc[:, 'id'], right_on=Pop_Lres.loc[:, 'id'], how='left')
    Pop.rename(columns={'SUM': 'pop_lr' + year}, inplace=True)
    Pop = Pop.drop(columns=['key_0'])

    Pop.iloc[:, 4:].describe()

    # 3.4 PopDis
    PopDis = fishnetPopDis.merge(PopDis_Hres.loc[:, ['SUM']],
                                   left_on=fishnetPopDis.loc[:, 'id_target'],
                                   right_on=PopDis_Hres.loc[:, 'id_target'],
                                   how='left')
    PopDis.rename(columns={'SUM': 'popDis_hr' + year}, inplace=True)
    PopDis = PopDis.drop(columns=['key_0'])

    PopDis = PopDis.merge(PopDis_Lres.loc[:, ['SUM']],
                              left_on=PopDis.loc[:, 'id_target'], right_on=PopDis_Lres.loc[:, 'id_target'],
                              how='left')
    PopDis.rename(columns={'SUM': 'popDis_lr' + year}, inplace=True)
    PopDis = PopDis.drop(columns=['key_0'])

    PopDis.iloc[:, 5:].describe()

    # 3.5. Fill NA with 0
    NTL = NTL.fillna(0)
    NTLDis = NTLDis.fillna(0)
    Pop = Pop.fillna(0)
    PopDis = PopDis.fillna(0)

    # 4. Area
    NTL.loc[:, ['NTL_hr' + year]] = NTL.loc[:, ['NTL_hr' + year]] * 9
    NTL.loc[:, ['NTL_lr' + year]] = NTL.loc[:, ['NTL_lr' + year]] * 9
    NTL.loc[:, ['NTL_nr' + year]] = NTL.loc[:, ['NTL_nr' + year]] * 9
    NTL.loc[:, ['NTL_bg' + year]] = NTL.loc[:, ['NTL_bg' + year]] * 9

    NTLDis.loc[:, ['NTLDis_hr' + year]] = NTLDis.loc[:, ['NTLDis_hr' + year]] * 9
    NTLDis.loc[:, ['NTLDis_lr' + year]] = NTLDis.loc[:, ['NTLDis_lr' + year]] * 9
    NTLDis.loc[:, ['NTLDis_nr' + year]] = NTLDis.loc[:, ['NTLDis_nr' + year]] * 9
    NTLDis.loc[:, ['NTLDis_bg' + year]] = NTLDis.loc[:, ['NTLDis_bg' + year]] * 9

    Pop.loc[:, ['pop_hr' + year]] = Pop.loc[:, ['pop_hr' + year]] * 9
    Pop.loc[:, ['pop_lr' + year]] = Pop.loc[:, ['pop_lr' + year]] * 9

    PopDis.loc[:, ['popDis_hr' + year]] = PopDis.loc[:, ['popDis_hr' + year]] * 9
    PopDis.loc[:, ['popDis_lr' + year]] = PopDis.loc[:, ['popDis_lr' + year]] * 9

    # 5. Verification
    ComparNTL = NTL.merge(NTLDis.groupby(['id_NTL']).sum(),
                            left_on=NTL.loc[:, 'Id'], right_on=NTLDis.groupby(['id_NTL']).sum().index, how='left')
    ComparNTL = ComparNTL.drop(columns=['key_0'])

    print('Error NTLArea_hr... ' + str((ComparNTL['NTL_hr' + year] - ComparNTL['NTLDid_hr' + year]).sum()))
    print('Error NTLArea_lr... ' + str((ComparNTL['NTL_lr' + year] - ComparNTL['NTLDis_lr' + year]).sum()))
    print('Error NTLArea_nr... ' + str((ComparNTL['NTL_nr' + year] - ComparNTL['NTLDis_nr' + year]).sum()))
    print('Error NTLArea_bg... ' + str((ComparNTL['NTL_bg' + year] - ComparNTL['NTLDis_bg' + year]).sum()))

    ComparPop = Pop.merge(PopDis.groupby(['id_census']).sum(),
                            left_on=Pop.loc[:, 'id'], right_on=PopDis.groupby(['id_census']).sum().index,
                            how='left')
    ComparPop = ComparPop.drop(columns=['key_0'])

    print('Error popArea_hr... ' + str((ComparPop['pop_hr' + year] - ComparPop['popDis_hr' + year]).sum()))
    print('Error popArea_lr... ' + str((ComparPop['pop_lr' + year] - ComparPop['popDis_lr' + year]).sum()))

    NTL['NTL_Area'] = NTL.loc[:, ['NTL_hr' + year, 'NTL_lr' + year, 'NTL_nr' + year, 'NTL_bg' + year]]\
        .astype('float').sum(axis=1)
    Pop['pop_Area'] = Pop.loc[:, ['pop_hr' + year, 'pop_lr' + year]].astype('float').sum(axis=1)

    NTL.to_csv('Resources/Temp/NTL' + year + '.csv')
    NTLDis.to_csv('Resources/Temp/NTLDis' + year + '.csv')
    Pop.to_csv('Resources/Temp/Pop' + year + '.csv')
    PopDis.to_csv('Resources/Temp/PopDis' + year + '.csv')

    NTLDisJoin = NTLDis.merge(NTL.loc[:, ['Id', 'NTL_Area']],
                                  left_on=NTLDis.loc[:, 'id_NTL'], right_on=NTL.loc[:, 'Id'], how='left')
    NTLDisJoin = NTLDisJoin.drop(columns=['key_0'])

    PopDisJoin = PopDis.merge(Pop.loc[:, ['id', 'Pop_Area']],
                                  left_on=PopDis.loc[:, 'id_census'], right_on=Pop.loc[:, 'id'], how='left')
    PopDisJoin = PopDisJoin.drop(columns=['key_0'])

    # Nightlight
    for index, row in NTLDisJoin.iterrows():
        if row['NTLDis_hr' + year] > 0:
            NTLDisJoin.loc[index, 'NTL_hr_final' + year] = (row['NTLDis_hr' + year] / row['NTL_Area']) * row['NTL']
        else:
            NTLDisJoin.loc[index, 'NTL_hr_final' + year] = 0

    for index, row in NTLDisJoin.iterrows():
        if row['NTLDis_lr' + year] > 0:
            NTLDisJoin.loc[index, 'NTLDis_lr_final' + year] = (row['NTLDis_lr' + year] / row['NTL_Area']) * row['NTL']
        else:
            NTLDisJoin.loc[index, 'NTLDis_lr_final' + year] = 0

    for index, row in NTLDisJoin.iterrows():
        if row['NTLDis_nr' + year] > 0:
            NTLDisJoin.loc[index, 'NTLDis_nr_final' + year] = (row['NTLDis_nr' + year] / row['NTL_Area']) * row['NTL']
        else:
            NTLDisJoin.loc[index, 'NTLDis_nr_final' + year] = 0

    for index, row in NTLDisJoin.iterrows():
        if row['NTLDis_bg' + year] > 0:
            NTLDisJoin.loc[index, 'NTLDis_bg_final' + year] = (row['NTLDis_bg' + year] / row['NTL_Area']) * row['NTL']
        else:
            NTLDisJoin.loc[index, 'NTLDis_bg_final' + year] = 0

    # population
    for index, row in PopDisJoin.iterrows():
        if row['popDis_hr' + year] > 0:
            PopDisJoin.loc[index, 'popDis_hr_final' + year] = (row['popDis_hr' + year] / row['Pop_Area']) * row[
                'estPop2013']
        else:
            PopDisJoin.loc[index, 'popDis_hr_final' + year] = 0

    for index, row in PopDisJoin.iterrows():
        if row['popDis_lr' + year] > 0:
            PopDisJoin.loc[index, 'popDis_lr_final' + year] = (row['popDis_lr' + year] / row['Pop_Area']) * row[
                'estPop2013']
        else:
            PopDisJoin.loc[index, 'popDis_lr_final' + year] = 0

    vars = PopDisJoin.loc[:,
           ['id_target', 'id_census', 'id_NTL', 'estPop2013', 'NTL', 'popDis_hr' + year, 'popDis_lr' + year,
            'popDis_hr_final' + year, 'popDis_lr_final' + year, 'Pop_Area']]. \
        merge(NTLDisJoin.loc[:,
              ['NTLDis_hr' + year, 'NTLDis_lr' + year, 'NTLDis_nr' + year, 'NTLDis_bg' + year, 'NTLDis_hr_final' + year,
               'NTLDis_lr_final' + year, 'NTLDis_nr_final' + year, 'NTLDis_bg_final' + year, 'NTL_Area']],
              left_on=PopDisJoin.loc[:, 'id_target'], right_on=NTLDisJoin.loc[:, 'id_target'], how='left')
    vars = vars.drop(columns=['key_0'])
    vars.to_csv('Results/vars_all' + year + '.csv')

    vars_id_target = PopDisJoin.loc[:, ['id_target', 'popDis_hr_final' + year, 'popDis_lr_final' + year]]. \
        merge(
        NTLDisJoin.loc[:,
        ['NTLDis_hr_final' + year, 'NTLDis_lr_final' + year, 'NTLDis_nr_final' + year, 'NTLDis_bg_final' + year]],
        left_on=PopDisJoin.loc[:, 'id_target'], right_on=NTLDisJoin.loc[:, 'id_target'], how='left')
    vars_id_target = vars_id_target.drop(columns=['key_0'])
    vars_id_target.to_csv('Results/vars_id_target' + year + '.csv')
    vars_id_target.iloc[:, 1:].corr().to_csv('Results/corr_id_target' + year + '.csv')

    vars_id_NTL = PopDisJoin.loc[:, ['popDis_hr_final' + year, 'popDis_lr_final' + year]] \
        .merge(NTLDisJoin.loc[:,
               ['id_NTL', 'NTLDis_hr_final' + year, 'NTLDis_lr_final' + year, 'NTLDis_nr_final' + year, 'NTLDis_bg_final' + year]],
               left_on=PopDisJoin.loc[:, 'id_target'], right_on=NTLDisJoin.loc[:, 'id_target'], how='left')
    vars_id_NTL = vars_id_NTL.drop(columns=['key_0'])
    vars_id_NTL_grouped = vars_id_NTL.groupby(['id_NTL']).sum()
    vars_id_NTL_grouped.to_csv('Results/vars_id_NTL_grouped' + year + '.csv')
    vars_id_NTL_grouped.iloc[:, :].corr().to_csv(
        'Results/corr_id_NTL_grouped' + year + '.csv')

    vars_id_Pop = PopDisJoin.loc[:, ['id_census', 'popDis_hr_final' + year, 'popDis_lr_final' + year]] \
        .merge(NTLDisJoin.loc[:,['NTLDis_hr_final' + year, 'NTLDis_lr_final' + year, 'NTLDis_nr_final' + year, 'NTLDis_bg_final' + year]],
        left_on=PopDisJoin.loc[:, 'id_target'], right_on=NTLDisJoin.loc[:, 'id_target'], how='left')
    vars_id_Pop = vars_id_Pop.drop(columns=['key_0'])

    vars_id_Pop_grouped = vars_id_Pop.groupby(['id_census']).sum()
    vars_id_Pop_grouped.to_csv('Results/vars_id_Pop_grouped' + year + '.csv')
    vars_id_Pop_grouped.iloc[:, :].corr().to_csv('Results/corr_id_Pop_grouped' + year + '.csv')

# 6. model:
# variables: NTLDis15_Hres_final,NTLDis15_Lres_final,NTLDis15_Road_final
vars_id_NTL_grouped = pd.read_csv('Results/vars_id_NTL_grouped.csv')
vars_id_NTL_grouped.rename(columns={'NTLDis_hr_final2013': 'Hres'}, inplace=True)
vars_id_NTL_grouped.rename(columns={'NTLDis_lr_final2013': 'Lres'}, inplace=True)
vars_id_NTL_grouped.rename(columns={'NTLDis_nr_final2013': 'Nres'}, inplace=True)
vars_id_NTL_grouped.rename(columns={'NTLDis_bg_final2013': 'BG'}, inplace=True)
vars_id_NTL_grouped['popDis_r_final2013'] = vars_id_NTL_grouped.loc[:, 'popDis_hr_final2013'] + \
                                            vars_id_NTL_grouped.loc[:,'popDis_hr_final2013']

censuspop13 = vars_id_NTL_grouped['popDis_r_final2013'].to_frame('pop13census')
censuspop13['id_NTL'] = range(1, len(censuspop13)+1)
censuspop13.to_csv('Results/CensusPop13.csv')

model3 = ols("popDis_r_final2013 ~ Hres + Lres + Nres + BG",
             vars_id_NTL_grouped).fit()
print(model3.summary())
print("\nRetrieving manually the parameter estimates:")
print(model3._results.params)

fig = sm.graphics.plot_partregress_grid(model3)
fig.tight_layout(pad=1.0)

fig = sm.graphics.plot_ccpr(model3)
fig.tight_layout(pad=1.0)
fig = sm.graphics.plot_ccpr_grid(model3)
fig.tight_layout(pad=1.0)

# predictor
for year in years:
    vars_id_NTL_grouped = pd.read_csv('Results/vars_id_NTL_grouped.csv')
    vars_id_NTL_grouped.rename(columns={'NTLDis_hr_final'+ year: 'Hres'}, inplace=True)
    vars_id_NTL_grouped.rename(columns={'NTLDis_lr_final'+ year: 'Lres'}, inplace=True)
    vars_id_NTL_grouped.rename(columns={'NTLDis_nr_final'+ year: 'Nres'}, inplace=True)
    vars_id_NTL_grouped.rename(columns={'NTLDis_bg_final'+ year: 'BG'}, inplace=True)
    pop = model3.predict(vars_id_NTL_grouped.loc[:,['Hres', 'Lres', 'Nres', 'BG']])
    print('population of the ' + year + ': ' + pop.sum())

    pop = pop.to_frame('pop13')
    pop['id_NTL'] = range(1, len(pop)+1)

    pop.to_csv('Results/EstimatedPop13.csv')