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
# locations = df.groupby(['id', 'year', 'month', 'type']).mean().loc[:, ['lat', 'long']].groupby('id').max()
# data = families.join(locations, lsuffix='_families', rsuffix='_locations')
#
# mask = data['type'] == 'displacement'
# displacements = data[mask]
# mask = data['type'] == 'return'
# returns = data[mask]
#
# displacements.to_csv(field + 'displacements.csv')
# returns.to_csv(field + 'return.csv')

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

returns = gp.read_file(field + 'return.shp')
returns.columns = aux1
returnsUnits = gp.read_file(field + 'field_ntl_thiessen.shp')
displacements = gp.read_file(field + 'displacement.shp')
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

sns.set(rc={'figure.figsize': (30, 11)})
sns.set_style("whitegrid", {'axes.grid' : False})
overall.transpose().plot(kind="bar", width=0.8)
plt.xlabel(xlabel="Month", size=20)
plt.ylabel(ylabel = "Number of Families", size=20)
plt.xticks(rotation = 45, size=15)
plt.yticks(size=15)
plt.title('Current Returnee & Displaced Family Population', size=20)
plt.savefig(figures + 'Current_Returnee_Population_barchart.png', dpi=500, bbox_inches='tight')


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


average_returnee_2017.plot(column='average_returnee_2017', cmap='Spectral_r', linewidth=0.1, edgecolor='white', legend=True)
plt.title('average_returnee_2017')
average_returnee_2018.plot(column='average_returnee_2018', cmap='Spectral_r', linewidth=0.1, edgecolor='white', legend=True)
plt.title('average_returnee_2018')


fig, axs = plt.subplots(2, 1, figsize=(10, 10))
average_returnee_displaced_2017.plot(column='average_returnee_displaced_2017', cmap='Spectral_r', linewidth=0.1, edgecolor='white', legend=True, ax=axs[0])
axs[0].get_xaxis().set_visible(False)
axs[0].get_yaxis().set_visible(False)
axs[0].title.set_text('Average Returnee-Displaced Family Population (2017)')
average_returnee_displaced_2018.plot(column='average_returnee_displaced_2018', cmap='Spectral_r', linewidth=0.1, edgecolor='white', legend=True, ax=axs[1])
axs[1].get_xaxis().set_visible(False)
axs[1].get_yaxis().set_visible(False)
axs[1].title.set_text('Average Returnee-Displaced Family Population (2018)')






sns.set(rc={'figure.figsize': (20, 15)}, style="whitegrid")
fig, ax = plt.subplots(2, 1)
for n in range(0, ret.shape[0]):
    pp = sns.lineplot(x=list(ret.columns[-82:-23]), y=(ret.iloc[n, :][-82:-23]).astype('float'), ax=ax[0])
    ax[0].set(ylabel='Returned Family population')
for n in range(0, disp.shape[0]):
    pp = sns.lineplot(x=list(disp.columns[-128:-69]), y=(disp.iloc[n, :][-105:-46]).astype('float'), ax=ax[1])
    ax[1].set(ylabel='Displaced Family population')
fig.suptitle('Current Returned and Displaced Population (Not cumulative)')
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation = 90)

pp.get_figure().savefig(figures + 'Current_Returned_and_Displaced_Population.png')

ff = pd.DataFrame(index=aux2)
ff = ret.loc[:, aux1[2:-3]].apply(lambda x: x[x >= 0].sum(), axis=0)

lables = list(ff.index)
fig, ax = plt.subplots(figsize=(20,15))
ff.plot(cmap="Spectral", ax = ax)
plt.xlabel('Month/Year', fontsize=18)
plt.ylabel('Sum of Nightlight For The City', fontsize=16)
plt.xticks(range(0,len(list(ff.index))), lables, rotation='vertical')

# Map values
# Ret
vmin = np.nanmin(ret.iloc[:, -82:-23].min())
vmax = np.nanmax(ret.iloc[:, -82:-23].max())

fig, axs = plt.subplots(2, 3, figsize=(18, 10))
pp = ret.plot(column='Ret2017_6Prop', cmap='Spectral_r', linewidth=0.1, ax=axs[0,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,0].get_xaxis().set_visible(False)
axs[0,0].get_yaxis().set_visible(False)
axs[0,0].title.set_text('Ret2017_6Prop Returned Population')
pp = ret.plot(column='Ret2017_12Prop', cmap='Spectral_r', linewidth=0.1, ax=axs[0,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,1].get_xaxis().set_visible(False)
axs[0,1].get_yaxis().set_visible(False)
axs[0,1].title.set_text('Ret2017_12Prop Returned Population')
pp = ret.plot(column='Ret2018_1Prop', cmap='Spectral_r', linewidth=0.1, ax=axs[0,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,2].get_xaxis().set_visible(False)
axs[0,2].get_yaxis().set_visible(False)
axs[0,2].title.set_text('Ret2018_1Prop Returned Population')
pp = ret.plot(column='Ret2018_5Prop', cmap='Spectral_r', linewidth=0.1, ax=axs[1,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,0].get_xaxis().set_visible(False)
axs[1,0].get_yaxis().set_visible(False)
axs[1,0].title.set_text('Ret2018_5Prop Returned Population')
pp = ret.plot(column='Ret2018_6Prop', cmap='Spectral_r', linewidth=0.1, ax=axs[1,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,1].get_xaxis().set_visible(False)
axs[1,1].get_yaxis().set_visible(False)
axs[1,1].title.set_text('Ret2018_6Prop Returned Population')
pp = ret.plot(column='Ret2018_7Prop', cmap='Spectral_r', linewidth=0.1, ax=axs[1,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,2].get_xaxis().set_visible(False)
axs[1,2].get_yaxis().set_visible(False)
axs[1,2].title.set_text('Ret2018_7Prop Returned Population')
fig.suptitle('Current Returnee Population for representative months')
pp.get_figure().savefig(figures + 'Current_Returnee_Population.png')

# Disp amd Ret
vmin = np.nanmin(disp.iloc[:, -69:-46].min())
vmax = np.nanmax(disp.iloc[:, -69:-46].max())

fig, axs = plt.subplots(3, 3, figsize=(18, 10))
pp = disp.plot(column='RetDisp2017_5Prop', cmap='Spectral_r', linewidth=0.1, ax=axs[0,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,0].get_xaxis().set_visible(False)
axs[0,0].get_yaxis().set_visible(False)
axs[0,0].title.set_text('RetDisp2017_5 Returned Population Change')
pp = disp.plot(column='RetDisp2017_6Prop', cmap='Spectral_r', linewidth=0.1, ax=axs[0,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,1].get_xaxis().set_visible(False)
axs[0,1].get_yaxis().set_visible(False)
axs[0,1].title.set_text('RetDisp2017_6 Returned Population Change')
pp = disp.plot(column='RetDisp2017_12Prop', cmap='Spectral_r', linewidth=0.1, ax=axs[0,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,2].get_xaxis().set_visible(False)
axs[0,2].get_yaxis().set_visible(False)
axs[0,2].title.set_text('RetDisp2017_12 Returned Population Change')
pp = disp.plot(column='RetDisp2018_1Prop', cmap='Spectral_r', linewidth=0.1, ax=axs[1,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,0].get_xaxis().set_visible(False)
axs[1,0].get_yaxis().set_visible(False)
axs[1,0].title.set_text('RetDisp2018_1 Returned Population Change')
pp = disp.plot(column='RetDisp2018_5Prop', cmap='Spectral_r', linewidth=0.1, ax=axs[1,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,1].get_xaxis().set_visible(False)
axs[1,1].get_yaxis().set_visible(False)
axs[1,1].title.set_text('RetDisp2018_5 Returned Population Change')
pp = disp.plot(column='RetDisp2018_6Prop', cmap='Spectral_r', linewidth=0.1, ax=axs[1,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,2].get_xaxis().set_visible(False)
axs[1,2].get_yaxis().set_visible(False)
axs[1,2].title.set_text('RetDisp2018_6 Returned Population Change')
pp = disp.plot(column='RetDisp2018_7Prop', cmap='Spectral_r', linewidth=0.1, ax=axs[2,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[2,0].get_xaxis().set_visible(False)
axs[2,0].get_yaxis().set_visible(False)
axs[2,0].title.set_text('RetDisp2018_7 Returned Population Change')
pp = disp.plot(column='RetDisp2018_8Prop', cmap='Spectral_r', linewidth=0.1, ax=axs[2,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[2,1].get_xaxis().set_visible(False)
axs[2,1].get_yaxis().set_visible(False)
axs[2,1].title.set_text('RetDisp2018_8 Returned Population Change')
pp = disp.plot(column='RetDisp2018_9Prop', cmap='Spectral_r', linewidth=0.1, ax=axs[2,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[2,2].get_xaxis().set_visible(False)
axs[2,2].get_yaxis().set_visible(False)
axs[2,2].title.set_text('RetDisp2018_9 Returned Population Change')
fig.suptitle('Current Return and Displacement Population for representative months')
pp.get_figure().savefig(figures + 'Current_Return_and_Displacement.png')

# Change
# Ret
vmin = np.nanmin(ret.iloc[:, -23:].min())
vmax = np.nanmax(ret.iloc[:, -23:].max())

fig, axs = plt.subplots(2, 3, figsize=(18, 10))
pp = ret.plot(column='Ret2017_12Prop_Mchange', cmap='Spectral_r', linewidth=0.1, ax=axs[0,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,0].get_xaxis().set_visible(False)
axs[0,0].get_yaxis().set_visible(False)
axs[0,0].title.set_text('Ret2017_12Prop_Mchange Returned Population Change')
pp = ret.plot(column='Ret2018_1Prop_Mchange', cmap='Spectral_r', linewidth=0.1, ax=axs[0,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,1].get_xaxis().set_visible(False)
axs[0,1].get_yaxis().set_visible(False)
axs[0,1].title.set_text('Ret2018_1Prop_Mchange Returned Population Change')
pp = ret.plot(column='Ret2018_5Prop_Mchange', cmap='Spectral_r', linewidth=0.1, ax=axs[0,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,2].get_xaxis().set_visible(False)
axs[0,2].get_yaxis().set_visible(False)
axs[0,2].title.set_text('Ret2018_5Prop_Mchange Returned Population Change')
pp = ret.plot(column='Ret2018_6Prop_Mchange', cmap='Spectral_r', linewidth=0.1, ax=axs[1,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,0].get_xaxis().set_visible(False)
axs[1,0].get_yaxis().set_visible(False)
axs[1,0].title.set_text('Ret2018_6Prop_Mchange Returned Population Change')
pp = ret.plot(column='Ret2018_7Prop_Mchange', cmap='Spectral_r', linewidth=0.1, ax=axs[1,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,1].get_xaxis().set_visible(False)
axs[1,1].get_yaxis().set_visible(False)
axs[1,1].title.set_text('Ret2018_7Prop_Mchange Returned Population Change')
pp = ret.plot(column='Ret2018_9Prop_Mchange', cmap='Spectral_r', linewidth=0.1, ax=axs[1,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,2].get_xaxis().set_visible(False)
axs[1,2].get_yaxis().set_visible(False)
axs[1,2].title.set_text('Ret2018_9Prop_Mchange Returned Population Change')
fig.suptitle('Change in Return for representative months in comparison to their previous month')
pp.get_figure().savefig(figures + 'Change_in_Return.png')

# Disp amd Ret
vmin = np.nanmin(disp.iloc[:, -23:].min())
vmax = np.nanmax(disp.iloc[:, -23:].max())

fig, axs = plt.subplots(3, 3, figsize=(18, 10))
pp = disp.plot(column='Sum2017_6Prop', cmap='Spectral_r', linewidth=0.1, ax=axs[0,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,0].get_xaxis().set_visible(False)
axs[0,0].get_yaxis().set_visible(False)
axs[0,0].title.set_text('Sum2017_6Prop_Mchange Returned Population Change')
pp = disp.plot(column='Sum2017_12Prop', cmap='Spectral_r', linewidth=0.1, ax=axs[0,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,1].get_xaxis().set_visible(False)
axs[0,1].get_yaxis().set_visible(False)
axs[0,1].title.set_text('Sum2017_12Prop_Mchange Returned Population Change')
pp = disp.plot(column='Sum2018_1Prop', cmap='Spectral_r', linewidth=0.1, ax=axs[0,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[0,2].get_xaxis().set_visible(False)
axs[0,2].get_yaxis().set_visible(False)
axs[0,2].title.set_text('Sum2018_1Prop_Mchange Returned Population Change')
pp = disp.plot(column='Sum2018_5Prop', cmap='Spectral_r', linewidth=0.1, ax=axs[1,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,0].get_xaxis().set_visible(False)
axs[1,0].get_yaxis().set_visible(False)
axs[1,0].title.set_text('Sum2018_5Prop_Mchange Returned Population Change')
pp = disp.plot(column='Sum2018_6Prop', cmap='Spectral_r', linewidth=0.1, ax=axs[1,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,1].get_xaxis().set_visible(False)
axs[1,1].get_yaxis().set_visible(False)
axs[1,1].title.set_text('Sum2018_6Prop_Mchange Returned Population Change')
pp = disp.plot(column='Sum2018_7Prop', cmap='Spectral_r', linewidth=0.1, ax=axs[1,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[1,2].get_xaxis().set_visible(False)
axs[1,2].get_yaxis().set_visible(False)
axs[1,2].title.set_text('Sum2018_7Prop_Mchange Returned Population Change')
pp = disp.plot(column='Sum2018_8Prop', cmap='Spectral_r', linewidth=0.1, ax=axs[2,0], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[2,0].get_xaxis().set_visible(False)
axs[2,0].get_yaxis().set_visible(False)
axs[2,0].title.set_text('Sum2018_8Prop_Mchange Returned Population Change')
pp = disp.plot(column='Sum2018_9Prop', cmap='Spectral_r', linewidth=0.1, ax=axs[2,1], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[2,1].get_xaxis().set_visible(False)
axs[2,1].get_yaxis().set_visible(False)
axs[2,1].title.set_text('Sum2018_9Prop_Mchange Returned Population Change')
pp = disp.plot(column='Sum2018_10Prop', cmap='Spectral_r', linewidth=0.1, ax=axs[2,2], edgecolor='white', legend=True, vmin=vmin, vmax=vmax)
axs[2,2].get_xaxis().set_visible(False)
axs[2,2].get_yaxis().set_visible(False)
axs[2,2].title.set_text('Sum2018_10Prop_Mchange Returned Population Change')
fig.suptitle('Change in Return and Displacement for representative months in comparison to their previous month')
pp.get_figure().savefig(figures + 'Change_in_Return_and_Displacement.png')

# Annual change
ret['F2015_1_change'] = ret['F2015_1'] - ret['F2014_1']
ret['F2016_1_change'] = ret['F2016_1'] - ret['F2015_1']
ret['F2017_1_change'] = ret['F2017_1'] - ret['F2016_1']
ret['F2017_11_change'] = ret['F2017_11'] - ret['F2017_1']
ret['F2017_12_change'] = ret['F2017_12'] - ret['F2017_1']

ret.iloc[:, -5:].sum()

vmin = np.nanmin(ret.iloc[:, -5:].min())
vmax = np.nanmax(ret.iloc[:, -5:].max())

fig, axs = plt.subplots(1, 2, figsize=(20, 10))
ret.plot(column='F2017_11_change', cmap='Spectral_r', linewidth=0.1, ax=axs[0], edgecolor='white', legend=True)
axs[0].get_xaxis().set_visible(False)
axs[0].get_yaxis().set_visible(False)
axs[0].title.set_text('F2017_11-F2017_1 Returned Population Change')
ret.plot(column='F2017_12_change', cmap='Spectral_r', linewidth=0.1, ax=axs[1], edgecolor='white', legend=True)
axs[1].get_xaxis().set_visible(False)
axs[1].get_yaxis().set_visible(False)
axs[1].title.set_text('F2017_12-F2017_1 Returned Population Change')



disp['F2015_1_change'] = disp['F2015_1'] - disp['F2014_1']
disp['F2016_1_change'] = disp['F2016_1'] - disp['F2015_1']
disp['F2017_1_change'] = disp['F2017_1'] - disp['F2016_1']
disp['F2018_1_change'] = disp['F2018_1'] - disp['F2017_1']
disp['F2018_12_change'] = disp['F2018_12'] - disp['F2018_1']

disp.iloc[:, -5:].sum()

