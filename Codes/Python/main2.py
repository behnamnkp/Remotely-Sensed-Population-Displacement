import pandas as pd
import numpy as np
import zipfile
import time
import csv
import multiprocessing
from multiprocessing import *
from multiprocessing.connection import wait
from multiprocessing import Process, Pipe, current_process
import timeit

source_pth = '/users/bnikparv/Axle/'

def create_trajectory(dd, years):

    '''
    :param dd: A dataframe of family records
    :param years: years of study
    :return: Trajectory of families
    '''

    family_record_counts = dd.groupby('FAMILYID').size()

    trajectories = pd.DataFrame(columns=['LOCATIONID'] + years, index=family_record_counts.index)
    for idx, item in dd.loc[:,
                    ['LOCATIONID', 'RECENCY_DATE', 'ZIP', 'LATITUDE', 'LONGITUDE']].iterrows():
       trajectories.loc[idx, item['RECENCY_DATE']] = str(item['LATITUDE']) + ', ' + str(item['LONGITUDE'])
       trajectories.loc[idx, 'LOCATIONID'] = trajectories.loc[idx, 'LOCATIONID'] + ', ' + item['RECENCY_DATE'] + '_' + str(item['LOCATIONID'])

    return trajectories

def search(dd, tj, q):
    '''
    :param dd: A dataframe of family records
    :param tj: A dataframe of family displacements
    :param q: Shared memory
    :return: List of family locations
    '''
    dd['RECENCY_DATE2'] = dd.loc[:, 'RECENCY_DATE'].astype(str).str[:4]
    dd = dd[dd['RECENCY_DATE2'] != '2004']
    dd = dd[dd['RECENCY_DATE2'] != '2003']
    dd = dd.sort_values(['FAMILYID', 'archive_version_year']).drop_duplicates(subset='FAMILYID', keep='last')
    dd.set_index('FAMILYID', inplace=True)
    families = dd[dd.index.isin(tj.index)]

    print('check:')
    print(families.loc[:, ['RECENCY_DATE2', 'LATITUDE', 'LONGITUDE']])

    traj = tj
    aux = []
    for idx, family in families.iterrows():
        try:
            if str(traj.loc[idx, family['RECENCY_DATE2']]) == 'nan' or str(
                    traj.loc[idx, family['RECENCY_DATE2']]) == 'NaN':
                location = str(family['LATITUDE']) + ', ' + str(family['LONGITUDE'])
                aux.append(list([idx, family['RECENCY_DATE2'], location, family['LOCATIONID']]))

        except:
            print('problem with: ', idx)
    q.put(aux)

def main():

    NUM_PROCESSORS = 36
    ROWS = 200000
    ALL = (2000000000 // (NUM_PROCESSORS*ROWS) + 1) * ROWS * NUM_PROCESSORS

    col_list = ["FAMILYID", "RECENCY_DATE", "DOWNGRADE_REASON_CODE", "archive_version_year", "LOCATION_TYPE",
                "HEAD_HH_AGE_CODE", "CHILDRENHHCOUNT", "LOCATIONID", "LATITUDE", "LONGITUDE", "ZIP"]
    col_list2 = ['FAMILYID', 'DOWNGRADE_REASON_CODE', 'DOWNGRADE_DATE', 'RECENCY_DATE',
                 'LOCATION_TYPE', 'PRIMARY_FAMILY_IND', 'HOUSEHOLDSTATUS',
                 'TRADELINE_COUNT', 'HEAD_HH_AGE_CODE', 'LENGTH_OF_RESIDENCE',
                 'CHILDRENHHCOUNT', 'CHILDREN_IND', 'SUPPRESS_FTC', 'SUPPRESS_TPS',
                 'TELE_RESTRICTED_IND', 'SUPPRESS_MPS', 'ADDRESSTYPE',
                 'MAILABILITY_SCORE', 'WEALTH_FINDER_SCORE', 'FIND_DIV_1000',
                 'OWNER_RENTER_STATUS', 'ESTMTD_HOME_VAL_DIV_1000', 'MARITAL_STATUS',
                 'PPI_DIV_1000', 'MSA2000_CODE', 'MSA2000_IDENTIFIER', 'CSA2000_CODE',
                 'CBSACODE', 'CBSATYPE', 'CSACODE', 'LOCATIONID', 'HOUSE_NUM',
                 'HOUSE_NUM_FRACTION', 'STREET_PRE_DIR', 'STREET_NAME',
                 'STREET_POST_DIR', 'STREET_SUFFIX', 'UNIT_TYPE', 'UNIT_NUM', 'BOX_TYPE',
                 'BOX_NUM', 'ROUTE_TYPE', 'ROUTE_NUM', 'CITY', 'STATE', 'ZIP', 'ZIP4',
                 'DPBC', 'VACANT', 'USPSNOSTATS', 'BATHROOM_CNT', 'BEDROOM_CNT',
                 'CONSTRUCTION_TYPE_CODE', 'BUILT_YEAR', 'BUILDING_AREA', 'ROOM_CNT',
                 'LATITUDE', 'LONGITUDE', 'MATCHLEVEL', 'CENSUSSTATECODE',
                 'CENSUSCOUNTYCODE', 'CENSUSTRACT', 'CENSUSBLOCKGROUP',
                 'CENSUS2010COUNTYCODE', 'CENSUS2010TRACT', 'CENSUS2010BLOCK',
                 'archive_version_year']

    data = pd.read_csv(source_pth + 'Data/74b0365c06d4479a.csv', low_memory=False, usecols=col_list, dtype=object)
    data.iloc[:,3] = data.iloc[:,3].astype(str).str[:4]
    years = ['2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017','2018', '2019']

    # sample year
    data_frames = []
    data_refined = pd.DataFrame()
    for year in years:
       print("Let's start with year " + year + '...')
       # Filter the current year
       data.iloc[:, 2] = data.iloc[:, 2].astype(str).str[:4]
       df = data.loc[data.iloc[:, 2] == year, :]
      # # Filter out records with DOWNGRADE_REASON_CODE != NULL
       # df = df[df['DOWNGRADE_REASON_CODE'].isnull()]
       # from the duplicate family IDs, use the one with later archive date
       df = df.sort_values(['FAMILYID', 'archive_version_year']).drop_duplicates(subset='FAMILYID', keep='last')
       data_refined = pd.concat([data_refined, df])

       data_frames.append(df)

    data_refined.set_index('FAMILYID', inplace=True)

    trajectories = create_trajectory(data_refined, years)
    trajectories.to_csv(source_pth + 'Results/trajectories.csv')
    trajectories = pd.read_csv(source_pth + 'Results/trajectories.csv', dtype=object)

    print(multiprocessing.cpu_count())

    for item in range(1, ALL, NUM_PROCESSORS*ROWS):

        print('Progress(%): ', ((item + (NUM_PROCESSORS*ROWS)) / ALL) * 100)
        strt = timeit.default_timer()
        df = pd.read_csv(source_pth + 'Data/hs55wxjneeuraajp_csv.zip', compression='zip', low_memory=False,
                         nrows=NUM_PROCESSORS*ROWS, skiprows=item, dtype=object, names=col_list2, encoding_errors='ignore',
                         usecols=col_list)
        trajectories = pd.read_csv(source_pth + 'Results/trajectories.csv', dtype=object)
        trajectories.set_index('FAMILYID', inplace=True)

        processes = []
        qq = []

        for jj in range(0, df.shape[0], ROWS):
            q = Queue()
            p = Process(target=search, args=(df.iloc[jj:jj + ROWS, :], trajectories, q))
            processes.append(p)
            qq.append(q)

        for process in processes:
            process.start()

        for i in qq:
            loc = i.get()
            for element in loc:
                trajectories.loc[element[0], element[1]] = element[2]
                trajectories.loc[element[0], 'LOCATIONID'] = trajectories.loc[element[0], 'LOCATIONID'] + ', ' + \
                                                             element[1] + '_' + str(element[3])

        trajectories.reset_index(inplace=True)
        trajectories.to_csv(source_pth + 'Results/trajectories.csv', index=False)
        print(timeit.default_timer() - strt)

        for process in processes:
            process.join()

if __name__ == "__main__":
    main()