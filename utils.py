# general imports
import os
import wrds
import json
import numpy as np
import pandas as pd
import shutil
from datetime import date


def compustat_wrds(variables, dataset, db):
    """
    Args:
        variables (str): string containing the Compustat variables to retrieve
        dataset (pandas.DataFrame): ECL.csv file as dataframe
        db: WRDS Python API connection object.

    Returns:
        pandas.DataFrame: ECL dataset merged with Compustat
    """
    
     # SQL query to extract the CompuStat data.
    query = """
    SELECT gvkey, datadate FROM comp_na_annual_all.funda 
    WHERE datafmt = 'STD'
    AND indfmt = 'INDL'
    AND consol = 'C'
    AND popsrc = 'D'
    AND datadate BETWEEN '1993-01-01' AND '2023-05-01'
    """
    query = query[:27] + ',' + variables + ' ' + query[27:]
    
    
    # fetch the data.
    try:
        wrds_comp = db.raw_sql(query)
    except:
        print('Connection with CompuStat failed.')

    # convert to appropriate data types
    dataset['datadate'] = pd.to_datetime(dataset['datadate'], dayfirst=True)
    wrds_comp['datadate'] = pd.to_datetime(wrds_comp['datadate'])
    dataset['gvkey'] = dataset['gvkey'].astype(float)
    wrds_comp['gvkey'] = wrds_comp['gvkey'].astype(float)

    # merge compustat and ECL datasets
    merged_dataset = dataset.merge(wrds_comp, on = ['gvkey', 'datadate'], how = 'left', indicator=True)
    left_only_records = len(merged_dataset.loc[merged_dataset['_merge'] == 'left_only'])
    print(f'{left_only_records} records in the dataset do not have an accompanying CompuStat record.')
    
    return merged_dataset


def compustat_local (path, dataset, update):
    
    """
    Args:
        path (str): path to local compustat .csv file.
        dataset (pandas.DataFrame): ECL.csv file as dataframe
        update (bool): If "True", return filtered compustat copy - can be used to update ECL.csv - deprecated

    Returns:
        pandas.DataFrame: ECL dataset merged with Compustat
    """
    
    # load compustat
    local_comp = pd.read_csv(path, low_memory=False)
    
    # convert to appropriate data types
    local_comp['datadate'] = pd.to_datetime(local_comp['datadate'])
    local_comp['gvkey'] = local_comp['gvkey'].astype(float)
    
    # filter on screening variables - get indices
    indfmt = local_comp.loc[~(local_comp['indfmt'] == 'INDL')].index
    consol = local_comp.loc[~(local_comp['consol'] == 'C')].index
    popsrc = local_comp.loc[~(local_comp['popsrc'] == 'D')].index
    datafmt = local_comp.loc[~(local_comp['datafmt'] == 'STD')].index
    datadate = local_comp.loc[(local_comp['datadate']  < '1993-01-01') | 
                              (local_comp['datadate']  > '2023-05-01')].index
    local_comp['src'] = local_comp['src'].fillna('0').astype(int)
    src = local_comp.loc[(local_comp['src'].isin([3,4,9,13,14,26,37,43,99,88]))].index

    # filter on screening variables - drop selected indices
    indices = list(indfmt) + list(consol) + list(popsrc) + list(datafmt) + list(datadate) + list(src)
    local_comp = local_comp.drop(indices).reset_index(drop=True)
    
    # return data for update
    if update:
        return local_comp
    
    # convert to appropriate data types
    dataset['datadate'] = pd.to_datetime(dataset['datadate'], dayfirst=True)
    dataset['gvkey'] = dataset['gvkey'].astype(float)
    
    # drop duplicates
    local_comp = local_comp.drop('cik', axis=1)

    # merge compustat and ECL datasets
    merged_dataset = dataset.merge(local_comp, on = ['gvkey', 'datadate'], how = 'left', indicator=True)
    left_only_records = len(merged_dataset.loc[merged_dataset['_merge'] == 'left_only'])

    
    return merged_dataset


def compute_features(dataset):
    """
    Args:
        dataset (pandas.DataFrame): ECL dataset merged with Compustat as a dataframe
        
    Returns: 
        ECL dataset with 28 financial predictors and the column indices of these predictors
    """

    # based on Mai et al. (2019) 
    # predictors for business failure prediction
    
    dataset['actlct'] = dataset['act'] / dataset['lct']
    dataset['apsale'] = dataset['ap'] / dataset['sale']
    dataset['cashat'] = dataset['che'] / dataset['at']
    dataset['chat'] = dataset['ch'] / dataset['at']
    dataset['chlct'] = dataset['ch'] / dataset['lct']
    dataset['ebit_dp_at'] = (dataset['ebit']+dataset['dp']) / dataset['at']
    dataset['ebitat'] = dataset['ebit'] / dataset['at']
    dataset['ebitsale'] = dataset['ebit'] / dataset['sale']
    dataset['fat'] = (dataset['dlc'] + (0.5*dataset['dltt'])) / dataset['at']
    dataset['invchinvt'] = dataset['invch'] / dataset['invt']
    dataset['invtsale'] = dataset['invt'] / dataset['sale']
    dataset['lct_ch_at'] = (dataset['lct']-dataset['ch']) / dataset['at']
    dataset['lctat'] = dataset['lct'] / dataset['at']
    dataset['lctlt'] = dataset['lct'] / dataset['lt']
    dataset['lctsale'] = dataset['lct'] / dataset['sale']
    dataset['ltat'] = dataset['lt'] / dataset['at']
    dataset['log_at'] = np.log(dataset['at'])
    dataset['log_sale']  = np.log(dataset['sale'])
    dataset['niat'] = dataset['ni'] / dataset['at']
    dataset['nisale'] = dataset['ni'] / dataset['sale']
    dataset['oiadpat'] = dataset['oiadp'] / dataset['at']
    dataset['oiadpsale'] = dataset['oiadp'] / dataset['sale']
    dataset['qalct'] = (dataset['act'] - dataset['invt']) / dataset['lct']
    dataset['reat'] = dataset['re'] / dataset['at']
    dataset['relct'] = dataset['re'] / dataset['lct']
    dataset['saleat'] = dataset['sale'] / dataset['at']
    dataset['seqat'] = dataset['seq'] / dataset['at']
    dataset['wcapat'] = dataset['wcap'] / dataset['at']
    
    # replace infinite values with NAs
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

    # store column indices
    predictors = dataset.iloc[:,-28:].columns

    return dataset, predictors