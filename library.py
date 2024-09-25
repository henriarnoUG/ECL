import numpy as np
import pandas as pd
import wrds
import json
import os
import shutil
import json
from datetime import date
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS 
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score

def compustat_wrds(variables, dataset, db):
    """
    Retrieve CompuStat variables and merge them into the ECL dataset.

    Args:
        variables (str): A string containing the CompuStat variables to retrieve.
        dataset (pandas.DataFrame): The ECL dataset.
        db: The WRDS Python API Connection object.

    Returns:
        pandas.DataFrame: The ECL dataset with CompuStat variables added.
    """
    
     # SQL query to extract the specified CompuStat variables.
    query = """
    SELECT gvkey, datadate FROM comp_na_annual_all.funda 
    WHERE datafmt = 'STD'
    AND indfmt = 'INDL'
    AND consol = 'C'
    AND popsrc = 'D'
    AND datadate BETWEEN '1993-01-01' AND '2023-05-01'
    """
    query = query[:27] + ',' + variables + ' ' + query[27:]
    
    
    # Fetch the data.
    try:
        wrds_comp = db.raw_sql(query)
    except:
        print('Connection with CompuStat failed.')

    # Convert datadate and gvkey columns to appropriate data types.
    dataset['datadate'] = pd.to_datetime(dataset['datadate'], dayfirst=True)
    wrds_comp['datadate'] = pd.to_datetime(wrds_comp['datadate'])
    dataset['gvkey'] = dataset['gvkey'].astype(float)
    wrds_comp['gvkey'] = wrds_comp['gvkey'].astype(float)

    # Merge CompuStat and the ECL dataset.
    merged_dataset = dataset.merge(wrds_comp, on = ['gvkey', 'datadate'], how = 'left', indicator=True)
    left_only_records = len(merged_dataset.loc[merged_dataset['_merge'] == 'left_only'])
    print(f'{left_only_records} records in the dataset do not have an accompanying CompuStat record.')
    
    return merged_dataset


def compustat_local (path, dataset, update):
    
    """
    Retrieve CompuStat data from a local CSV file and merge it with the ECL dataset.

    Args:
        path (str): The path to the local copy of CompuStat, stored as a .csv file.
        dataset (pandas.DataFrame): The ECL dataset.
        update (bool): If True, return CompuStat data for an update; otherwise, merge the data into the dataset.

    Returns:
        pandas.DataFrame: The updated ECL dataset or the CompuStat data for an update.
    """
    
    # Read in the local CompuStat data from the CSV file.
    local_comp = pd.read_csv(path, low_memory=False)
    
    # Read in the local CompuStat data from the CSV file.
    local_comp['datadate'] = pd.to_datetime(local_comp['datadate'])
    local_comp['gvkey'] = local_comp['gvkey'].astype(float)
    
    # Define indices to remove based on screening variables.
    indfmt = local_comp.loc[~(local_comp['indfmt'] == 'INDL')].index
    consol = local_comp.loc[~(local_comp['consol'] == 'C')].index
    popsrc = local_comp.loc[~(local_comp['popsrc'] == 'D')].index
    datafmt = local_comp.loc[~(local_comp['datafmt'] == 'STD')].index
    datadate = local_comp.loc[(local_comp['datadate']  < '1993-01-01') | 
                              (local_comp['datadate']  > '2023-05-01')].index
    
    # Additionally, remove records that do not correspond with relevant 10k records.
    local_comp['src'] = local_comp['src'].fillna('0').astype(int)
    src = local_comp.loc[(local_comp['src'].isin([3,4,9,13,14,26,37,43,99,88]))].index


    # Additionally, remove records that do not correspond with relevant 10k records.
    indices = list(indfmt) + list(consol) + list(popsrc) + list(datafmt) + list(datadate) + list(src)
    local_comp = local_comp.drop(indices).reset_index(drop=True)
    print('Dropped ' + str(len(indices)) + ' rows from CompuStat based on screening variables')
    
    # return data for update
    if update:
        return local_comp
    
    # Adjust the datatypes of the variables that we will use to merge CompuStat records and ECL records.
    dataset['datadate'] = pd.to_datetime(dataset['datadate'], dayfirst=True)
    dataset['gvkey'] = dataset['gvkey'].astype(float)
    
    # Drop duplicate column
    local_comp = local_comp.drop('cik', axis=1)

    # Merge CompuStat and the ECL dataset.
    merged_dataset = dataset.merge(local_comp, on = ['gvkey', 'datadate'], how = 'left', indicator=True)
    left_only_records = len(merged_dataset.loc[merged_dataset['_merge'] == 'left_only'])
    print(f'{left_only_records} records in the dataset do not have an accompanying CompuStat record.')

    
    return merged_dataset


def compute_features(dataset):
    """
    dataset (pandas.DataFrame): A Pandas dataframe containing the ECL dataset including the CompuStat variables.
    returns: (1) The ECL dataset with 28 financial ratios added and (2) the column indices of these ratios.
    """

    # Calculate 28 financial ratios based on Mai et al. (2019) for DL-based bankruptcy prediction.
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
    
    # Calculate 28 financial ratios based on Mai et al. (2019) for DL-based bankruptcy prediction.
    dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Store the columns.
    predictors = dataset.iloc[:,-28:].columns

    return dataset, predictors