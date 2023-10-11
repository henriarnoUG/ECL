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

def get_CompuStat_WRDS (variables, dataset, db):
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


def get_CompuStat_local (path, dataset, update):
    
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
    
    # Adjust the datatypes of the variables that we will use to merge CompuStat records and ECL records.
    local_comp = local_comp.drop(['cik', 'at'], axis=1)

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


def update_config_file(path_config, start_year, end_year, user_agent, demo):
    '''
    Update the configuration file with new settings.

    Args:
        path_config (str): The path to the config.json file.
        start_year (int): The start year of the filings to crawl.
        end_year (int): The end year of the filings to crawl.
        user_agent (str): A string containing the user's email, which will be declared to the SEC when collecting the data.
        demo (bool): If True, only a limited number of examples will be crawled for demonstration purposes.

    Returns:
        dict: A copy of the original config file as a dictionary. The updated version overwrites the original config.json file.
    '''

    # Open the file.
    with open(path_config) as fp:    
        config = json.load(fp)

    # Store a copy
    copied = config.copy()

    # Update settings in the configuration file.
    config['edgar_crawler']['filing_types'] = ['10-K', '10-K405', '10-KT', '10KSB', '10KSB40']
    config['edgar_crawler']['start_year'] = start_year
    config['edgar_crawler']['end_year'] = end_year
    config['edgar_crawler']['quarters'] = [1,2,3,4]
    config['edgar_crawler']['user_agent'] = user_agent

    # Add a list of CIKs to extract to the config file in the demo.
    # In the update process, this can be left blank to crawl all data on the website.
    if demo:
        ciks = ["1318605", "1018724", "77476", "22701"]
        config['edgar_crawler']['cik_tickers'] = ciks

    # save the adjustments
    with open(path_config, 'w') as fp:    
        json.dump(config,fp)  
        
    return copied


def move_extracted_filings(path_records, path_corpus, demo):
    '''
    Move newly extracted filings to the EDGAR-corpus and return their metadata.

    Args:
        path_records (str): The path to the newly extracted records.
        path_corpus (str): The path to the EDGAR-corpus.
        demo (bool): If True, work with existing files for demonstration purposes.

    Returns:
        pandas.DataFrame: A DataFrame containing metadata for the newly extracted filings.

    The code moves the newly extracted filings to the EDGAR-corpus and removes them from the /update folder.
    '''
    
    # Create a list to store the meta-data of the newly extracted records in.
    meta_data = []

    # loop over the records
    for file in os.listdir(path_records):

        # Extract the year
        substring = file.split('10K')[1]
        year = substring.split('_')[1]

        # Check if the file is new.
        # In the demo, we work with existing files.
        target = path_corpus + year + '/'
        if (not os.path.exists(target + file)) or (demo):

            # if file does not exist, move to dataset (except in demo).
            if not demo:
                shutil.move(path_records + file, target)        

            # Read in the new file.
            with open(target + file, "r") as f:
                data = json.load(f)

            # Store metadata.
            filing = []
            cols = ['cik', 'company', 'filing_type', 'filing_date', 'period_of_report', 'state_of_inc', 'state_location', 'sic']
            for variable in cols:
                filing.append(data[variable])          
            filing.append('/' + year + '/' + file)
            meta_data.append(filing)


    # Store the metadata as a DataFrame.
    cols.append('filename')
    meta_data = pd.DataFrame(meta_data, columns=cols)

    # clear the remaining files from the update folder
    dataset_path = os.getcwd() + '/update/edgar-crawler/datasets/'
    for file in os.listdir(dataset_path):
        if os.path.isfile(dataset_path + file):
            os.remove(dataset_path + file)
        else:
            shutil.rmtree(dataset_path + file)
            
    return meta_data



def fuzzy_match_period_of_report(dataset, edgar, compustat, offset = 7):
    '''
    Match EDGAR and CompuStat records with the same CIK based on the period_of_report within a specified offset.

    Args:
        dataset (pandas.DataFrame): Result of an outer merge between EDGAR header records and CompuStat records.
        edgar (pandas.DataFrame): DataFrame containing EDGAR headers.
        compustat (pandas.DataFrame): DataFrame containing CompuStat headers.
        offset (int): The maximum number of days that the period_of_report between two records (with the same CIK) may differ. If the difference is smaller, they are matched.

    Returns:
        pandas.DataFrame: A DataFrame where EDGAR and CompuStat records with the same CIK are matched if their period_of_report lies close to each other (within the offset).

    The function extracts records that can be matched and keeps relevant columns, constructs a window for period_of_report matching, and returns a DataFrame with matched records.
    '''
    
    # Extract records that can be matched.
    compusub = dataset.loc[(~dataset['cik'].isna()) & (dataset['_merge'] == 'left_only')]
    edgarsub = dataset.loc[(~dataset['cik'].isna()) & (dataset['_merge'] == 'right_only')]

    # Keep relevant columns.
    edgarsub = edgarsub[['cik', 'company','period_of_report']]
    compusub = compusub[['cik', 'conm','period_of_report']]
    
    # Construct a window for period_of_report matching.
    edgarsub['lower_date'] = edgarsub['period_of_report'] + pd.DateOffset(days = offset*-1)
    edgarsub['upper_date'] = edgarsub['period_of_report'] + pd.DateOffset(days = offset)
    
    # Match the reports on CIK, keeping only records with dates within the window.
    df = compusub.merge(right=edgarsub, on='cik', how="left", indicator=True)
    df = df.loc[(df['period_of_report_x'] >= df['lower_date']) & (df['period_of_report_x'] <= df['upper_date'])]

    # Adjust the columns for the matched records.
    df = df.drop(['lower_date', 'upper_date', '_merge'], axis = 1)
    
    # Get the matched records from CompuStat and EDGAR databases.
    compustat_records = pd.merge(compustat.reset_index(), df, left_on = ['cik', 'period_of_report'], right_on = ['cik', 'period_of_report_x'] , how = 'inner').set_index('index')
    edgar_records = pd.merge(edgar.reset_index(), df, left_on = ['cik', 'period_of_report'], right_on = ['cik', 'period_of_report_y'] , how = 'inner').set_index('index')

    # Adjust the columns for the EDGAR records.
    compustat_records['period_of_report'] = compustat_records['period_of_report_y']
    compustat_records['period_of_report_compu'] = compustat_records['period_of_report_x']
    compustat_records['conm'] = compustat_records['conm_x']
    compustat_records = compustat_records.drop(['period_of_report_x', 'period_of_report_y', 'conm_x', 'conm_y', 'company'], axis = 1)

     # Adjust the columns for the EDGAR records.
    edgar_records['period_of_report'] = edgar_records['period_of_report_y']
    edgar_records['company'] = edgar_records['company_x']
    edgar_records = edgar_records.drop(['period_of_report_x', 'period_of_report_y', 'company_x', 'company_y', 'conm'], axis = 1)

    # Match on CIK/period_of_report.
    matched = pd.merge(compustat_records, edgar_records, on=['cik', 'period_of_report'], how='outer', indicator=True)
    matched = matched.drop_duplicates()

    # Get the indices of the unmatched records in the dataset.
    compustat_in_dataset = pd.merge(dataset.reset_index(), matched, left_on = ['cik', 'period_of_report'], right_on = ['cik', 'period_of_report_compu'] , how = 'inner').set_index('index')
    edgar_in_dataset = pd.merge(dataset.reset_index(), matched, left_on = ['cik', 'period_of_report'], right_on = ['cik', 'period_of_report'] , how = 'inner').set_index('index')

    comp_indices = compustat_in_dataset.index
    edgar_indices = edgar_in_dataset.index

    # Remove extra column 'matched' from the matched records.
    matched = matched.drop('period_of_report_compu', axis = 1)

    # Drop the matched records from the dataset.
    dataset = dataset.drop(comp_indices)
    dataset = dataset.drop(edgar_indices)
    
    # Add the matched records to the dataset.
    result = pd.concat([dataset, matched])
        
    return result


def fuzzy_match_name(dataset, edgar, compustat):
    '''
    Match EDGAR and CompuStat records based on the period_of_report and company name.

    Args:
        dataset (pandas.DataFrame): Result of an outer merge between EDGAR header records and CompuStat records.
        edgar (pandas.DataFrame): DataFrame containing EDGAR headers.
        compustat (pandas.DataFrame): DataFrame containing CompuStat headers.

    Returns:
        pandas.DataFrame: A DataFrame where EDGAR and CompuStat records are matched if their period_of_report is the same and they have the same name.

    The function extracts records that can be matched, keeps relevant columns, and returns a DataFrame with matched records.
    '''
    
    
    # Extract records that can be matched.
    compusub = dataset.loc[(dataset['_merge'] == 'left_only') & ((dataset['cik'] == 999999))]
    edgarsub = dataset.loc[(dataset['_merge'] == 'right_only')]

    # Keep relevant columns.
    edgarsub = edgarsub[['company','period_of_report']]
    compusub = compusub[['conm','period_of_report']]
    
    
    # Merge the CompuStat and EDGAR records on company name and period_of_report.
    matched = pd.merge(left=compusub, right=edgarsub, how='inner', left_on=['conm', 'period_of_report'], right_on=['company', 'period_of_report'], indicator=True)


    # Get the matched records from CompuStat and EDGAR databases.
    compustat_records = pd.merge(compustat.reset_index(), matched, left_on = ['conm', 'period_of_report'], right_on = ['conm', 'period_of_report'] , how = 'inner', indicator=False).set_index('index')
    edgar_records = pd.merge(edgar.reset_index(), matched, left_on = ['company', 'period_of_report'], right_on = ['company', 'period_of_report'] , how = 'inner', indicator=False).set_index('index')

    # Adjust the columns for the EDGAR records.
    compustat_records = compustat_records.drop(['company', '_merge'], axis = 1)
    edgar_records = edgar_records.drop(['conm', '_merge'], axis = 1)

    # Match on company name/period_of_report and adjust CIK.
    matched = pd.merge(compustat_records, edgar_records, left_on=['conm', 'period_of_report'], right_on=['company', 'period_of_report'], how='outer', indicator=True)
    matched['cik'] = matched['cik_y']
    matched = matched.drop(['cik_x', 'cik_y'], axis=1)

    # Get the indices of the unmatched records.
    compustat_in_dataset = pd.merge(dataset.reset_index(), matched, left_on = ['conm', 'period_of_report'], right_on = ['conm', 'period_of_report'] , how = 'inner').set_index('index')
    edgar_in_dataset = pd.merge(dataset.reset_index(), matched, left_on = ['company', 'period_of_report'], right_on = ['company', 'period_of_report'] , how = 'inner').set_index('index')

    comp_indices = compustat_in_dataset.index
    edgar_indices = edgar_in_dataset.index

    # Drop the matched records from the dataset.
    dataset = dataset.drop(comp_indices)
    dataset = dataset.drop(edgar_indices)

    # Add the matched records.
    result = pd.concat([dataset, matched])
    
    return result