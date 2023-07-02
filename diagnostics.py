''' 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Udacity :  Machine Learning DevOps Engineer (MLOps) Nano-degree
  Project :  #5 - A Dynamic Risk Assessment System
  Step    :  #5 - Model and Data Diagnostics
  Author  :  Rakan Yamani
  Date    :  05 June 2023
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import pandas as pd
import numpy as np
import timeit
import os
import pickle
import logging
import json
import subprocess

logging.basicConfig(filename='logging.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s - %(message)s', datefmt="%m/%d/%y %I:%M:%S %p")

###[Task-1: Load config.json and get input and output paths]###
with open('config.json','r') as f:
    config = json.load(f) 
    dataset_csv_path = os.path.join(os.path.abspath('./'),'data',config['output_folder_path']) 
    test_data_path = os.path.join(os.path.abspath('./'),'data',config['test_data_path']) 
    prod_deployment_path = os.path.join(os.path.abspath('./'),'model',config['prod_deployment_path']) 
    logging.info("SUCCESS: Reading configuration file 'config.json' for Model and Data Diagnostics step") 


###[Task-2: Method for getting model predictions]###
def model_predictions(X):
    '''
    This is a method to read the deployed model and a test dataset, calculate predictions before returning all predictions as list
    
    Args:
        X: feature as a Dataframe

    Returns:
        y_pred: model predictions
    '''
    model = pickle.load(open(os.path.join(prod_deployment_path,'trainedmodel.pkl'),'rb'))
    logging.info(f"SUCCESS: Loading required test data 'trainedmodel.pkl' from [{config['prod_deployment_path']}]") 

    y_pred = model.predict(X)
    logging.info("SUCCESS: Predicting provided data")
    
    return y_pred 


###[Task-3: Method for getting summary statistics]###
def dataframe_summary():
    '''
    This is a method to get model's summary statistics before returning a list containing all summary statistics
    
    Returns:
        list[dict]: list of dictionaries, each contains {column name, mean, median & std}
    '''
    
    summary_stats_dict = {}
    
    df = pd.read_csv(os.path.join(dataset_csv_path,'finaldata.csv')).drop(['corporation', 'exited'], axis=1)
    df = df.select_dtypes('number')
    logging.info(f"SUCCESS: Loading and preparing 'finaldata.csv' './data/{config['output_folder_path']}' folder")
    
    for col in df.columns:
        summary_stats_dict[col] = {'mean': df[col].mean(), 'median': df[col].median(), 'std': df[col].std()}
    
    logging.info("SUCCESS: Calculating summary statistics for prod model")
    
    return summary_stats_dict


###[Task-4: Method for counting missing data]### 
def missing_data():
    '''
    This is a method to calculating missing data in percentage
    '''
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv')).drop(['corporation', 'exited'], axis=1)
    logging.info(f"SUCCESS: Loading and preparing 'finaldata.csv' from './data/{config['output_folder_path']}' folder")

    missings = {col: {'percentage': perc} for col, perc in zip(df.columns, df.isna().sum() / df.shape[0] * 100)}
    logging.info("SUCCESS: Calculating missing data in percentage")

    return missings


###[Task-5: Method for getting timings]### 
def execution_time():
    '''
    This is a method to calculating timings of training and ingestion processes
    '''
    ingestion_time = []
    training_time = []
    
    for _ in range(25):
        ingestion_starttime = timeit.default_timer()
        _ = subprocess.run(['python', 'ingestion.py'], capture_output=True)
        ingestion_time.append(timeit.default_timer() - ingestion_starttime)
    
        training_starttime = timeit.default_timer()
        _ = subprocess.run(['python', 'training.py'], capture_output=True)
        training_time.append(timeit.default_timer() - training_starttime)
    
    logging.info("SUCCESS: Calculating training and ingestion timings")

    return [{'ingestion_mean_time': np.mean(ingestion_time)},{'training_mean_time': np.mean(training_time)}]
    

###[Task-6: Method for checking dependencies]###  
def outdated_packages_list():
    outdated = subprocess.check_output(['pip', 'list','--outdated'])
    outdated = outdated.decode("utf-8").split('\n')
    
    required_packages = []
    outdated_packages = []
    with open('requirements.txt','r') as f:
        requirements = f.read()
        requirements = requirements.split()
        for pkg in requirements:
            required_packages.append(pkg.split('==')[0])
        for out_pkg in outdated[2:-1]:
            out_pkg = out_pkg.split()
            if out_pkg[0] in set(required_packages):
                outdated_packages.append({'Package':out_pkg[0], 'Version':out_pkg[1],'Latest': out_pkg[2]})

    return outdated_packages
        
if __name__ == '__main__':
    
    # run main script
    logging.info("SUCCESS: ========= Sarting 'diagnostics.py' =========")
    
    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    X = test_df.drop(['corporation', 'exited'], axis=1)
    
    model_predictions(X)
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()
    
    logging.info("SUCCESS: ========= End of 'diagnostics.py' =========")