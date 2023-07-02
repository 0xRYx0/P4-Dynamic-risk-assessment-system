''' 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Udacity :  Machine Learning DevOps Engineer (MLOps) Nano-degree
  Project :  #5 - A Dynamic Risk Assessment System
  Step    :  #1 - Data Ingestion
  Author  :  Rakan Yamani
  Date    :  04 June 2023
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

import pandas as pd
import os
import json
import logging
from datetime import datetime
import re

logging.basicConfig(filename='logging.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s - %(message)s', datefmt="%m/%d/%y %I:%M:%S %p")

logging.info("SUCCESS: Creating logging file named 'logging.log'") 


###[Task-1: Load config.json and get input and output paths]###
with open('config.json','r') as f:
    config = json.load(f) 
    input_folder_path = os.path.join(os.path.abspath('./'),'data',config['input_folder_path'])
    output_folder_path = os.path.join(os.path.abspath('./'),'data',config['output_folder_path'])
    test_data_path = os.path.join(os.path.abspath('./'),'data',config['test_data_path'])    
    logging.info("SUCCESS: Reading configuration file 'config.json' for Data Ingestion step") 

###[Task-2: Method for data ingestion]###
def merge_multiple_dataframe():
    '''
    This is a method to ingest required data and perform multiple operations:
        - Checking for input datasets 
        - Merging datasets 
        - Dropping duplicated values
        - Saving ingested metadata into 'ingestedfiles.txt'
        - Writing processed dataset into 'finaldata.csv'  
    '''
    
    # check for datasets, compile them together, and write to an output file
    merged_df = pd.DataFrame()
    used_data_files = []
    
    for file in os.listdir(input_folder_path):
    
        # arranging file path for ingestion metadata:
        path=os.path.join(input_folder_path, file)
        used_data_files.append(os.path.join(*path.split(os.path.sep)[-3:]))
        logging.info(f"SUCCESS: Ingesting file '{file}'")
        
        # merging csv files: 
        temp_df = pd.read_csv(path)
        merged_df = pd.concat([merged_df, temp_df], ignore_index=True)
        
    logging.info(f"SUCCESS: Merging data source files {used_data_files}")
    
    # dropping duplicated values: 
    merged_df = merged_df.drop_duplicates().reset_index(drop=1) 
    logging.info("SUCCESS: Dropping duplicated values from merged dataframe")

    # writing the final dataset:
    merged_df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)
    logging.info("SUCCESS: Saving ingested data into 'finaldata.csv'")

    
    # saving ingested metadata:
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), "w") as file:
        file.write(f"SUCCESS: Data Ingestion @ [{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}]\n")
        file.write(f"\n# Total ingested files: [{len(used_data_files)}]\n")
        file.write("\n".join(used_data_files))
        
    logging.info("SUCCESS: Saving ingested metadata into 'ingestedfiles.txt'")
    logging.info("SUCCESS: Required data has been ingested and merged!")


if __name__ == '__main__':
    
    # run main script
    logging.info("SUCCESS: ========= Sarting 'ingestion.py' =========")
    merge_multiple_dataframe()
    logging.info("SUCCESS: ========= End of 'ingestion.py' =========")


