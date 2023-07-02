''' 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Udacity :  Machine Learning DevOps Engineer (MLOps) Nano-degree
  Project :  #5 - A Dynamic Risk Assessment System
  Step    :  #4 - Model Deployment
  Author  :  Rakan Yamani
  Date    :  04 June 2023
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

import os
import shutil
import logging
import json

logging.basicConfig(filename='logging.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s - %(message)s', datefmt="%m/%d/%y %I:%M:%S %p")

###[Task-1: Load config.json and get input and output paths]###
with open('config.json','r') as f:
    config = json.load(f) 
    dataset_csv_path = os.path.join(os.path.abspath('./'),'data',config['output_folder_path']) 
    output_model_path = os.path.join(os.path.abspath('./'),'model',config['output_model_path']) 
    prod_deployment_path = os.path.join(os.path.abspath('./'),'model',config['prod_deployment_path']) 
    logging.info("SUCCESS: Reading configuration file 'config.json' for Model Deployment step") 

###[Task-2: Method for deploying the model]###
def store_model_into_pickle():
    '''
    This is a method to deploy trained model to production environment
    '''
    shutil.copy(os.path.join(dataset_csv_path,'ingestedfiles.txt'),prod_deployment_path)
    shutil.copy(os.path.join(output_model_path,'trainedmodel.pkl'),prod_deployment_path)
    shutil.copy(os.path.join(output_model_path,'latestscore.txt'),prod_deployment_path)
    
    logging.info("SUCCESS: Deploying trained model to production")
    logging.info("SUCCESS: Copying required production files: 'trainedmodel.pkl', 'ingestfiles.txt' and 'latestscore.txt'")

if __name__ == '__main__':

    # run main script
    logging.info("SUCCESS: ========= Sarting 'deployment.py' =========")
    store_model_into_pickle()
    logging.info("SUCCESS: ========= End of 'deployment.py' =========")
    