''' 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Udacity :  Machine Learning DevOps Engineer (MLOps) Nano-degree
  Project :  #5 - A Dynamic Risk Assessment System
  Step    :  #8 - Process Automation
  Author  :  Rakan Yamani
  Date    :  09 June 2023
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
import os
import json
import logging
import pandas as pd
from sklearn.metrics import f1_score

import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting

logging.basicConfig(filename='logging.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s - %(message)s', datefmt="%m/%d/%y %I:%M:%S %p")

###[Task-1: Load config.json and get input and output paths]###
with open('config.json','r') as f:
    config = json.load(f) 
    input_data_path = os.path.join(os.path.abspath('./'),'data',config['input_folder_path']) 
    output_folder_path = os.path.join(os.path.abspath('./'),'data',config['output_folder_path']) 
    test_data_path = os.path.join(os.path.abspath('./'),'data',config['test_data_path']) 
    output_model_path = os.path.join(os.path.abspath('./'),'model',config['output_model_path']) 
    prod_deployment_path = os.path.join(os.path.abspath('./'),'model',config['prod_deployment_path']) 
    logging.info("SUCCESS: Reading configuration file 'config.json' for Process Automation step") 

def fullprocess():
    '''
        This is a method to trigger the full processing pipeline when a new data is discovered
    '''
###[Task-2: Check and read new data]###

    # read ingestedfiles.txt in deployment folder 
    with open(os.path.join(prod_deployment_path, "ingestedfiles.txt")) as file:
        ingested_files = [line.strip('\n') for line in file.readlines()[3:]]

    # determine whether the source data folder has files that aren't listed in ingestedfiles.txt  
    source_files = []
    for file in os.listdir(input_data_path):
        path=os.path.join(input_data_path, file)
        source_files.append(os.path.join(*path.split(os.path.sep)[-3:]))

    ###[Deciding whether to proceed, part 1]###
    # check for differences: 
    if len(set(source_files).difference(set(ingested_files))) == 0:
        logging.info("SUCCESS: Validating input data. No new data was found") 
        return None
    
    # proceed by ingesting new datasets: 
    logging.info("SUCCESS: Validating input data. NEW data was found!") 
    ingestion.merge_multiple_dataframe()
    logging.info("SUCCESS: Deciding to proceed with processing the new datasets") 


    ###[Checking for model drift]### 
    # check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    logging.info("SUCCESS: Checking for model drift") 
    with open(os.path.join(prod_deployment_path, "latestscore.txt")) as file:
        deployed_score = float(file.read().split()[-1])
    
           
    df = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))
    X = df.drop(['corporation', 'exited'], axis=1)
    y = df.pop('exited')
    logging.info("SUCCESS: Preparing testing data (X) and prediction data (y)")
    
    y_pred = diagnostics.model_predictions(X)
    new_score = f1_score(y.values, y_pred)
    logging.info(f"Note: Deployed score = [{deployed_score:.4f}], New score = [{new_score:.4f}]")

    ###[Deciding whether to proceed, part 2]###
    # If the deployed score is lower than the new score, then model drift has occurred. Otherwise, it has not.
    if new_score >= deployed_score:
        logging.info("SUCCESS: No drifting has occured to the deployed model. Decision not to proceed")
        return None
    
    logging.info("SUCCESS: An evidence for model drift has been noticed to the deployed model. Decision to proceed with a new retraining")
    
    # Re-training model by running the training.py script
    training.train_model()
    logging.info("SUCCESS: Re-training model")
    
    # Re-scoring model by running the scoring.py script
    scoring.score_model()
    logging.info("SUCCESS: Re-scoring model")

    # Re-deployment by running the deployment.py script
    deployment.deploy_model()
    logging.info("SUCCESS: Re-deploying model")

    # Diagnostics and reporting by running both diagnostics.py and reporting.py for the re-deployed model
    reporting.plot_confusion_matrix()
    os.system("python apicalls.py")
    logging.info("SUCCESS: Running diagnostics and reporting")


if __name__ == '__main__':
    
    # run main script
    logging.info("SUCCESS: ========= Sarting 'fullprocess.py' =========")
    fullprocess()
    logging.info("SUCCESS: ========= End of 'fullprocess.py' =========")
