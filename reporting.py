''' 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Udacity :  Machine Learning DevOps Engineer (MLOps) Nano-degree
  Project :  #5 - A Dynamic Risk Assessment System
  Step    :  #6 - Reporting
  Author  :  Rakan Yamani
  Date    :  06 June 2023
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''


import pandas as pd
import seaborn as sns
import json
import os
import logging
from diagnostics import model_predictions
from sklearn.metrics import confusion_matrix


logging.basicConfig(filename='logging.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s - %(message)s', datefmt="%m/%d/%y %I:%M:%S %p")

###[Task-1: Load config.json and get input and output paths]###
with open('config.json','r') as f:
    config = json.load(f) 
    # dataset_csv_path = os.path.join(os.path.abspath('./'),'data',config['output_folder_path']) 
    test_data_path = os.path.join(os.path.abspath('./'),'data',config['test_data_path']) 
    prod_deployment_path = os.path.join(os.path.abspath('./'),'model',config['prod_deployment_path']) 
    logging.info("SUCCESS: Reading configuration file 'config.json' for Reporting step") 


###[Task-2: Method for reporting]###
def score_model():
    '''
    This is a method to calculate a confusion matrix using the test data and the deployed model and write it to the workspace
    '''
    
    df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    X = df.drop(['corporation', 'exited'], axis=1)
    y = df.pop('exited')
    
    # calling deployed model from diagnostics.py
    y_pred = model_predictions(X)
    logging.info("SUCCESS: Predicting provided data using [diagnostics.py > model_predictions()] ")

    # calculate and write the confusion matrix to the workspace
    cm = confusion_matrix(y, y_pred)
    figure = sns.heatmap(cm, annot=True, cmap="crest").get_figure()  
    figure.savefig(os.path.join(prod_deployment_path, 'confusionmatrix.png'), dpi=400)
    logging.info("SUCCESS: Generating and saving confusion matrix ")
    

if __name__ == '__main__':
    
    # run main script
    logging.info("SUCCESS: ========= Sarting 'reporting.py' =========") 
    score_model()
    logging.info("SUCCESS: ========= End of 'reporting.py' =========")
