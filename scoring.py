''' 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Udacity :  Machine Learning DevOps Engineer (MLOps) Nano-degree
  Project :  #5 - A Dynamic Risk Assessment System
  Step    :  #3 - Model Scoring
  Author  :  Rakan Yamani
  Date    :  04 June 2023
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

import pandas as pd
import pickle
import os
import logging 
from sklearn.metrics import f1_score
import json

logging.basicConfig(filename='logging.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s - %(message)s', datefmt="%m/%d/%y %I:%M:%S %p")

###[Task-1: Load config.json and get input and output paths]###
with open('config.json','r') as f:
    config = json.load(f) 
    test_data_path = os.path.join(os.path.abspath('./'),'data',config['test_data_path']) 
    output_model_path = os.path.join(os.path.abspath('./'),'model',config['output_model_path']) 
    logging.info("SUCCESS: Reading configuration file 'config.json' for Model Scoring step") 


###[Task-2: Method for scoring the model]###
def score_model():
    '''
    This is a method to score the trained Logistic Regression model and perform multiple operations:
    - Loading the model 
    - Loading test data
    - Calculating model's F1 score
    - Writing the results ointo 'latestscore.txt'
    '''
    
    # load test data:
    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    logging.info(f"SUCCESS: Loading required test data 'testdata.csv'") 

    # load traing model:
    model = pickle.load(open(os.path.join(output_model_path,'trainedmodel.pkl'),'rb'))
    logging.info(f"SUCCESS: Loading required test data 'trainedmodel.pkl' from './model/{config['output_model_path']}'") 
    
    # prepare modeling data (X) and prediction data (y) as required:
    X = test_df.drop(['corporation', 'exited'], axis=1)
    y = test_df.pop('exited')
    logging.info("SUCCESS: Preparing testing data (X) and prediction data (y)")
    
    # predict and calculate f1 score: 
    y_pred = model.predict(X)
    logging.info("SUCCESS: Predicting testing data")

    f1 = f1_score(y, y_pred)
    logging.info("SUCCESS: Calculating F1 score")

    # writing f1 scores to file:
    with open(os.path.join(output_model_path, 'latestscore.txt'), 'w') as file:
        file.write(f"f1 score = {f1:.4f}")
    logging.info("SUCCESS: Saving F1 scores to 'latestscore.txt' file")
    
    return print(f"f1 score = {f1:.4f}")


if __name__ == '__main__':
    
    # run main script
    logging.info("SUCCESS: ========= Sarting 'scoring.py' =========")
    score_model()
    logging.info("SUCCESS: ========= End of 'scoring.py' =========")

