''' 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Udacity :  Machine Learning DevOps Engineer (MLOps) Nano-degree
  Project :  #5 - A Dynamic Risk Assessment System
  Step    :  #2 - Model Training
  Author  :  Rakan Yamani
  Date    :  04 June 2023
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

import pandas as pd
import pickle
import os
import logging 
import json
from sklearn.linear_model import LogisticRegression

logging.basicConfig(filename='logging.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s - %(message)s', datefmt="%m/%d/%y %I:%M:%S %p")

###[Task-1: Load config.json and get input and output paths]###
with open('config.json','r') as f:
    config = json.load(f) 
    dataset_csv_path = os.path.join(os.path.abspath('./'),'data',config['output_folder_path']) 
    output_model_path = os.path.join(os.path.abspath('./'),'model',config['output_model_path']) 
    logging.info("SUCCESS: Reading configuration file 'config.json' for Model Training step") 
    

###[Task-2: Method for training the model]###
def train_model():
    '''
    This is a method to train a Logistic Regression model on the ingested date 'finaldata.csv' and save the model as 'trainedmodel.pkl'  
    '''
    
    # load required dataset:
    df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    logging.info(f"SUCCESS: Loading traing data 'finaldata.csv'")
    
    # prepare modeling data (X) and prediction data (y) as required:
    X = df.drop(['corporation', 'exited'], axis=1)
    y = df.pop('exited')
    logging.info("SUCCESS: Preparing modeling data (X) and prediction data (y)")

    # use this logistic regression for training:
    lr_model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, 
                                  l1_ratio=None, max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                                  random_state=0, solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
    
    # fit the logistic regression to your data:
    lr_model.fit(X, y)
    logging.info("SUCCESS: Training required logistic regression model")
    
    # write the trained model to your workspace in a file called trainedmodel.pkl
    pickle.dump(lr_model, open(os.path.join(output_model_path,'trainedmodel.pkl'),'wb'))
    logging.info(f"SUCCESS: Exporting trained model 'trainedmodel.pkl' to './model/{config['output_model_path']}' folder")


if __name__ == '__main__':
    
    # run main script
    logging.info("SUCCESS: ========= Sarting 'training.py' =========")
    train_model()
    logging.info("SUCCESS: ========= End of 'training.py' =========")