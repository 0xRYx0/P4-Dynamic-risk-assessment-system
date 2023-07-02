''' 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Udacity :  Machine Learning DevOps Engineer (MLOps) Nano-degree
  Project :  #5 - A Dynamic Risk Assessment System
  Step    :  #7 - API Setup 
  Author  :  Rakan Yamani
  Date    :  06 June 2023
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import diagnostics 
import json
import os
import logging 
import subprocess
import re

###[Task-1: Set up variables for use in our script]###
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

###[Task-2: Load config.json and get input and output paths]###
with open('config.json','r') as f:
    config = json.load(f) 
    dataset_csv_path = os.path.join(os.path.abspath('./'),'data',config['output_folder_path']) 
    prediction_model = None
    logging.info("SUCCESS: Reading configuration file 'config.json' for app.py step") 

###[Task-3: Index Endpoint]###
logging.info("SUCCESS: API route for - Index Endpoint") 
@app.route('/')
def index():
    return "Here we go! Welcome"


###[Task-4: Prediction Endpoint]###
logging.info("SUCCESS: API route for - Prediction Endpoint") 
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    '''
    This is a method to call the prediction function created earlier and return value for prediction outputs
    '''
    filepath = request.get_json()['filepath']
    df = pd.read_csv(filepath).drop(['corporation', 'exited'], axis=1)
    return jsonify(diagnostics.model_predictions(df).tolist())

###[Task-5: Scoring Endpoint]### 
logging.info("SUCCESS: API route for - Scoring Endpoint") 
@app.route("/scoring", methods=['GET','OPTIONS'])
def score(): 
    '''
    This is a method to check the score of the deployed model and return value (a single F1 score number)
    '''       
    output = subprocess.run(['python', 'scoring.py'], capture_output=True).stdout
    return re.findall(r'f1 score = \d*\.?\d+', output.decode())[0]
    
###[Task-6: Summary Statistics Endpoint]### 
logging.info("SUCCESS: API route for - Summary Statistics Endpoint") 
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    '''
    This is a method to check means, medians, and modes for each column and return a list of all calculated summary statistics
    '''   
    return jsonify(diagnostics.dataframe_summary())

###[Task-7: Diagnostics Endpoint]### 
logging.info("SUCCESS: API route for - Diagnostics Endpoint") 
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnos():        
    '''
    This is a method to check timing and percent NA values and return value for all diagnostics
    ''' 
    return jsonify({
        'missing_percentage': diagnostics.missing_data(),
        'execution_time': diagnostics.execution_time(),
        'outdated_packages': diagnostics.outdated_packages_list()
    })

if __name__ == "__main__":    
    
    # run main script
    logging.info("SUCCESS: ========= Sarting 'app.py' =========") 
    
    app.run(host='127.0.0.1', port=8000, debug=True, threaded=True)
    
    
    logging.info("SUCCESS: ========= End of 'app.py' =========")