''' 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Udacity :  Machine Learning DevOps Engineer (MLOps) Nano-degree
  Project :  #5 - A Dynamic Risk Assessment System
  Step    :  #8 - API Calls 
  Author  :  Rakan Yamani
  Date    :  06 June 2023
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

import requests
import json
import os
import logging

###[Task-1: Load config.json and get input and output paths]###
with open('config.json','r') as f:
    config = json.load(f) 
    test_data_path = os.path.join(os.path.abspath('./'),'data',config['test_data_path']) 
    prod_deployment_path = os.path.join(os.path.abspath('./'),'model',config['prod_deployment_path']) 
    logging.info("SUCCESS: Reading configuration file 'config.json' for appcalls.py step") 

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

#Call each API endpoint and store the responses
predict = requests.post(f'{URL}/prediction',json={'filepath': os.path.join(test_data_path, 'testdata.csv')}).text
logging.info("SUCCESS: Retrieving prediction response - API") 

score = requests.get(f'{URL}/scoring').text
logging.info("SUCCESS: Retrieving scoring response - API") 

stats = requests.get(f'{URL}/summarystats').text 
logging.info("SUCCESS: Retrieving summary stats response - API") 

diagnos = requests.get(f'{URL}/diagnostics').text 
logging.info("SUCCESS: Retrieving diagnostics response - API") 

#combine all API responses
# responses = [predict, score, stats, diagnos]

#write the responses to your workspace
with open(os.path.join(prod_deployment_path, 'apireturns.txt'), 'w') as file:
    file.write('Summary of all responses for testing our API')
    file.write('\n#1# Model Predictions:\n')
    file.write(predict)
    file.write('\n\n==================\n\n')
    file.write('#2# Model Score:\n') 
    file.write(score)
    file.write('\n\n==================\n\n')
    file.write('#3# Statistics Summary:\n')
    file.write(stats)
    file.write('\n\n==================\n\n')
    file.write('#4# Diagnostics Summary:\n')
    file.write(diagnos)
    file.write('\n\n=========| END OF API CALLS |=========\n\n')