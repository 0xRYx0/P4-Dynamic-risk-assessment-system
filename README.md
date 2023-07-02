## Project 4: Dynamic Risk Assessment System
The fourth project of [ML DevOps Engineer Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821) by Udacity. This project aims to create, deploy, and monitor a risk assessment ML model to estimate the attrition risk of each company's 10,000 clients. the project consists of five steps: 

1. **Data ingestion**: Automatically check a database for new data, compile it into a training dataset, and save it. Write metrics to persistent storage.
2. **Training, scoring, and deploying**: Develop scripts to train an ML model for attrition risk prediction, score the model, and save both the model and metrics.
3. **Diagnostics**: Generate dataset summary statistics, time model training and scoring, and check for dependency changes.
4. **Reporting**: Automatically create plots and documents for model metrics and provide an API endpoint for predictions and metrics.
5. **Process Automation**: Regularly use a script and cron job to run all steps.

### Project files
```
📦 P4-Dynamic-risk-assessment-system
 ┣ 📂 data                         # ~~~[ Data folder ]~~~
 ┃  ┣ 📂 ingesteddata              # Ingested and processed datasets (reference for validating new data)
 ┃  ┃  ┣ 📜 finaldata.csv
 ┃  ┃  ┗ 📜 ingestedfiles.txt
 ┃  ┣ 📂 practicedata              # Practice dataset (used to generate first model)
 ┃  ┃  ┣ 📜 dataset1.csv
 ┃  ┃  ┗ 📜 dataset2.csv
 ┃  ┣ 📂 sourcedata                # Initial dataset (used for model training)
 ┃  ┃  ┣ 📜 dataset3.csv
 ┃  ┃  ┗ 📜 dataset4.csv
 ┃  ┗ 📂 testdata                  # Testing dataset (used for validating the score of deployed vs new model)
 ┃     ┗ 📜 testdata.csv
 ┣ 📂 model                        # ~~~[ Models folder ]~~~
 ┃  ┣ 📂 models                    # Models summary and metrics
 ┃  ┃  ┣ 📜 trainedmodel.pkl
 ┃  ┃  ┣ 📜 apireturns.txt
 ┃  ┃  ┣ 📜 confusionmatrix.png
 ┃  ┃  ┣ 📜 latestscore.txt
 ┃  ┃  ┗ 📜 logging-sourcedata.log
 ┃  ┣ 📂 practicemodels            # Models used during the training stage with their matrics
 ┃  ┃  ┣ 📜 trainedmodel.pkl
 ┃  ┃  ┗ 📜 latestscore.txt
 ┃  ┗ 📂 production_deployment     # Trained serialized models with matrics (Production)
 ┃     ┣ 📜 trainedmodel.pkl
 ┃     ┣ 📜 apireturns.txt
 ┃     ┣ 📜 confusionmatrix.png
 ┃     ┣ 📜 latestscore.txt
 ┃     ┗ 📜 logging-sourcedata.log  
 ┣ 📜 ingestion.py                 # Step (1): Data Ingestion - to ingest required data and perform multiple operations
 ┣ 📜 training.py                  # Step (2): Model Training - to train a Logistic Regression model and save the model 
 ┣ 📜 scoring.py                   # Step (3): Model Scoring - to score the trained & deployed Logistic Regression model 
 ┣ 📜 deployment.py                # Step (4): Model Deployment - to deploy a trained model to the production environment
 ┣ 📜 diagnostics.py               # Step (5): Model and Data Diagnostics - to generate the model's statistical diagnostics 
 ┣ 📜 reporting.py                 # Step (6): Reporting - to generate a confusion matrix using the test data
 ┣ 📜 app.py                       # Step (7): API Setup - Flask application to establish the required API endpoints 
 ┣ 📜 apicalls.py                  # Step (8): API Calls - to test API endpoints and write the results on a file 
 ┣ 📜 fullprocess.py               # Main script to trigger the full processing pipeline when new data is discovered
 ┣ 📜 config.json                  # Project's configuration file
 ┣ 📜 cronjob.txt                  # Scheduled cron job 
 ┣ 📜 requirements.txt             # Project's required dependencies       
 ┗ 📜 README.md   
```

### Usage
1. Edit `config.json` file to use practice data:
    ```
    { 
      "input_folder_path": "practicedata",
      " output_folder_path": "ingesteddata", 
      "test_data_path": "testdata", 
      "output_model_path": "practicemodels", 
      "prod_deployment_path": "production_deployment"
    }
    ```
2. Run the following scripts in sequence:
    * Data Ingestion:
        * script:  
            ```bash
            > python ingestion.py
            ```
         * Artifacts output: 
            ``` 
            data/ingesteddata/finaldata.csv 
            data/ingesteddata/ingestedfiles.txt
            ```
            
    * Model Training:
        * script:  
            ```bash
            > python training.py
            ```
         * Artifacts output: 
            ``` 
            models/practicemodels/trainedmodel.pkl
            ```
            
    * Model Scoring:
        * script:  
            ```bash
            > python scoring.py
            ```
         * Artifacts output: 
            ``` 
            models/practicemodels/latestscore.txt
            ```
               
    * Model Deployment:
        * script:  
            ```bash
            > python deployment.py
            ```
         * Artifacts output: 
            ``` 
            models/production_deployment/ingestedfiles.txt
            models/production_deployment/trainedmodel.pkl
            models/production_deployment/latestscore.txt
            ```

    * Model Diagnostics:
        * script:
            ```bash
            > python diagnostics.py
            ```
         * Artifacts output: None

    * Model Reporting:
        * script:  
            ```bash
            > python reporting.py
            ```
         * Artifacts output: 
            ``` 
            models/practicemodels/confusionmatrix.png
            ```
    * Flask Application:
        * script:  
            ```bash
            > python app.py
            ```
         * Artifacts output: None

    * API Endpoints
        * script:  
            ```bash
            > python apicalls.py
            ```
         * Artifacts output: 
            ``` 
            models/practicemodels/apireturns.txt
            ```
3. Edit `config.json` file to use production data:
    ```
    { 
      "input_folder_path": "sourcedata",
      "output_folder_path": "ingesteddata", 
      "test_data_path": "testdata", 
      "output_model_path": "models", 
      "prod_deployment_path": "production_deployment"
    }
    ```

4. Full process automation:
   ```bash
   > python fullprocess.py
   ```


### License

Distributed under the [MIT](https://choosealicense.com/licenses/mit/) license. 
