----RUNNING CrimeForecaster----
1. Use python 3.7 to create a virtual environment and install all the dependencies from requirements.txt
2. In CRIME_CHICAGO/ use generate_dataset.ipynb file to generate train, val, test files 
3. Make subfolder and keep the train, val, test files in the subfolder
4. Generate yaml file accordingly
5. change default value of config_filename to newly generate yaml file path
6. Run cf_train.py
7. Change regression/classification by
    - choosing loss func in dcrnn_supervisor\91
    - commnent/uncomment appopriate portion of code from dcrnn_supervisor\295-348
8. Metrics are saved in runs/logs/[run-time log directory] 