----RUNNING HAGEN----
1. Keep dataset in crime-data/CRIME_CHICAGO/
2. Make subfolder and keep the train, val, test files in the subfolder
3. Generate yaml file accordingly
4. change default value of config_filename to newly generate yaml file path
5. Run hagen_train.py
6. Change regression/classification by
    - choosing loss func in hagen_supervisor\300
    - commnent/uncomment appopriate portion of code from hagen_supervisor\134-180
7. Metrics are saved in runs/logs/[run-time log directory] 