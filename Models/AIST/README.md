-----Running AIST-----

1. change dataset path in three places:
    - train.py\61
    - utils.py\110
    - utils.py\245

2. change the value of 'div' in utils.py\23
    div = 24/(width of timestep)

3. For regression/classification
    - change loss function in train.py\131-133
    - check "for regression/classification" and comment/uncomment necessary portions

4. run code by running the run.ipynb notebook