# -*- coding: utf-8 -*-
"""POI_Segregation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OepT8x4YVwuc2OiI0idEyxItU_48orUP
"""

# UNCOMMENT IF USING COLAB

# from google.colab import drive
# drive.mount('/content/gdrive')

import pandas as pd
import numpy as np
import shapely.wkt
from shapely.geometry import Point, Polygon, MultiPolygon
import re

# Reads the point-of-interest dataset from New York City
# Important locations, classified as following, with the geographical data and date of creation/updation to records

# 1 Residential
# 2 Education Facility
# 3 Cultural Facility
# 4 Recreational Facility
# 5 Social Services
# 6 Transportation Facility
# 7 Commercial
# 8 Government Facility (non public safety)
# 9 Religious Institution
# 10 Health Services
# 11 Public Safety
# 12 Water
# 13 Miscellaneous

filePOI = 'RawData/Point_Of_Interest.csv'
fileRegions = "RawData/Regions.csv"


poi = pd.read_csv(filePOI,low_memory=False)
locations = pd.read_csv(fileRegions, low_memory=False)

relevantColumns = ['the_geom','CREATED','FACILITY_T']
finalColumns = ['Precincts','FACILITY_T']
poi = poi[relevantColumns]

# Reading geographical divisions of precincts to localize the POIs to their respective regions
precincts = {}
for index, row in locations.iterrows():
  precincts[row['Precinct']] = shapely.wkt.loads(row['the_geom'])

# Extracting longitude and latitude details of various POIs
# Extracting date of creation or updation to records for various POIs
# This information is used later to localize each point

lat = []
longt = []
date = []

for idx,row in poi.iterrows():
  location = re.split(r' |\)|\(' ,row['the_geom'])
  year = int(row['CREATED'].split()[0].split('/')[-1])
  lat = lat + [location[-2]]
  longt = longt + [location[2]]
  date = date + [year]

poi['Year'] = date
poi['Latitude'] = lat
poi['Longitude'] = longt


# Assigning the precinct number to a location in which the property is
pos = 0
prec = np.ndarray((poi.shape[0],))

for index,row in poi.iterrows():

  poo = Point(float(row['Longitude']),float(row['Latitude']))  
  for key,val in precincts.items():
    if poo.within(val):
      prec[pos] = key
      break

  pos=pos+1
  if(pos%1000 == 0):
    print ("processed "+str(pos)+" records!!")
    
poi['Precincts'] = prec.astype(int)
print("Done!!")

poi['Precincts'] = poi['Precincts'].astype(np.int64)
poi = poi[poi['Precincts'] >= 0]


# Inverted dictionaries for easy conversion of the above dataframe to a matrix that is usable later in the neural architecture

# For categories of the locality, listed above
inv_categories = {}
n_categories = poi['FACILITY_T'].nunique()
uniq_categories = poi['FACILITY_T'].unique()
for i in range(n_categories):
  inv_categories[uniq_categories[i]] = i

# For geographical division of precincts
n_precincts = poi['Precincts'].nunique()
uniq_precincts = poi['Precincts'].unique()
inv_prec = {}
for i in range(n_precincts):
  inv_prec[uniq_precincts[i]] = i

poi = poi[poi['Year'] <= 2008]
poi = poi[finalColumns]

matrices = np.zeros((n_precincts,n_categories),dtype=np.int64)
exceptions = 0

for idx, row in poi.iterrows():
  try:
    id1 = inv_prec[row['Precincts']]
    id2 = inv_categories[row['FACILITY_T']]
    matrices[id1][id2]= matrices[id1][id2] + 1
  except:
    print("Exception!!!")
    print("Precincts",id1)
    print("FACILITY_T",id2)
    exceptions = exceptions + 1

import pickle

outputPOI = 'poiMatrices'

# Pickle dump this file
file = open(outputPOI'wb')
pickle.dump(matrices,file)

# To load this files, use:
# with open(filename,'rb') as toOpen:
#     data = pickle.load(toOpen)