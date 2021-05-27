# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:59:09 2017

@author: Xueou
"""

import os
import glob
import numpy as np
import pandas as pd
import sys
import re
import time
import math
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
#from mpl_toolkits.basemap import Basemap
from operator import itemgetter, attrgetter


TdriveFolder  = '/Users/rubenrdp/Desktop/Tdrive/taxi_log_2008_by_id/'   # Location of Tdrive files
SaveFolder    = '/Users/rubenrdp/Desktop/Tdrive/results/'               # Location where to save the resulting matrixes
PTaxi5File    = 'aggregated_5th_transition_data.pkl'                    # Transition matrix from taxis with 5th ring
PTaxi3File    = 'aggregated_3rd_Pmat.npy'                               # Transition matrix from taxis with 3rd ring


# This function give transition probability dataframe for any single taxi 
def transMatSingleTaxi(idTaxi, latMin = 39.73, latMax = 40.10, lonMin = 116.15, lonMax = 116.60, unit = 0.005):
    TdriveFile = TdriveFolder + str(idTaxi) + '.txt'
    if (os.stat(TdriveFile).st_size == 0):
        raise ValueError('This file is empty')
        
    TdriveFrame_raw = pd.read_csv(TdriveFile, sep=',', header=None)

    TdriveFrame_raw.columns = ['id', 'date', 'longitude', 'latitude']  
    TdriveFrame_raw.insert(loc=1, column='timestamp', value=TdriveFrame_raw['date'])
    TdriveFrame_raw[['date','time']] = TdriveFrame_raw['date'].str.split(' ',expand=True)
    TdriveFrame = TdriveFrame_raw[['id', 'date', 'time', 'longitude', 'latitude']]
    #print TdriveFrame

    nla = (latMax - latMin)/unit
    nlo = (lonMax - lonMin)/unit
    TdriveFrame = TdriveFrame[(TdriveFrame['longitude'] >= lonMin) & (TdriveFrame['longitude'] <= lonMax) & (TdriveFrame['latitude'] >= latMin) & (TdriveFrame['latitude'] <= latMax)]
    
    TdriveFrame.index = range(len(TdriveFrame.index))
    nrowTdriveFrame = TdriveFrame.shape[0]
    if(nrowTdriveFrame < 2):
        raise ValueError('No available data within 5 ring!')
        
    transMat = np.zeros((nrowTdriveFrame, 6))

    for i in range(nrowTdriveFrame):
        #if(i%10==0):
         #   print 'i = ', i
        xlon = math.ceil((TdriveFrame['longitude'][i] - lonMin)/unit)
        ylat = math.ceil((TdriveFrame['latitude'][i] - latMin)/unit)
        transMat[i, 0] = int(float(TdriveFrame['date'][i][-1]))
        if((TdriveFrame['longitude'][i] - lonMin) / unit == int((TdriveFrame['longitude'][i] - lonMin))):
            transMat[i, 1] = xlon
        else:
            transMat[i, 1] = xlon - 1
        if((TdriveFrame['latitude'][i] - latMin) / unit == int((TdriveFrame['latitude'][i] - latMin))):
            transMat[i, 1] = ylat
        else:
            transMat[i, 2] = ylat - 1
        
        transMat[i, 3] = transMat[i, 2] * nlo + transMat[i, 1]
    
    for i in range(nrowTdriveFrame-1):
        if(transMat[i+1, 0] == transMat[i, 0]):
            transMat[i, 4] = transMat[i+1, 3]
        else:
            transMat[i, 4] = float('nan')
            
    transMat[-1, 4] = float('nan')
    
    transMat = transMat[~np.isnan(transMat).any(axis=1)] #remove rows containing 'nan', i.e., last observation for a particular cell
    if(transMat.shape[0] < 1):
        raise ValueError('No trajectory data')
    
    sortTransMat = np.array(sorted(transMat, key=itemgetter(3,4))) #sort the matrix rows according to 'from' cells, then 'to' cells
    uniqCellFrom = list(np.unique(sortTransMat[:,3], return_counts = True)) #calculate unique 'start' cells and their frequency
    freqTrans = 1.0/uniqCellFrom[1] #assume same probability of transforming from the same 'start' cell to its various 'to' cells
    uniqCellFrom.append(freqTrans)
    sortTransMat[:,5] = np.repeat(freqTrans, np.array(uniqCellFrom[1]))
    transprob = pd.DataFrame(uniqCellFrom)
    transprob = transprob.transpose()
    transprob.columns = ['from', 'freq', 'prob']
    transprob = transprob.drop(['freq'], axis=1)

    uniqTransMatPair = np.unique(sortTransMat[:,3:5], return_counts=True, axis = 0)
    uniqTransMatPair = list(uniqTransMatPair)
    uniqTransMatPairdf = np.zeros((len(uniqTransMatPair[0]), len(uniqTransMatPair)+1))

    uniqTransMatPairdf[:,0],uniqTransMatPairdf[:,1] = zip(*uniqTransMatPair[0])
    uniqTransMatPairdf[:,2] = uniqTransMatPair[1]

    transMatDF = pd.DataFrame(uniqTransMatPairdf)
    transMatDF.columns = ['from', 'to', 'freq']
    transMatDF=transMatDF.merge(transprob, on='from', how='left')
    transMatDF['probTrans'] =  transMatDF['prob'] * transMatDF['freq']
    transMatDF.insert(loc = 0, column = 'id', value = idTaxi)
    return(transMatDF)



#*************************************************#
#                 Main Program                    #
#*************************************************#
# Produce the transition matrix aggregating all taxis together
transMTaxiAgg = pd.DataFrame()
nFiles = len(glob.glob1(TdriveFolder,"*.txt"))
for j in range(1, nFiles+1):
    #if(j%100 == 0): # for monitoring progress
        #print ('Taxi: ', j) # for monitoring progress
    try:
        transM = transMatSingleTaxi(j)
    except ValueError as e:
        print ('Error:',e)
        print ('Taxi', j, 'has no data or trajectory data')
        print ('')
        continue
    transMTaxiAgg = transMTaxiAgg.append(transM)
#Save dataframe
if not os.path.exists(SaveFolder):
    os.makedirs(SaveFolder)
transMTaxiAgg.to_pickle(SaveFolder+PTaxi5File)  # where to save it, usually as a .pkl

#####################################################################

#transMTaxiAgg = pd.read_pickle(PTaxi5File) #load it again


#########################################
# Define map bounds
#########################################
#idTaxi=1
# Start: Beijing within 5th ring #
latMin = 39.8
latMax = 40.05
lonMin = 116.230 
lonMax = 116.55
unit = 0.05
# End: Beijing within 5th ring #
# Start: Beijing within 4th ring #
latMin = 39.83
latMax = 39.987
lonMin = 116.26
lonMax = 116.49
unit = 0.005
# End: Beijing within 4th ring #
# Start: Beijing within 3rd ring, it takes --- 11651.151 seconds ---#
latMin = 39.84
latMax = 39.97
lonMin = 116.29
lonMax = 116.46
unit = 0.005
# End: Beijing within 3rd ring #
# a unit of 0.05 corresponds to 5000 meters, with an average speed of 30 kilometers/hr,
# and an average sampling rate of 177 secs, the grid size should be better at about 1450 meters,
# i.e., 0.015
#unit = 0.005#0.005

#####################################
# Build the transition matrix from the transition data frame
#####################################
dimPmat = int(math.ceil((latMax - latMin)/unit) * math.ceil((lonMax - lonMin)/unit))
Pmat = np.zeros((dimPmat, dimPmat))
start_time = time.time()
for j in range(1, nFiles+1):
    temp = np.zeros((dimPmat, dimPmat))
    if(j%100 == 0): # for monitoring progress
        print ('Taxi: ', j) # for monitoring progress
    try:
        taxi = transMatSingleTaxi(j, latMin = 39.84, latMax = 39.97, lonMin = 116.29, lonMax = 116.46, unit = 0.005)
    except ValueError as e:
        print ('Error:',e)
        print ('Taxi', j, 'has no data or trajectory data')
        print ('')
        continue
    for pair in range(taxi.shape[0]):
        temp[int(round(taxi[['from']].values[pair][0], 0))][int(round(taxi[['to']].values[pair][0], 0))] = taxi[['probTrans']].values[pair]
    Pmat = Pmat + temp
    
print("--- %s seconds ---" % (time.time() - start_time))
np.save(SaveFolder + PTaxi3File, Pmat)  # where to save it

#Pmat = np.load(PTaxi3File) # load the matrix again

normed_Pmat = normalize(Pmat, axis=1, norm='l1') #normalize the matrix to be a transition probability matrix
normed_Pmat.sum(axis=1)
diag = np.diagonal(normed_Pmat) #extract diagonal entries
np.linalg.det(normed_Pmat) #compute determinant of the transition matrix

