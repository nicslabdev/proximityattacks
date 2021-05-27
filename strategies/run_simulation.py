# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 15:43:44 2019

@author: idswx
"""

import math
from collections import Counter
import numpy as np, numpy.random
from scipy.special import entr
import matplotlib.pyplot as plt

import os
import glob
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import sys
import re
import time
from random import randint
from sklearn.preprocessing import normalize
from operator import itemgetter, attrgetter

savePath = './cGUASresults5/'
if not os.path.exists(savePath):
    os.makedirs(savePath) 


#from strategy_functions_editing import genTransMat, getEnt, strategyLJS, strategyGUAS, strategyCstrtGUAS, runStrategy, strategyOutput, runSimul
from strategy_functions import genTransMat, getEnt, strategyLJS, strategyGUAS, strategyCstrtGUAS, strategyCstrtGUASBJ, runStrategy, strategyOutput, runSimul


#if __name__ == "__main__":
# Does the attacker know real B?
#tORf = [False, True]
tORf = [True] # name is short for true OR false
setSeed = False

# list of all implemented strategies
dict_strategy = {'GUAS':strategyGUAS, 'steps':strategyLJS, 'cGUAS':strategyCstrtGUAS, 'cGUASBJ':strategyCstrtGUASBJ}

# select the strategy you want to run    
#stratToTest = ['GUAS','steps']
stratToTest = ['cGUASBJ']


# P generated using Dirichlet Distribution 
# for all-but-one zero:
#Magic = 1/1000
# for all equal:
#Magic = 1
# for 'normal' randomness:
#Magic = 0.01    
#dirichletMagic = [0.001]
dirichletMagic = [1e13, 10, 1, 0.1, 0.01, 0.001]

useBJ = True

if useBJ:
    space2Search = [884]  #only required if useBeijing = True
    _resultname = 'BJcGUAS'
    
    from attackerConstraintBJ import attackerConstraint
    
else:
    # a list of search space size set to whatever you want
    space2Search = [100, 500, 2000]
    #space2Search = [100]
    _resultname = 'cGUAS'
    from attackerConstraint import attackerConstraint

    
start_time0 = time.time()

for knowledge in tORf:
    
    if knowledge == True:
        knowB = 'knw'
    else:
        knowB = 'unknw'

    for _searchSpace in space2Search:
        for dltmgc in dirichletMagic:
            #for constraintstep in [2, 10, 50]:
            for constraintstep in [3]:

                start_time = time.time()
                print("Running with configuration:")
                print("Attacker knowledge of B: %s"%str(knowledge))
                print("Strategy to test: %s"%str(stratToTest))
                print("Dirichlet parameters for B: %s"%str(dirichletMagic))
                print("Using constant random seed: %s"%str(setSeed))
                print("Using Beijing transformation: %s"%str(useBJ))
                print("Size of B is: %s"%str(_searchSpace))
                print("constraint step is : %s "%str(constraintstep))

                pd_cGUAS = []
                for counter in range(100):
#                     cGUAS = runSimul([dltmgc], [_searchSpace], dict_strategy, stratToTest, useBJ, knowledge, setSeed, maxrounds = 1, 
#                                      suc_thresh = 0.5, cGUASi=0, cGUASj=randint(0, _searchSpace-1), cGUASd=constraintstep, cGUASm=1, cGUASn=_searchSpace)
                    if useBJ:
                        cGUAS = runSimul([dltmgc], [_searchSpace], dict_strategy, stratToTest, useBJ, knowledge, setSeed, maxrounds = 1,
                                         suc_thresh = 0.5, cGUASi=0, cGUASj=5, cGUASd=constraintstep, cGUASm=26, cGUASn=34)
                    else:
                        cGUAS = runSimul([dltmgc], [_searchSpace], dict_strategy, stratToTest, useBJ, knowledge, setSeed, maxrounds = 1, 
                                     suc_thresh = 0.5, cGUASi=0, cGUASj=5, cGUASd=constraintstep, cGUASm=1, cGUASn=_searchSpace)
                        
                    pd_cGUAS.append(cGUAS)

                pd_cGUAS_df = pd.concat(pd_cGUAS, ignore_index=True)
                pd_cGUAS_df.to_csv(savePath+str(_resultname)+str(constraintstep)+'_space'+str(_searchSpace)+'_'+str(knowB)+'_alpha_'+  str(dltmgc) + '.csv', index=True, header=True)
                print("--- %s seconds --- for space size %s, dltmgc %s" % ((time.time() - start_time), _searchSpace, dltmgc))              
    #pd_cGUAS_df.describe()
    #stratToTest = ['GUAS']
    #resultGUAS = runSimul(dirichletMagic, space2Search, dict_strategy, stratToTest, useBJ, knowledge, setSeed, maxrounds = 1, suc_thresh = 0.5,
    #                  cGUASi=0, cGUASj=5, cGUASd=2, cGUASm=1, cGUASn=100)
    print("--- %s seconds --- for knowledge %s" %((time.time() - start_time0), knowB))
              