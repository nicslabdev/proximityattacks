#!/usr/bin/env python3
# Nils+Xueou, SUTD, 2018
""" 
import math
import numpy as np
from scipy.special import entr
import matplotlib.pyplot as plt

import os
import glob
import pandas as pd
import sys
import re
import time
import pickle
from sklearn.preprocessing import normalize
from operator import itemgetter, attrgetter
"""

def genTransMat(n):
    """
    Generate transition matrix
    Arguments: n - dimension of transition matrix
    Return: transMat - an n*n transition matrix 
    """
    import numpy as np
    
    transMat = np.random.rand(n,n) #generate a square matrix consisting random numbers 
    
    #Normalize transMat to be a transition matrix
    row_sums = transMat.sum(axis=1) 
    transMat = transMat/row_sums[:,None] #transMat__ = np.divide(transMat, row_sums): seems not correct
       
    return transMat

def getEnt(probs):
    """
    Compute the Shannon's entropy a pmf array
    Arguments: probs - a pmf
    """
    import math
    from scipy.special import entr
    import numpy as np
    
    for r in [probs.sum()]:
        if not math.isclose(r, 1.0, rel_tol=1e-3):
            print('Warning: Sum of probs is not 1: %f'%r)
            continue
        
    return entr(probs).sum()/np.log(2)
    


'''
Different stategies. Each should return the index of the value in B they would like to check
'''

def strategyLJS(B, r, cGUASi=0, cGUASj=5, cGUASd=2, cGUASm=1, cGUASn=10):  #strategyLJS(B, T, r): B and T seems no use, T is transition matrix, 
    '''
    Strategy of the proof: pick 2*r for round r    
    
    Arguments: r - $r$th round  
               B - pmf matrix of Bob (either attacker's belief or Bob real. No use in LJS, just to be consistent in strategy function)
               cGUASi, cGUASj, cGUASd, cGUASm1, cGUASn - strategyCstrtGUAS parameters for consistency
    Return: a list of one item - Attacker's guess for round r      
    '''
    return 2*r

def strategyGUAS(B, r, cGUASi=0, cGUASj=5, cGUASd=2, cGUASm=1, cGUASn=10): #strategyGUAS(B, T, r): T and r seems no use
    '''
    Pick the max likelihood of B    
    Arguments: B - B is pmf of Bob (either attacker's belief or Bob real pmf)
               r - $r$th round (no use in GUAS, just to be consistent in strategy function)
               cGUASi, cGUASj, cGUASd, cGUASm1, cGUASn - strategyCstrtGUAS parameters for consistency
    Return: index/position - Attacker's guess
    '''
    import numpy as np
    
    return np.argmax(B)


def strategyCstrtGUAS(B, r, cGUASi=0, cGUASj=5, cGUASd=2, cGUASm=1, cGUASn=10):
    '''
    Pick the max likelihood of B with constraint   
    Arguments: B - B is pmf of Bob (either attacker's belief or Bob real pmf)
               r - $r$th round (no use in GUAS, just to be consistent in strategy function)
               cGUASi, cGUASj, cGUASd, cGUASm1, cGUASn - strategyCstrtGUAS parameters for consistency
    Return: index/position - Attacker's guess
    '''
    import sys
    #sys.path.append('D:/idswx/locpriv-master/esorics_cleanCode/')
    from attackerConstraint import attackerConstraint
    
    cstrtAttacker = attackerConstraint(i=cGUASi, j=cGUASj, d=cGUASd, m=cGUASm, n=cGUASn)
    #print("def strategyCstrtGUAS:")
    #print("i :", cGUASi, "j: ", cGUASj, "d: ", cGUASd, "m: ", cGUASm, "n: ",cGUASn)
    return cstrtAttacker.strategyGUASconstraint(B)


def strategyCstrtGUASBJ(B, r, cGUASi=0, cGUASj=5, cGUASd=2, cGUASm=26, cGUASn=34):
    '''
    Pick the max likelihood of B with constraint   
    Arguments: B - B is pmf of Bob (either attacker's belief or Bob real pmf)
               r - $r$th round (no use in GUAS, just to be consistent in strategy function)
               cGUASi, cGUASj, cGUASd, cGUASm1, cGUASn - strategyCstrtGUAS parameters for consistency
    Return: index/position - Attacker's guess
    '''
    import sys
    #sys.path.append('D:/idswx/locpriv-master/esorics_cleanCode/')
    from attackerConstraintBJ import attackerConstraint
    
    cstrtAttacker = attackerConstraint(i=cGUASi, j=cGUASj, d=cGUASd, m=cGUASm, n=cGUASn)
    #print("def strategyCstrtGUAS:")
    #print("i :", cGUASi, "j: ", cGUASj, "d: ", cGUASd, "m: ", cGUASm, "n: ",cGUASn)
    return cstrtAttacker.strategyGUASconstraint(B)



def runStrategy(initP, transMat, strategy, maxrounds, knowledge, suc_thresh = 0.5,
                   cGUASi=0, cGUASj=5, cGUASd=2, cGUASm=1, cGUASn=10):    
    
    ''' 
    Given a setup, perform an attacker's strategy
    
    Argumemts: initP - a pmf(in matrix) used as initial state. 
               transMat - ransition matrix.
               strategy - Attacker's strategy function
               maxrounds - max rounds performed by Attacker
               suc_thresh - The successful probability threshold, default is 0.5
               knowledge - True or False, indicate if Attacker's knows initial position pmf of Bob
               
    Return: rent - a list consisting of every round's entropy
            rsuc - a list consisting of probability of successful attack after each round
            r - The round number until which Attacker acheives successful threshold
            guesses - a list consisting of Attacker's guess for each round
    '''    
    import numpy as np
    
    Breal = initP.copy() # will contain evolving P minus tested locations
    # See if attacker knows initial distribution
    if knowledge == True:
        Bbelief = initP.copy() # will contain evolving P minus tested locations
        cGUASj_cur = np.array(Bbelief).argmax()
    else:
        # if not, attacker assumes uniform distribution
        Bbelief = np.matrix([1/initP.shape[1]]*initP.shape[1])
        cGUASj_cur = cGUASj #5
    suc = 0. # initialize vector for probability that we were successful until this round (inclusive this round)

    # rsuc and rent will have one value per round, for each P we are testing
    rsuc = [] #np.zeros((len(initP), maxrounds))
    rent = [] #np.zeros((len(initP), maxrounds))
    guesses = []
    #for r in range(maxrounds):
    r = 0
    while suc < suc_thresh:
        # print("Round %d: B is %s"%(r,str(Breal)))
        #print("Bbelief.shape: ", Bbelief.shape)
        #print("Bbelief: ", Bbelief)
        #print("def runStrategy: ")
        #print("i :", cGUASi, "j: ", cGUASj, "d: ", cGUASd, "m: ", cGUASm, "n: ",cGUASn)
            
        guess = strategy(Bbelief, r, cGUASi=cGUASi, cGUASj=cGUASj_cur, cGUASd=cGUASd, cGUASm=cGUASm, cGUASn=cGUASn)
        
        if guess >= Breal.shape[1]:
            #return rent, rsuc, maxrounds, guesses
            guess = guess % Breal.shape[1]
            
        guesses.append(int(guess))
        # hack to make steps work nicely for non-uniform
        cGUASj_cur = guess
        
        guessProbs = Breal[0, guess]  #Breal[range(len(Breal)), guess]        
        #suc = (1 - suc) * guessProbs  + suc
        #print ('suc is %f'%suc)
        suc = guessProbs  + suc
        #print("r: ", r, " === suc: ", suc, " === cGUASj_cur: ", cGUASj_cur, "=== guessProbs", guessProbs)
        #print("Guess is %d: guessProbs is is %f. Updated suc is %f"%(int(guess[0]),guessProbs, suc))

        ent = getEnt(Breal)

        rsuc.append(suc)
        rent.append(ent)
        
        #Breal[np.arange(Breal.shape[0]), guess] = 0
        if (strategy == strategyGUAS) or (strategy == strategyCstrtGUAS) or (strategy == strategyCstrtGUASBJ):
            Bbelief[np.arange(Bbelief.shape[0]), guess] = 0    
            # re-normalize Bbelief
            Bbelief = Bbelief/Bbelief.sum() #np.divide(Bbelief, Bbelief.sum())

            
        # Compute next round's B
        Breal = Breal.dot(transMat)
        Bbelief = Bbelief.dot(transMat)
        r = r + 1
        
        if suc >= suc_thresh:
            return rent, rsuc, r, guesses




def strategyOutput(Magic, searchSpace, dict_strategy, stratToTest, transMat, maxrounds, knowledge, suc_thresh = 0.5,
                   cGUASi=0, cGUASj=5, cGUASd=2, cGUASm=1, cGUASn=10):
    
    ''' 
    Run the strategy(ies) desired and give a pandas df output for a intial distribution
    
    Argumemts: Magic - dirichlet concentration parameter to generate initial distribution
               searchSpace - searchSpace size
               dict_strategy - a dictionary with strategy name and the strategy function name, e.g., {'GUAS':strategyGUAS, 'steps':strategyLJS}
               stratToTest - list of strategy name to test in the "dict_strategy" argument
               maxrounds - max rounds performed by Attacker
               knowledge - True or False, indicate if Attacker's knows initial position pmf of Bob
               
    Return: (1)a pandas dataframe of result for each setup and strategy, and (2)a dict of entropy (3) a dict of success for each strategy
    ''' 
    import pandas as pd
    import numpy as np
    from collections import defaultdict 


    #create a dataframe summarizing simulation result
    out_list = []
    pd.DataFrame(columns=['Strategy', 'Magic', 'Search space', 'Round number', 'Probability', 'Initial Entropy'])
 
    # Generate intial distribution with Magic
    P = []
    # P generation using Dirichlet distribution + Magic
    if Magic > 0:
        P.append(np.matrix(np.random.dirichlet(np.ones(searchSpace)*Magic, size=1)))
    else:
        P.append(np.matrix([1/searchSpace]*searchSpace))

    # initializing dict with lists 
    allEnt = defaultdict(list)
    allSuc = defaultdict(list)
    i = 0
    for s in stratToTest:
        print('Strategy %s'%s)
        for mp in P:
            #print(mp)
            #print("def strategyOutput: ")
            #print("i :", cGUASi, "j: ", cGUASj, "d: ", cGUASd, "m: ", cGUASm, "n: ",cGUASn)
            entropy, success, minr, guesses = runStrategy(initP = mp.copy(), transMat = transMat.copy(),
                                                          strategy = dict_strategy[s], maxrounds = maxrounds,
                                                          knowledge = knowledge, cGUASi=cGUASi, cGUASj=cGUASj,
                                                          cGUASd=cGUASd, cGUASm=cGUASm, cGUASn=cGUASn)
            print('With Magic %f and size %d: Minimal number of rounds to reach 0.5: %d. Initial Ent is %f'%(Magic, searchSpace, minr, getEnt(mp)))
            #with open("guess.txt", "a") as guess_file:
            print("Guesses were: ", guesses)
            out_ = pd.DataFrame([[s, Magic, searchSpace, minr, suc_thresh, float(getEnt(mp))]])
            out_list.append(out_)
        
        out_df = pd.concat(out_list, ignore_index=True)
        out_df.columns = ['Strategy', 'Magic', 'Search space', 'Round number', 'Probability', 'Initial Entropy'] 
        allEnt[s] = entropy
        allSuc[s] = success
        i += 1

    return out_df, allSuc, allEnt



def runSimul(dirichletMagic, space2Search, dict_strategy, stratToTest, useBJ, knowledge, setSeed, maxrounds = 1, suc_thresh = 0.5,
             cGUASi=0, cGUASj=5, cGUASd=2, cGUASm=1, cGUASn=10):
    
    ''' 
    Run the simulation for the paper
    
    Argumemts: dirichletMagic - a list of dirichlet concentration parameters to generate initial distribution
               space2Search - a list of searchSpace size to simulate (for searchSpace in space2Search)
               dict_strategy - a dictionary with strategy name and the strategy function name, e.g., {'GUAS':strategyGUAS, 'steps':strategyLJS}
               stratToTest - list of strategy name to test in the "dict_strategy" argument
               useBJ - if True, use BJ derived transition matrix
               maxrounds - default 1
               knowledge - True or False, indicate if Attacker's knows initial position pmf of Bob
               setSeed - if True set seed for random generating
               suc_thresh - success probability
               
    Return: (1)a pandas dataframe of result for each setup and strategy, and (2)a dict of entropy (3) a dict of success for each strategy
    ''' 
    import pandas as pd
    import numpy as np
    import time
    from sklearn.preprocessing import normalize
    from collections import defaultdict
    
    transMat_dict = defaultdict(list)
    
    if useBJ:
        Pmat = np.load('./aggregated_3rd_Pmat.npy') # load the matrix
        transMat = normalize(Pmat, axis=1, norm='l1') #normalize the matrix to be a transition probability matrix
        space2Search = [884]
        for searchSpace in space2Search:
            transMat_dict[searchSpace] = transMat
    else:
        # mT is the random walk transition matrix, i.e., A matrix with 1/2 transition probability to neighboring cells        
        for searchSpace in space2Search:
            transMat =  np.diag(np.ones(searchSpace-1), 1)*1/2 +  np.diag(np.ones(searchSpace-1), -1)*1/2 
            #Normalize for the top and bottom row
            row_sums = transMat.sum(axis=1)
            transMat = transMat / row_sums[:, np.newaxis]
            transMat_dict[searchSpace] = transMat

    output_list = []
    count = 0
    #start_time = time.time()
    for Magic in dirichletMagic:
        for searchSpace in space2Search:
                        
            maxrounds = searchSpace*1000
            if setSeed:
                np.random.seed(count)
            count += 1
            transMat = transMat_dict[searchSpace]
            #print("def runSimul: ")
            #print("i :", cGUASi, "j: ", cGUASj, "d: ", cGUASd, "m: ", cGUASm, "n: ",cGUASn)
            out_= strategyOutput(Magic, searchSpace, dict_strategy, stratToTest, transMat, maxrounds, knowledge, suc_thresh,
                                 cGUASi=cGUASi, cGUASj=cGUASj, cGUASd=cGUASd, cGUASm=cGUASm, cGUASn=cGUASn)[0]
            output_list.append(out_)
    
    out_df = pd.concat(output_list, ignore_index=True)
    out_df.columns = ['Strategy', 'Magic', 'Search space', 'Round number', 'Probability', 'Initial Entropy']  
    #print("--- %s seconds ---" % (time.time() - start_time))
    
    return out_df
    
