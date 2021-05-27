# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 12:19:44 2019

@author: idswx
"""

def getProbs(data):
    
    from collections import Counter
    
    if len(data) <= 1:
        return 0

    counts = Counter()

    for d in data:
        counts[d] += 1

    probs = [float(c) / len(data) for c in counts.values()]
    probs = [p for p in probs if p > 0.]

    return probs

def strategyConst(B, T, r):
    '''Pick a constant value in middle of B'''
    import numpy as np
    return np.shape(B)[1]//2
