# -*- coding: utf-8 -*-
"""
One Dimensional Case.
We assume Bob is doing random walk in a one dimensional search space of size n.
A location proximity service is available for Alice to query if Bob is within a 
certain distance from her.
Alice can cheat with her location while using this service.
Her goal is to locate Bob.
We use B to denote the victim Bob and A to denote the attacker Alice.
We are interested in the following question:
    
    "how many queries are needed for Alice so that the probability of A 
    locating B is at least 0.5?"
    
This script can be used to calculate a lower bound, k_lower, such that 

In a one dimensional search space of size n, no matter what attacker strategy is
used by Alice. k_lower+1 is a lower bound on the minimum number of quereis 
Alice needs to locate Bob with a probability>=0.5.
"""

import numpy as np
import time

"""
Function to calculate the initial matrix B, B_i is the probability B starts 
from position i.
We assume this probability is 1/n, where n is the size of the space
"""
def MatrixB(n):
    B=[1.0/n]*n
    B = np.matrix(B)
    return B

"""
Function to generate the transition matrix P, where P_{i,j} is the probability
for B to go from i to j in one step, where we assume B takes one step between
any two queries made from Alice.
"""
def MatrixP(n):
    P =[[0 for i in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if ((i+1==j)|(i-1==j)):
                P[i][j]=1/2.0
    P[0][1]=1
    P[n-1][n-2]=1
    P = np.matrix(P)
    return P

def StepsNeeded(n):
    
    #Blist=ComputeBlist(n)
    Probability = 0
    #Defining initial position matrix B
    B=MatrixB(n)
              
    #Defining transition matrix P
    P = MatrixP(n)
    Btemp=B
    for k in range(n):
        listvector=[]
        for i in range(n):
            listvector.append(Btemp[0,i])
        Btemp=Btemp*P
        m = max(listvector)
        Probability = Probability + m
        if Probability>=0.5:
            WriteToFile("n {0},k {1},Probability {2}\n".format(n,k,Probability))
            return k


"""
This function takes input (min,max), where min corresponds to the minimum 
space size and max corresponds to the maximum space size that will be computed.

Output of the function: steplist[]
steplist[j]=k, where j=n-min and k is such that for a search space of size n, 
k+1 is a lower bound on the number of queries needed by A in order to locate B 
with probability>=0.5 no matter what attacker strategy she uses
"""
def Steplist(min,max):
    steplist=[]
    for n in range(min,max+1):
        time1 = time.clock()
        steplist.append(StepsNeeded(n))
        print("running for n",n,"time",time.clock()-time1)
        WriteToFile("running for n, {0}, time, {1}\n\n".format(n,time.clock()-time1))
        if n%5==0:
            WriteToFile("steplist:{0}\n".format(steplist))
    return steplist
    
#Search Space list
def Spacelist(min,max):
    spacelist=[]
    for n in range(min,max+1):
        spacelist.append(n)
    #print("spacelist", spacelist)
    return(spacelist)

"""
This function takes input (min,max), where min corresponds to the minimum 
space size and max corresponds to the maximum space size that will be computed.

Output of the function: 
    spacelist[] such that spacelist[j]=n,
    steplist[] such that steplist[j]=k, where n=min+j and and k is such that 
    for a search space of size n, k+1 is a lower bound on the number of queries 
    needed by A in order to locate B with probability>=0.5 no matter what 
    attacker strategy she uses
"""
def CalculateList(min,max):
    with open('DatafileDim1klower.txt','w') as f:
        f.write('Dimension One\n')
    WriteToFile('Running for k_lower\n')
    spacelist=Spacelist(min,max)
    steplist=Steplist(min,max)
    print("spacelist", spacelist)
    print("steplist:", steplist)
    WriteToFile("\n\nspacelist:{0}\nsteplist:{1}\n".format(spacelist,steplist))

"""
This function defines which file to write the data to.
Modify the second line to have output file with a differnt name.
"""
def WriteToFile(str):
    with open('DatafileDim1klower.txt', 'a') as f:
        f.write(str)
    
    
    
    
CalculateList(10,20)
    
    
    
    
    
    
    
    