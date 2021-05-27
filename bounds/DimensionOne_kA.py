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
    
In this script, we define two possible attacker strategies (see lines 58-69).
Line 73 defines which attacker strategy is used for the computation.
For more attacker strategy not defined here, pls implement and change the first
line of AttackerV fucntion (line 73).

For any implemented attacker strategy, say A_st, for any n, this script can be 
used to calculate k such that 

In a one dimensional search space of size n, Alice follows attacker strategy 
A_st, k+1 is the number of quereis Alice needs to locate Bob with a probability>=0.5.
"""

import numpy as np
import itertools
import time
import copy
from multiprocessing import Process, Array

#Number of CPU cores for threading
NUMCORES = 30

#Global cache for IndexList
try:
    IndexListCache = np.load('IndexList.npy').item()
except IOError: 
    IndexListCache = {}

"""
Function to calculate the initial matrix B, B_i is the probability B starts 
from position i.
We assume this probability is 1/n, where the search space is of size n

"""
def MatrixB(n):
    B=[1.0/n]*n
    return B

"""
The attacker strategies are defined in the following functions.
For more attacker strategy not defined here, pls implement and change the first
line of AttackerV fucntion
"""

def Attackerjp(n):
    Alist=[]
    for i in range(0,n,2):
        Alist.append(i)

#function to generate static attacker Position vector
def StaticAttacker(pos,n):
    Alist=[]
    for i in range(n*3):
        Alist.append(pos)
    #print("Static Attacker at position",Alist)
    return Alist

#function to generate attacker Position vector
def AttackerV(n):
    Alist=Attackerjp(n)
    
    for i in range(n):
        B = Alist[:-1]
        B = list(reversed(B))
        Alist.extend(B)

    return Alist




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

"""
Two lists are computed in this function:
    Blist: Bllist[i]=B^(i)=B*P^i, and B^(i)_j is the probability of B being at 
    position j at step i
    Plist: Plist[i]= P^i, and P^i_{x,y} gives the probability of B going from 
    x to y in i steps
"""
def ComputeBandP(n):
    #Defining initial position matrix B
    B=MatrixB(n)
    B = np.matrix(B)
  #  print B
              
    #Defining transition matrix P
    P = MatrixP(n)

    #Printing the two matrices
    #print ("Matrix B",B,"\n")
    #print ("Matrix P",P,"\n")
    

    #Calculate the powers of P and the vectors B^{m)}
    Blist=[]
    Plist=[]
    IdentityMatrix = [[0 for i in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if (i==j):
                IdentityMatrix[i][j]=1
    IdentityMatrix = np.matrix(IdentityMatrix)
                              
    Blist.append(B)
    Plist.append(IdentityMatrix)
    for m in range(1,3*n):
        #Pm=LA.matrix_power(P, m)
        Pm = Plist[m-1]*P
        #print("Pm", Pm, "PmPpower", Pmpower)
        Plist.append(Pm)
        Blist.append(B*Pm)    
    return Blist, Plist

#Partial sum from start to end
def ProductInSumFunc(m,IndexList,l,Alist,Plist,start,end,result,index):
    LongSum = 0
    for j in range(int(start),int(end)):
        ProductIndex = (m,)+ IndexList[j]
        ProductInSum = 1
        for i in range(l):
            a = ProductIndex[i]
            b = ProductIndex[i+1]
            Aa = Alist[a]
            Ab = Alist[b]
            ProductInSum = ProductInSum*Plist[b-a][Aa,Ab]
            if ProductInSum == 0:
                break
        LongSum = ProductInSum + LongSum
    result[index] = LongSum 


#Function to calculate the long sum
def CalLongSum(IndexList,m,l,Alist,Plist):
    NoOfComb = len(IndexList)
    #print("NoOfComb",NoOfComb)
    LongSum = 0
 
    if NoOfComb < NUMCORES:
 #   Parallelize:
        for j in range(NoOfComb):
            ProductIndex = (m,)+ IndexList[j]
            ProductInSum = 1
            for i in range(l):
                a = ProductIndex[i]
                b = ProductIndex[i+1]
                Aa = Alist[a]
                Ab = Alist[b]
                ProductInSum = ProductInSum*Plist[b-a][Aa,Ab]
                if ProductInSum == 0:
                    break

            LongSum = ProductInSum + LongSum
  
    else: 
        
        #Change to number of real cores!
        threads = [None] * NUMCORES
        results =  Array('d', [0]*len(threads))
        
        for i in range(len(threads)):
            threads[i] = Process(target=ProductInSumFunc, args=(m,IndexList,l,Alist,Plist,i*NoOfComb/len(threads),(i+1)*NoOfComb/len(threads),results, i))

        for i in range(len(threads)):
            threads[i].start()


        for i in range(len(threads)):
            threads[i].join()

        LongSum = sum(results)


    return LongSum
        
    
def ComputeIndexListRec(m,l,k):
    id = str(m)+","+str(l)+","+str(k)
    if id not in IndexListCache.keys():
        #print("Not generated",m,l,k)
        if l ==1 :
            IndexListCache[id] = [ (x,) for x in range(m+1,k+1) ]
        elif (k-m-1 >= l):
            s = ComputeIndexListRec(m,l,k-1)
            n = copy.deepcopy(s)
            t = ComputeIndexListRec(m,l-1,k-1)
            t = [ x+(k,) for x in t]
            n = [ x[:-1]+(k,) for x in n]
            IndexListCache[id] = list(set(s + n + t))
        else: 
            IndexListCache[id] = [tuple(range(m+1,k+1))]
    return IndexListCache[id]    



def ComputeIndexList(m,l,k):
    id = str(m)+","+str(l)+","+str(k)
    if id not in IndexListCache.keys():
        M = range(m+1,k+1)
        IndexListCache[id]=list(itertools.combinations(M,l))
    return IndexListCache[id]   


"""
Function to calculate the term in the big bracket, for calculation of 
exact value of probability
"""
def CalTotalSum(n,m,k,Alist,Plist):
    #print ("Computing ranges for m =", m, '\n')
    TotalSum = 0
    for l in range(1,k-m+1):
        IndexList = ComputeIndexListRec(m,l,k)
        TotalSum = TotalSum+(-1)**l*CalLongSum(IndexList,m,l,Alist,Plist)
    TotalSum = 1+ TotalSum
    return TotalSum

def CalTotalSumTimesBlist(n,k,Alist,Plist,Blist,m):
    return Blist[m][0,Alist[m]]*CalTotalSum(n,m,k,Alist,Plist)




"""
This function takes input (n,k,pos,Blist,Plist), where the size of the space is 
n, pos=-1 corresponds to a non-static attacker and if pos is between 0 and n, 
it corresponds to a static attacker who stays at pos all the time. 
Blist, Plist as described above
Output of the program: Probability
    for a one dimensional space n, after k+1 queries, the probability of
    A locating B
"""
def CalProbability(n,k,pos,Blist,Plist):    
    #Defining Attacker positions
    if pos == -1:
        Alist = AttackerV(n)
        #print("LinearAttacker")
    else:
        Alist = StaticAttacker(pos,n)
        #print("StaticAttacker at pos",pos)
    
    Probability=0
    """
    To parallelize this:
    """
    for m in range(k+1):
            #print(TotalSum)
            #print ("Bmam",Blist[m][0,Alist[m]])
            Probability= CalTotalSumTimesBlist(n,k,Alist,Plist,Blist,m) + Probability
        #print("Probability",Probability)

    return Probability




    
"""
This function takes input (n,k,pos,Blist,Plist), where the search space is of 
size n, pos=-1 corresponds to a non-static attacker and if pos is between 0 and n-1,
it corresponds to a static attacker who stays at pos all the time. 
Blist, Plist as described above
Output of the program: k
    for a one dimensional search space of size n, after k+1 queries, the probability of
    A locating B is at least 0.5
"""
def StepsNeeded(n,pos):
        steps = 0
        Blist,Plist = ComputeBandP(n)
        pos=int(n/2) #try to specify position
        for k in range(n):
            Probability = CalProbability(n,k,pos,Blist,Plist)
            if Probability>=1/2.0:
                steps = k
                print("n ",n,"k ",k, "Probability", Probability)
                WriteToFile("For n={0},k={1}, the probability is {2}\n".format(n,k,Probability))       
                break
        return steps
    
    
    

"""
This function takes input (min,max,pos), where min corresponds to the minimum 
space size and max corresponds to the maximum space size that will be computed.
pos=-1 corresponds to a non-static attacker and if 0\leq pos<n, pos corresponds to 
a static attacker who stays at pos all the time. 

Output of the function: steplist[]
steplist[j]=k, where j=n-min and k is such
that for a search space of size n, the probability of A locating B is at 
least 0.5 after k+1 queries
"""
def Steplist(min,max,pos):
    steplist=[]
    for n in range(min,max+1):
        time1 = time.clock()
        steplist.append(StepsNeeded(n,pos))
        print("running for n",n,"time",time.clock()-time1)
        WriteToFile("running for n, {0}, time, {1}\n\n".format(n,time.clock()-time1))
        if n%5==0:
            WriteToFile("steplist:{0}\n".format(steplist))
         
 
    return(steplist)

#Search Space list
def Spacelist(min,max):
    spacelist=[]
    for n in range(min,max+1):
        spacelist.append(n)
    #print("spacelist", spacelist)
    return(spacelist)

    
"""
This function takes input (min,max,pos), where min corresponds to the minimum 
space size and max corresponds to the maximum space size that will be computed.
pos=-1 corresponds to a non-static attacker and if 0\leq pos<n, pos corresponds to 
a static attacker who stays at pos all the time. 

Output of the function: 
    spacelist[] such that spacelist[j]=n,
    steplist[] such that steplist[j]=k, where n=min+j and for a space of size n, 
    the probability of A locating B is at least 0.5 after k+1 queries
"""
def CalculateList(min,max,pos):
    with open('DatafileDim1.txt','w') as f:
        f.write('Dimension One\n')
    if pos==-1:
        WriteToFile('Computation of k_A\n')
    spacelist=Spacelist(min,max)
    steplist=Steplist(min,max,pos)
    print("spacelist", spacelist)
    print("steplist:", steplist)
    WriteToFile("\n\nspacelist:{0}\nsteplist:{1}\n".format(spacelist,steplist))
    """
    the following line saves the python dictionary IndexListCache in order to 
    reduce the time for running next program again
    """
    np.save('IndexList.npy', IndexListCache)


"""
This function defines which file to write the data to.
Modify the second line to have output file with a differnt name.
"""
def WriteToFile(str):
    with open('DatafileDim1.txt', 'a') as f:
        f.write(str)


CalculateList(4,8,-1)

        
        
        
        

