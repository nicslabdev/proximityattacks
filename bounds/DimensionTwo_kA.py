# -*- coding: utf-8 -*-
"""
Two Dimensional Case.
We assume Bob is doing random walk in a two dimensional search space of size n by n.
A location proximity service is available for Alice to query if Bob is within a 
certain distance from her.
Alice can cheat with her location while using this service.
Her goal is to locate Bob.
We use B to denote the victim Bob and A to denote the attacker Alice.
We are interested in the following question:
    
    "how many queries are needed for Alice so that the probability of A 
    locating B is at least 0.5?"
    
In this script, we define several possible attacker strategies (see lines 61-127).
Line 131 defines which attacker strategy is used for the computation.
For more attacker strategy not defined here, pls implement and change the first
line of AttackerV fucntion (line 131).

For any implemented attacker strategy, say A_st, for any n, this script can be 
used to calculate k such that 

In a two dimensional search space of size n by n, Alice follows attacker strategy 
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
We assume this probability is 1/n^2, where the search space is of size n by n

"""
def MatrixB(n):
    n2=n*n
    B=[1.0/n2]*n2
    B = np.matrix(B)
    return B

"""
The attacker strategies are defined in the following functions.
For more attacker strategy not defined here, pls implement and change the first
line of AttackerV fucntion
"""

#linear attacker
def LinearAttacker(n):
    Alist=[]
    for i in range(n):
        if i%2==0:
            for j in range(n):
                Alist.append(n*i+j)
        else:
            for j in range(n-1,-1,-1):
                Alist.append(n*i+j)
    return Alist

#function to generate static attacker Position vector
def StaticAttacker(pos,n):
    Alist=[]
    for i in range(n*n*3):
        Alist.append(pos)
    #print("Static Attacker at position",Alist)
    return Alist

#attacker at (1,1)
def Attacker11(n):
    Alist=[]
    for i in range(n*n):
        Alist.append(n+1)
    return Alist

#attacker at (1,1),(2,2)..
def AttackerDia(n):
    Alist=[]
    for i in range(n):
        Alist.append(n*i+i)
    return Alist

#attacker at two diagonals
#4-9, 5-14
def Attacker2Dia(n):
    Alist=[]
    for i in range(1,n-1):
        Alist.append(n*i+i)
    for i in range(1,n-1):
        #for j in range(1,n-1):
        j=n-1-i
        Alist.append(n*i+j)
        #print("i,j",i,j)
    return Alist
#Attacker2Dia(6)

#attacker at (1,1)-(1,3)-(1,5)...
#result: 4-11, 5-15, 6-22, 7-27
def AttackerJp(n):
    Alist=[]
    if n%2==1:
        for j in range(1,n,2):
            for i in range(1,n-1,2):
                Alist.append(n*i+j)
                #print("i,j",i,j)
    else:
        for j in range(1,n-1,2):
            for i in range(1,n,2):
                Alist.append(n*i+j)
                #print("i,j",i,j)
    return Alist
#AttackerJp(5)

def Attackercorner(n):
    Alist=[n+1,2*n-2,n*n-n-2,n*n-2*n+1]
    return Alist

#function to generate attacker Position vector
def AttackerV(n):
    Alist=AttackerJp(n)
    
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
def in_range(i,j,n):
    #r = 0
    if ((0 <= i <= (n-1)) and (0 <= j <= (n-1))):
        return True
    return False

def MatrixP(n):
    n2=n*n
    P=[[0 for i in range(n2)] for i in range(n2)]
    
    for i in range (0,n):
        for j in range (0,n):
            counter = 0
            if in_range(i-1,j,n):
                counter = counter + 1
            if in_range(i,j-1,n):
                counter = counter + 1
            if in_range(i+1,j,n):
                counter = counter + 1
            if in_range(i,j+1,n):
                counter = counter + 1

            if in_range(i-1,j,n):
                P[n*i+j][n*(i-1)+j] = 1.0/counter
            if in_range(i,j-1,n):
                P[n*i+j][n*i+j-1] = 1.0/counter
            if in_range(i+1,j,n):
                P[n*i+j][n*(i+1)+j] = 1.0/counter
            if in_range(i,j+1,n):
                P[n*i+j][n*i+j+1] = 1.0/counter
    P=np.matrix(P)
    return P

"""
Two lists are computed in this function:
    Blist: Bllist[i]=B^(i)=B*P^i, and B^(i)_j is the probability of B being at 
    position j at step i
    Plist: Plist[i]= P^i, and P^i_{x,y} gives the probability of B going from 
    x to y in i steps
"""
def ComputeBandP(n):
    
    n2=n*n
    #Defining initial position matrix B
    B=MatrixB(n)
    noofloop = n*n
              
    #Defining transition matrix P
    P = MatrixP(n)

    

    #Calculate the powers of P and the vectors B^{m)}
    Blist=[]
    Plist=[]
    IdentityMatrix = [[0 for i in range(n2)] for i in range(n2)]
    for i in range(n2):
        for j in range(n2):
            if (i==j):
                IdentityMatrix[i][j]=1
    IdentityMatrix = np.matrix(IdentityMatrix)
                              
    Blist.append(B)
    Plist.append(IdentityMatrix)
    for m in range(1,noofloop):
        
        Pm = Plist[m-1]*P
        
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
n by n, pos=-1 corresponds to a non-static attacker and if pos is between 0 and 
n^2-1, it corresponds to a static attacker who stays at pos all the time. 
Blist, Plist as described above
Output of the program: Probability
    for a two dimensional space of size n by n, after k+1 queries, the probability of
    A locating B
"""
def CalProbability(n,k,pos,Blist,Plist):    
    #Defining Attacker positions
    if pos == -1:
        Alist = AttackerV(n)
        
    else:
        Alist = StaticAttacker(pos,n)
        
    
    Probability=0
    """
    To parallelize this:
    """
    for m in range(k+1):
            
            
            Probability= CalTotalSumTimesBlist(n,k,Alist,Plist,Blist,m) + Probability
        

    return Probability




    
"""
This function takes input (n,k,pos,Blist,Plist), where the search space is of 
size n by n, pos=-1 corresponds to a non-static attacker and if pos is between 0 and n^2-1, 
it corresponds to a static attacker who stays at pos all the time. 
Blist, Plist as described above
Output of the program: k
    for a two dimensional space of size n by n, after k+1 queries, the probability of
    A locating B is at least 0.5
"""
def StepsNeeded(n,pos):
        steps = 0
        Blist,Plist = ComputeBandP(n)
        for k in range(n,n*n):
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
pos=-1 corresponds to a non-static attacker and if 0\leq pos<n^2-1, pos corresponds to 
a static attacker who stays at pos all the time. 

Output of the function: steplist[]
steplist[j]=k, where j=n-min and k is such
that for a search space of size n by n, the probability of A locating B is at 
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
    
    return(spacelist)

    
"""
This function takes input (min,max,pos), where min corresponds to the minimum 
space size and max corresponds to the maximum space size that will be computed.
pos=-1 corresponds to a non-static attacker and if 0\leq pos<n^2, pos corresponds to 
a static attacker who stays at pos all the time. 

Output of the function: 
    spacelist[] such that spacelist[j]=n,
    steplist[] such that steplist[j]=k, where n=min+j and for a space of size n by n, 
    the probability of A locating B is at least 0.5 after k+1 queries
"""
def CalculateList(min,max,pos):
    with open('DatafileDim2.txt','w') as f:
        f.write('Dimension Two\n')
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
    with open('DatafileDim2.txt', 'a') as f:
        f.write(str)


CalculateList(4,5,-1)

        
        
        
        

