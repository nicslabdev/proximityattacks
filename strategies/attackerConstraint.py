# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 16:07:41 2019

@author: idswx
"""
import numpy as np 
 
class attackerConstraint:
    """
    Create a class of attacker with constraint
    
    """
    
    def __init__(self, i, j, d, m, n):
        
        """
        Arguements: (i,j) - attacker is at location (i, j)
                    d - allowed distance to move. Attacker can move within range (i-d,j-d) to (i+d,j+d)
                    (m,n) 2-D space A.shape = (m,n)
        """
        
        self.i = i
        self.j = j
        self.d = d
        self.m = m
        self.n = n
        
    def allowed_moves(self, to1D = False): #Ruben
        """
        Find out the positions which are allowed for the adversary to move given a restriction 
        on the number of cells (distance) he/she can move per iteration. 
        Arguements: (i,j) - attacker is at location (i, j)
                    d - allowed distance to move. Attacker can move within range (i-d,j-d) to (i+d,j+d)
                    (m,n) 2-D space A.shape = (m,n)
                    to1D - If True, we need project 2D to 1D
                    
        Return: Two arrays for the x and y values of the coordinates where the attacker is allowed to move if to1D = False
                If to1D = True, return the corresponding 1D index

        """
         
    
        r = [[x,y] for x in range(self.i-self.d, self.i+self.d+1)\
             for y in range(self.j-self.d,self.j+self.d+1)\
             if ((x >= 0) & (x < self.m) & (y >= 0) & (y < self.n))] 
        
        #print("i :", self.i, "j: ", self.j, "d: ", self.d, "m: ", self.m, "n: ",self.n)
        #print("r is :", r)
        
        x, y = np.array(r).transpose()
    
        if to1D == False:
            return x, y
        else:
            idx = np.ravel_multi_index([x, y], (self.m,self.n))
            return idx
    
    
    def strategyGUASconstraint(self, B):
        """
        Run GUAS with constraint
        Arguments: B-a matrix(if 2 dim) of pmf
        
        Return: Attacker's belief by GUAS in the grid of allowed moves 
        """
        x, y = self.allowed_moves()       
        return y[np.argmax(B[x,y])]
    

"""
#Example: 
test = attackerConstraint(i=0, j=5, d=2, m=1, n=10)
x, y = test.allowed_moves()
test.strategyGUASconstraint(B=np.array([[0,1,2,3,4,5,6,7,8,9]]))
"""