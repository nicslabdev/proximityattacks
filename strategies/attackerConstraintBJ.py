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
        
    def allowed_moves(self):
        
        i2D,j2D = np.unravel_index(self.j, (26, 34))[0], np.unravel_index(self.j, (26, 34))[1]
        
        r = [[x, y] for x in range(i2D-self.d, i2D+self.d+1) \
             for y in range(j2D-self.d, j2D+self.d+1) \
             if ((x >= 0) & (x < self.m) & (y >= 0) & (y < self.n))] 
#         print(r)
        x, y = np.array(r).transpose()
        
#         print("self ijdmn: ", self.i, self.j, self.d, self.m, self.n)
#         print("x range: ", self.i-self.d, self.i+self.d+1)
#         print("y range: ", self.j-self.d, self.j+self.d+1)
#         print("x: ", x)
#         print("y: ", y)
#         print("r is : ", r)
        x, y = np.array(r).transpose()

        return x, y
    
    def strategyGUASconstraint(self, B):
        """
        Run GUAS with constraint
        Arguments: B-a matrix(if 2 dim) of pmf
        
        Return: Attacker's belief by GUAS in the grid of allowed moves 
        """
        x, y = self.allowed_moves() 
        ravel_xy = np.ravel_multi_index(np.array([x, y]), (self.m,self.n))
        #print("x: ", x)
        #print("y: ", y)
        #return y[np.argmax(B[x,y])]
        #return np.ravel_multi_index([x, y], (self.m,self.n))
        return ravel_xy[np.argmax(B[0,ravel_xy])]