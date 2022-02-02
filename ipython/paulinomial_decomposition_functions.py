#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import sympy as sp
from sympy import *
from sympy.physics.quantum import TensorProduct
import functools
from operator import mul

def get_pauli_matrices():
    nPauli = 4
    II = Matrix([[1+0*I,0],[0,1+0*I]])
    Z = Matrix([[1,0],[0,-1]])
    X = Matrix([[0,1],[1,0]])
    Y = Matrix([[0,0-I],[0+I,0]])
    Pauli = [II,X,Y,Z]
    Pauli_Names = ["I","X","Y","Z"]
    return nPauli, Pauli, Pauli_Names

def pauli_decomp(nqbits,M,Pauli,Pauli_Names,nPauli):
    
    pauli_decomp = []
    
    # ((qbit1),(qbit2),(qbit3),...)
    # ranges=((0,nPauli),(0,nPauli),(0,nPauli)) #<--- 3 qubit
    ranges=[[0,nPauli]] # this is a list
    for i in range(1,nqbits):
        ranges.append([0,nPauli])
    # now convert to tuples
    ranges = [tuple(l) for l in ranges]
    ranges = tuple(ranges)
    #print(ranges)

    operations=functools.reduce(mul,(p[1]-p[0] for p in ranges))-1
    result=[i[0] for i in ranges]
    pos=len(ranges)-1
    increments=0

    #P = TensorProduct(Pauli[result[0]],Pauli[result[1]],Pauli[result[2]]) # 3qbit
    P = TensorProduct(Pauli[result[0]],Pauli[result[1]])
    for ii in range(2,nqbits):
        P = TensorProduct(P,Pauli[result[ii]]);
    
    coeff = (1/(2**(nqbits))) * trace(P*M)
    if (abs(coeff) > 1e-12):
        
        #pauli_decomp.append([coeff,Pauli_Names[result[0]] + Pauli_Names[result[1]] + Pauli_Names[result[2]],[result[0],result[1],result[2]]]) # 3bit
        #print(str(coeff) + "*" + Pauli_Names[result[0]] + Pauli_Names[result[1]] + Pauli_Names[result[2]])  
        
        # nqbit data entry
        name = Pauli_Names[result[0]]
        index = [result[0]]
        for ii in range(1,nqbits):
            name =  name + Pauli_Names[result[ii]]
            index.append(result[ii])
        pauli_decomp.append([coeff,name,index]) 

    while increments < operations:
        if result[pos]==ranges[pos][1]-1:
            result[pos]=ranges[pos][0]
            pos-=1
        else:
            result[pos]+=1
            increments+=1
            pos=len(ranges)-1 #increment the innermost loop
        
            #P = TensorProduct(Pauli[result[0]],Pauli[result[1]],Pauli[result[2]]) # 3qbit
            P = TensorProduct(Pauli[result[0]],Pauli[result[1]])
            for ii in range(2,nqbits):
                P = TensorProduct(P,Pauli[result[ii]]);
            
            coeff = (1/(2**(nqbits))) * trace(P*M)
            if (abs(coeff) > 1e-12):
                #pauli_decomp.append([coeff,Pauli_Names[result[0]] + Pauli_Names[result[1]] + Pauli_Names[result[2]],[result[0],result[1],result[2]]])
                #print(str(coeff) + "*" + Pauli_Names[result[0]] + Pauli_Names[result[1]] + Pauli_Names[result[2]])    
                
                # nqbit data entry
                name = Pauli_Names[result[0]]
                index = [result[0]]
                for ii in range(1,nqbits):
                    name =  name + Pauli_Names[result[ii]]
                    index.append(result[ii])
                pauli_decomp.append([coeff,name,index]) 
                               
    return pauli_decomp

def print_pauli_decomp_list(pauli_decomp):
    for i in range(0,len(pauli_decomp)):
        print(N(pauli_decomp[i][0],6)," ",pauli_decomp[i][1])
        
def build_matrix_from_pauli_decomp(pauli_decomp, Pauli):
    nqbits = len(pauli_decomp[0][2])
    print("nqbits: ",nqbits)
    
    P = TensorProduct(Pauli[pauli_decomp[0][2][0]],Pauli[pauli_decomp[0][2][1]])
    for ii in range(2,nqbits):
        P = TensorProduct(P,Pauli[pauli_decomp[0][2][ii]]);
    P = pauli_decomp[0][0] * P
    
    #P = pauli_decomp[0][0] * TensorProduct(Pauli[pauli_decomp[0][2][0]],Pauli[pauli_decomp[0][2][1]],Pauli[pauli_decomp[0][2][2]])
    #print(N(pauli_decomp[0][0],6)," * ",Pauli_Names[pauli_decomp[0][2][0]] + Pauli_Names[pauli_decomp[0][2][1]] + Pauli_Names[pauli_decomp[0][2][2]])
    for i in range(1,len(pauli_decomp)):
        
        PP = TensorProduct(Pauli[pauli_decomp[i][2][0]],Pauli[pauli_decomp[i][2][1]])
        for ii in range(2,nqbits):
            PP = TensorProduct(PP,Pauli[pauli_decomp[i][2][ii]]);
        P = P + pauli_decomp[i][0] * PP
        
        #P = P + pauli_decomp[i][0] * TensorProduct(Pauli[pauli_decomp[i][2][0]],Pauli[pauli_decomp[i][2][1]],Pauli[pauli_decomp[i][2][2]])
        #print(N(pauli_decomp[i][0],6)," * ",Pauli_Names[pauli_decomp[i][2][0]] + Pauli_Names[pauli_decomp[i][2][1]] + Pauli_Names[pauli_decomp[i][2][2]])
    return P;


# In[17]:


# Test
# from scipy.sparse import diags
# def create_dirichlet_stiffness(n):
#     temp = [np.ones(n-1),-2*np.ones(n),np.ones(n-1)]
#     offset = [-1,0,1]
#     K = diags(temp,offset).toarray()
#     K[0,1:n] = 0
#     K[n-1,1:n] = 0
#     K[0,0] = 1
#     K[n-1,n-1] = 1
#     return K

# # Create 4 bit (16 grid node) stiffness mmatrix
# nqbit = 4
# K = create_dirichlet_stiffness(2**nqbit)

# nPauli, Pauli, Pauli_Names = get_pauli_matrices()
# print("calling paulinomial decomposition")
# result = pauli_decomp(nqbit,K,Pauli,Pauli_Names,nPauli)
# TEST = build_matrix_from_pauli_decomp(result)
# TEST = np.array(TEST).astype(np.float64)
# print(TEST)
# print(K)
# DIFF = TEST - K
# assert(abs(max(DIFF))<1e-12)

