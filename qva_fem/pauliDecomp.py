import sys
sys.path.append('/Users/corey/.pyenv/versions/3.5.4/lib/python3.5/site-packages')

from sympy import *
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.dagger import Dagger
from IPython.core.display import display, HTML
import numpy as np
import functools
from operator import mul

# Pauli Unitary Transformations
nPauli = 4
II = Matrix([[1+0*I,0],[0,1+0*I]])
Z = Matrix([[1,0],[0,-1]])
X = Matrix([[0,1],[1,0]])
Y = Matrix([[0,0-I],[0+I,0]])
Pauli = [II,X,Y,Z]
Pauli_Names = ["I","X","Y","Z"]

def pauli_decomp(nqbits,M,Pauli,Pauli_Names,nPauli):
    
    pauli_decomp = []
    new_pauli_decomp = []
    
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
        new_pauli_decomp.append(name) 
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
                new_pauli_decomp.append(name)
                pauli_decomp.append([coeff,name,index]) 
                               
    return pauli_decomp, new_pauli_decomp

def print_decomp_list(pauli_decomp):
    for i in range(0,len(pauli_decomp)):
        print(N(pauli_decomp[i][0],6)," ",pauli_decomp[i][1])
        
def build_matrix_from_decomp(pauli_decomp):
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
    

