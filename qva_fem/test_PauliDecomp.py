#!/usr/bin/env python3
import sys
sys.path.append('/Users/corey/.pyenv/versions/3.5.4/lib/python3.5/site-packages')

#from pauliDecomp import *
from fem_1d import *
from qva import *
import numpy as np
import matplotlib.pyplot as plt

nqbits = 2
nlayers = 3
maxiter = 200
reduced = 1
method = "COBYLA"
rhobeg = np.pi/10.0

#random.seed()
nparameters = nqbits + 2*(nqbits-1)*(nlayers-1)
#parameters = [float(random.randint(-3100,3100))/1000 for i in range(0, nparameters)]
parameters = [-0.271168, -1.418207,  3.262556, -3.699268, -1.848239,  2.757174]

# Define and Solve the FEM Problem
xl = 0; xr = 1; nx = 2**(nqbits) + 2;
tl = 1; tr = 3; nt = 11
u, x, t, M, A, dt, h = heat_solve(xl,xr,nx,tl,tr,nt,reduced)

# Get FEM Matrix Pauli Decomposition
#result = pauli_decomp(nqbits,A,Pauli,Pauli_Names,nPauli)
#c = []
#g = []
#for i in range(len(result)):
#    c.append(result[i][0])
#    g.append(result[i][2])
#print(c)
#print(g)
g = [[0, 0],[0, 1],[1, 1],[2, 2]]
c = [2.13333333333333, -0.966666666666667, -0.483333333333333,-0.483333333333333]


ul = u_heat(xl,tl)
ur = u_heat(xr,tl)
uscale = u_heat(x[1],t)

if reduced == 1:
    b = u_heat(x[1:nx-1],tl)
else:
    b = u_head[x,tl]

u, nit, parameters = run_qva(nqbits,nlayers,maxiter,c,g,b,parameters,method,rhobeg,ul,ur,uscale,reduced)

print(u)


