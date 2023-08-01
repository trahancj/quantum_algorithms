import numpy as np
from qva import *
from scipy.linalg import ishermitian
import matplotlib.pyplot as plt
import random
from fem_1d import *
from pauliDecomp import *
from scipy.sparse import diags
import pprint
pp = pprint.PrettyPrinter(width=110, compact=True)
import qiskit.quantum_info as qi
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
#from qiskit_aer.primitives import Estimator  # import change!!!
from qiskit.primitives import Estimator
from scipy.optimize import minimize



###############################
#     <0| U^d An V(k) |0>     #
###############################
def b_psi(parameters,nqbits, nlayers, g):
    ni = 2**(nqbits)
    nx = ni + 2

    qubits = []
    for i in range(0,nqbits):
        qubits.append(i)

    ## Getting the U operator ##

    uc = QuantumCircuit(nqbits)
    uc.h(0)
    uc.h(1)
    U = qi.Operator(uc)
  
    ## Getting the V(k) operator ##

    Vcirc = QuantumCircuit(nqbits)
    ## parameters = [1, 2, 3, 4]
    Vcirc = ansatz_RYZ(Vcirc, qubits, parameters, nlayers)
    ###############print(Vcirc)
    V = qi.Operator(Vcirc)

    ## Getting the An operator ##

    #coeffs = c
    labels = g
    An = SparsePauliOp(labels)
    An = qi.Operator(An)
 
    ## Making the operator ##

    op = V.compose(An.compose(U.adjoint()))  # <-- our circuit matches this order

    # NOTE: if op is hermitian, then it's expectation values should be all real

    ## prepare |0> state ##
    state = QuantumCircuit(nqbits)

    ## Run estimator ##

    estimator = Estimator()
    expectation_value = estimator.run(state, op).result().values
    ##expectation_value = estimator.run(state, op, shots=10000).result().values
    #print("*********************************************")
    #print("USING OPERATORS")
    #print("REAL[ E[<0| U^d An V(k) |0>] ] = ",expectation_value[0].real)
    #print("IMAG[ E[<0| U^d An V(k) |0>] ] = ",expectation_value[0].imag)
    #print("*********************************************")


    return expectation_value
    ###########################################################################
    ###########################################################################



###############################
# <0| V(k)^D An^D Am V(k) |0> #
###############################

def psi_psi(parameters, nqbits, nlayers, g1, g2):
    qubits = []
    for i in range(0,nqbits):
        qubits.append(i)

    state = QuantumCircuit(nqbits)
    state = ansatz_RYZ(state, qubits, parameters, nlayers)

    An = SparsePauliOp.from_list([(g1, 1)])
    An = An.adjoint()
    An = qi.Operator(An)

    Am = SparsePauliOp.from_list([(g2, 1)])
    AM = qi.Operator(Am)

    op = An.compose(Am)

    estimator = Estimator()
    expectation_value = estimator.run(state, op).result().values
    ##expectation_value = estimator.run(state, op, shots=10000).result().values
    #print("*********************************************")
    #print("USING OPERATORS")
    #print("REAL[ E[<0| V(k)^D Am^D An V(k) |0>] ] = ",expectation_value[0].real)
    #print("IMAG[ E[<0| V(k)^D Am^D An V(k) |0>] ] = ",expectation_value[0].imag)
    #print("*********************************************")


    return expectation_value



cost_values = []



def new_cost_function(parameters, nqbits, nlayers, my_gate_set, my_coefficient_set):
    #print(f'Parameters in new cost function {parameters}')
    #print("IN NEW COST FUNCTION")
    norm = complex(0,0)
    cost = complex(0,0)
    for i in range(0,len(my_gate_set)):
        for j in range(0,len(my_gate_set)):
            #print("IN FOR LOOP")
            norm += my_coefficient_set[i] * my_coefficient_set[j].conjugate() * complex(psi_psi(parameters, nqbits, nlayers, my_gate_set[i],my_gate_set[j])[0].real, psi_psi(parameters, nqbits, nlayers, my_gate_set[i],my_gate_set[j])[0].imag)
            t1 = complex(b_psi(parameters, nqbits, nlayers, my_gate_set[i])[0].real, b_psi(parameters, nqbits, nlayers, my_gate_set[i])[0].imag)
            t2 = complex(b_psi(parameters, nqbits, nlayers, my_gate_set[j])[0].real, b_psi(parameters, nqbits, nlayers, my_gate_set[j])[0].imag)
            cost +=  my_coefficient_set[i] * my_coefficient_set[j] * t1 * t2
    cost = complex(cost)
    #print(cost)
    norm = complex(norm)

    result = 1-float(cost.real/norm.real)
    cost_values.append(result)
    print("iteration: ",len(cost_values)," || cost: ",result) #," || w: ",parameters)
    return result



def new_run_qva(nqbits, nlayers, maxiter,c,g,b,parameters,method,rhobeg,ul,ur,uscale,reduced):
    #print("NEW RUN QVA")
    n = 2**nqbits
    nparameters = nqbits + 2*(nqbits-1)*(nlayers-1)
    print(f"Number of parameters {nparameters} and length {len(parameters)}")
    assert(nparameters == len(parameters))
    
    aux = 0
    qubitsRHS = [1]
    for i in range(1,nqbits):
        qubitsRHS.append(i+1)

    # Normalize this vector
    b = b/np.linalg.norm(b)
    #print("NORMALIZE THIS VECTOR")
    #b = np.array(b,dtype=complex)
    
    # Get a circuit that takes 0 state and gives the vector
    automated_circ = QuantumCircuit(nqbits+1)
    automated_circ = obtain_circuit_from_vec(b,automated_circ,qubitsRHS)
    circ_RHS = QuantumCircuit(nqbits+1)
    circ_RHS = control_version(automated_circ,circ_RHS,aux)
    
    nit = 0
    cost_values = []
    maxiter = 4000
    #print("BEFORE MINIMIZE")
    out = minimize(new_cost_function, parameters, args=(nqbits, nlayers, g, c), method=method, options={'maxiter':maxiter,'rhobeg':rhobeg,'disp':False}) # Works great for 3 qubit
    #print("AFTER MINIMIZE")
    #print("cost values: ",cost_values)
    #print(out)
    final_parameters = out['x'][0:len(parameters)]
    parameters=final_parameters
    
    qubits = []
    for i in range(0,nqbits):
        qubits.append(i)
    
    circ = QuantumCircuit(nqbits, nqbits)
    circ = ansatz_RYZ(circ, qubits, final_parameters, nlayers)
    circ.save_statevector()
    backend = Aer.get_backend('aer_simulator')
    t_circ = transpile(circ, backend)
    qobj = assemble(t_circ)
    job = backend.run(qobj)

    result = job.result()
    o = result.get_statevector(circ, decimals=3)
    
    u_internal = np.absolute(np.real(o))
    u_internal = u_internal * (uscale/u_internal[0])
    
    n = n + 2
    u = np.zeros(n)
    if reduced == 0:
        u = u_internal
    else:
        u[0] = ul
        u[n-1] = ur
        u[1:n-1] = u_internal
        
    #print("cost_Values: ".cost_values)
    return u,parameters,cost_values