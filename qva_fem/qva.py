import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.extensions import UnitaryGate
from qiskit.circuit.library import *
from qiskit import transpile, assemble
from qiskit import Aer, IBMQ
import math
import cmath
import random
import sys
from scipy.optimize import minimize
import time

##############################################################################################################
##############################################################################################################
# Ansantz from: Variational Quantum Linear Solver, Carlos Bravo-Prieto (2020)
def ansatz_layer_RYZ(circ, qubits, parameters, parameter_count):

    for iz in range (0, len(qubits)-1,2):
        circ.cz(qubits[iz], qubits[iz+1])
    
    for iz in range (0, len(qubits)):
        circ.ry(parameters[parameter_count], qubits[iz])
        parameter_count += 1
    
    for iz in range (1, len(qubits)-1,2):
        circ.cz(qubits[iz], qubits[iz+1])
    
    for iz in range (1, len(qubits)-1):
        circ.ry(parameters[parameter_count], qubits[iz])
        parameter_count += 1
    return circ, parameter_count

def ansatz_RYZ(circ, qubits, parameters, nlayers):
    
    parameter_count = 0;
    
    for iz in range (0, len(qubits)):
        circ.ry(parameters[parameter_count], qubits[iz])
        parameter_count += 1
    circ.barrier()
    
    for ilayer in range(1,nlayers):
        circ, parameter_count = ansatz_layer_RYZ(circ, qubits, parameters, parameter_count)
        circ.barrier()
    
    #print(f'Parameter count: {parameter_count} and length of parameters {len(parameters)}')
    assert(parameter_count == len(parameters))
    
    #print("parameter_count: ",parameter_count)
    
    return circ

def ansatz_layer_RYZ_controlled(circ, qubits, parameters, aux, anc, parameter_count):
    
    for iz in range (0, len(qubits)-1,2):
        circ.ccx(aux, qubits[iz+1], anc)
        circ.cz(qubits[iz], anc)
        circ.ccx(aux, qubits[iz+1], anc)
    
    for iz in range (0, len(qubits)):
        circ.cry(parameters[parameter_count], aux, qubits[iz])
        parameter_count += 1
    
    for iz in range (1, len(qubits)-1,2):
        circ.ccx(aux, qubits[iz+1], anc)
        circ.cz(qubits[iz], anc)
        circ.ccx(aux, qubits[iz+1], anc)
    
    for iz in range (1, len(qubits)-1):
        circ.cry(parameters[parameter_count], aux, qubits[iz])
        parameter_count += 1
    circ.barrier()
    
    return circ, parameter_count

def ansatz_RYZ_controlled(circ, qubits, parameters, nlayers, aux, anc):
    
    parameter_count = 0
    
    for iz in range (0, len(qubits)):
        circ.cry(parameters[parameter_count], aux, qubits[iz])
        parameter_count += 1
    circ.barrier()
    
    for ilayer in range(1,nlayers):
        circ, parameter_count = ansatz_layer_RYZ_controlled(circ, qubits, parameters, aux, anc, parameter_count)
    
    assert(parameter_count == len(parameters))
    return circ
    
 # Ansantz from: Variational Quantum Linear Solver, Carlos Bravo-Prieto (2020)
def ansatz_layer_ZXZ(circ, qubits, parameters, parameter_count):
    
    for iz in range (0, len(qubits)):
        circ.rz(parameters[parameter_count], qubits[iz])
        parameter_count += 1
        
    for iz in range (0, len(qubits)):
        circ.rx(parameters[parameter_count], qubits[iz])
        parameter_count += 1
    
    for iz in range (0, len(qubits)):
        circ.rz(parameters[parameter_count], qubits[iz])
        parameter_count += 1
        
    for iz in range (0, len(qubits)-1,2):
        circ.cz(qubits[iz], qubits[iz+1])
    
    for iz in range (1, len(qubits)-1,2):
        circ.cz(qubits[iz], qubits[iz+1])
    
    return circ, parameter_count

def ansatz_layer_ZXZ_controlled(circ, qubits, parameters, aux, anc, parameter_count):
    
    for iz in range (0, len(qubits)):
        circ.crz(parameters[parameter_count], aux, qubits[iz])
        parameter_count += 1
        
    for iz in range (0, len(qubits)):
        circ.crx(parameters[parameter_count], aux, qubits[iz])
        parameter_count += 1
    
    for iz in range (0, len(qubits)):
        circ.crz(parameters[parameter_count], aux, qubits[iz])
        parameter_count += 1
        
    for iz in range (0, len(qubits)-1,2):
        circ.ccx(aux, qubits[iz+1], anc)
        circ.cz(qubits[iz], anc)
        circ.ccx(aux, qubits[iz+1], anc)
    
    for iz in range (1, len(qubits)-1,2):
        circ.ccx(aux, qubits[iz+1], anc)
        circ.cz(qubits[iz], anc)
        circ.ccx(aux, qubits[iz+1], anc)
    
    circ.barrier()
    return circ, parameter_count

def ansatz_ZXZ(circ, qubits, parameters, nlayers):
    
    parameter_count = 0;
    
    for iz in range (0, len(qubits)):
        circ.rx(parameters[parameter_count], qubits[iz])
        parameter_count += 1
    circ.barrier()
    
    for ilayer in range(1,nlayers):
        circ, parameter_count = ansatz_layer_ZXZ(circ, qubits, parameters, parameter_count)
        circ.barrier()
    
    #print("parameter_count: ",parameter_count," nparameters: ",len(parameters))
    assert(parameter_count == len(parameters))
    
    return circ

def ansatz_ZXZ_controlled(circ, qubits, parameters, nlayers, aux, anc):
    
    parameter_count = 0
    
    for iz in range (0, len(qubits)):
        circ.crx(parameters[parameter_count], aux, qubits[iz])
        parameter_count += 1
    circ.barrier()
    
    for ilayer in range(1,nlayers):
        circ, parameter_count = ansatz_layer_ZXZ_controlled(circ, qubits, parameters, aux, anc, parameter_count)
    
    #print("parameter_count: ",parameter_count," nparameters: ",len(parameters))
    assert(parameter_count == len(parameters))
    
    return circ

##############################################################################################################
##############################################################################################################

#give the RHS vector and state preparation routine will automatically generate the correct circuit
def obtain_circuit_from_vec(desired_vec,circ, qubits):
    #circ.isometry(desired_vec,qubits,[])
    controlled_gate = StatePreparation(desired_vec)
    
    circ.append(controlled_gate,qubits)
    #circ.initialize(desired_vec,qubits)
    circ = circ.decompose().decompose().decompose().decompose()
    circ = transpile(circ,basis_gates=['rx','ry','rz', 'h', 'cx'])#basis_gates=['u1', 'u2', 'u3', 'cx'])
    return circ

def swap_gates(a,control,new_circ):
    #coded for basis_gates=['rx','ry','rz', 'h', 'cx']
    if a[0].name == 'rx':
        new_name ='crx'
        new_bits = [control,a[1][0].index]
        new_circ.crx(a[0].params[0],new_bits[0],new_bits[1])
    elif a[0].name == 'ry':
        new_name ='cry'
        new_bits = [control,a[1][0].index]
        new_circ.cry(a[0].params[0],new_bits[0],new_bits[1])
    elif a[0].name == 'rz':
        new_name= 'crz'
        new_bits = [control,a[1][0].index]
        new_circ.crz(a[0].params[0],new_bits[0],new_bits[1])
    elif a[0].name == 'h':
        new_name = 'ch'
        new_bits = [control,a[1][0].index]
        new_circ.ch(a[0].params[0],new_bits[0],new_bits[1])
    elif a[0].name == 'cx':
        new_name = 'ccx'
        new_bits = [control,a[1][0].index,a[1][1].index]
        new_circ.ccx(new_bits[0],new_bits[1],new_bits[2])
    return new_name, new_bits

# shifts the auto-generated solution to all real
def Ph(quantum_circuit, theta, qubit,control_bit):
    quantum_circuit.cp(theta,control_bit, qubit)
    quantum_circuit.cx(control_bit,qubit)
    quantum_circuit.cp(theta,control_bit, qubit)
    quantum_circuit.cx(control_bit,qubit)
    return 0
def control_version(circ,new_circ,control_bit):
    #global phase shift
    Ph(new_circ,circ.global_phase,1,control_bit)
    
    for a in circ:
        new_name,new_bits = swap_gates(a,control_bit,new_circ)
        #my_gate = Gate(name=new_name, num_qubits=a[0].num_qubits+1,  params=a[0].params)
        #new_circ.append(my_gate, new_bits)
    #new_circ = new_circ.inverse() # CJT TAKEN OUT
    return new_circ

def control_b(circ,auxilary,qubits,input_circ):
    #nq = circ.num_qubits
    circ=circ.compose(input_circ)
    return circ
    
    
##############################################################################################################
##############################################################################################################

def had_test(circ, gate_type, qubits, aux, parameters, part, nlayers):
    
    assert(len(qubits)==len(gate_type[0]))
    assert(len(qubits)==len(gate_type[1]))

    # First Hadamard gate applied to the ancillary qubit.
    circ.h(aux)
    circ.barrier()
    
    # For estimating the imaginary part of the coefficient, we must add a "-i"
    # phase gate.
    if part == "Im" or part == "im":
        circ.sdg(aux)
        #circ.p(-np.pi / 2, aux)
    circ.barrier()
    
    # Variational circuit generating a guess for the solution vector |x>
    circ = ansatz_RYZ(circ, qubits, parameters, nlayers)
    circ.barrier()
    
    # Controlled application of the unitary component A_l of the problem matrix A.
    # Note A^{\dagger} = A for hermitian Pauli Decomops
    for ie in range (0, len(gate_type[0])): # loop over the qubits in a gate
        if (gate_type[0][ie] == 1):
            circ.cx(aux, qubits[ie])
        if (gate_type[0][ie] == 2):
            circ.cy(aux, qubits[ie])
        if (gate_type[0][ie] == 3):
            circ.cz(aux, qubits[ie])
    circ.barrier()
    
    # Controlled application of the unitary component A_m of the problem matrix A.
    for ie in range (0, len(gate_type[1])):
        if (gate_type[1][ie] == 1):
            circ.cx(aux, qubits[ie])
        if (gate_type[1][ie] == 2):
            circ.cy(aux, qubits[ie])
        if (gate_type[1][ie] == 3):
            circ.cz(aux, qubits[ie])
    circ.barrier()
    
    # Second Hadamard gate applied to the ancillary qubit.
    circ.h(aux)
    
    return circ

#############################################################################################
#############################################################################################
def special_had_test(circ, gate_type, qubits, aux, anc, parameters, reg, part, nlayers,circ_RHS):
    
    # First Hadamard gate applied to the ancillary qubit.
    circ.h(aux)
    circ.barrier()
    
    # For estimating the imaginary part of the coefficient, we must add a "-i"
    # phase gate.
    if part == "Im" or part == "im":
        circ.sdg(aux)
        #circ.p(-np.pi / 2, aux)
    circ.barrier()
    
    # Variational circuit generating a guess for the solution vector |x>
    circ = ansatz_RYZ_controlled(circ, qubits, parameters, nlayers, aux, anc)
    circ.barrier()
    
    # Controlled application of the unitary component A_l of the problem matrix A.
    for ty in range (0, len(gate_type)):
        if (gate_type[ty] == 1):
            circ.cx(aux, qubits[ty])
        if (gate_type[ty] == 2):
            circ.cy(aux, qubits[ty])
        if (gate_type[ty] == 3):
            circ.cz(aux, qubits[ty])
    circ.barrier()
    
    # Adjoint of the unitary U_b associated to the problem vector |b>.
    # In this specific example Adjoint(U_b) = U_b.
    circ = control_b(circ, aux, qubits,circ_RHS)
    circ.barrier()

    # Second Hadamard gate applied to the ancillary qubit.
    circ.h(aux)
    
    return circ

##############################################################################################################
##############################################################################################################

# Returns the the real or imaginary part of the normalization constant <psi|psi>, where |psi> = A |x>, by evaulating a Hadamard test
# Note: Hardwired for 3 qbit problems
def psi_norm_hadamard(nqbits,gate_1,gate_2,part,parameters,nlayers):

    aux = 0
    qubits = []
    for i in range(0,nqbits):
        qubits.append(i+1)
    
    qctl = QuantumRegister(nqbits+1)  # I think this is only needs 4 qubits, since the controlled fixed ansatz is not a part of it
    qc   = ClassicalRegister(nqbits+1)
    circ = QuantumCircuit(qctl, qc)
    backend = Aer.get_backend('aer_simulator')
            
    circ = had_test(circ, [gate_1,gate_2], qubits, aux, parameters, part, nlayers)
    ##print(circ)
    circ.save_statevector()
    t_circ = transpile(circ, backend)
    qobj = assemble(t_circ)
    job = backend.run(qobj)
    result = job.result()
    #outputstate = np.real(result.get_statevector(circ, decimals=100))
    outputstate = result.get_statevector(circ, decimals=100)
    o = outputstate
    ##print(f'RESULT 1: {result}')

    #print("o: ",o)

    # 5 qbit Psi = ['|00000>', '|00001>', '|00010>', '|00011>', '|00100>', '|00101>', '|00110>', '|00111>', /
    #               '|01000>', '|01001>', '|01010>', '|01011>', '|01100>', '|01101>', '|01110>', '|01111>',
    #               '|10000>', '|10001>', '|10010>', '|10011>', '|10100>', '|10101>', '|10110>', '|10111>',
    #               '|11000>', '|11001>', '|11010>', '|11011>', '|11100>', '|11101>', '|11110>', '|11111>']
    # The auxiliary test qbit is the first qbit on the RHS.
    # Below we loop over all 32 (2^5) states and add the square of all states where the auxiliary = 1 which is every other state
    # In the Hadamard test, the expection value is the difference between the |0> and |1> control probabilities. We are looking at |1>, so
    # this difference can be calculated as
    # Re[<psi|U|psi>] = P(0) - P(1) = 1 - 2 P(1)
    m_sum = 0
    for l in range (0, len(o)):
        if (l%2 == 1):
            #n = o[l]**2
            n = o[l] * o[l].conjugate()
            m_sum+=n

                    
    result = (1-(2*m_sum))

    return result # ths is Re/Im[  <0|V^{dagger} A^{dagger}_i A_j V|0>  ]
    
#############################################################################################
#############################################################################################
    
# Calculated Re/Im[  <0|U^{dagger} A_i V|0> ] using a special Hadamard test. There are better ways to do this.
# Note: This is the unnormalized cost value
def bpsi_hadamard(nqbits,gate,part,parameters,nlayers,circ_RHS):
 
    aux = 0
    qubits = []
    for i in range(0,nqbits):
        qubits.append(i+1)
    anc = nqbits+1

    qctl = QuantumRegister(nqbits+2)
    qc = ClassicalRegister(nqbits+2)
    circ = QuantumCircuit(qctl, qc)
    backend = Aer.get_backend('aer_simulator')
    
    circ = special_had_test(circ, gate, qubits, aux, anc, parameters, qctl, part, nlayers,circ_RHS)
    
    circ.save_statevector()
    ##print(circ)
    t_circ = transpile(circ, backend)
    qobj = assemble(t_circ)
    job = backend.run(qobj)
    result = job.result()
    #outputstate = np.real(result.get_statevector(circ, decimals=100))
    outputstate = result.get_statevector(circ, decimals=100)
    o = outputstate
    ##print(f'The state: {o}')
    ##print(f'RESULT 2: {o}')

    
    m_sum = 0
    for l in range (0, len(o)):
        if (l%2 == 1):
            ##print(o[l] * o[l].conjugate())
            #n = o[l]**2
            n = o[l] * o[l].conjugate()
            m_sum+=n
    ##print(m_sum)
    result = (1-(2*m_sum))
    ##print(f"Result: {result}")
    
    return result

##############################################################################################################
##############################################################################################################

# Implements the entire cost function on the quantum circuit
def cost_function(parameters,nqbits,my_gate_set,my_coefficient_set,nlayers,circ_RHS, cost_values, nit):
    
    norm = complex(0,0) # Calculate <\psi|\psi>
    cost = complex(0,0) # Calculate |<b|\psi>|^2
    for i in range(0, len(my_gate_set)):
        for j in range(0, len(my_gate_set)):
            norm += my_coefficient_set[i] * my_coefficient_set[j].conjugate() * complex(psi_norm_hadamard(nqbits,my_gate_set[i],my_gate_set[j],"Re",parameters,nlayers), \
                                                                                        psi_norm_hadamard(nqbits,my_gate_set[i],my_gate_set[j],"Im",parameters,nlayers))
            t1 = complex(bpsi_hadamard(nqbits,my_gate_set[i],"Re",parameters,nlayers,circ_RHS),bpsi_hadamard(nqbits,my_gate_set[i],"Im",parameters,nlayers,circ_RHS))
            t2 = complex(bpsi_hadamard(nqbits,my_gate_set[j],"Re",parameters,nlayers,circ_RHS),bpsi_hadamard(nqbits,my_gate_set[j],"Im",parameters,nlayers,circ_RHS))
            #cost +=  my_coefficient_set[i] * my_coefficient_set[j].conjugate() * t1 * t2
            cost +=  my_coefficient_set[i] * my_coefficient_set[j] * t1 * t2
    cost = complex(cost)
    norm = complex(norm)
        
    if (abs(cost.imag) > 1e-10):
        print("TEST FAILED: abs(np.imag(cost) > 1e-10 :: result = ",abs(np.imag(cost)))
        sys.exit("ERRORS!")
    if (abs(norm.imag) > 1e-10):
        print("TEST FAILED: abs(np.imag(norm) > 1e-10 :: result = ",abs(np.imag(norm)))
        sys.exit("ERRORS!")

    #print("cost: ",cost)
    #print("np.real(cost): ",np.real(cost))
    #print("norm: ",norm)
    result = 1-float(cost.real/norm.real)
    cost_values.append(result)
    print("iteration: ",len(cost_values)," || cost: ",result) #," || w: ",parameters)
    return result

##############################################################################################################
##############################################################################################################

def run_qva(nqbits,nlayers,maxiter,c,g,b,parameters,method,rhobeg,ul,ur,uscale,reduced):
    
    n = 2**nqbits
    nparameters = nqbits + 2*(nqbits-1)*(nlayers-1)
    assert(nparameters == len(parameters))
    
    aux = 0
    qubitsRHS = [1]
    for i in range(1,nqbits):
        qubitsRHS.append(i+1)

    # Normalize this vector
    b = b/np.linalg.norm(b)
    #b = np.array(b,dtype=complex)
    
    # Get a circuit that takes 0 state and gives the vector
    automated_circ = QuantumCircuit(nqbits+1)
    automated_circ = obtain_circuit_from_vec(b,automated_circ,qubitsRHS)
    circ_RHS = QuantumCircuit(nqbits+1)
    circ_RHS = control_version(automated_circ,circ_RHS,aux)
    
    nit = 0
    cost_values = []
    out = minimize(cost_function, parameters, args=(nqbits, g, c, nlayers, circ_RHS, cost_values, nit), method=method, options={'maxiter':maxiter,'rhobeg':rhobeg,'disp':False}) # Works great for 3 qubit
    #print("cost values: ",cost_values)
    print(f'OUT: {out}')
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
    print(f"RESULT 3: {o}")
    
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
