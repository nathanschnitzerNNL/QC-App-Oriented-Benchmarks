import sys
sys.path[1:1] = ["_common", "_common/qiskit"]
sys.path[1:1] = ["../../_common", "../../_common/qiskit"]
import time
import math
import os
import execute as ex
import metrics as metrics
from collections import defaultdict
import numpy as np
import json
np.random.seed(0)
import cProfile

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_nature.drivers import PySCFDriver, UnitsType, Molecule
from qiskit_nature.circuit.library import HartreeFock as HF
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.transformers import ActiveSpaceTransformer
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit.opflow import PauliTrotterEvolution, CircuitStateFn, Suzuki
from qiskit.opflow.primitive_ops import PauliSumOp
from qiskit import Aer, execute

# Function that converts a list of single and double excitation operators to Pauli Operators
def readPauliExcitation(norb, circuit_id=0):
    # load precomputed data
    filename = os.path.join(os.getcwd(), f'../qiskit/ansatzes/{norb}_qubit_{circuit_id}.txt')
    with open(filename) as f:
        data = f.read()
    ansatz_dict = json.loads(data)
    
    # initialize vars
    pauli_list = []
    cur_coeff = 1e5
    cur_list = []

    # Loop over excitations
    for ext in ansatz_dict:
        if cur_coeff > 1e4:
            cur_coeff = ansatz_dict[ext]
            cur_list = [(ext, ansatz_dict[ext])]
        elif abs(abs(ansatz_dict[ext]) - abs(cur_coeff)) > 1e-4:
            pauli_list.append(PauliSumOp.from_list(cur_list))
            cur_coeff = ansatz_dict[ext]
            cur_list = [(ext, ansatz_dict[ext])]
        else:
            cur_list.append((ext, ansatz_dict[ext]))

    pauli_list.append(PauliSumOp.from_list(cur_list))

    return pauli_list

# Get the Hamiltonian by reading in pre-computed file
def ReadHamiltonian(nqubit):

    # load pre-computed data
    filename = os.path.join(os.getcwd(), f'../qiskit/Hamiltonians/{nqubit}_qubit.txt')
    with open(filename) as f:
        data = f.read()
    ham_dict = json.loads(data)

    # pauli list
    pauli_list = []
    for p in ham_dict:
        pauli_list.append( (p, ham_dict[p]) )

    # build Hamiltonian
    ham = PauliSumOp.from_list(pauli_list)

    # return Hamiltonian
    return ham

def VQEEnergy(n_spin_orbs, na, nb, circuit_id=0, method=1):

    '''
    Construct a Qiskit circuit for VQE Energy evaluation with UCCSD ansatz
    :param n_spin_orbs:The number of spin orbitals
    :return: return a Qiskit circuit for this VQE ansatz
    '''

    # allocate qubits
    num_qubits = n_spin_orbs
    qr = QuantumRegister(num_qubits); cr = ClassicalRegister(num_qubits); qc = QuantumCircuit(qr, cr, name="main")

    # number of alpha spin orbitals
    norb_a = int(n_spin_orbs / 2)

    # number of beta  spin orbitals
    norb_b = norb_a

    # construct the Hamiltonian
    qubit_op = ReadHamiltonian(n_spin_orbs)

    # initialize the HF state
    qc = HartreeFock(n_spin_orbs, na, nb)

    # form the list of single and double excitations
    singles = []
    doubles = []
    for occ_a in range(na):
        for vir_a in range(na, norb_a):
            singles.append((occ_a, vir_a))


    for occ_b in range(norb_a, norb_a+nb):
        for vir_b in range(norb_a+nb, n_spin_orbs):
            singles.append((occ_b, vir_b))

    for occ_a in range(na):
        for vir_a in range(na, norb_a):
            for occ_b in range(norb_a, norb_a+nb):
                for vir_b in range(norb_a+nb, n_spin_orbs):
                    doubles.append((occ_a, vir_a, occ_b, vir_b))

    # get cluster operators in Paulis
    pauli_list = readPauliExcitation(n_spin_orbs, circuit_id)

    # loop over the Pauli operators
    for index, PauliOp in enumerate(pauli_list):

        # get circuit for exp(-iP)
        cluster_qc = ClusterOperatorCircuit(PauliOp)

        # add to ansatz
        qc.compose(cluster_qc, inplace=True)

    # method 2, only compute the last term in the Hamiltonian
    if method == 2:
        # last term in Hamiltonian
        qc_with_mea, is_diag = ExpectationCircuit(qc, qubit_op[1], num_qubits)

        # return the circuit
        return qc_with_mea

    # now we need to add the measurement parts to the circuit
    # circuit list
    qc_list = []
    diag = []
    off_diag = []
    for p in qubit_op:

        # get the circuit with expectation measurements
        qc_with_mea, is_diag = ExpectationCircuit(qc, p, num_qubits)

        # add to circuit list
        qc_list.append(qc_with_mea)

        # diagonal term
        if is_diag:
            diag.append(p)
        # off-diagonal term
        else:
            off_diag.append(p)

    return qc_list

def ClusterOperatorCircuit(pauli_op):
    
    # compute exp(-iP)
    exp_ip = pauli_op.exp_i()

    # Trotter approximation
    qc_op = PauliTrotterEvolution(trotter_mode=Suzuki(order=1, reps=1)).convert(exp_ip)

    # convert to circuit
    qc = qc_op.to_circuit()

    # return this circuit
    return qc

# Function that adds expectation measurements to the raw circuits
def ExpectationCircuit(qc, pauli, nqubit, method=1):

    # a flag that tells whether we need to perform rotation
    need_rotate = False

    # copy the unrotated circuit
    raw_qc = qc.copy()

    # whether this term is diagonal
    is_diag = True

    # primitive Pauli string
    PauliString = pauli.primitive.to_list()[0][0]

    # coefficient
    coeff = pauli.coeffs[0]

    # basis rotation
    for i, p in enumerate(PauliString):

        target_qubit = nqubit - i - 1
        if (p == "X"):
            need_rotate = True
            is_diag = False
            raw_qc.h(target_qubit)
        elif (p == "Y"):
            raw_qc.sdg(target_qubit)
            raw_qc.h(target_qubit)
            need_rotate = True
            is_diag = False

    # perform measurements
    raw_qc.measure_all()

    # name of this circuit
    raw_qc.name = PauliString + " " + str(np.real(coeff))

    return raw_qc, is_diag


# Function that implements the Hartree-Fock state
def HartreeFock(norb, na, nb):

    # initialize the quantum circuit
    qc = QuantumCircuit(norb)

    # alpha electrons
    for ia in range(na):
        qc.x(ia)

    # beta electrons
    for ib in range(nb):
        qc.x(ib+int(norb/2))

    # return the circuit
    return qc

backend = Aer.get_backend("qasm_simulator")

precalculated_data = {}

def run(min_qubits=4, max_qubits=8, max_circuits=3, num_shots=4092 * 2**8, method=1):

    print(f"... using circuit method {method}")

    # validate parameters (smallest circuit is 4 qubits)
    max_qubits = max(4, max_qubits)
    min_qubits = min(max(4, min_qubits), max_qubits)
    if min_qubits % 2 == 1: min_qubits += 1  # min_qubits must be even
    if method == 1: max_circuits = 1

    # Execute Benchmark Program N times for multiple circuit sizes
    # Accumulate metrics asynchronously as circuits complete
    for input_size in range(min_qubits, max_qubits + 1, 2):

        # determine the number of circuits to execute fo this group
        num_circuits = max_circuits

        num_qubits = input_size

        # decides number of electrons
        na = int(num_qubits/4)
        nb = int(num_qubits/4)

        # decides number of unoccupied orbitals
        nvira = int(num_qubits/2) - na
        nvirb = int(num_qubits/2) - nb

        # determine the size of t1 and t2 amplitudes
        t1_size = na * nvira + nb * nvirb
        t2_size = na * nb * nvira * nvirb

        # random seed
        np.random.seed(0)

        # create the circuit for given qubit size and simulation parameters, store time metric
        ts = time.time()

        # circuit list 
        qc_list = []

        # method 1 (default)
        if method == 1:
            # sample t1 and t2 amplitude
            t1 = np.random.normal(size=t1_size)
            t2 = np.random.normal(size=t2_size)

            # construct all circuits
            qc_list = VQEEnergy(num_qubits, na, nb, 0, method)
        else:
            # loop over circuits
            for circuit_id in range(num_circuits):
                # sample t1 and t2 amplitude
                t1 = np.random.normal(size=t1_size)
                t2 = np.random.normal(size=t2_size)

                # construct circuit 
                qc_single = VQEEnergy(num_qubits, na, nb, circuit_id, method)               
                qc_single.name = qc_single.name + " " + str(circuit_id) 

                # add to list 
                qc_list.append(qc_single)
            

        print(f"************\nExecuting VQE with num_qubits {num_qubits}")

        for qc in qc_list:

            # get circuit id
            if method == 1:
                circuit_id = qc.name.split()[0]
            else:
                circuit_id = qc.name.split()[2]

            # collapse the sub-circuits used in this benchmark (for qiskit)
            qc2 = qc.decompose()

            # submit circuit for execution on target (simulator, cloud simulator, or hardware)
            job = execute(qc, backend, shots=num_shots)
            
            # executation result
            result = job.result()
            
            # get measurement counts
            counts = result.get_counts(qc)

            # initialize empty dictionary
            dist = {}
            for key in counts.keys():
                prob = counts[key] / num_shots
                dist[key] = prob

            # add dist values to precalculated data for use in fidelity calculation
            precalculated_data[f"{circuit_id}"] = dist
            
        with open(f'precalculated_data_qubit_{num_qubits}_method1.json', 'w') as f:
            f.write(json.dumps(
                precalculated_data,
                sort_keys=True,
                indent=4,
                separators=(',', ': ')
                ))


if __name__ == '__main__':
    cProfile.run('run()')
    #run()

