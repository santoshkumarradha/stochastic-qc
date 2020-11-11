from qiskit.circuit import Parameter
from qiskit import (QuantumRegister, ClassicalRegister,
                    QuantumCircuit, Aer, execute)
from qiskit.circuit.library import (
    YGate, CYGate, CRYGate, RYGate, XGate, CXGate, CRXGate, RXGate,
    ZGate, CZGate, CRZGate, RZGate)
import numpy as np
from itertools import combinations
from qiskit.quantum_info import state_fidelity as distance
from scipy.optimize import minimize
from tqdm import tqdm as tqdm


class wick:
    def __init__(self, n, seed=0, depth=2, verbose=False):
        self.n = n
        self.verbose = verbose
        self.circuit = None
        self.main_circuit = None
        self.seed = seed
        self.depth = depth
        self.anc = QuantumRegister(1, "ancilla")
        self.basis = QuantumRegister(n, "basis")
        self.meassure = ClassicalRegister(1, "meassure")
        self.combi = None
        self.theta = None
        self.gates = None
        self.set_gates()
        self.parameter_two_qubit = 0  # As of now no two qubit gates
        self.meassure_at_end = False
        self.make_random_gate_anzats()
        self.initial = None
        self.set_initial()
        self.angles = []
        self.initial_closeness = None
        self.num_parameters = len(self.circuit.parameters)
        self.circuit_aij = [
            [0 for i in range(self.num_parameters)] for j in range(self.num_parameters)]
        self.calculate_aij()

    def set_initial(self, mu=0.5, sigma=0.01):
        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        compare = gaussian(np.linspace(0, 1, 2**self.n), mu, sigma)
        compare /= np.linalg.norm(compare)
        self.initial = compare

    def get_random_gates(self, num_gates):
        import random
        random.seed(self.seed)
        gates = [[YGate, CYGate, CRYGate, RYGate], [XGate, CXGate,
                                                    CRXGate, RXGate], [ZGate, CZGate, CRZGate, RZGate]]
        return random.choices(gates, k=num_gates)

    def set_gates(self):
        self.combi = list(combinations(range(self.n), 2))
        self.theta = [Parameter(f't-{i}')
                      for i in range(len(self.combi)+self.n*self.depth)]
        self.gates = self.get_random_gates(len(self.theta))

    def get_final_state(self, angles):
        circ_params_wo_meassure = self.main_circuit.remove_final_measurements(
            inplace=False)
        values = {i: angles[j] for j, i in enumerate(
            circ_params_wo_meassure.parameters)}
        circ_params_wo_meassure.assign_parameters(values, inplace=True)
        simulator = Aer.get_backend('statevector_simulator')
        result = execute(circ_params_wo_meassure, simulator).result()
        statevector = result.get_statevector(circ_params_wo_meassure)
        return statevector

    def get_final_state_lm(self, angles, ij):
        """Returns the value of the curcit for Aij

        Args:
            angles (float array): angles for the anzats 
            ij (1d array): [i,j] values for Aij element

        Returns:
            array,dict: State vector and probablity of ancilla being 1 or 0
        """
        circ_params_wo_meassure = self.circuit_aij[ij[0]][ij[1]].remove_final_measurements(
            inplace=False)
        values = {i: angles[j] for j, i in enumerate(
            circ_params_wo_meassure.parameters)}
        circ_params_wo_meassure.assign_parameters(values, inplace=True)
        simulator = Aer.get_backend('statevector_simulator')
        result = execute(circ_params_wo_meassure, simulator).result()
        statevector = result.get_statevector(circ_params_wo_meassure)
        temp = statevector*statevector.conj()
        p_1 = np.real(temp[int(len(temp)/2):].sum())
        results = {'1': p_1, '0': 1-p_1}
        return statevector, results

    def get_cost(self, angles, compare=None):
        a = self.get_final_state(angles)
        if compare == None:
            compare = self.initial
        # return np.arccos(distance(a,compare))
        return 2*(1-distance(a, compare))

    def get_initial_angles(self, method="SLSQP", maxiter=1000, tol=1e-7):
        angle0 = np.random.uniform(0, 2*np.pi, len(self.circuit.parameters))
        bnds = [(0, 2*np.pi)] * len(angle0)
        result = minimize(self.get_cost, angle0, method=method,
                          tol=tol, bounds=bnds, options={"maxiter": maxiter})
        if result.success == False:
            print("Warning: Angles not converged within given iterations")
        self.angles.append(result.x)
        self.initial_closeness = result.fun

    def make_random_gate_anzats(self, lm=None):

        # Initialize local variables and functions
        cnt = 0
        set_ancila = True

        def check_lm(lm, qubit):
            '''
            Need to check if the two qubit derivative calculatin is correct, I am sure it is not
            eq.33/34 in 1804.03023
            '''
            if cnt == lm[0]:
                self.circuit.x(self.anc)
                self.circuit.append(self.gates[cnt][1](), [self.anc, qubit])
                self.circuit.x(self.anc)
            if cnt == lm[1]:
                self.circuit.append(self.gates[cnt][1](), [self.anc, qubit])

        # Start constructiong the circuit
        if lm == None:
            set_ancila = 0
            lm = [-1, -1]
        if set_ancila:
            self.circuit = QuantumCircuit(self.basis, self.anc, self.meassure)
            self.circuit.h(self.anc)
        else:
            self.circuit = QuantumCircuit(self.basis)

        self.circuit.h(self.basis)
        self.circuit.barrier()

        if self.parameter_two_qubit:
            for i in self.combi:
                qubits = [self.basis[i[0]], self.basis[i[1]]]
                check_lm(lm, qubits[1])
                self.circuit.append(
                    self.gates[cnt][2](self.theta[cnt]), qubits)
                cnt += 1
            self.circuit.barrier()

        # single qubit gates
        for _ in range(self.depth):
            if self.parameter_two_qubit == 0:
                # Two qubit entangling gate series before single qubit parameterized gates
                for i in self.combi:
                    self.circuit.cx(self.basis[i[0]], self.basis[i[1]])
                self.circuit.barrier()
            for i in range(self.n):
                qubits = [self.basis[i]]
                check_lm(lm, qubits[0])
                self.circuit.append(
                    self.gates[cnt][3](self.theta[cnt]), qubits)
                cnt += 1
            self.circuit.barrier()

        # Final H on ancila
        if set_ancila:
            self.circuit.h(self.anc)
            if self.meassure_at_end:
                self.circuit.measure(self.anc, self.meassure)
        if lm == [-1, -1]:
            self.main_circuit = self.circuit.copy()
        else:
            self.circuit_aij[int(lm[0])][int(lm[1])] = self.circuit.copy()

    def calculate_aij(self):
        if self.verbose:
            iterations = tqdm(range(self.num_parameters),
                              desc="Calculating ij parameterized circuits\n")
        else:
            iterations = range(self.num_parameters)
        for i in iterations:
            for j in range(self.num_parameters):
                self.make_random_gate_anzats(lm=[i, j])
