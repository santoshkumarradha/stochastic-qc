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
from qiskit.quantum_info import Pauli
from qiskit.extensions import UnitaryGate as ug
import itertools
import functools
import numba
from scipy.sparse import diags
from tqdm.auto import tqdm


def get_paulis(H, hide_identity=True):
    """Claculate the pauli decomposition of given matrix

    Args:
        H (2D array): Hamiltonian matrix
        hide_identity (bool, optional): Do you want to hide pure indentity term ?. Defaults to True.

    Returns:
        Dict: Coeefecients and Pauli string ex: {'II':2,'XY':1.2}
    """
    n = int(np.log2(len(H)))
    N = 2 ** n

    paulis = [Pauli.from_label('I'), Pauli.from_label(
        'X'), Pauli.from_label('Y'), Pauli.from_label('Z')]
    obs = []
    coeffs = []
    final = {}
    for term in itertools.product(paulis, repeat=n):
        matrices = [i.to_matrix() for i in term]
        coeff = np.trace(functools.reduce(np.kron, matrices) @ H) / N
        coeff = np.real_if_close(coeff).item()
        if not np.allclose(coeff, 0):
            coeffs.append(coeff)

            if not all([t == Pauli.from_label('I') for t in term]) and hide_identity:
                final["".join([i.to_label() for i in term])] = coeff
            else:
                if hide_identity == False:
                    final["".join([i.to_label() for i in term])] = coeff
    return final


def get_diff_mat(N, a=1, b=1, dx=1, boundary=1):
    D = 1  # In Black-Scholes we have:
    # D = ( 0.5 * (sigma**2) * (s**2) ),
    C = 0  # C = ( r * s )
    dt = .01
    h = 6.1875
    N -= 1
    # Define ratios for matrix algorithm
    Lambda = ((D*dt) / (h**2))

    mu = ((C*dt) / (2*h))
    # print mu, Lambda

    # define the tridiagonal matrix A
    A = np.zeros((N+1, N+1), dtype=float)
    # need to eventually implement this form: (1 + 2*Lambda + dt * r**(m+1)), since we have r = 0
    A[0, 0] = (1 + 2*Lambda)
    A[0, 1] = (- Lambda - mu)
    for n, l in enumerate(((mu - Lambda), (1 + 2*Lambda), (- Lambda - mu))):
        np.fill_diagonal(A[1:, n:], l)
    A[N, N-1] = (mu - Lambda)
    A[N, N] = (1 + 2*Lambda)
    return A
# print(A)


class wick:
    def __init__(self, n, terminal_time=10, num_time_steps=30,
                 interest_rate=0, strike_price=25, volatility=0.18,
                 stock_start=0.5, stock_end=100, seed=0, depth=2, verbose=False):

        # ** General declarations
        self.verbose = verbose

        # ** Declarations involving system
        self.n = int(n)
        self.N = 2**self.n
        self.T = terminal_time
        self.M = num_time_steps
        self.r = interest_rate
        self.sigma = volatility
        self.k = strike_price
        self.s_init = stock_start
        self.s_end = int(stock_end/self.k)*self.k
        self.t_init = 0
        self.t_end = self.T
        self.h = float(self.s_end - self.s_init) / self.N  # step-size

        # arrange grid in s, with N equally spaced nodes
        self.s = np.zeros(self.N)
        self.s = np.linspace(self.s_init, self.s_end,
                             num=self.N, endpoint=True)
        self.s[0] = self.s_init
        # time discretization
        self.dt = float(self.t_end - self.t_init) / self.M

        # arrange grid in time with step size dt
        self.t = np.arange(self.t_init, self.t_end, self.dt)
        self.t[0] = self.t_init

        # Define Diffusion and Drift coefficients/constants we can change it to better represent later on
        self.D = 1  # In Black-Scholes we have:
        # D = ( 0.5 * (sigma**2) * (s**2) ),
        self.C = 0  # C = ( r * s )

        # Define ratios for matrix algorithm
        self.Lambda = ((self.D*self.dt) / (self.h**2))
        self.mu = ((self.C*self.dt) / (2*self.h))

        # mesh storing time and space wavefunctions
        self.u = np.zeros((self.N, self.M+1))

        # ** Declarations involving calcualtion of circuit
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
        self.initial_closeness = None
        self.num_parameters = len(self.circuit.parameters)
        self.angles = np.zeros((self.M+1, self.num_parameters))
        self.circuit_aij = [
            [0 for i in range(self.num_parameters)] for j in range(self.num_parameters)]
        print("Calculating Aij circuits")
        self.calculate_aij()
        print("Done")
        self.ham_pauli = None
        self.make_pauli_ham()
        self.num_ham_terms = len(self.ham_pauli)
        # number of terms in ham * number of parameters
        self.circuit_cik = [
            [0 for i in range(len(self.ham_pauli))] for j in range(self.num_parameters)]
        print("Calculating Cik circuits")
        self.circuit_cik_a_values = [
            [0j for i in range(len(self.ham_pauli))] for j in range(self.num_parameters)]
        self.calculate_cik()
        print("Done")

    def get_diff_mat(self):
        """Calculates the differential^2 operator for given BM system

        Returns:
            2d Array: D^2 operator
        """
        N = self.N-1
        A = np.zeros((N+1, N+1), dtype=float)
        # need to eventually implement this form: (1 + 2*self.Lambda + dt * r**(m+1)), since we have r = 0
        A[0, 0] = (1 + 2*self.Lambda)
        A[0, 1] = (- self.Lambda - self.mu)
        for n, l in enumerate(((self.mu - self.Lambda), (1 + 2*self.Lambda), (- self.Lambda - self.mu))):
            np.fill_diagonal(A[1:, n:], l)
        A[N, N-1] = (self.mu - self.Lambda)
        A[N, N] = (1 + 2*self.Lambda)
        return A

    def set_initial(self):
        """Initilize the time 0 boundary condition on the option price
        """
        for j in range(0, self.N):
            # **      s[j] - k, for a Call-Option
            # ** OR:  (k - s[j]) for a PUT-Option
            if (self.s[j] - self.k) <= 0:
                self.u[j, 0] = 0
            elif (self.s[j] - self.k) > 0:
                self.u[j, 0] = self.s[j] - self.k
        self.u = np.sqrt(self.u)
        self.u = np.nan_to_num(self.u/np.linalg.norm(self.u, axis=0))

        self.initial = self.u[:, 0]

    def get_random_gates(self, num_gates):
        """Generate random gates from X,Y,Z combination according to seed

        Args:
            num_gates (int): Number of gates needed

        Returns:
            Array(gates): list of randomg gates
        """
        import random
        random.seed(self.seed)
        gates = [[YGate, CYGate, CRYGate, RYGate], [XGate, CXGate,
                                                    CRXGate, RXGate], [ZGate, CZGate, CRZGate, RZGate]]
        # ? Seems like only Y gates alone does it much better ?
        gates = [[YGate, CYGate, CRYGate, RYGate]]
        return random.choices(gates, k=num_gates)

    def set_gates(self):
        """set gates generated by get_random_gates function
        """
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

    # ! Check if final state claulation, ie Im{e^{it} <0|U|0>} is correct

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

    # ! Check if final state claulation, ie Im{e^{it} <0|U|0>} is correct

    def get_final_state_ik(self, angles, ij):
        """Returns the value of the curcit for Cik

        Args:
            angles (float array): angles for the anzats
            ij (1d array): [i,j] values for Aij element

        Returns:
            array,dict: State vector and probablity of ancilla being 1 or 0
        """
        circ_params_wo_meassure = self.circuit_cik[ij[0]][ij[1]].remove_final_measurements(
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

    def make_pauli_ham(self):
        """Make the hamiltonian part interms of pauli decomposition
        """
        # As of now only \partial^2 k included in hamiltonian
        # ham_mat = get_diff_mat(2**self.n, self.a, self.b,
        #                        self.dx, self.boundary)
        ham_mat = self.get_diff_mat()
        self.ham_pauli = get_paulis(ham_mat)

    def get_cost(self, angles, compare=None):
        a = self.get_final_state(angles)
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
        self.initial_closeness = result.fun
        self.angles[0] = result.x

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
        for iter_depth in range(self.depth):
            if self.parameter_two_qubit == 0:
                # Two qubit entangling gate series before single qubit parameterized gates
                if iter_depth > 0:
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

    def make_cik_circ(self, ik, h_pauli, coeff):

        # Initialize local variables and functions
        cnt = 0

        def check_ik(ik, qubit):
            '''
            Need to check if the two qubit derivative calculatin is correct, I am sure it is not
            eq.33/34 in 1804.03023
            '''
            if cnt == ik[0]:
                self.circuit.x(self.anc)
                self.circuit.append(self.gates[cnt][1](), [self.anc, qubit])
                self.circuit.x(self.anc)

        # Start constructiong the circuit
        self.circuit = QuantumCircuit(self.basis, self.anc, self.meassure)
        self.circuit.h(self.anc)
        # ! Check If the angle rotation is correct.
        # ! we want to have |0> + e^{ik[1]} |1>
        # ! Damn seems to be working without it. So where does the hamiltonian parts go ?
        # ! otherwise there seems to be no place in putting the Hamiltonian coeefcients !
        theta_ham = np.angle(coeff[ik[1]]*(-1j/2))
        # theta_ham = np.angle(coeff[ik[1]])
        # self.circuit.rz(theta_ham, self.anc)

        self.circuit.h(self.basis)
        self.circuit.barrier()

        if self.parameter_two_qubit:
            for i in self.combi:
                qubits = [self.basis[i[0]], self.basis[i[1]]]
                check_ik(ik, qubits[1])
                self.circuit.append(
                    self.gates[cnt][2](self.theta[cnt]), qubits)
                cnt += 1
            self.circuit.barrier()

        # single qubit gates
        for iter_depth in range(self.depth):
            if self.parameter_two_qubit == 0:
                # Two qubit entangling gate series before single qubit parameterized gates
                if iter_depth > 0:
                    for i in self.combi:
                        self.circuit.cx(self.basis[i[0]], self.basis[i[1]])
                self.circuit.barrier()
            for i in range(self.n):
                qubits = [self.basis[i]]
                check_ik(ik, qubits[0])
                self.circuit.append(
                    self.gates[cnt][3](self.theta[cnt]), qubits)
                cnt += 1
            self.circuit.barrier()
            qs = [self.anc]
            for i in self.basis:
                qs.append(i)
        self.circuit.append(h_pauli[ik[1]], qs)
        self.circuit.h(self.anc)
        if self.meassure_at_end:
            self.circuit.measure(self.anc, self.meassure)
        self.circuit_cik[int(ik[0])][int(ik[1])] = self.circuit.copy()

    def calculate_aij(self):
        if self.verbose:
            iterations = tqdm(range(self.num_parameters),
                              desc="Calculating A_ij parameterized circuits\n")
        else:
            iterations = range(self.num_parameters)
        for i in iterations:
            for j in range(self.num_parameters):
                self.make_random_gate_anzats(lm=[i, j])

    def calculate_cik(self):
        # ! ----------------------------------
        # ! Check if this needs to be refersed ?
        # ! ----------------------------------

        ham_terms_circ = [ug(Pauli.from_label(i).to_operator(), label=" ("+i+")").control(
            1) for i in list(self.ham_pauli.keys())]
        coeff = list(self.ham_pauli.values())
        if self.verbose:

            iterations = tqdm(range(self.num_parameters),
                              desc="Calculating C_ij parameterized circuits\n")
        else:
            iterations = range(self.num_parameters)

        for i in iterations:
            for j in range(len(ham_terms_circ)):
                self.make_cik_circ(
                    ik=[i, j], h_pauli=ham_terms_circ, coeff=coeff)
                self.circuit_cik_a_values[i][j] = np.abs(-1j/2 * coeff[j])

    def evolve_system(self, verbose=True):
        """Evolve the system with time gap dt up to steps N

        Args:
            verbose (bool, optional): Want to show progreess bar ?. Defaults to True.
        """

        for ntime in tqdm(range(self.M)):
            angles = self.angles[ntime]
            if verbose:
                iter_range = tqdm(range(self.num_parameters))
            else:
                iter_range = range(self.num_parameters)
            A = np.zeros((self.num_parameters, self.num_parameters))
            for i in iter_range:
                for j in range(self.num_parameters):
                    # if j <= i:
                    state, p = self.get_final_state_lm(angles, [j, i])
                    A[i, j] = (p['0']-p['1']) * \
                        np.abs(-1j/2 * 1j/2)  # 2*p['1']-1
                    # A[j, i] = p['0']-p['1']

            C = np.zeros((self.num_parameters, self.num_ham_terms))
            for i in iter_range:
                for j in range(self.num_ham_terms):
                    state, p = self.get_final_state_ik(angles, [i, j])
                    # 2*p['1']-1  # p['0']-p['1']  # 2*p['0']-1
                    C[i, j] = (p['0']-p['1']) * \
                        np.abs(self.circuit_cik_a_values[i][j] * -1j/2)
            try:
                theta_dot = np.linalg.solve(A, C.sum(axis=1))
            except:
                print("diag did not work, going with lstq")
                theta_dot, residuals, rank, s = np.linalg.lstsq(
                    A, C.sum(axis=1))
            self.angles[ntime+1] = (angles+self.dt*theta_dot)
            state = self.get_final_state(self.angles[ntime+1])
            self.u[:, ntime+1] = state
