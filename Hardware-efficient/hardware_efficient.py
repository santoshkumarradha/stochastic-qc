from qiskit.circuit import Parameter
from qiskit import (QuantumRegister, ClassicalRegister,
                    QuantumCircuit, Aer, execute)
from qiskit.circuit.library import (
    YGate, CYGate, CRYGate, RYGate, XGate, CXGate, CRXGate, RXGate,
    ZGate, CZGate, CRZGate, RZGate)

from qiskit.aqua.operators import PauliExpectation, CircuitSampler, StateFn, CircuitStateFn, AerPauliExpectation, ExpectationFactory
from qiskit.aqua.operators import X, Y, Z, I

from qiskit.quantum_info import state_fidelity as distance
from qiskit.quantum_info import Pauli

from qiskit.aqua.operators.primitive_ops import MatrixOp
from qiskit.aqua.operators.primitive_ops import PauliOp

from scipy.optimize import minimize
import functools
import numba
import numpy as np
from scipy.sparse import diags


def get_u(p1, p2, label=None):
    q = QuantumRegister(2)
    anc = QuantumRegister(1)
    c = QuantumCircuit(q, anc)
    c.cry(p2, anc, q[0])
    c.ry(p1, q[0])
    c.cnot(q[0], q[1])
    return c.to_gate(label=label)


def get_two_qubit_pqc(n, depth=1, label="2-qubit-PQC"):
    two_qbit_pairs_1 = [list(i) for i in [np.arange(n)[i:i + 2]
                                          for i in range(0, len(np.arange(n)), 2)] if len(i) == 2]
    two_qbit_pairs_2 = [list(i) for i in [np.arange(n)[i:i + 2]
                                          for i in range(1, len(np.arange(n)), 2)] if len(i) == 2]
    q = QuantumRegister(n, 'q')
    anc = QuantumRegister(1, "anc")
    c = QuantumCircuit(anc, q)
    for i in two_qbit_pairs_1:
        p1 = Parameter("t_{}{}_{}".format(i[0], i[1], depth))
        p2 = Parameter("opt_{}{}_{}".format(i[0], i[1], depth))
        qubits = [q[i[0]], q[i[1]]]``
        qubits.append(anc)
        c.append(get_u(p1, p2, label="U"), qubits)

    for i in two_qbit_pairs_2:
        p1 = Parameter("t_{}{}_{}".format(i[0], i[1], depth))
        p2 = Parameter("opt_{}{}_{}".format(i[0], i[1], depth))
        qubits = [q[i[0]], q[i[1]]]
        qubits.append(anc)
        c.append(get_u(p1, p2, label="U"), qubits)
    return c.to_gate(label=label)


def get_u_withoutmeassure(p1, label=None):
    q = QuantumRegister(2)
    c = QuantumCircuit(q)
    c.ry(p1, q[0])
    c.cnot(q[0], q[1])
    return c.to_gate(label=label)


def get_two_qubit_pqc_withoutmeassure(n, depth=1, label="2-qubit-PQC"):
    two_qbit_pairs_1 = [list(i) for i in [np.arange(n)[i:i + 2]
                                          for i in range(0, len(np.arange(n)), 2)] if len(i) == 2]
    two_qbit_pairs_2 = [list(i) for i in [np.arange(n)[i:i + 2]
                                          for i in range(1, len(np.arange(n)), 2)] if len(i) == 2]
    q = QuantumRegister(n, 'q')
    c = QuantumCircuit(q)
    for i in two_qbit_pairs_1:
        p1 = Parameter("t_{}{}_{}".format(i[0], i[1], depth))
        qubits = [q[i[0]], q[i[1]]]
        c.append(get_u_withoutmeassure(p1, label="U"), qubits)

    for i in two_qbit_pairs_2:
        p1 = Parameter("t_{}{}_{}".format(i[0], i[1], depth))
        qubits = [q[i[0]], q[i[1]]]
        c.append(get_u_withoutmeassure(p1, label="U"), qubits)
    return c.to_gate(label=label)


def get_circuit(n=4, depth=2, Imaginary=True, without_measure=False):
    if without_measure == False:
        q = QuantumRegister(n, 'q')
        anc = QuantumRegister(1, "anc")
        c = QuantumCircuit(anc, q)
        c.h(anc)
        if Imaginary:
            c.s(anc)
        c.barrier()
        for i in range(depth):
            pqc = get_two_qubit_pqc(n, depth=i, label=f"2-qubit-PQC-{i+1}")
            c.append(pqc, range(n+1))
            c.barrier()
        c.h(anc)
        return c
    else:
        q = QuantumRegister(n, 'q')
        c = QuantumCircuit(q)
        c.barrier()
        for i in range(depth):
            pqc = get_two_qubit_pqc_withoutmeassure(
                n, depth=i, label=f"2-qubit-PQC-{i+1}")
            c.append(pqc, range(n))
            c.barrier()
        return c


def get_exp(H, circ):
    psi = CircuitStateFn(circ)
    backend = Aer.get_backend('statevector_simulator')
    measurable_expression = StateFn(H, is_measurement=True).compose(psi)
    expectation = AerPauliExpectation().convert(measurable_expression)
    sampler = CircuitSampler(backend).convert(expectation)
    return sampler.eval().real


def get_random_param_dict(circ, seed=4):
    values = {}
    np.random.seed(seed)
    for i in sorted(circ.parameters, key=lambda x: x.name):
        values[i] = np.random.uniform(0, np.pi)
    return values


def set_angle(circ, angle):
    values = {}
    for j, i in enumerate(sorted(circ.parameters, key=lambda x: x.name)):
        values[i] = angle[j]
    return circ.assign_parameters(values).copy()


def get_exp_angle(H, circ, angle):
    calc_circ = set_angle(circ, angle)
    return get_exp(H, calc_circ)


def get_diff_mat(N, a=1, b=1, dx=10, boundary=1):
    '''
    Returns the differential oppertor matrix for equation given by a(d/dx)+b(d/dx)^2 for log(N) qubits discritized by dx
    '''
    D = a*diags([1, -2, 1], [-1, 0, 1], shape=(N, N))*dx
    D = D.toarray()
    D[0][0] = D[-1][-1] = -boundary
    return D


def set_angle(circ, angle):
    values = {}
    for j, i in enumerate(sorted(circ.parameters, key=lambda x: x.name)):
        values[i] = angle[j]
    return circ.assign_parameters(values).copy()


def get_exp_angle(H, circ, angle):
    calc_circ = set_angle(circ, angle)
    return get_exp(H, calc_circ)
