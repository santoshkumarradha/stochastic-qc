import numpy as np
import itertools
from scipy import optimize as opt
from pauli import pauli_action
from binary_functions import Bas2Int, Int2Bas, Opp2Str
from numpy import linalg as LA
from scipy import linalg as SciLA
from tools import print_state, fidelity, dgr, dpbc, dobc
from pauli import sigma_matrices


def Hpsi(H_, psi_):
    phi = np.zeros(psi_.shape, dtype=complex)

    for (A, h, imp, gmp) in H_:
        for m in np.where(np.abs(h) > 1e-8)[0]:
            phi += h[m]*gmp[m, imp[m, :]]*psi_[imp[m, :]]

    return phi.copy()


def Hmat(H_):
    N = H_[0][2].shape[1]
    Hm = np.zeros((N, N), dtype=complex)
    for i in range(N):
        ei = np.zeros(N, dtype=complex)
        ei[i] = 1.0
        Hm[:, i] = Hpsi(H_, ei).copy()
    return Hm


def Hmoms(H_, psi_):
    phi_ = Hpsi(H_, psi_)
    ea = np.vdot(psi_, phi_)
    ev = np.vdot(phi_, phi_)
    return np.real(ea), np.real(ev-ea**2)


def print_Hamiltonian(H_):
    mu = 0

    for (A, h, imp, gmp) in H_:
        #print('A: ',A)
        nact = len(A)
        print("active qubits ", A)
        print("operators: ")
        for m in np.where(np.abs(h) > 1e-8)[0]:
            print(Opp2Str(Int2Bas(m, 4, nact)), h[m])
        mu += 1


def Hii(H_, i):
    N = H_[0][2].shape[1]
    nbit = int(np.log2(N))
    hii = 0.0
    xi = Int2Bas(i, 2, nbit)
    for (A, h, imp, gmp) in H_:
        nact = len(A)
        for m in np.where(np.abs(h) > 1e-8)[0]:
            sm = Int2Bas(m, 4, nact)
            smx = [sigma_matrices[xi[A[w]], xi[A[w]], sm[w]]
                   for w in range(nact)]
            hii += np.real(h[m]*np.prod(smx))
    return hii


def TransverseIsing(nspin, R, J, h):
    H = []

    for i in range(nspin):
        j = (i+1) % nspin
        active = [k for k in range(nspin) if dpbc(
            i, k, nspin) < R or dpbc(j, k, nspin) < R]
        active = np.asarray(active)
        print(active)
        nact = len(active)
        h_alpha = np.zeros(4**nact)
        ii = np.where(active == i)[0][0]
        jj = np.where(active == j)[0][0]

        idx = [0]*nact
        idx[ii] = 1
        h_alpha[Bas2Int(idx, 4)] = h
        idx = [0]*nact
        idx[ii] = 3
        idx[jj] = 3
        if np.abs(i-j) == 1 and j != 0:
            h_alpha[Bas2Int(idx, 4)] = J
        imap, gmap = pauli_action(active, nspin)
        H.append((active, h_alpha, imap, gmap))
    Hm = Hmat(H)
    print()
    print_Hamiltonian(H)
    return H
