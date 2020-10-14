from logging import raiseExceptions
from qiskit.extensions import UnitaryGate as u
from qiskit import *
import numpy as np
from qiskit.circuit.library.basis_change import QFT
hbar = 1  # do we set it to 1?


def binary(x, n): return ''.join(
    reversed([str((x >> i) & 1) for i in range(n)]))


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class ham():
    def __init__(self, n, dt=.01, m=1, start_mu=0, start_sigma=.1, ntimes=1):
        """Class to simulate a 1D hamiltonian

        Args:
            n (int): Number of qubits (discritization of x axis into 2^n)
            dt (float, optional): time step. Defaults to .01
            m (float, optional): Mass of the particle. Defaults to 1.
            start_mu (float, optional): Starting gaussian wave function mu. Defaults to 0.
            start_sigma (float, optional): Starting Gaussian wave function varience. Defaults to .1.
            ntimes (int, optional): Number of time steps to simulate the hamiltonian evolution. Defaults to 1.
        """
        self.n = n
        self.x = np.arange(-2**self.n, 2**self.n, 2)/2
        self.q = QuantumRegister(n, 'q')
        self.circ = QuantumCircuit(self.q)
        self.circuit = self.circ
        self.dt = dt
        self.m = m
        self.start_sigma = start_sigma
        self.start_mu = start_mu
        self.P = self.get_p()
        self.ntimes = ntimes
        self.V = None
        self.result = None
        self.initial = None

    def add_H_once(self):
        """Adds the quantum gate
        1. QFT to go to momentum space
        2. add p, which is diognal in momentum sapce
        3. Inverse QFT back to real space
        4. add V(x) which is diognal in real space
        """
        self.circ.barrier()
        # QFT|p|QFT^-1
        self.circ.append(QFT(self.n), self.q)
        self.circ.append(self.P, self.q)
        self.circ.append(QFT(self.n, inverse=True, name="iqft"), self.q)

        # V(x)
        self.circ.append(self.V, self.q)

    def get_p(self):
        """Momentum operator in momentum basis

        Returns:
            Unitary: Momentum operator
        """
        # H_p=p^2/2m *dt/hbar
        H_p = self.x**2 / (2*self.m)
        exp = np.exp(-1j*H_p*self.dt/hbar)
        p = np.diag(exp)
        return u(p, label="P^2/2m")

    def get_v(self):
        """Potential operator

        Returns:
            Unitary: Potenaital operator
        """
        # seems like I need to reverse the potential, why??
        exp = np.exp(-1j*self.v*self.dt/hbar)[::-1]
        p = np.diag(exp)
        return u(p, label="V(x)")

    def make_ham(self):
        """Make the hamiltonian n times after initialization
        """
        if self.V is None:
            raiseExceptions("Please set V with ham.set_V()")
        self.circ = QuantumCircuit(self.q)
        self.initilize(self.start_mu, self.start_sigma)
        # self.initilize(self.start_mu, self.start_sigma)
        for i in range(self.ntimes):
            self.add_H_once()
        self.circ.measure_all()

    def initilize(self, mu, sig):
        """Initialize the hamiltonian with a gaussian state

        Args:
            mu (float): center of gaussian
            sig (flaot): varience of gaussian
        """
        init = gaussian(self.x, mu, sig)
        init /= np.linalg.norm(init)
        tab = {}
        for i in range(len(self.x)):
            tab[binary(i, self.n)] = init[i]
        self.initial = tab
        # initilize
        self.circ.initialize(init, self.q)

    def set_V(self, v):
        """Set the potential, use self.x to get the discritized real space
        example - for quadratic potential, 
        set_V(a*self.x**2)

        Args:
            v (array): Array of dimension 2^n
        """
        if len(v) != len(self.x):
            raiseExceptions(
                "length of v array must be eqaul to x mesh -> (2^n) ")
        self.v = v
        self.V = self.get_v()

    def simulate(self, shots=1028):
        """Simulate the system 

        Args:
            shots (int, optional): Number of simulation shots. Defaults to 1028.
        """
        simulator = Aer.get_backend('qasm_simulator')
        result = execute(self.circ, simulator, shots=shots).result()
        self.result = result.get_counts(self.circ)

    def plot_potential(self, ax, s=10, color="k"):
        """Plot the potential function

        Args:
            ax (Axis): Maptlotlib axis
            s (int, optional): Size of the scatter plot. Defaults to 10.
            color (str, optional): Color of the plot. Defaults to "k".
        """
        ax.set_xticklabels(self.x, fontsize=10)
        ax_potential = ax.twinx()
        ax_potential.scatter(range(len(self.v)), self.v,
                             s=s, c=color, label="V(x)")
        ax_potential.grid("off")
