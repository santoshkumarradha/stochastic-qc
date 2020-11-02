from scipy import fft, ifft
import numpy as np
hbar = 1


def gaussian(x, mu, sig, p0):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))*np.exp(1j*p0*(x-mu))


class classical_ham():
    def __init__(self, n, dt=.01, m=1, start_mu=0, start_p0=0, start_sigma=.1, ntimes=1, xmin=-1, xmax=1):
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
        # self.x = np.arange(-2**self.n, 2**self.n, 2)/2
        # rescale x ? this might probably help
        self.x = np.linspace(xmin, xmax, 2**n)
        self.dt = dt
        self.m = m
        self.start_sigma = start_sigma
        self.start_mu = start_mu
        self.P = self.get_p()
        self.ntimes = ntimes
        self.V = None
        self.result = None
        self.initial = None
        self.p0 = start_p0

    def get_p(self):
        """Momentum operator in momentum basis

        Returns:
            Unitary: Momentum operator
        """
        # H_p=p^2/2m *dt/hbar
        H_p = self.x**2 / (2*self.m)
        exp = np.exp(-1j*H_p*self.dt/hbar)
        p = np.diag(exp)
        return p

    def get_v(self, divide=True):
        """Potential operator

        Returns:
            Unitary: Potenaital operator
        """
        # seems like I need to reverse the potential, why??
        k = 1
        if divide:
            k = 2
        exp = np.exp(-1j*self.v*self.dt/(hbar*k))[::-1]
        p = np.diag(exp)
        return p

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

    def ham_once(self, x):
        v = self.V
        p2m = self.get_p()
        return v@ifft(p2m@fft(v@x))

    def get_result(self, T=1, dt=0.1):
        self.dt = dt
        self.ntimes = int(T/dt)
        x0 = gaussian(self.x, self.start_mu, self.start_sigma, self.p0)
        x0 /= np.linalg.norm(x0)
        x = x0
        for i in range(self.ntimes):
            x = self.ham_once(x)
        return np.real(x*x.conj())
