from scipy.fftpack import ifftshift, fftshift
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
        self.ntimes = ntimes
        self.V = None
        self.result = None
        self.initial = None
        self.p0 = start_p0
        self.V_opp = None
        self.x0 = gaussian(self.x, self.start_mu, self.start_sigma, self.p0)
        self.x0 /= np.linalg.norm(self.x0)
        self.cnt = 0

    def set_V(self, v):
        self.v = v
        self.V_opp = np.diag(np.exp(-1j * self.v * self.dt/2))
        self.p2m = np.diag(np.exp(-1j * self.x**2 / (2*self.m) * self.dt))

    def simple_v(self, t, dt=None):
        if dt is not None:
            self.dt = dt
        self.set_V(self.v)
        # def applyh(x): return self.V_opp@ifft(self.p2m@fft(self.V_opp@x))

        def applyh(x): return self.V_opp@np.fft.ifft(np.fft.fftshift((self.p2m @
                                                                      np.fft.fftshift(np.fft.fft(self.V_opp@x)))))
        x = self.x0
        t = t * 2**self.n
        self.cnt = t
        # in total does 2^n/(dt) steps between each time steps
        for _ in range(int(t/self.dt)):
            x = applyh(x)
        return np.real(x*x.conj())

    def H_BS(self, t, dt=None, sigma=.1, r=1):
        if dt is not None:
            self.dt = dt
        po_1 = -(self.x**2)*(sigma**2 / 2)
        po_2 = (self.x)*((sigma**2 / 2)-r)
        p2m = np.diag(np.exp(-1j*(po_1+po_2) * self.dt))

        v = np.diag(np.exp(-1j*np.ones_like(self.x)*r * self.dt/2))

        def applyh(x): return v@np.fft.ifft(np.fft.fftshift(p2m @
                                                            np.fft.fftshift(np.fft.fft(v@x))))
        x = self.x0
        t = t * 2**self.n
        self.cnt = t
        # in total does 2^n/(dt) steps between each time steps
        for _ in range(int(t/self.dt)):
            x = applyh(x)
        return np.real(x*x.conj())
