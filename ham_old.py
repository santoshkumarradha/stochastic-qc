def p(dt, n, m=1):
    # p^2/2m *dt/hbar
    k = 2**n
    pmesh = np.arange(-k, k, 2)/2
    H = pmesh**2 / (2*m)
    exp = np.exp(-1j*H*dt/hbar)
    p = np.diag(exp)
    P = u(p, label="P^2")
    return P


def v_x2(o, dt, n, m=1, get_vals=0):
    # 1/2 m \omega^2 x^2
    k = 2**n
    xmesh = np.arange(-k, k, 2)/2
    H = 0.5 * m * o**2 * xmesh**2

    exp = np.exp(-1j*H*dt/hbar)
    p = np.diag(exp)
    P = u(p, label="X^2")
    if get_vals:
        return H
    else:
        return P


def v_x_barrier(n, dt, pot=1, d=2, get_vals=0):
    # step potential
    xmesh = np.arange(-2**n, 2**n, 2)/2
    v = np.zeros(xmesh.shape)
    v[np.where((xmesh > -d) & (xmesh < d))] = pot
    exp = np.exp(-1j*v*dt/hbar)
    p = np.diag(exp)
    P = u(p, label="X^2")
    if get_vals:
        return v
    else:
        return P


def make_new_circ(n=2):
    q = QuantumRegister(n, 'q')
    circ = QuantumCircuit(q)
    return circ, q


def add_H_once(circ, o=0, m=1, dt=5e-2, pot="quad", d=None):
    n = circ.n_qubits
    q = circ.qubits

    circ.barrier()

    # QFT|p|QFT^-1
    circ.append(QFT(n), q)
    circ.append(p(dt=dt, n=n, m=m), q)
    circ.append(QFT(n, inverse=True, name="iqft"), q)

    # V(x)
    if pot == "quad":
        circ.append(v_x2(o=o, dt=dt, n=n, m=m), q)
    if pot == "step":
        circ.append(v_x_barrier(n, pot=o, d=d, dt=dt), q)


def make_H(n=2, o=0, m=1, dt=5e-2, ntimes=1, start_mu=0, start_sigma=0.1, pot="quad", d=None):
    circ, q = make_new_circ(n)
    initilize(circ, start_mu, start_sigma)
    for i in range(ntimes):
        add_H_once(circ, o=o, m=m, dt=dt, pot=pot, d=d)
    circ.measure_all()
    return circ


def initilize(circ, mu=0, sig=.1):
    n = circ.n_qubits
    q = circ.qubits
    tab = {}
    for i, j in enumerate(np.arange(-2**n, 2**n, 2)/2):
        tab[binary(i, n)] = j
    init = np.zeros(2**n)
    init[np.where(np.arange(-2**n, 2**n, 2)/2 == 0)] = 1

    x = np.arange(-2**n, 2**n, 2)/2
    init = gaussian(x, mu, sig)
    init /= np.linalg.norm(init)

    # initilize
    circ.initialize(init, q)


def simulate(circ, shots=1028):
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(circ, simulator, shots=shots).result()
    counts = result.get_counts(circ)
    return counts


def get_initial_counts(n, mu=0, sig=.1):
    x = np.arange(-2**n, 2**n, 2)/2
    init = gaussian(x, mu, sig)
    init /= np.linalg.norm(init)
    initial = {}
    for i in range(2**n):
        initial[binary(i, n)] = init[i]
    return initial
