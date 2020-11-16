from wick import wick as wick_class
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm as tqdm


from wick import wick as wick_class

n = 4
depth = 2
wick = wick_class(n, depth=depth)
print("Optimizing the initial angles")
wick.get_initial_angles(maxiter=2000, method="COBYLA")
print("Initial closeness : ", wick.initial_closeness)
wick.evolve_system(dt=.2, t=10)

states = []
for j, i in enumerate(wick.angles):
    states.append(wick.get_final_state(i))

np.save("states.npy", states)
