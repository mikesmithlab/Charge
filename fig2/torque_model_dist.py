
from re import X
import numpy as np
import matplotlib.pyplot as plt


def projection_dipole(N, r):
    # np.random.seed(3)
    theta = 2.0*np.pi*np.random.rand(N)
    phi = 2.0*np.pi*np.random.rand(N)

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(phi)

    Px = np.sum(x)/N
    Py = np.sum(y)/N
    Pz = np.sum(z)/N

    projection_dipole_length = (Px**2+Py**2)**0.5

    return projection_dipole_length


N = 20*60*5
r = 5e-3

d = []
for i in range(20000):
    d.append(projection_dipole(N, r))

d = np.array(d)

freq, bin_edges = np.histogram(d, bins=50)
bins = bin_edges[:-1] + (bin_edges[1] - bin_edges[0])/2

plt.figure()
plt.plot(bins*1000, freq)
plt.xlabel('Projection length (mm)')
plt.show()
