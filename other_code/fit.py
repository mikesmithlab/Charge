import numpy as np
import matplotlib.pyplot as plt

dtheta = np.linspace(-140, 0, 10000)
a = -0.002122615
b = 159.11519968
d = 0.13

V = dtheta / (a * np.sin(np.pi/180*(b + dtheta)))

np.savetxt('dtheta_fit.txt', np.c_[dtheta, V])

plt.figure()
plt.plot(dtheta, V, 'b-')
plt.ylim([-100, 100000])
plt.show()
