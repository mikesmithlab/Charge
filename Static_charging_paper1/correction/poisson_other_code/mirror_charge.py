m=5E-4
L=1
V=1000
S=0.13
eps0 = 8.854E-12
E=V/S

import numpy as np
import matplotlib.pyplot as plt

qf = np.linspace(0,5E-9, 100)
qc_qf = (1 - (2*(qf**2)*L)/(np.pi*eps0*S**3*m*9.18))**-1 

qc_qf2 = 1 + qf/(4*np.pi*eps0*S**2*E)

plt.figure()
plt.xlabel('qf')
plt.ylabel('qc/qf')
plt.plot(qf/1E-9, qc_qf, 'x')
plt.plot(qf/1E-9, qc_qf, 'r-')
plt.plot(qf/1E-9, qc_qf2, 'o')
plt.plot(qf/1E-9, qc_qf2, 'g-')
plt.show()
