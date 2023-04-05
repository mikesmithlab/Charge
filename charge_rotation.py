
import numpy as np
import matplotlib.pyplot as plt

d_gap = 0.1
d_1 = 0.05
R=0.02
V=0
V1=50
V2=100
q=-1E-9
eps_0 = 8.854E-12

fig, axes = plt.subplots(2,1,sharex=True)

axes[0].set_xlabel('theta')
axes[0].set_ylabel('F')
axes[1].set_ylabel('M')

def calc_M(V, marker='r-'):
    theta = np.linspace(0,2*np.pi,360)
    d_2 = d_gap - d_1

    f_image_1 = -q*q/(16*np.pi*eps_0*(d_1 - R*np.cos(theta))**2)
    f_image_2 = -q*q/(16*np.pi*eps_0*(d_2 + R*np.cos(theta))**2)
    F_image = f_image_1 + f_image_2
    F_ext = q*V/d_gap
    F=F_image+F_ext
    M = F*R*np.sin(theta)
    axes[0].plot(theta, F, marker)
    axes[1].plot(theta, M, marker)

calc_M(V)
calc_M(V1,marker='b-')
calc_M(V2,marker='g-')

plt.show()