import matplotlib.pyplot as plt
import numpy as np


def get_theta(L_px):
    d = 0.012 #Ball diameter
    d_pixels = 801-550
    theta = (np.pi/180)*np.linspace(0,179.9, 1800)
    t=1.5
    W=7.7
    L = W*np.cos(theta)+t*np.sin(theta)
    plt.figure(1)
    plt.plot(theta,L,'*')
    plt.show()
    index = np.argmin(np.abs(L_px - L))
    theta_measured = theta[index]*180/np.pi
    return theta_measured

print(get_theta(7))

