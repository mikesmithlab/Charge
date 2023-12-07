import matplotlib.pyplot as plt
import numpy as np


def plot_theta(t=1.3E-3, W=7.7E-3, scale=1):
    theta = (np.pi / 180) * np.linspace(-30, 30, 1800)
    L = np.abs(W * np.cos(theta)) + np.abs(t * np.sin(theta))
    plt.figure()
    plt.plot(theta * 180 / np.pi, L, 'r')
    plt.show()


plot_theta()
