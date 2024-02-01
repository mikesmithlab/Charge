# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:31:20 2023

@author: ppzmrs
"""
import numpy as np
import matplotlib.pyplot as plt

N = 10*60*5

e0 = 8.85e-12
q = 5.e-9/N
r = 5e-3
gap = 0.5e-3
L = 3*r

np.random.seed(3)
theta = 2.0*np.pi*np.random.rand(N)
theta = theta-theta[0]
th0 = 0.0

# yc=np.zeros(N)
rxz = r

phi = np.arccos(2*np.random.rand(N)-1)
yc = r*np.cos(phi)
rxz = r*np.sin(phi)

rx = rxz*np.cos(theta)
rz = -rxz*np.sin(theta)
xc = -gap-r+rx
yc = yc
zc = rz

plt.close('all')

plt.figure(2, figsize=(7, 7))
ax = plt.axes(projection='3d')
ax.scatter(xc, yc, zc, 'bo')
ax.set_xlim(-gap-r-r, -gap-r+r)
ax.set_ylim(-r, r)
ax.set_zlim(-r, r)
plt.pause(5)

# %%

fig = plt.figure(1, figsize=(7, 7))
ax = plt.axes([0.1, 0.5, 0.8, 0.4])
ax.set_xlim(-L, L)
ax.set_ylim(-L/2, L/2)
tx = plt.axes([0.1, 0.1, 0.8, 0.3])

plt.pause(1)

while (th0 < 2*np.pi):
    theta = theta+0.05
    rx = rxz*np.cos(theta)
    rz = -rxz*np.sin(theta)
    xc = -gap-r+rx
    yc = yc
    zc = rz
    xi = -xc
    yi = yc
    zi = zc

    tau = 0.0

    for i in np.arange(N):

        dx = xi[i]-xc
        dy = yi[i]-yc
        dz = zi[i]-zc
        d2 = dx*dx+dy*dy+dz*dz
        d = np.sqrt(d2)
        nx = dx/d
        ny = dy/d
        nz = dz/d

        f = (q*q)/(4.0*np.pi*e0*d*d)
        fx = np.sum(f*nx)
        fz = np.sum(f*nz)

        tau += rz[i]*fx-rx[i]*fz

    ax.clear()
    ax.set_xlim(-L, L)
    ax.set_ylim(-L/2, L/2)
    for i in np.arange(N):
        cir0 = plt.Circle((-gap-r, 0), r, color='k', fill=False)
        ax.add_patch(cir0)
        cir2 = plt.Circle((xc[i], zc[i]), r*0.05, color='b', fill=True)
        ax.add_patch(cir2)
        cir0 = plt.Circle((gap+r, 0), r, color='k', fill=False)
        ax.add_patch(cir0)
        cir2 = plt.Circle((xi[i], zi[i]), r*0.05, color='r', fill=True)
        ax.add_patch(cir2)

    tp = th0-2.0*np.pi*np.int32(th0/(2.0*np.pi))
    tx.plot(tp, tau, '*b')
    tx.set_xlabel('Delta Theta')
    tx.set_ylabel('Tau (Nm)')

    plt.pause(0.0001)
    th0 = theta[0]
