"""Takes the output of the patch simulation and converts into format for Ovito visualisation software."""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:31:20 2023

@author: ppzmrs
"""
import numpy as np
import matplotlib.pyplot as plt


def write_header(filename='output.dat'):
    with open('output.dat', 'a') as f:
        f.write(str(np.size(xc))+'\n')
        f.write('frame, object, x, y, z, red, green, blue\n')
    
def add_data(filename, frame, particle_type, xc, yc, zc, charge):
    #Particle_type = 0 for actual ball, 1 for charge on ball and 2 for charge on surface
    #xc, yc, zc = cartesian coords of charges.
    #Charge must be between 0 and 1
    with open('output.dat', 'a') as f:
        red = charge
        green = np.zeros(np.size(zc))
        blue = np.zeros(np.size(zc))
        for i in range(np.size(xc)):
            f.write('%d, %d, %f, %f, %f, %f, %f, %f\n' %
                (frame, particle_type, xc[i], yc[i], zc[i], red[i], green[i], blue[i]))

def scale_charge(charge, QMax, QMin):
    #Normalises values between 0 and 1 for colormapping
    charge = (charge - QMin)/(QMax - QMin)
    return charge

def dummy_data():
    N = 100
    e0 = 8.85e-12
    q = 5.e-9/N
    r = 5e-3
    gap = 0.5e-3
    L = 3*r


    theta1 = 2.0*np.pi*np.linspace(0, 1, N)
    theta1 = theta1-theta1[0]
    th0 = 0.0

    # yc=np.zeros(N)
    rxz = r

    phi1 = np.pi*np.linspace(0, 1, N)

    phi, theta = np.meshgrid(phi1, theta1)
    phi = phi.flatten()
    theta = theta.flatten()

    yc = r*np.cos(phi)
    rxz = r*np.sin(phi)

    rx = rxz*np.cos(theta)
    rz = -rxz*np.sin(theta)
    element = '0'

    xc = rx
    yc = yc
    zc = rz

    return xc, yc, zc, r

filename='output.dat'
xc, yc, zc, r = dummy_data()

for i in range(10):
    write_header(filename='output.dat')
    xc, yc, zc, r = dummy_data()
    charge = scale_charge(zc, np.max(zc), np.min(zc))
    add_data(filename, i, 1, xc, yc, zc + 0.1*np.max(zc)*i, charge)


