# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 18:18:12 2024

@author: Mike
"""

import numpy as np
import matplotlib.pyplot as plt

qmax=6.e-11
dq=2.e-11

N=128

q=np.zeros(N)

for n in range(4000):
    ir=np.int32(N*np.random.rand(1))
    
    q[ir]=q[ir]+dq
    if q[ir] > qmax: 
        q[ir]=qmax
    
    if n%10 == 0:
        qtot=np.sum(q)      
        plt.plot(n,qtot,'b*')
        plt.pause(0.1)

plt.show()

""""       

qmax=3.e-11
dq=1.e-11

N=256

q=np.zeros(N)

for n in range(4000):
    ir=np.int32(N*np.random.rand(1))
    
    q[ir]=q[ir]+dq
    if q[ir] > qmax: 
        q[ir]=qmax
    
    if n%10 == 0:
        qtot=np.sum(q)      
        plt.plot(n,qtot,'g*')
        plt.pause(0.1)

qmax=12.e-11
dq=1.e-11

N=64

q=np.zeros(N)

for n in range(4000):
    ir=np.int32(N*np.random.rand(1))
    
    q[ir]=q[ir]+dq
    if q[ir] > qmax: 
        q[ir]=qmax
    
    if n%10 == 0:
        qtot=np.sum(q)      
        plt.plot(n,qtot,'r*')
        plt.pause(0.1)
plt.show()

"""