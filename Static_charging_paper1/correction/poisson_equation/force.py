import numpy as np
import pandas as pd

from utils import load


filename='bead_plate_dz0_01.txt'

E,phi,state=load(filename)
Ez = E[2]
ball=state['ball']

Q=ball['Q']
charge_coords = ball['charge_coords']
N_charges = len(charge_coords)
#dQ=1.0/N_charges
dQ = Q/N_charges
print(dQ)
print(N_charges)


force = 0.0
for charge in charge_coords:
    print('charge', charge, Ez[charge[0], charge[1], charge[2]])
    force +=dQ*(Ez[charge[0], charge[1], charge[2]])

print(f"Force = {force}")

V=2000
D=0.13
dz=0.02
print(f"simple Force {Q*V/D}")

img_force = (Q**2/(4*np.pi*8.854E-12))*((1/(D-2*dz)**2) - (1/(D+2*dz)**2))

print(f"Image force = {img_force}")


