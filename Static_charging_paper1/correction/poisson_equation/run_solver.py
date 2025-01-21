import numpy as np
import pandas as pd


from poisson_solver import *
from visuals import *
from calc_charge import *
from utils import save, write_dict

"""---------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------
SETUP
-----------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------"""
#output
filename = 'bead_plate_dz0_01.txt'

#Define grid and plate physical dimensions
L_box = 0.25
N = 101#num grid points per dimension
BC='dirichlet'

dx=L_box/(N-1) # Grid point spacing, assume symmetric.
dR = np.sqrt(3)*dx # dR is the shell thickness for finding pixels that are at the surface to add charge to.

#plates
L_plate = 0.1
D_plate = 0.13
V_plate = 0
t_plate = 0.002

#Sphere
R_sphere = 0.005
Q = 3E-9
dz=0.02 # dz needs to be a whole number of grid points. Enter a value and it will change it to the nearest value

dz = dx*int(dz/dx)

#Turn features on and off
plates = True
bead = True

#How many iterations to perform of minimisation
iters = 3000

#Constants
eps0 = 8.854E-12


"""---------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------
DEFINE PLATES AND PARTICLE
-----------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------"""
lh_plate = {'show':plates,'Ltop': L_plate/2,'Lbottom':-L_plate/2, 'Z': -D_plate/2 - dz, 't':t_plate, 'V': +V_plate/2}
rh_plate = {'show':plates,'Ltop': L_plate/2, 'Lbottom':-L_plate/2, 'Z': +D_plate/2 -dz, 't':t_plate,  'V': -V_plate/2}
ball = {'show':bead,'R':R_sphere, 'Q':Q, 'dR': dR}
sim = {'Lbox':L_box,'N':N, 'dx':dx, 'iters':iters, 'plates':True, 'ball':True, 'eps0':8.854E-12}

state={'lh_plate':lh_plate, 'rh_plate':rh_plate, 'ball':ball, 'sim':sim}

if plates:
    mask_pos = define_plate_mask(lh_plate, sim)
    mask_neg = define_plate_mask(rh_plate, sim)
else:
    mask_pos=None
    mask_neg=None


if bead:
    rho, charge_coords=charge_ball(ball, sim)
    ball['charge_coords'] = charge_coords
else:
    rho = np.zeros((sim['N'],sim['N'],sim['N']))

"""
-------------------------------------------------------
Solve the equations
---------------------------------------------------------
"""


grid, err =solve_poisson(rho, sim, lh_plate, rh_plate, mask_pos, mask_neg, edge=BC, plates=plates)
E = calc_Efield(grid, sim)

save(filename[:-4] + '_potential.txt', grid)
save(filename[:-4] + '_field.txt', np.array([E[0],E[1],E[2]]))

state = {'sim':sim,'lh_plate': lh_plate, 'rh_plate':rh_plate, 'ball':ball}

write_dict(state, filename[:-4] + "_state.txt")

