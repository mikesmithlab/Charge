import numpy as np
import pandas as pd

from utils import load
from poisson_solver import *
from visuals import *

"""
-------------------------------------------------------
Display the results
---------------------------------------------------------
"""
filename='bead_plate_dz0_01.txt'

E,phi,state=load(filename)

lh_plate = state['lh_plate']
rh_plate = state['rh_plate']
ball = state['ball']
sim = state['sim']

ax=plot_2Dpotential(phi, sim, zorder=1)
#ax=plot_plate(ax, sim, lh_plate, zorder=2)
#ax=plot_plate(ax, sim, rh_plate, zorder=2)
#ax=plot_ball(ax, ball, zorder=2)
plot_grid(ax, sim)

plot_rad_potential(phi, ball, sim)

plates=lh_plate['show']
bead=ball['show']

plot_field(phi, E, sim, lh_plate, rh_plate, ball, plates=plates, bead=bead)
#gauss_surf(ax, gaussian_surface)
plt.show()