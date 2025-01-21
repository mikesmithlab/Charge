
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, generate_binary_structure



def abscoord_to_grididx(displacement, sim):
    #converts absolute coord assuming (0,0) at centre of box to a phi grid index
    value = int(0.5+(sim['Lbox']/2 + displacement)/sim['dx'])
    return value

"""
-----------------------------------------------
Setting Boundary Conditions
------------------------------------------------
"""


def neumann_boundary(grid):
    """This sets the boundary pixels and neighbour pixels to the same value. ie derivative = 0"""
    grid[0,:,:] = grid[1,:,:] 
    grid[-1,:,:] = grid[-2,:,:]
    grid[:,0,:] = grid[:,1,:] 
    grid[:,-1,:] = grid[:,-2,:]
    grid[:,:,0] = grid[:,:,1] 
    grid[:,:,-1] = grid[:,:,-2]
    return grid

def dirichlet_boundary(grid, V):
    """This sets the simulation boundary pixels to the potential V"""
    grid[0,:,:] = V
    grid[:,0,:] = V
    grid[:,:,0] = V
    grid[-1,:,:] = V
    grid[:,-1,:] = V
    grid[:,:,-1] = V
    return grid

def dirichlet(phi, V, mask):
    """This sets the value of the potential to V at the points labelled True in the mask"""
    phi[mask] = V
    return phi


def solve_poisson(rho, sim, lh_plate, rh_plate, mask_pos, mask_neg, edge='dirichlet', plates=True):
    """Solves the poisson equation numerically for a 3D grid
    args:
    rho - a 3D numpy grid of the spatially varying charge density
    sim - a dictionary of key simulation parameters
    lh_plate / rh_plate - params associated with the left/right hand capacitor
    mask_pos / mask_neg - 3D numpy array which defines where the positive / negative capacitor plate is, used for fixing the potential at certain locations
    
    kwargs:
    edge - defines the boundary condition at the edge of simulation box. Can be neumann (derivative = 0) or dirichlet (value = 0)
    plates - defines whether to include the plates in the simulation

    
    """
    #Numerically solve
    err = []
    #Create a kernel to do the averaging of neighbours
    kern = generate_binary_structure(3,1).astype(float)/6
    kern[1,1,1] = 0
    phi = np.zeros((sim['N'],sim['N'],sim['N']))

    #iterate towards solution
    for _ in tqdm(range(sim['iters'])):
        phi_updated = convolve(phi, kern, mode='constant')
        phi_updated += sim['dx']**2 * rho / (6 * sim['eps0'])
        
        # Boundary conditions zero gradient (neumann)
        if edge == 'neumann':
            phi_updated = neumann_boundary(phi_updated)
        # Boundary condition fixed potential (dirichlet)
        elif edge == 'dirichlet':
            phi_updated = dirichlet_boundary(phi_updated, 0)

        # Boundary conditions (dirchlett)
        if plates:
            phi = dirichlet(phi_updated, lh_plate['V'], mask_pos)
            phi = dirichlet(phi_updated, rh_plate['V'], mask_neg)
        
        # See what error is between consecutive arrays
        err.append(np.mean((phi-phi_updated)**2))
        phi = phi_updated
    return phi, err


def define_plate_mask(plate, sim):
    """Create a mask indicating where a plate is in the grid"""
    N=sim['N']
    grid = np.zeros((N,N,N))
    plate_bottom = abscoord_to_grididx(plate['Lbottom'], sim)
    plate_top = abscoord_to_grididx(plate['Ltop'], sim)
    plate_pos = abscoord_to_grididx(plate['Z'], sim)

    #x,y,z
    grid[plate_bottom:plate_top, plate_bottom:plate_top, plate_pos] = 1
    mask = grid==1
    return mask

def calc_Efield(phi, sim):
    """Compute the components of the electric field from the potential.
    phi - a 3D numpy array of potential at each point
    sim - a dictionary describing key simulation params
    """
    Ex, Ey, Ez = np.gradient(-phi, sim['dx'])
    return (Ex, Ey, Ez)

def convergence(err, sim):
    #Check how it is converging
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(range(sim['iters']), err,'r-')
    ax.set_ylim(ymin=0, ymax=0.01)
    plt.show()


