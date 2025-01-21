import numpy as np
from poisson_solver import abscoord_to_grididx

def find_charge_coords(ball, sim):
    R=ball['R']
    dR=ball['dR']
    N=sim['N']

    #Works out which pixels are at the surface of sphere. ie between R and R+dR.
    coords=[]
    dist=[]
    
    lower_bound = abscoord_to_grididx(-(R+dR), sim)
    upper_bound = abscoord_to_grididx(+(R+dR), sim)

    for i in range(lower_bound, upper_bound+1):
        for j in range(lower_bound, upper_bound+1):
            for k in range(lower_bound, upper_bound+1):
                dist_2_centre = ((i-N//2)**2 + (j-N//2)**2 + (k-N//2)**2)**0.5
                if (dist_2_centre >= R/sim['dx']) and (dist_2_centre < (R+dR)/sim['dx']):
                    coords.append((i,j,k))
                    dist.append(dist_2_centre)
    return coords

def charge_ball(ball, sim):
    """Add charge to the surface of the ball"""
    N=sim['N']
    rho = np.zeros((N,N,N))
    
    #Coords of surface charges on ball
    charge_coords = find_charge_coords(ball, sim)

    #Create a matrix of little charges at the surface of sphere
    for charge in charge_coords:
        rho[charge[0], charge[1], charge[2]] = ball['Q']/(len(charge_coords)*sim['dx']**3)
    return rho , charge_coords 

def Int_EdA_face(E_vec, Na, Nb):
    """Function calcs integral of E.dA for one face of a cube
    The surface normal of cube face is in direction of E_vec component
    The Tuples define the area of the face in directions orthogonal to N.

    Input should be eg. Ez[Nz[0]], (Ny1, Ny2), (Nx1, Nx2)

    edge_factor compensates for edges and corners.
    """
    flux = 0
    #For each surface pixel in face
    
    for i in np.arange(Na[0], Na[1]+1):
        for j in np.arange(Nb[0], Nb[1]+1):
            if (i==Na[0] or i ==Na[1]) and (j==Nb[0] or j ==Nb[1]):
                edge_factor = 0.25
            elif (i==Nb[0] or i ==Nb[1]) or (j==Nb[0] or j ==Nb[1]):
                edge_factor = 0.5
            else:
                edge_factor = 1
        
            flux += (edge_factor*E_vec[i, j]*dx**2)
    return flux

def calc_charge(coords, E):
    """
    To calc charge we take region which defines a cube in real coords.
    ((bottomlefty, bottomleftz),(toprighty, toprightz))
    x is assumed to be identical to y by symmetry
    Calc flux through each face and then multiply by eps0 to get Q in nC
    """
    Nx, Ny, Nz = coords 
    Ex,Ey,Ez = E
    
    flux = 0
    #dA is negative surface vector
    flux -= Int_EdA_face(Ex[Nx[0]], Ny, Nz)
    flux -= Int_EdA_face(Ey[:,Ny[0],:], Nz, Nx)
    flux -= Int_EdA_face(Ez[:,:,Nz[0]], Ny, Nx)
    
    #dA is positive surface vector
    flux += Int_EdA_face(Ex[Nx[1]], Nz, Ny)
    flux += Int_EdA_face(Ey[:,Ny[1],:], Nx, Nz)
    flux += Int_EdA_face(Ez[:,:,Nz[1]], Ny, Nx)   
    
    Q = flux * 8.854E-12 / 1E-9
    return Q

def charge_distribution_plate(E, plate, sim):
    """This function assumes the plates are vertical"""
    charge_z = []
    box = ((plate['Z']-sim['dz']-2*plate['t_plate'],plate['Lbottom']),(plate['Z']-sim['dz']+2*plate['t_plate'], plate['Ltop']))
    coords = abscoord_to_grididx(box, sim)
    top = plate['Ltop']
    for coord in range(coords[0][1], coords[1][1]):
        section = ((coords[0][0],coord),(coords[1][0], coord+1))
        charge_z.append(calc_charge(section, E))
    return charge_z

def get_grid_coords(region, sim):
    Ny = (abscoord_to_grididx(region[0][1], sim), abscoord_to_grididx(region[1][1], sim))
    Nz = (abscoord_to_grididx(region[0][0], sim), abscoord_to_grididx(region[1][0], sim))
    Nx = Ny
    return (Nx, Ny, Nz)
