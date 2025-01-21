import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle, Circle
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import convolve, generate_binary_structure


def plot_2Dpotential(grid, sim, zorder=1):
    #Create a contour map of the potential
    # slc specifies the slice in 3D cube to look at in the y-z plane
    fig, ax = plt.subplots(figsize=(6,5))
    #CS = plt.contour([np.arange(-sim['N']//2, sim['N']//2)*sim['dx'] , np.arange(-sim['N']//2, sim['N']//2)*sim['dx']], grid[sim['N']//2], levels=40, zorder=3)
    Z,Y = np.meshgrid(np.arange(-sim['N']//2, sim['N']//2,1)*sim['dx'],np.arange(-sim['N']//2, sim['N']//2,1)*sim['dx'])
    CS = plt.contour(Z,Y, grid[sim['N']//2], levels=40, zorder=3)
    ax.clabel(CS, CS.levels, inline=True, fontsize=6)
    ax.set_title('Potential contour map')
    ax.set_xlabel('$z$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal')
    return ax

def plot_plate(ax, sim, plates, zorder=2):
    if plates['V'] > 0:
        colour='r'
    else:
        colour='k'
    
    plate=Rectangle((plates['Z'] - plates['t']/2, plates['Lbottom']), plates['t'], plates['Ltop']-plates['Lbottom'],
                                linewidth=1, edgecolor='k', facecolor=colour, linestyle='-', zorder=2)
    ax.add_patch(plate)
    return ax

def plot_ball(ax, ball, zorder=2):
    sphere = Circle((0,0), ball['R'], linewidth=2, edgecolor='k', facecolor='w', linestyle='-', zorder=2)
    ax.add_patch(sphere)
    return ax

def plot_grid(ax, sim):
    #Set up grid to match the grid where values are calculated
    ax.xaxis.set_minor_locator(MultipleLocator(sim['dx']))
    ax.yaxis.set_minor_locator(MultipleLocator(sim['dx']))
    ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
    return ax

def plot_rad_potential(grid, ball, sim):
    fig, ax = plt.subplots(figsize=(6,5))
    potential = grid[sim['N']//2, sim['N']//2, sim['N']//2:]
    rad=np.linspace(0, np.size(potential), num=np.size(potential))*sim['dx']

    ax.plot(rad, potential)
    ax.plot([ball['R'], ball['R']], [0, np.max(potential)], 'g--')
    ax.plot(rad[rad>ball['R']], (ball['Q']/(4*np.pi*sim['eps0']*rad))[rad>ball['R']], 'r--')


#Plot the electric field and show the gaussian surface
def plot_field(grid, E, sim, lh_plate, rh_plate, ball, plates=True, bead=True):
    fig, ax = plt.subplots(figsize=(6,5))
    # Create a grid for plotting
    z = np.linspace(0, grid[sim['N']//2].shape[0]*sim['dx'], grid[sim['N']//2].shape[1])-sim['N']*sim['dx']/2
    y = np.linspace(0, grid[sim['N']//2].shape[0]*sim['dx'], grid[sim['N']//2].shape[1])-sim['N']*sim['dx']/2
    Z, Y = np.meshgrid(z, y)

    Ex, Ey, Ez = E
    E_magnitude = (Ex**2 + Ey**2 + Ez**2)**0.5

    # Plotting the electric field magnitude as a contourf plot
    contour = ax.contourf(Z, Y, E_magnitude[sim['N']//2], cmap='viridis', levels=40)
    plt.colorbar(contour, label='E Magnitude')

    # Overlaying the electric field vectors as a streamplot
    ax.streamplot(Z, Y, Ez[sim['N']//2], Ey[sim['N']//2], color='g', linewidth=1, density=1)
    
    #Set up grid to match the grid where values are calculated
    ax.xaxis.set_minor_locator(MultipleLocator(sim['dx']))
    ax.yaxis.set_minor_locator(MultipleLocator(sim['dx']))
    ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
    
    #Show plates
    if plates:
        pos_plate = Rectangle((lh_plate['Z'] - lh_plate['t']/2, lh_plate['Lbottom']), lh_plate['t'], lh_plate['Ltop']-lh_plate['Lbottom'],
                            linewidth=1, edgecolor='k', facecolor='r', linestyle='-', zorder=2)
        neg_plate = Rectangle((rh_plate['Z'] - rh_plate['t']/2, rh_plate['Lbottom']), rh_plate['t'], rh_plate['Ltop']-rh_plate['Lbottom'],
                            linewidth=1, edgecolor='k', facecolor='k', linestyle='-', zorder=2)

        ax.add_patch(pos_plate)
        ax.add_patch(neg_plate)

    #Show Sphere
    if bead:
        sphere = Circle((0,0), ball['R'], linewidth=2, edgecolor='k', facecolor='w', linestyle='-', zorder=2)
        ax.add_patch(sphere)
    
    ax.set_aspect('equal')
    

    ax.set_xlabel('Z-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Electric Field Magnitude and Vectors (2D Slice)')
    return ax

def gauss_surf(ax, gaussian_surface):
    # Add a semi-transparent rectangular patch defined by gauss_surf coords
    rect = Rectangle(gaussian_surface[0], 
                        gaussian_surface[1][0] - gaussian_surface[0][0], 
                        gaussian_surface[1][1] - gaussian_surface[0][1],
                        linewidth=2, edgecolor='k', facecolor='g', alpha=0.5, linestyle='--', hatch='\/')
    ax.add_patch(rect)
    return ax
