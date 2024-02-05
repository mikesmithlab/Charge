# -*- coding: utf-8 -*-
"""
This file does the calculations for a sphere covered with N randomly distributed charged dots interacting with its image charges separated by a distance d. This represents the glass in the experiment. It then creates the animated images of the simulation. assemble.py can be used to create a movie.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import io
import os

from labvision.video import WriteVideo


"""-----------------------------------------------------------------------
Visualization
----------------------------------------------------------------------------"""

def Sphere(centre=[0., 0., 0.], radius=1.,
           n_meridians=200, n_circles_latitude=None):
    """
    Create the arrays of values to plot the surface of a sphere.    
    """
    if n_circles_latitude is None:
        n_circles_latitude = max(n_meridians/2, 4)
    u, v = np.mgrid[0:2*np.pi:n_meridians*1j, 0:np.pi:n_circles_latitude*1j]
    sphere_x = centre[0] + radius * np.cos(u) * np.sin(v)
    sphere_y = centre[1] + radius * np.sin(u) * np.sin(v)
    sphere_z = centre[2] + radius * np.cos(v)
    return sphere_x, sphere_y, sphere_z


def plot_sphere(ax, xc, yc, zc, rad, gap, offset=1.5, mirror=False, projection=True, plot_dipole=True):
    # Renders a semitransparent sphere with charges on surface. Then produces a projection onto the xy plane.
    marker_colour = 'b'
    colour = 'r'
    r = rad
    if mirror:
        marker_colour = 'r'
        colour = 'b'
        xc = -xc
        r = -rad
        gap = -gap

    ax.scatter(xc, yc, zc + offset*rad, color=marker_colour,
               marker='o', edgecolors='k')
    sphere = ax.plot_surface(*Sphere(centre=[gap+r, 0, offset*rad], radius=rad), color=colour, alpha=0.4)
    
    if projection:
        ax.scatter(xc, yc, 0, color=marker_colour,
                   marker='o', edgecolors='k', alpha=0.3)
        for x, z, y in zip(xc, yc, zc):
            ax.plot([x, x], [z, z], [0, offset*rad + y], 'k--', alpha=0.25)
        p = plt.Circle((gap+r, 0), rad, color=colour,
                       fill=True, alpha=0.15, edgecolor='k')
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p)
    ax.set_box_aspect((2*(gap+rad)/rad, 1, 1))
    
    if plot_dipole:
        dipole = calc_dipole(xc, yc, zc)
        ax.plot([gap+r, dipole[0]], [0, dipole[1]],
                [offset*rad, offset*rad + dipole[2]], 'k-', alpha=1)
        ax.plot([gap+r, dipole[0]], [0, dipole[1]],
                [offset*rad, offset*rad + dipole[2]], 'k*', alpha=1)
        ax.plot([gap+r, dipole[0]], [0, dipole[1]],
                [0, 0], 'k-', alpha=1)
    return ax


def plot_charged_sphere(ax, xc, yc, zc, r, gap):
    # Plot the sphere with its charges in 3D. Also project onto the horizontal xz plane.
    ax = plot_sphere(ax, xc, yc, zc, r, gap, mirror=False)
    ax = plot_sphere(ax, xc, yc, zc, r, gap, mirror=True)
    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    # Turn off tick labels
    ax.set_zticklabels([])

    return ax

def display_plot(ax, fig, i, theta, phi, dipole_angle, total_tau, sticking_force, r, gap, seed, dtheta, path, N, Q):
    """Creates a composite figure with the spheres with charges at the top and a plot of the torque etc at the bottom"""
    max_tau = np.max(np.abs(1e6*total_tau))
    max_F = -np.max(np.abs(1e3*sticking_force))
    ax1,ax2,ax3=ax
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax2.set_ylim(-max_tau, max_tau)
    ax2.set_xlim(0, 2*np.pi)
    ax3.set_ylim(max_F, 0)
    ax3.set_xlim(0, 2*np.pi)
    ax2.set_xlabel(u'\u0394\u03B8 (rad)')
    ax2.set_ylabel(u'\u03C4 (\u03bcNm)')
    ax3.set_xlabel(u'\u0394\u03B8 (rad)')
    ax3.set_ylabel('F (mN)')

    #Add the charged spheres images to figure
    pos_charges, _ = calc_charge_coords(theta, phi, r, gap)
    xc, yc, zc = pos_charges
    ax1 = plot_charged_sphere(
        ax1, 1000*xc, 1000*zc, 1000*yc, 1000*r, 1000*gap)  # Convert to mm

    #Plot the force and  torque as function angle
    ax2.plot(dipole_angle[:i], 1e6*total_tau[:i], '.r')
    ax2.plot(dipole_angle[i], 1e6*total_tau[i], 'g.')
    ax3.plot(dipole_angle[:i], 1e3*sticking_force[:i], '.r')
    ax3.plot(dipole_angle[i], 1e3*sticking_force[i], 'g.')

    fig.canvas.draw()
    fig.savefig(path + '/torque_model_N' + str(N)  + '_Q' + str(Q) + '_seed' + str(seed) + '.png', format='png')

    plt.pause(0.001)
    theta = theta+dtheta


def display_anim(N, dipole_angle, total_tau, sticking_force, r, gap, path, seed=3, dtheta=0.05, DPI=128, final=False):
    """This uses plot_charged_sphere to create the animation. If final is True you get the final point and the full graph. If final is False it will export all the images at different angles so you can create a little movie animation
    """
    
    fig = plt.figure(figsize=(4, 5), constrained_layout=False, dpi=DPI)
    gs = fig.add_gridspec(nrows=20, ncols=20, left=0, right=1, bottom=0, top=1)

    # set up the axes for the first plot
    ax1 = fig.add_subplot(gs[:11, 1:], projection='3d')
    ax1.view_init(elev=10., azim=90)
    # set up the axes for the second plot
    ax2 = fig.add_subplot(gs[9:12, 4:18])
    ax3 = fig.add_subplot(gs[13:16, 4:18])
    ax=(ax1,ax2,ax3)
    
    theta, phi = angles(N, seed=seed)
    
    if final:
        display_plot(ax, fig, len(dipole_angle)-1, theta, phi, dipole_angle, total_tau, sticking_force, r, gap, seed, dtheta, path, N, Q)
    else:
        for i in range(len(dipole_angle)):
            display_plot(ax, fig, i, theta, phi, dipole_angle, total_tau, sticking_force, r, gap, seed, dtheta, path, N, Q)

"""-----------------------------------------------------------------------
Calculation
----------------------------------------------------------------------------"""
def angles(N, seed=3):
    #Creates random theta and phi coordinates for N charges on the surface of a sphere
    np.random.seed(seed)
    theta = 2.0*np.pi*np.random.rand(N)
    theta = theta-theta[0]
    phi = np.arccos(2*np.random.rand(N)-1)
    return theta, phi
"""
    
def angles(N, seed=None):
    # Make data for plot
    u = np.linspace(0, 2*np.pi, 10)
    v = np.linspace(0, np.pi, 10)
    u,v = np.meshgrid(u,v)
    theta = u.flatten()
    phi = v.flatten()
    return theta, phi
"""
def calc_dipole(xc, yc, zc):
    """Calculate the dipole moment"""
    dipole = np.zeros(3)
    for i in np.arange(len(xc)):
        dipole[0] += xc[i]
        dipole[1] += yc[i]
        dipole[2] += zc[i]
    dipole = dipole/len(xc)

    return dipole


def calc_charge_coords(theta, phi, r, gap):
    """theta is the azimuthal angle, theta (0,2pi), phi is the elevation angle. r is radius of sphere, gap is gap between spheres. The horizontal plane (which contains theta) is x-z ."""
    rxz = r
    yc = r*np.cos(phi)
    rxz = r*np.sin(phi)

    rx = rxz*np.cos(theta)
    rz = rxz*np.sin(theta)  # rz = -rxz*np.sin(theta) # Why is this negative?
    xc = gap + r + rx  # xc = -gap-r+rx
    yc = yc
    zc = rz
    pos_charges = (xc, yc, zc)
    dipole = calc_dipole(*pos_charges)

    return pos_charges, dipole


def calc_torque_curve(N, r, gap, Q, seed=3, dtheta=0.05, e0=8.85e-12):

    # charge per patch
    q = Q*1E-9/N

    #randomly place charges on the sphere
    theta, phi = angles(N, seed=seed)
    pos_charges, dipole = calc_charge_coords(theta, phi, r, gap)



    # setup lists to receive calc values at each value of theta
    th0 = 0.0
    dipole_angle = []
    total_tau = []
    sticking_force = []

    #Here we store the force and torque due to the total charge positioned at the effective dipole position so we can compare with full result
    dipole_force = []
    dipole_torque = []

    central_charge_force = []

    #rotate the spheres
    while (th0 < 2*np.pi):
        pos_charges, dipole = calc_charge_coords(theta, phi, r, gap)
        xc, yc, zc = pos_charges

        xi = -xc
        yi = yc
        zi = zc

        tau = 0.0
        normal_force = 0.0
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

            tau += zc[i]*fx-(xc[i]+gap+r)*fz
            normal_force += fx

        # Store torque and angle in lists
        sticking_force.append(normal_force)
        total_tau.append(tau)
        dipole_angle.append(th0)

        dipole_separation = (2*dipole[0])
        dp_force = (-Q*Q*1E-18)/(4.0*np.pi*e0*dipole_separation**2)
        dipole_force.append(dp_force)
        dipole_torque.append(dp_force*dipole[2])

        central_charge_force.append((Q*1E-9)**2/(4*np.pi*e0*(2*(gap + r))**2))

        theta = theta+dtheta
        th0 = th0+dtheta
    projected_dipole_length = 1000*((dipole[0]-gap - r)**2 + dipole[1]**2)**0.5

    return np.array(dipole_angle), np.array(total_tau), np.array(sticking_force), np.array(dipole_force),np.array(dipole_torque), projected_dipole_length, central_charge_force


if __name__ == "__main__":
    import os

    path = os.environ['USERPROFILE']+ '/OneDrive - The University of Nottingham/Documents/Papers/Charge/Static_Charging/Figures/Figure4/'
    # file = path + 'torque_model.mp4'
    # vid = lv.video.WriteVideo(file, )


    #Simulation parameters
    r = 5e-3 # m - rad of sphere
    Q = 5.57  # nC
    dQ = 1.81
    new_Q = Q
    dtheta = 0.05  # increment to calc talk at
    gap = 0.5e-3 # thickness of glass slide
    L = 3*r
    scale = 1000  # Plot everything in mm
    num_charges = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    num_seeds = 500
    num_view = 50 # This is the seed you want to view

    for N in num_charges:
        print(N)
        #Output folder for a particular number of charges
        path2 = path + 'model_output/N' + str(N) + '_Qrandom' #'_Q' + str(Q)

        if os.path.exists(path2) == False:
            os.mkdir(path2)

        #Different seeds for random number generator
        for i in range(num_seeds):
            new_Q = np.random.normal(Q, dQ)
            #Run simulation for single realisation
            dipole_angle, total_tau, sticking_force, dipole_force, dipole_torque, projected_dipole_length, central_charge_force = calc_torque_curve(
                N, r, gap, new_Q, dtheta=dtheta, seed=i)
            
            #Store results for single realisation of N charges
            output_file = path2 + '/torque_model_N' + \
                str(N) + '_Q' + str(Q) + '_seed' + str(i) + '_d' + \
                str(projected_dipole_length)
            pd.DataFrame({'dipole_angle': dipole_angle, 'total_tau': total_tau,
                          'sticking_force': sticking_force, 'dipole_force':dipole_force, 'dipole_torque': dipole_torque, 'dipole_length':projected_dipole_length, 'central_charge_force':central_charge_force}).to_csv(output_file + '.csv')

            """if i==num_view:
                #If you want an animation set final=False but make sure you only do it for one seed and one N.
                display_anim(N, dipole_angle, total_tau,
                             sticking_force, r, gap, path2, seed=i, dtheta=dtheta, final=False)
            """
