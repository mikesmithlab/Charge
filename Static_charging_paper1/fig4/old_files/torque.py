# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:31:20 2023

@author: ppzmrs
"""
import numpy as np
import matplotlib.pyplot as plt
import labvision as lv


def WireframeSphere(centre=[0.,0.,0.], radius=1.,
                    n_meridians=20, n_circles_latitude=None):
    """
    Create the arrays of values to plot the wireframe of a sphere.

    Parameters
    ----------
    centre: array like
        A point, defined as an iterable of three numerical values.
    radius: number
        The radius of the sphere.
    n_meridians: int
        The number of meridians to display (circles that pass on both poles).
    n_circles_latitude: int
        The number of horizontal circles (akin to the Equator) to display.
        Notice this includes one for each pole, and defaults to 4 or half
        of the *n_meridians* if the latter is larger.

    Returns
    -------
    sphere_x, sphere_y, sphere_z: arrays
        The arrays with the coordinates of the points to make the wireframe.
        Their shape is (n_meridians, n_circles_latitude).

    Examples
    --------
    >>> fig = plt.figure()
    >>> ax = fig.gca(projection='3d')
    >>> ax.set_aspect("equal")
    >>> sphere = ax.plot_wireframe(*WireframeSphere(), color="r", alpha=0.5)
    >>> fig.show()

    >>> fig = plt.figure()
    >>> ax = fig.gca(projection='3d')
    >>> ax.set_aspect("equal")
    >>> frame_xs, frame_ys, frame_zs = WireframeSphere()
    >>> sphere = ax.plot_wireframe(frame_xs, frame_ys, frame_zs, color="r", alpha=0.5)
    >>> fig.show()
    """
    if n_circles_latitude is None:
        n_circles_latitude = max(n_meridians/2, 4)
    u, v = np.mgrid[0:2*np.pi:n_meridians*1j, 0:np.pi:n_circles_latitude*1j]
    sphere_x = centre[0] + radius * np.cos(u) * np.sin(v)
    sphere_y = centre[1] + radius * np.sin(u) * np.sin(v)
    sphere_z = centre[2] + radius * np.cos(v)
    return sphere_x, sphere_y, sphere_z

def plot_dots():
N=10

e0=8.85e-12
q=5.e-9/N
r=5e-3
gap=0.5e-3
L=3*r

np.random.seed(3)
theta=2.0*np.pi*np.random.rand(N)
theta=theta-theta[0]
th0=0.0

#yc=np.zeros(N)
rxz=r

phi=np.arccos(2*np.random.rand(N)-1)
yc=r*np.cos(phi)
rxz=r*np.sin(phi)

rx=rxz*np.cos(theta)
rz=-rxz*np.sin(theta)
xc=-gap-r+rx
yc=yc
zc=rz

plt.close('all')

plt.figure(2,figsize=(7,7))
ax=plt.axes(projection='3d')
ax.scatter(xc,yc,zc,'bo')
ax.set_xlim(-gap-r-r,-gap-r+r)
ax.set_ylim(-r,r)
ax.set_zlim(-r,r)
plt.pause(5)

#%%

fig=plt.figure(1,figsize=(7,7))
ax=plt.axes([0.1,0.5,0.8,0.4])
ax.set_xlim(-L,L)
ax.set_ylim(-L/2,L/2)
tx=plt.axes([0.1,0.1,0.8,0.3])

plt.pause(1)

while(th0<2*np.pi):
    theta=theta+0.05
    rx=rxz*np.cos(theta)
    rz=-rxz*np.sin(theta)
    xc=-gap-r+rx
    yc=yc
    zc=rz
    xi=-xc
    yi=yc
    zi=zc
    
    tau=0.0
    
    for i in np.arange(N):

        dx=xi[i]-xc
        dy=yi[i]-yc
        dz=zi[i]-zc
        d2=dx*dx+dy*dy+dz*dz
        d=np.sqrt(d2)
        nx=dx/d
        ny=dy/d
        nz=dz/d
        
        f=(q*q)/(4.0*np.pi*e0*d*d)
        fx=np.sum(f*nx)
        fz=np.sum(f*nz)
    
        tau+=rz[i]*fx-rx[i]*fz
            
           
    ax.clear()
    ax.set_xlim(-L,L)
    ax.set_ylim(-L/2,L/2)
    for i in np.arange(N):
        cir0 = plt.Circle((-gap-r, 0), r, color='k',fill=False)
        ax.add_patch(cir0)
        cir2 = plt.Circle((xc[i], zc[i]), r*0.05, color='b',fill=True)
        ax.add_patch(cir2)
        cir0 = plt.Circle((gap+r, 0), r, color='k',fill=False)
        ax.add_patch(cir0)
        cir2 = plt.Circle((xi[i], zi[i]), r*0.05, color='r',fill=True)
        ax.add_patch(cir2)
        
        
    tp=th0-2.0*np.pi*np.int32(th0/(2.0*np.pi))
    tx.plot(tp,tau,'*b')
    tx.set_xlabel('Delta Theta')
    tx.set_ylabel('Tau (Nm)')

    plt.pause(0.0001)
    th0=theta[0]
