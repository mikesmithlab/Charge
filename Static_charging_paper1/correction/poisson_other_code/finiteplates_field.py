import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


class ElectricField:
    def __init__(self, Q_plate, D, L, Q):
        #self.V = V
        self.D = D#separation plate
        self.L = L#dimension plate
        self.Q = Q
        N = 11#Needs to be odd
        self.mid = 1+N//2 # Only interested in saving the z=0 plane
        self.sigma = Q_plate/(L**2)
        self.dL=L/N
        self.x_range = np.linspace(-D/2, D/2, N) # direction between plates
        self.y_range = np.linspace(-L/2, L/2, N)
        self.z_range = np.linspace(-L/2, L/2, N)
        self.X, self.Y, self.Z = np.meshgrid(self.x_range, self.y_range, self.z_range)
                
    def charge_field(self):
        #Charge positioned at (0,0,0)
        k = 1/(4*np.pi*8.854E-12)
        # Field due to point charge
        r = np.sqrt(self.X**2 + self.Y**2)
        Ex = k*self.Q * self.X / (r**3)
        Ey = k*self.Q * self.Y / (r**3)
        return Ex, Ey
    
    def plate_element_field(self,y,z):
        k = 1/(4*np.pi*8.854E-12)
        
        # Field due to surface charge
        r_p = np.sqrt((self.D/2 + self.X)**2 + (self.Y-y)**2 + (self.Z-z)**2) # Distance from +ve plate charge (y,z) to a point (X,Y,Z)
        r_n = np.sqrt((self.D/2 - self.X)**2 + (self.Y-y)**2 + (self.Z-z)**2)  # Distance from -ve plate charge (y,z) to a point (X,Y,Z)

        # surface charge from the positive plate then the negative plate
        Ex = k*self.sigma*self.dL**2 * (self.X + self.D/2) / (r_p**3) + k*self.sigma*self.dL**2 * (self.D/2 - self.X) / (r_n**3)
        Ey = k*self.sigma*self.dL**2 * (self.Y-y) / (r_p**3) - k*self.sigma*self.dL**2 * (self.Y-y) / (r_n**3)
        return Ex, Ey


    def compute_field_vectors(self):
        Ex_tot = np.zeros_like(self.X[:,:,self.mid])
        Ey_tot = np.zeros_like(self.Y[:,:,self.mid])

        # Uniform field between plates
        for y in self.y_range:
            for z in self.z_range:
                Ex, Ey = self.plate_element_field(y,z)
                Ex_tot += Ex
                Ey_tot += Ey

        #Central point charge
        Ex,Ey = self.charge_field()
        Ex_tot += Ex
        Ey_tot += Ey
        return Ex_tot, Ey_tot

    def plot(self):
        # Calculate the electric field vectors
        Ex, Ey = self.compute_field_vectors()
        E_magnitude = np.sqrt(Ex**2 + Ey**2)

        # Plotting the electric field
        plt.figure(figsize=(6, 6))
        plt.title(f"{self.Q*1E9} nC")
        mid = 1+self.N//2
        print(np.shape(E_magnitude))
        print(np.shape(Ex))
        print(self.X[:,:,mid])
        print(self.Y[:,:,mid])
        print(E_magnitude)
        
        plt.contourf(self.X[:,:,mid], self.Y[:,:,mid], E_magnitude[:,:,mid], cmap='viridis',
                     levels=np.logspace(np.log10(E_magnitude.min()), np.log10(E_magnitude.max()), 50), norm=LogNorm())
        plt.colorbar(label='Electric Field Magnitude')
        plt.streamplot(self.X[:,:,mid], self.Y[:,:,mid], Ex[:,:,mid], Ey[:,:,mid], color='k', linewidth=1, density=1.5)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        #plt.axhline(0, color='black', linewidth=0.5, ls='--')
        #plt.axvline(0, color='black', linewidth=0.5, ls='--')
        plt.xlim([np.min(self.x_range), np.max(self.x_range)])
        plt.ylim([np.min(self.y_range), np.max(self.y_range)])
        plt.axis('equal')
        plt.grid()
        plt.show()


if __name__ == '__main__':
    E = ElectricField(2000, 0.13, 0.1, 3E-9)
    E.plot()
