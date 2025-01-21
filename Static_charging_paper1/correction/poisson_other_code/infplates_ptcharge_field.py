import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


class ElectricField:
    def __init__(self, V, D, Q):
        self.V = V
        self.D = D
        self.Q = Q
        N=100
        self.mid = 1+N//2
        self.x_range = np.linspace(-D/2, D/2, N)
        self.y_range = np.linspace(-D/2, D/2, N)

    def compute_field_vectors(self, X, Y):
        Ex = np.zeros_like(X)
        Ey = np.zeros_like(Y)

        # Uniform field between plates
        Ex += self.V / self.D

        k = 1/(4*np.pi*8.854E-12)
        # Field due to point charge
        r = np.sqrt(X**2 + Y**2)
        Ex += k*self.Q *  X / (r**3)
        Ey += k*self.Q * Y / (r**3)
        return Ex, Ey

    def plot(self):
        X, Y = np.meshgrid(self.x_range, self.y_range)

        # Calculate the electric field vectors
        Ex, Ey = self.compute_field_vectors(X, Y)
        E_magnitude = np.sqrt(Ex**2 + Ey**2)

        # Plotting the electric field
        plt.figure(figsize=(6, 6))
        plt.title(f"{self.Q*1E9} nC")
        plt.contourf(X, Y, E_magnitude, cmap='viridis',
                     levels=np.logspace(np.log10(E_magnitude.min()), np.log10(E_magnitude.max()), 50), norm=LogNorm())
        plt.colorbar(label='Electric Field Magnitude')
        plt.streamplot(X, Y, Ex, Ey, color='k', linewidth=1, density=1.5)
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
    E = ElectricField(2000, 0.13, 0.1E-9)
    E.plot()
   
