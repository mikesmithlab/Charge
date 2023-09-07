import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma


def analytical_model(N_vals, q=1E-12, R=0.005, rho=2000, theta_c = np.pi/8, ):
    "https://math.stackexchange.com/questions/103142/expected-value-of-random-walk"
    total_charge = q*N_vals

    prefactor = q*R*np.sqrt(2/3)*gamma([(3+1)/2])/gamma([3/2])
    dipole = prefactor * np.sqrt(N_vals)

    #Multiply by R since only relevant lengthscale to make dimensionless.
    ratio = total_charge/dipole


    t_bounce = np.sqrt(2*0.01*R/9.81)
    I_ball =(2/5)*(4/3)*np.pi*R**3 * rho * R**2





    plt.figure(1)
    plt.plot(N_vals,total_charge,'b-')
    plt.plot(N_vals,dipole,'r-')


    plt.figure(2)
    plt.plot(N_vals, ratio,'g')
    plt.show()





if __name__ == '__main__':
    N_vals = np.linspace(1,10000,10000)
    analytical_model(N_vals)
