import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from DataAnalysis.fitting import Fit
from DataAnalysis.fit_models import dipole_fit, dipole_fit_offset


d_plates = 0.13  # m
mg_bead = 0.568E-3*9.81  # mass in kg
R_bead = 4.9E-3  # 9.8E-3 # m
kappa = 2.56E-8  # SI
L_string = 1  # m


def calc_charge(df):
    df['E'] = 1000*df['voltage']/d_plates
    df['theta'] = (df['beadx_m'] - df['beadx_m'].iloc[0])/L_string
    charge, b = np.polyfit(df['E'], df['theta']*mg_bead, 1)
    return charge


def calc_dipole(df, charge):
    df['dtheta'] = df['adjusted_angle'] - df['adjusted_angle'].iloc[0]
    dtheta = df['dtheta'].to_numpy()
    index = np.argsort(dtheta)
    voltage = 1000*df['voltage'].to_numpy()
    fit = Fit('dipole_fit', x=dtheta[index], y=voltage[index])

    qr_sk = (charge*R_bead/(d_plates*kappa))

    guess = [0.5*qr_sk, 160]

    fit.add_params(guess=guess,
                   lower=[1.1*qr_sk, 0], upper=[0, 360])
    fit.fit()
    fit.plot_fit(xlabel='dtheta (deg)', ylabel='voltage (V)')
    dipole = fit.fit_params[0] * (d_plates*kappa)
    theta_0 = fit.fit_params[1]

    dtheta_fine = np.linspace(-140, 0, 1000)
    np.savetxt('dtheta_fit.csv', np.c_[dtheta_fine, dipole_fit(dtheta_fine, fit.fit_params[0], fit.fit_params[1])/(1000*0.13)], fmt='%.7f', delimiter=',')

    
    


    # plt.figure()
    # plt.plot(dtheta[index], voltage[index], 'go')
    # plt.plot(dtheta[index], dipole_fit(
    #    dtheta[index],  guess[0], guess[1]), 'r-')
    # plt.show()

    return dipole, theta_0


if __name__ == '__main__':
    path = 'Z:/GranularCharge/WhiteBead/2023_09_07/'
    filename = 'img5.csv'

    df = pd.read_csv(path + filename)
    charge = calc_charge(df)
    dipole, theta_0 = calc_dipole(df, charge)

    print(f'Charge is {charge *1e9 :2.3} nC')
    print(
        f'Dipole is {dipole *1e12 :2.3} nCmm and Theta_0 is {theta_0:3.2} deg')
    print(f'Dipole_length = {1000*dipole/charge:2.3} mm')

    # bead1_165_15min ==> (1.4638459245098259, 101.17907238495917) = 7.17 mm
    # bead1_235_15min ==> (0.4455277028218308, -175.44168350310724) = 2.18 mm
    # bead2_165_15min ==> (0.6196387750942645, 114.21982410188674) = 3.04 mm
    # bead2_235_15min ==> (0.21233363365295416, -179.09896039747315) = 1.04 mm
