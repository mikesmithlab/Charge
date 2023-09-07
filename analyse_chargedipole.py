import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from DataAnalysis.fitting import Fit
from DataAnalysis.fit_models import dipole_fit

path = 'Z:/GranularCharge/BlueBeads10mm/2023_05_25/bead1_165_15min/'
filename = 'img.csv'

d_plates = 0.16  # m
mg_bead = 0.56E-3*9.81  # mass in kg
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

    fit.add_params(guess=[qr_sk, 101],
                   lower=[1.5*qr_sk, -180], upper=[0, 180])
    fit.fit()
    fit.plot_fit(xlabel='dtheta (deg)', ylabel='voltage (kV)')
    dipole = fit.fit_params[0] * (d_plates*kappa)
    theta_0 = fit.fit_params[1]

    print(fit.fit_params[0]/qr_sk)
    print(fit.fit_params[1])
    # plt.figure()
    # plt.plot(dtheta[index], voltage[index], 'go')
    # plt.plot(dtheta[index], dipole_fit(dtheta[index],  0.999*qr_sk, 103.8), 'r-')
    # plt.show()

    return dipole, theta_0


df = pd.read_csv(path + filename)
charge = calc_charge(df)
dipole, theta_0 = calc_dipole(df, charge)


print(f'Charge is {charge *1e9 :2.3} nC')
print(f'Dipole is {dipole *1e12 :2.3} nCmm and Theta_0 is {theta_0:3.2} deg')
print(f'Dipole_length = {1000*dipole/charge:2.3} mm')


# bead1_165_15min ==> (1.4638459245098259, 101.17907238495917) = 7.17 mm
# bead1_235_15min ==> (0.4455277028218308, -175.44168350310724) = 2.18 mm
# bead2_165_15min ==> (0.6196387750942645, 114.21982410188674) = 3.04 mm
# bead2_235_15min ==> (0.21233363365295416, -179.09896039747315) = 1.04 mm

