import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from filehandling import BatchProcess

initial = True  # Take raw data and summarise max min etc
final = True  # collate above into file for plotting

path = os.environ['USERPROFILE'] + \
    '/OneDrive - The University of Nottingham/Documents/Papers/Charge/Static_Charging/Figures/Figure4/model_output/'

ending = '_Qrandom'

if initial:

    N = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    # N = [5]

    for n in N:

        Fmax = []
        Fmin = []
        Tmax = []
        Tmin = []
        dipole_Fmax = []
        dipole_Fmin = []
        dipole_Tmax = []
        dipole_Tmin = []
        dipole_length = []

        for file in BatchProcess(path + 'N' + str(n) + ending + '/*.csv'):
            print(file)
            df = pd.read_csv(file)
            Fmax.append(df['sticking_force'].abs().max())
            Fmin.append(df['sticking_force'].abs().min())
            Tmin.append(df['total_tau'].abs().min())
            Tmax.append(df['total_tau'].abs().max())
            dipole_Fmax.append(df['dipole_force'].abs().max())
            dipole_Fmin.append(df['dipole_force'].abs().min())
            dipole_Tmin.append(df['dipole_torque'].abs().min())
            dipole_Tmax.append(df['dipole_torque'].abs().max())
            dipole_length.append(df['dipole_length'].max())

        pd.DataFrame({'Fmin': Fmin, 'Fmax': Fmax, 'Tmin': Tmin, 'Tmax': Tmax, 'dipole_Fmin': dipole_Fmin, 'dipole_Fmax': dipole_Fmax,
                     'dipole_Tmin': dipole_Tmin, 'dipole_Tmax': dipole_Tmax, 'dipole_length': dipole_length}).to_csv(path + 'summary_N' + str(n) + ending + '.csv')


if final:
    nvals = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    results = {'N': nvals, 'Fmax': [], 'Fmin': [], 'Tmax': [], 'Tmin': [
    ], 'D_Tmax': [], 'D_Tmin': [], 'D_Fmax': [], 'D_Fmin': [], 'D_length': []}

    for N in nvals:

        df = pd.read_csv(path + 'summary_N' + str(N) + ending + '.csv')

        results['Fmax'].append(df['Fmax'].median())
        results['Fmin'].append(df['Fmin'].median())
        results['Tmax'].append(df['Tmax'].median())
        results['Tmin'].append(df['Tmin'].median())
        results['D_Fmax'].append(df['dipole_Fmax'].median())
        results['D_Fmin'].append(df['dipole_Fmin'].median())
        results['D_Tmax'].append(df['dipole_Tmax'].median())
        results['D_Tmin'].append(df['dipole_Tmin'].median())
        results['D_length'].append(df['dipole_length'].median())

    pd.DataFrame(results).to_csv(path + ending + 'complete_summary.csv')
