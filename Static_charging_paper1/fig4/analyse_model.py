
from matplotlib import markers
import pandas as pd
import numpy as np
from filehandling import BatchProcess, smart_number_sort
import matplotlib.pyplot as plt
import os

path = os.environ['USERPROFILE'] + "/OneDrive - The University of Nottingham/Documents/Papers/Charge/Figures/FIgure2/model_output/"
N_vals = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
if False:
    
    for i in N_vals:
        print(i)
        #These are the full model.
        max_torque = []
        min_torque = []
        max_force = []
        min_force = []
        #These are for the model of all charge at the dipole point
        max_dipole_force=[]
        min_dipole_force=[]
        max_dipole_torque=[]
        min_dipole_torque=[]
        dipole_length = []
        force_at_max_torque = []
        
            

        # for each realization of the N charge sphere we extract the max or min magnitudes.
        for file in BatchProcess(path + 'N' + str(i) + '_Q5/*.csv'):
            print(file)
            df = pd.read_csv(file)
            max_torque.append(df['total_tau'].abs().max())
            min_torque.append(df['total_tau'].abs().min())
            max_force.append(df['sticking_force'].abs().max())
            min_force.append(df['sticking_force'].abs().min())
            max_dipole_torque.append(df['dipole_torque'].abs().max())
            min_dipole_torque.append(df['dipole_torque'].abs().min())
            max_dipole_force.append(df['dipole_force'].abs().max())
            min_dipole_force.append(df['dipole_force'].abs().min())
            force_at_max_torque.append(
                df['sticking_force'].abs().iloc[np.argmax(df['total_tau'].abs())])
            dipole_length.append(file.split('_d')[1][:-5])
        pd.DataFrame({'max_torque': max_torque, 'min_torque': min_torque, 'max_force': max_force,
                      'min_force': min_force, 'max_dipole_torque': max_dipole_torque, 'min_dipole_torque': min_dipole_torque, 'max_dipole_force': max_dipole_force,
                      'min_dipole_force': min_dipole_force,'force_at_max_torque': force_at_max_torque, 'dipole_length': dipole_length}).to_csv(path + 'summary_N' + str(i) + '.csv')
        



if True:
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True,
                           figsize=(6, 6), layout='tight')
    dipole_lengths = []
    torques = []
    max_forces = []
    min_forces = []
    force_at_max_torque = []
    max_dipole_force=[]
    min_dipole_force=[]
    dipole_torques=[]
    
    max_torques = []
    
    # Here from the summary files we extract the "typical" values as a median
    for i,file in enumerate(BatchProcess(path + 'summary_N' + '*.csv', smart_sort=smart_number_sort)):
        df = pd.read_csv(file)
        settings = {'alpha': 0.5, 'bins': 10}
        dipole_lengths.append(df['dipole_length'].median())
        #Full charge distribution
        torques.append(df['max_torque'].median())
        max_forces.append(df['max_force'].median())
        min_forces.append(df['min_force'].median())
        #Due to the effective dipole
        dipole_torques.append(df['max_dipole_torque'].median())
        max_dipole_force.append(df['max_dipole_force'].median())
        min_dipole_force.append(df['min_dipole_force'].median())
        force_at_max_torque.append(df['force_at_max_torque'].median())
        if i==0:
            dipole_total = [np.log10(df['max_dipole_torque'].to_numpy())]
            force_total = [np.log10(df['max_dipole_force'].to_numpy())]
        else:
            dipole_total.append(np.log10(df['max_dipole_torque'].to_numpy()))
            force_total.append([np.log10(df['max_dipole_force'].to_numpy())])

    dipole_total = np.array(dipole_total)
    force_total = np.array(force_total)

    max_forces = np.array(max_forces)
    min_forces = np.array(min_forces)
    torques = np.array(torques)
    max_dipole_forces = np.array(max_dipole_force)
    min_dipole_forces = np.array(min_dipole_force)
    dipole_torques = np.array(dipole_torques)
    force_at_max_torque = np.array(force_at_max_torque)
    dipole_lengths = np.array(dipole_lengths)

    weight = 5E-4*9.81

    mu = 0.005
    R = 5E-3
    gap = 0.5E-3
    Q = 5E-9

    angle = 90
    Fc = Q**2/(4*np.pi*8.85E-12*(2*R+gap)**2)
    Tau = mu*R*(weight*np.cos(angle*np.pi/180) + Fc)

    Tau2 = weight*R
    # Plot Dipole length with typical range of values
    ax[0].loglog(N_vals, dipole_lengths, 'rx')
    ax[0].loglog([N_vals[0], N_vals[-1]], [2.5, 2.5], 'k--')
    ax[0].loglog([N_vals[0], N_vals[-1]], [0.4, 0.4], 'k--')

    # Plot torque with typical range of values
    #full charge dist torque
    ax[1].loglog(N_vals, torques, 'bx')
    #dipole torque
    ax[1].loglog(N_vals, dipole_torques, 'gx')
    ax[1].loglog([N_vals[0], N_vals[-1]], [1E-6, 1E-6], 'k--')
    ax[1].loglog([N_vals[0], N_vals[-1]], [5E-6, 5E-6], 'k--')
    # Reference Torque values
    # Rolling friction
    ax[1].loglog([N_vals[0], N_vals[-1]], [Tau, Tau], 'g--')
    # weight*R
    ax[1].loglog([N_vals[0], N_vals[-1]], [Tau2, Tau2], 'g--')

    # Plot max force, force at max_torque and min_force. Cf with weight of the ball
    ax[2].loglog(N_vals, max_forces, 'g-')
    ax[2].loglog(N_vals, force_at_max_torque, 'k--')
    ax[2].loglog(N_vals, min_forces, 'g-')
    ax[2].loglog(N_vals, max_dipole_forces, 'r--')
    ax[2].loglog(N_vals, min_dipole_forces, 'r--')
    #Weight of bead
    ax[2].loglog([N_vals[0], N_vals[-1]], [weight, weight], 'b--')
    # Charge at centre.
    ax[2].loglog([N_vals[0], N_vals[-1]],
                 [max_forces[-1], max_forces[-1]], 'r--')

    ax[2].set_xlabel('Number of charges')
    ax[0].set_ylabel('Dipole length (mm)')
    ax[1].set_ylabel('Max torque (Nm)')
    ax[2].set_ylabel('Force (N)')

    ax[0].set_title('Q=5nC')

    plt.show()

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].boxplot(dipole_total.T, sym='')
    ax[1].boxplot(force_total.T, sym='')
    plt.show()