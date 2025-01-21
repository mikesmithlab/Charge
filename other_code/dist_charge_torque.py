import numpy as np
import matplotlib.pyplot as plt


t = 1E-4
eps = 4*np.pi*8.85E-12
R = 5E-3
theta = np.linspace(0, 2*np.pi, 1000)
nC = 1E-9


def torque(phi, q):
    """phi is the angle of the charge from the dipole vector
    q is the charge in nC"""
    q *= nC
    F = (q**2 / eps) * 1 / (4 * (R*(1-np.cos(theta + phi)) + t)**2)
    return F * R * np.sin(theta + phi)


def total_torque(q_phis, charges):

    sum_torque = 0
    for q, phi in zip(charges, q_phis):
        sum_torque += torque(phi, q)
    return sum_torque


def align_dipole(phis, charges):
    # Rotate the sphere so dipole points in +x direction
    charge_vectors = np.array([np.cos(phis), np.sin(phis)])
    dipole_vec = np.sum(charge_vectors * charges, axis=1)
    x_vec = np.array([1, 0])
    angle = np.arccos(np.dot(dipole_vec, x_vec) /
                      (np.linalg.norm(dipole_vec)*np.linalg.norm(x_vec)))
    if dipole_vec[1] < 0:
        angle *= -1
    phis -= angle
    return phis


def generate_starting_pt(Q_total, n_charges):
    # Random position and charge adding to total
    phis = np.random.uniform(0, 2*np.pi, n_charges)
    # To generate random charges that add to total charge
    charges = np.diff(np.sort(np.append(np.random.choice(np.linspace(
        0, Q_total, int(1E6)), size=n_charges-1), np.array([0, Q_total]))))
    charges = np.array([0.5,0.5])
    phis = align_dipole(phis, charges)
    return phis, charges


def experiment(Q_total, n_charges, display=True):
    # Total charge on sphere Q_total in nC
    # Number of charges n_charges

    # Reduntant code for uniformly spaced charges.
    # -------------------------------------------------
    #charges = (Q_total/n_charges) * np.ones(n_charges)
    #phis = np.linspace(0, 2*np.pi, n_charges+1)[:-1]

    # Given a total charge, randomly generate charges and positions
    phis, charges = generate_starting_pt(Q_total, n_charges)
    # Calculate the torque
    T_calc = total_torque(phis, charges)
    # Calculate the dipole and dipole length
    dipole = np.abs(np.sum(R * charges * np.cos(phis)))
    dipole_length = dipole/Q_total

    print("phi values : ", phis)
    print("num charges : ", n_charges)
    print("total_charge nC: ", Q_total)
    print("dipole nCmm: ", 1E3*dipole)
    print("dipole length (mm): ", 1E3*dipole/Q_total)
    print("Max_torque (Nm): ", np.max(T_calc))

    if display:
        display_output(theta, T_calc, phis)

    output = [Q_total, n_charges, dipole, dipole_length, np.max(T_calc)]
    return output


def display_output(theta, T_calc, phis):
    fig, ax = plt.subplots(2, 1)
    ax[1].plot(theta, T_calc, 'r-')
    ax[1].set_xlabel('theta')
    ax[1].set_ylabel('torque')
    ax[0].plot([R+t, R+t], [-R, R], 'k')
    ax[0].plot([R, R], [-R, R], 'k')
    ax[0].plot(R*np.cos(theta), R*np.sin(theta), 'b')
    ax[0].plot(R*np.cos(phis), R*np.sin(phis), 'ro')
    ax[0].set_aspect('equal')
    plt.show()


if __name__ == '__main__':
    import pandas as pd

    Q_mean = 3
    #charges = np.random.normal(Q_mean, 0.75, size=1000)
    charges=np.array([1,1])
    n_charges = 2
    df = pd.DataFrame(columns=['Q_total', 'n_charges','dipole', 'dipole_length', 'max_torque'])
    for charge in charges:
        output = experiment(charge, n_charges, display=False)
        df.loc[len(df), :] = output
    df.to_csv('output.csv')

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(df['Q_total'], df['max_torque'], 'b.')
    ax[0].set_xlabel('Q_total')
    ax[0].set_ylabel('Torque_max')
    ax[1].plot(df['dipole'], df['max_torque'], 'b.')
    ax[1].set_xlabel('dipole')
    ax[1].set_ylabel('Torque_max')
    ax[2].plot(df['Q_total'], df['dipole'], 'b.')
    ax[2].set_xlabel('charge')
    ax[2].set_ylabel('dipole')

    dipole_lengths = [1.55, 1.62, 1.01, 4.63, 3.68, 2.14, 2.54, 1.95]
    bin_edges = [0, 1, 2, 3, 4, 5]

    fig, ax = plt.subplots(2, 1)
    ax[0].hist(df['dipole_length'])
    ax[1].hist(dipole_lengths, bin_edges, density=True)

    plt.show()
