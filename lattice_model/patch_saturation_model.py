import numpy as np
import matplotlib.pyplot as plt


def proportional(qs, qp):
    q_tot = qs+qp
    if Q_max < q_tot:
        qs = qs*(Q_max/q_tot)
        qp = qp*(Q_max/q_tot)
    return qs, qp


def split_diff(qs, qp):
    q_tot = qs+qp
    if Q_max < q_tot:
        q_subtract = (q_tot-Q_max)/2
        qs -= q_subtract
        qp -= q_subtract
    return qs, qp


if __name__ == '__main__':
    Q_max = 10
    q_s = 0
    q_p = 5
    dq = 0.1
    plt.figure(1)

    for i in range(500):
        q_s += dq
        q_p += dq
        q_s, q_p = proportional(q_s, q_p)
        #q_s, q_p = split_diff(q_s, q_p)
        print(q_s, q_p)
        plt.plot(i, q_s, 'ro')
        plt.plot(i, q_p, 'bo')

    plt.show()
