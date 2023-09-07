from labequipment import daq
import matplotlib.pyplot as plt
import numpy as np
#Make a Change


def smooth(data, kernel_size=2000):
    kernel = np.ones(kernel_size) / kernel_size
    data_convolved = np.convolve(data, kernel, mode='same')
    return data_convolved


def measure(filename=None, t=40, voltage=1.0, calibration_scale=9.71):
    aio = daq.analog(reset=True)
    aio.addInput(0)
    # aio.addOutput(0)
    aio.Rate = 10000
    aio.Nscans = aio.Rate*t
    print('get ready...')
    data, timestamps = aio.read()

    print('analyse')
    # aio.aoRate = 1000
    # output_signal = voltage
    # aio.write(output_signal)
    data_smoothed = smooth(data * calibration_scale)
    time = np.linspace(0, t, aio.Nscans)
    bg = np.mean(data_smoothed[:aio.Rate])
    np.savetxt(filename, np.c_[time, data_smoothed])
    dt = time[1] - time[0]



    charge1 = np.sum(data_smoothed[1000:10*aio.Rate] - bg) * dt
    charge2 = np.sum(data_smoothed[1000:20 * aio.Rate] - bg) * dt
    charge = np.sum(data_smoothed[1000:-1000] - bg) * dt
    print('charge t=10', charge1, 'nC')
    print('charge t=20', charge2, 'nC')
    print('charge t=inf', charge, 'nC')
    plt.figure()
    plt.plot(time[1000:-1000], data_smoothed[1000:-1000] - bg)
    plt.plot([0,t],[0,0],'r-')
    plt.xlabel('time (s)')

    plt.figure(2)
    plt.plot(time, data, 'b-')

    plt.show()



expt_name = '2022_01_25_FaradayCup'
path = 'Z:\\GranularCharge\\ChargeProject\\'+ expt_name+'\\'
filename = expt_name + '_faraday_danglingbead20mins.txt'

measure(filename=path + filename)
