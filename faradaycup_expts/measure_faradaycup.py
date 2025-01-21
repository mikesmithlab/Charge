import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import os
import threading
import time


from labequipment.picoscope import PicoScopeDAQ
from filehandling import datetime_stamp, BatchProcess, get_filename
from dataanalysis.fitting import Fit


def beep():
    print("ready")
    time.sleep(1)
    print("steady")
    time.sleep(1)
    print("go")


def func(x,a,b,c):
    return a*np.exp(-b*x)+c


def analyse_charge(file):
    df = pd.read_csv(file)

    #Numerical integration
    df['baseline_shifted'] = df['channelA'] - (df['channelA'][:rolling*5].mean() + df['channelA'][-rolling*5:].mean())/2
    df['mean'] = df['baseline_shifted'].rolling(rolling, center=True).mean()
    dt = df['time'][1]-df['time'][0]
    df['I_nA'] = calib*df['mean']
    Q = df['I_nA'].sum()*dt
    print("Q={}nC".format(Q))
    
    #Fitting charge curve
    max_index = df['I_nA'].abs().argmax()
    time_data = df['time'][df.index > max_index] - df['time'][df.index == max_index].values
    time_data=time_data[:-1000]
    fit_data = df['I_nA'][df.index > max_index][:-1000]
    a, b, c = curve_fit(func, time_data, fit_data)[0]
    Q = a/b
    print("Q_fit={}nC".format(Q))

    



    plt.figure(2)
    plt.title('Numerical')
    plt.plot(df['time'], calib*df['mean'], 'r-')
    plt.ylabel('I (nA)')
    plt.xlabel('time (s)')
    plt.figure(1)
    plt.title('Fit')
    plt.plot(df['time'], df['I_nA'], 'b-')
    plt.plot(time_data + df['time'][df.index == max_index].values, func(time_data, a, b, c), 'g--')
    plt.ylabel('I (nA)')
    plt.xlabel('time (s)')
    plt.show()


def collect_charge_measurement(pico, output_filename=""):

    # Set up the Picoscope DAQ
    time_s, channelA, _ = pico.start_streaming(collect_time=collect_time)

    # channelA=channelA[:-int(collect_time*fs)]
    df = pd.DataFrame({'time': time_s, 'channelA': channelA})
    # df['smooth'] = df['channelA'].rolling(window=rolling, center=True).mean()
    print('Measurement made')

    df.to_csv(output_filename, index=False)


if __name__ == '__main__':
    datafile_path = "U:/GranularCharge/GloveBox/ChargeMeasure/"

    measure = False

    rolling = 1000
    calib = 17.36*2  # converts measured voltage to current
    collect_time = 9
    fs = 100000

    if measure:
        folder = datetime_stamp(format_string="%Y%m%d")
        if not os.path.exists(datafile_path + folder):
            os.mkdir(datafile_path + folder)

        filename = datafile_path + folder + '/' + datetime_stamp() + '.csv'

        x = threading.Thread(target=beep)
        pico = PicoScopeDAQ()
        pico.setup_channel(voltage_range=5, coupling='DC', sample_rate=fs)
        x.start()
        collect_charge_measurement(pico, output_filename=filename)
        pico.close_scope()

    else:
        filename = get_filename(initialdir=datafile_path,
                                filetypes=(("csv", "*.csv"),))

    analyse_charge(filename)
