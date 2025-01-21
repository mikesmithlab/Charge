import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os

from labequipment.picoscope import PicoScopeDAQ
from filehandling import datetime_stamp, BatchProcess


def collect_charge_measurement(pico, output_filename=""):
    
    # Set up the Picoscope DAQ
    times, channelA, _ = pico.start_streaming(collect_time=collect_time)
    #channelA=channelA[:-int(collect_time*fs)]
    df = pd.DataFrame({'time': times, 'channelA': channelA})
    df['smooth'] = df['channelA'].rolling(window=rolling, center=True).mean()
    print('Measurement made')
    print(df.head(n=rolling))
    print(output_filename[:-4] + datetime_stamp() + '.csv')
    df.to_csv(output_filename[:-4] + datetime_stamp() + '.csv', index=False)


    plt.figure()
    plt.plot(times, df['channelA'], 'b.')
    plt.plot(times, df['smooth'], 'r.')
    plt.pause(10)
    plt.gcf()
    plt.close()

def check_for_new_measurement_file(filestub, n):
    f = BatchProcess(filestub)
    print(f.files)
    while f.num_files < n:
        time.sleep(0.2)
        f = BatchProcess(filestub)
    return True




if __name__ == '__main__':
    datafile_path = "C:/Users/ppzmis/Documents/Charge_Measurements/charge_contact/datafiles/"
    n_measurements=100 
    rolling = 1000
    collect_time=9
    fs = 100000
    
    pico = PicoScopeDAQ()
    pico.setup_channel(voltage_range=5, coupling='DC', sample_rate=fs)

    for file in BatchProcess(datafile_path+'*'):
        os.remove(file)


    for n in range(n_measurements):
        check_for_new_measurement_file(datafile_path + 'force*', n+1)
        collect_charge_measurement(pico, output_filename=datafile_path+"charge.csv")
    
    pico.close_scope()
    
