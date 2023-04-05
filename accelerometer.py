from filehandling import BatchProcess, smart_number_sort
import matplotlib.pyplot as plt
import numpy as np
from DataAnalysis.fitting import Fit

def get_g(file, scale = 4.447151614741587, show=False):
    with open(file, 'r') as f: 
        f.readline()
        if f.readline().split('(')[2][:-2] == 'mV':
            scale = scale/1000
    data = np.loadtxt(file,delimiter=',',skiprows=3)
    
    time, amp = data[:,0], data[:,1]
    logic = np.isfinite(amp)
    amp=amp[logic]
    time=time[logic]
    rms = np.sqrt(np.nanmean(amp**2))
    g = rms*scale
    
    fitobj = Fit('sin_cos',time, amp)
    fitobj.fit()
    #fitobj.plot()


    if show:
        plt.figure(1)
        plt.plot(time, amp)
        plt.plot([time[0],time[-1]],[1/scale,1/scale],'g--')
        plt.show()
    return g

if __name__ == '__main__':
    path = "W:\\Alice\\accelerometer_mike\\"

    v = []
    g = []

    for file in BatchProcess(path + '*mv.csv',smart_sort=smart_number_sort): 
        voltage = int(file.split('\\')[-1][:-6])
        v.append(voltage)
        g.append(get_g(file))
  
    

    plt.figure(2)
    plt.plot(v,g,'rx')
    plt.show()

    #rms = get_rms(path + file3 + '.csv')

    #g = rms*scale
    #print(g)
    #print(rms_up, rms_down)
    

    



    