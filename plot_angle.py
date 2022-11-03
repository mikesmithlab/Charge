import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import datestr2num
import numpy as np


def fft_power_spectrum(ydata, fps):
    """
    Calculates the power spectrum on a 1D signal. It also obtains
    a quick estimate of the peak freq.
    :param tdata: time series
    :param ydata: signal to fft
    :param limits: adjusts displayed freq axis. Either None - no limits
                    or a tuple (lower limit, upper limit)
    :param show: set to True if you want to visualise the fft
    :return: 3 part tuple (freqs,powerspectrum amplitudes, peak freq)
    """
    y_fft = np.fft.fft(ydata)
    powers = y_fft*np.conjugate(y_fft)
    #freqs =fps* np.linspace(0,ydata.size-1,ydata.size)/np.size(ydata)
    freqs = np.fft.fftfreq(np.size(ydata), d=(1/fps))

    return  freqs, powers


path = 'Z:\\GranularCharge\\pingpong\\angles\\2022_08_23\\'
expt_name = 'purgeonoff'
filename = path + expt_name + '.txt'

data = np.genfromtxt(filename,delimiter=',')
print(np.shape(data))
fps = 0.75
t=(1/fps)*np.linspace(0,np.size(data)-1,np.size(data))
angle = data

#t = (1/fps)*np.linspace(0,200000-1, 200000)

#angle = np.sin(2*np.pi*0.00025*t) + np.sin(2*np.pi*0.001*t)

freqs, powers=fft_power_spectrum(angle, fps)


period = np.linspace(0, (1/fps)*powers.size-1, powers.size)

plt.figure(1)
plt.plot(t/3600,angle,'r-')
plt.plot(t/3600,angle,'b.')
plt.savefig(path + expt_name + 'angle.png')

plt.figure(2)
plt.semilogy(freqs[1:], powers[1:],'r-')
plt.xlim([0,0.1])


plt.show()
