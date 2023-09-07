import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import datestr2num

path = 'Z:\\GranularCharge\\siliconeballs\\2022_09_16\\'
expt_name = 'Try2'
filename = path + expt_name + '_temphumidity.txt'

data = np.genfromtxt(filename,delimiter=',')

t = data[:,0]/3600
temp = data[:,2]
humidity = data[:,3]

plt.figure()
plt.plot(t,temp,'r-')
plt.plot(t,humidity,'b-')
plt.savefig(path + expt_name + 'temphumidity.png')
plt.show()
