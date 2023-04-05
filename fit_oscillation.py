from DataAnalysis.fitting import Fit
from DataAnalysis.fit_models import decay_oscillation
import numpy as np
import matplotlib.pyplot as plt


pathname = 'C:\\Users\\ppzmis\\Documents\\'
filename = 'bluebead_torsion3.csv'
data=np.loadtxt(pathname+filename,delimiter=',',encoding='utf-8')
time = data[:,0]
width = data[:,1]

width = width[time > 15]
time = time[time > 15]

#plt.figure()
#plt.plot(time,width)
#plt.show()

fit = Fit('decay_oscillation',time,width,xlabel='time',ylabel='width projection')
fit.add_params([-1/9,260,2,0,96])
fit.fit()
fit.plot_fit(show=True)