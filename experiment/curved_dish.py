import numpy as np
import matplotlib.pyplot as plt
from dataanalysis.fitting import Fit
from dataanalysis.fit_models import gaussian, flipped_exponential

def histogram_plot(data, name='', xmin=0, xmax=6,bins=20):
    print(name)
    print('mean = ', np.mean(data))
    print('std = ', np.std(data))
    fig = plt.figure()
    freq, binedges, _ =plt.hist(data, bins=bins,range=(xmin,xmax))
    bins = 0.5*(binedges[1:]+binedges[:-1])
    f=Fit('gaussian', x=bins, y=freq)
    f.add_params([5,np.mean(data), np.std(data)])
    _, x1,y1 = f.fit(interpolation_factor=0.1)
    #plt.plot(x1, y1, 'r-')
    #plt.title(name)
    return x1,y1, fig 

#5s
gamma2_5_5s=np.array([1.11,1.18,1.14,0.93,1.08,0.74,0.83,0.65,1.45,1.02])

#10s
gamma2_5_10s = np.array([1.31,1.66,1.06, 2.28, 1.99, 1.74, 1.84, 2.04, 1.80, 1.11])

#17s
gamma2_5_17s=np.array([2.99,2.60,2.89,2.97,3.51,3.29,2.77,4.08,2.51,2.62])

#30s
gamma2_5_30s = np.array([3.12,3.26,3.37,2.93,2.79,4.21,3.89,3.03,3.40,3.42])

#1min
gamma2_5_1min = np.array([3.97, 4.02, 3.49,3.76,3.62, 3.09, 4.28, 4.54, 4.16, 2.68,
                          3.60, 3.05, 3.03, 3.80, 3.97, 3.91, 3.54, 3.01,3.06, 3.12])

#5mins
gamma2 = np.array([2.7,3.32,2.85,2.87,3.12,3.2,2.74,2.76,3.82, 3.12,
                    2.41,3.27,2.81,2.39,2.76,2.66,2.78,3.00,3.56,2.58,
                    2.34,3.16,3.27,3.76,2.82,2.02,2.45,3.31,2.39,2.55,
                    4.00,3.22,2.72,1.77,2.88,2.54,1.96,2.49,2.40,2.21,
                    2.70,2.95,3.13,3.66,2.59,4.09,2.00,3.80,3.13,3.09])
gamma2_5 = np.array([4.83,3.13,4.67,4.08,2.64,2.21,3.03,3.70,4.27,3.66,
                    4.62,2.69,3.53,3.33,2.91,3.89,3.45,3.32,2.17,3.21,
                    4.71,4.03,4.08,3.92,4.29,3.53,5.01,3.09,4.14,3.68,
                    3.20,3.66,3.53,3.56,2.83,3.31,4.86,4.57,3.57,4.10,
                    3.74,4.02,3.50,2.69,3.94,4.09,3.92,3.80,1.92,4.01,
                    3.35,3.56,3.13,3.19,4.09,3.69,3.21,4.99,3.62,3.75,
                    3.32,3.26,3.90,3.87,4.26,3.33,3.23,4.01,3.27,3.60,
                    ])

gamma3 = np.array([5.58,4.55,4.30,4.38,5.18,5.01,4.69,5.11,4.04,5.49])

#10mins
gamma2_5_10min = np.array([2.29,3.8,3.76,3.09,4.14,3.19,2.96,3.38,3.53,3.24,
                        4.05,3.61,3.28,3.73,4.37,3.73,3.96,4.7,3.6,4.71])

#continuous no washing
gamma2_5_3min = np.array([3.61, 3.73, 4.16, 4.89, 3.94, 3.56,4.33,4.08,5.40])     
gamma2_5_5min = np.array([4.12,3.92,4.54,4.22,3.22,4.04,4.42,4.22,3.77,3.69])                      

#x1,y1,fig1 = histogram_plot(gamma2_5_1min, 'gamma2.5 1min')
#x2,y2,fig2 = histogram_plot(gamma2, 'gamma2.0 5mins')
#x3,y3,fig3 = histogram_plot(gamma2_5, 'gamma2.5 5mins')
#x4,y4,fig4 = histogram_plot(gamma2_5_10min, 'gamma2.5 10mins')
#x5,y5,fig5 = histogram_plot(gamma2_5_10s, 'gamma2.5 10s')
#x6,y6,fig6 = histogram_plot(gamma2_5_30s, 'gamma2.5 30s')
#x7,y7,fig7 = histogram_plot(gamma2_5_5s, 'gamma2.5 5s')
#x8,y8,fig8 = histogram_plot(gamma2_5_5s, 'gamma2.5 5s')

#waggler
gamma2_5_10s_wag = np.array([2.14, 2.11, 2.59, 2.24, 2.37, 2.11, 2.03, 2.36, 1.63, 2.43])
gamma2_5_17s_wag = np.array([4.60, 4.44, 3.64, 4.86, 3.99, 3.47, 4.23, 3.77, 5.06, 4.05])
gamma2_5_30s_wag = np.array([5.11, 6.11, 6.67, 6.60, 5.90, 6.44, 6.69, 6.17, 6.36, 6.36])
gamma2_5_1min_wag = np.array([7.03, 6.88, 5.68, 7.11, 7.68, 7.28, 6.08, 6.61, 6.74, 6.65])
gamma2_5_5min_wag = np.array([7.81, 7.86, 7.63, 7.14, 7.8, 7.78, 7.26, 7.69, 6.97, 7.01])
gamma2_5_10min_wag = np.array([7.97, 8.17, 7.99, 8.04, 8.38, 8.73, 8.51, 8.19, 7.69, 7.82])


#spotty
gamma2_5_10s_spotty = np.array([1.32,1.84,1.55,1.32,1.30,0.86,1.31,0.94,0.97,1.17])
gamma2_5_5min_spotty = np.array([3.76, 3.76, 2.79, 3.55, 3.94, 4.73, 2.87, 4.20, 4.67, 4.03])

#plt.figure(fig2)
#plt.plot(x3,y3,'g-')
#plt.figure(fig3)
#plt.plot(x2,y2,'g-')
#plt.figure(fig1)
#plt.plot(x3,y3*np.size(gamma2_5_1min)/np.size(gamma2_5),'g-')
#plt.figure(fig5)
#plt.plot(x3,y3*np.size(gamma2_5_10s)/np.size(gamma2_5),'g-')
#plt.figure(fig6)
#plt.plot(x3,y3*np.size(gamma2_5_10s)/np.size(gamma2_5),'g-')
#plt.figure(fig7)
#plt.plot(x3,y3*np.size(gamma2_5_10s)/np.size(gamma2_5),'g-')
#plt.figure(fig8)
#plt.plot(x3,y3*np.size(gamma2_5_10s)/np.size(gamma2_5),'g-')

t=np.array([0, 5, 10, 17, 30, 60, 300, 600])
Q_av=np.array([0, np.mean(gamma2_5_5s),np.mean(gamma2_5_10s), np.mean(gamma2_5_17s),np.mean(gamma2_5_30s),np.mean(gamma2_5_1min), np.mean(gamma2_5), np.mean(gamma2_5_10min)])
Q_std=np.array([0, np.std(gamma2_5_5s),np.std(gamma2_5_10s), np.std(gamma2_5_17s),np.std(gamma2_5_30s),np.std(gamma2_5_1min), np.std(gamma2_5), np.std(gamma2_5_10min)])
Q_max=np.array([0, np.max(gamma2_5_5s),np.max(gamma2_5_10s), np.max(gamma2_5_17s),np.max(gamma2_5_30s),np.max(gamma2_5_1min), np.max(gamma2_5), np.max(gamma2_5_10min)])
Q_min=np.array([0, np.min(gamma2_5_5s),np.min(gamma2_5_10s), np.min(gamma2_5_17s),np.min(gamma2_5_30s),np.min(gamma2_5_1min), np.min(gamma2_5), np.min(gamma2_5_10min)])


#waggle
t_wag=np.array([0, 10, 17, 30, 60, 300, 600])
Q_av_wag=np.array([0, np.mean(gamma2_5_10s_wag), np.mean(gamma2_5_17s_wag), np.mean(gamma2_5_30s_wag), np.mean(gamma2_5_1min_wag), np.mean(gamma2_5_5min_wag), np.mean(gamma2_5_10min_wag)])
Q_std_wag=np.array([0, np.std(gamma2_5_10s_wag), np.std(gamma2_5_17s_wag),np.std(gamma2_5_30s_wag), np.std(gamma2_5_1min_wag), np.std(gamma2_5_5min_wag), np.std(gamma2_5_10min_wag)])
Q_max_wag=np.array([0, np.max(gamma2_5_10s_wag), np.max(gamma2_5_17s_wag),np.max(gamma2_5_30s_wag),np.max(gamma2_5_1min_wag), np.max(gamma2_5_5min_wag), np.max(gamma2_5_10min_wag)])
Q_min_wag=np.array([0, np.min(gamma2_5_10s_wag), np.min(gamma2_5_17s_wag),np.min(gamma2_5_30s_wag),np.min(gamma2_5_1min_wag), np.min(gamma2_5_5min_wag), np.min(gamma2_5_10min_wag)])

print("Q_av 2_5g  {Q_av} +/- {Q_std}, max {Q_max} min {Q_min}")
print("Q_av_waggler 2_5g  {Q_av_wag} +/- {Q_std_wag}, max {Q_max_wag} min {Q_min_wag}")
print("Q_av 2g  {} +/- {Q}, max {} min {}",np.mean(gamma2),np.std(gamma2),np.max(gamma2),np.min(gamma2))
print("Q_av 3g  {} +/- {Q}, max {} min {}",np.mean(gamma3),np.std(gamma3),np.max(gamma3),np.min(gamma3))



#spotty
t_spotty=np.array([10,300])
Q_av_spotty=np.array([np.mean(gamma2_5_10s_spotty),np.mean(gamma2_5_5min_spotty)])
Q_std_spotty=np.array([np.std(gamma2_5_10s_spotty),np.std(gamma2_5_5min_spotty)])
Q_max_spotty=np.array([np.max(gamma2_5_10s_spotty),np.max(gamma2_5_5min_spotty)])
Q_min_spotty=np.array([np.min(gamma2_5_10s_spotty),np.min(gamma2_5_5min_spotty)])

print("Q_av_spotty  {Q_av_spotty} +/- {Q_std_spotty}, max {Q_max_spotty} min {Q_min_spotty}")

f2=Fit('flipped_exponential', x=t, y=Q_av)
f2.add_params([np.max(Q_av),1/60,0,0 ])
_,t_fit, Q_fit=f2.fit(interpolation_factor=0.01)

f3=Fit('flipped_exponential', x=t_wag, y=Q_av_wag)
f3.add_params([np.max(Q_av_wag),1/60,0,0 ])
_,t_fit_wag, Q_fit_wag=f3.fit(interpolation_factor=0.01)

plt.figure(9)
plt.title('Charging')
#gamma2_5_static
plt.errorbar(t, Q_av, yerr=Q_std,linestyle='',capsize=5,ecolor='k', marker='o', mfc='red',
         mec='black', ms=10, mew=2)
plt.errorbar(t, Q_av, yerr=Q_std, linestyle='',capsize=5,ecolor='k', marker='o', mfc='red',
         mec='black', ms=10, mew=2)
plt.plot(t, Q_max, 'k--')
plt.plot(t, Q_min, 'k--')
plt.plot(t_fit, Q_fit, 'r')

#gamma2_5_waggler
plt.errorbar(t_wag, Q_av_wag, yerr=Q_std_wag, linestyle='', capsize=5,ecolor='k', marker='o', mfc='white',
         mec='black', ms=10, mew=2)         
plt.plot(t_fit_wag, Q_fit_wag, 'r')         
plt.plot(t_wag, Q_max_wag, 'k-')         
plt.plot(t_wag, Q_min_wag, 'k-')         

plt.errorbar([300],[np.mean(gamma2)], yerr=[np.std(gamma2)], capsize=5,ecolor='k', marker='o', mfc='blue',
         mec='black', ms=10, mew=2)
plt.errorbar([300],[np.mean(gamma3)], yerr=[np.std(gamma3)], capsize=5,ecolor='k', marker='o', mfc='green',
         mec='black', ms=10, mew=2)
plt.errorbar([180, 300],[np.mean(gamma2_5_3min), np.mean(gamma2_5_5min)], yerr=[np.std(gamma2_5_3min),np.std(gamma2_5_5min)], capsize=5,ecolor='k', marker='o', mfc='yellow',
         mec='black', ms=10, mew=2)

plt.errorbar(t_spotty,Q_av_spotty, yerr=Q_std_spotty, capsize=5,ecolor='k', marker='o', mfc='magenta',
         mec='black', ms=10, mew=2)

plt.xlabel('Time (s)')
plt.xlim([0, np.max(t)+50])
plt.ylim([0, np.max(gamma2_5_10min_wag)])
plt.ylabel('Q (nC)')
plt.show()

