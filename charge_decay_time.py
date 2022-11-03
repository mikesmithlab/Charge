import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = 'Z:/GranularCharge/ChargeProject/'
expt_name = '2011_11_10_nocharge'

output_path = path + 'decay_graphs/'
file = path + expt_name + '/' + expt_name + '.hdf5'
print(file)

df = pd.read_hdf(file)
print(df.head(n=100))
#parameters
d=0.13 # gap between plates
V = 5000  # voltage
m=7E-4  # mass particle in kg
bead_diameter_m = 1e-2
bead_diam_pixels = 100
scale = bead_diameter_m / bead_diam_pixels  # image scale
l=1 # length cotton.
dt = 120  # time between pictures in s
g=9.81

df.index.name = 'index'
times = dt*df.index.values


df['t'] = times
E = V/d

x=df['x'].values
y=df['y'].values
print(x)
x0 = x[0]
y0 = y[0]

df['dx'] = df['x'] - x0
df['dy'] = df['y'] - y0
print(df['dx'])

df['Q'] = m*g*df['dx']*scale/(l*E)


#plt.semilogy(df['t'],-df['Q'],'x')
fig, (ax1,ax2) = plt.subplots(2,1)


ax1.plot(df['t']/60,df['Q'],'bx')
ax1.set_xlabel('t')
ax1.set_ylabel('Q')
ax1.set_title(expt_name)
ax2.plot(df['t']/60, df['dy'],'rx')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
plt.savefig(output_path + expt_name + '.png')
plt.show()

