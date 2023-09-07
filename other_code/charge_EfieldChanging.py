import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


expt_name = '2021_11_15_2beads'
path = 'Z:/GranularCharge/ChargeProject/'


df = pd.read_hdf(path + expt_name + '/' + expt_name + '.hdf5')

#parameters
d=0.1 # gap between plates
increment=250 # increment of voltage in V
m=3E-4  # mass particle in kg
scale = 8e-3/(915-679)  # image scale
l=1 # length cotton.


df.index.name = 'index'
indices = df.index.values
voltages = np.append(indices[indices <= 20],40 - indices[indices > 20])

df['voltages'] = voltages
df['E'] = increment*df['voltages']/d
df['E']

filtered_df = df[(df.y >220) & (df.y < 245)]
filtered_df.head(n=40)

plt.plot(filtered_df['E'],filtered_df['x']*scale,'x')
plt.ylabel('x (m)')
plt.xlabel('E (Vm^-1)')

grad=-0.023/38000
charge = (m*9.81/l)*grad
charge